"""Bounded CPU architecture engineering; no screening/Confirmation entry point."""

import copy
import hashlib
import json
import statistics
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import torch
from experiments.common.models import GlobalRouter
from minicells_clm002.engine import Engine as BaseEngine
from minicells_clm002.native import build
from minicells_fast001.parent import substrate
from minicells_r3e.common import fingerprint
from minicells_r3f.diagnostics import calibrate
from minicells_r3k.reuse import bind
from torch import nn
from torch.nn import functional as F

BASE = Path(__file__).resolve().parents[1]


def manifest():
    return json.loads((BASE / "EXPERIMENT_MANIFEST.json").read_text())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def tensor_hash(x):
    return fingerprint({"value": x})


def tiny_data(tokenizer):
    """Different template families and literal facts; original tokenizer, no old split."""
    templates = {
        "train": [
            "A toy box has {a} stones and {b} stones. Together there are {c} stones.",
            "Lena walked near a river and saw {a} small birds beside {b} trees.",
        ],
        "development": [
            "Count these beads: {a} plus {b} gives {c}.",
            "At sunset Omar carried {a} red books through {b} quiet rooms.",
        ],
        "calibration": [
            "Mira keeps {a} shells; she adds {b}; the new count is {c}.",
            "Beyond the garden {a} rabbits rested near {b} blue flowers.",
        ],
        "confirmation": [
            "The sum of {a} and {b} equals {c}.",
            "When rain stopped, {a} children found {b} tiny boats in a pond.",
        ],
    }
    rows, identities = {}, {}
    seen = set()
    for index, (role, family) in enumerate(templates.items()):
        rows[role], identities[role] = [], []
        for text in family:
            text = text.format(a=11 + index * 13, b=7 + index * 17, c=18 + index * 30)
            ids = tokenizer.encode(text).ids
            # Teacher-forced fixed input/target construction, never padding or packed QA.
            tokens = torch.tensor(ids[:9], dtype=torch.long)
            if len(tokens) != 9:
                raise RuntimeError("ENGINEERING_TEMPLATE_TOO_SHORT")
            hashes = {
                "raw": digest(text.encode()),
                "normalized": digest(
                    " ".join(unicodedata.normalize("NFKC", text).casefold().split()).encode()
                ),
                "window": tensor_hash(tokens),
                "input": tensor_hash(tokens[:-1]),
                "target": tensor_hash(tokens[1:]),
            }
            for kind, value in hashes.items():
                if (kind, value) in seen:
                    raise RuntimeError("ENGINEERING_SPLIT_DUPLICATE")
                seen.add((kind, value))
            rows[role].append(
                {"x": tokens[:-1][None], "y": tokens[1:][None], "valid": torch.ones(1, 8)}
            )
            identities[role].append(hashes)
    return rows, {
        "scope": "ENGINEERING_ONLY_NO_SCIENTIFIC_REGISTRATION",
        "template_sha256": digest(json.dumps(templates, sort_keys=True).encode()),
        "rows": identities,
        "confirmation_quality_reads": 0,
        "exact_hash_intersections": 0,
    }


class RoleStore:
    def __init__(self, rows):
        self._rows = rows
        self.accesses = []

    def read(self, role):
        if role == "confirmation":
            raise RuntimeError("CONFIRMATION_SEALED")
        if role not in {"train", "development", "calibration"}:
            raise ValueError("UNKNOWN_DATA_ROLE")
        self.accesses.append(role)
        return self._rows[role]


class Reference(nn.Module):
    """Common native input/context/head; global shared expert pool at every depth."""

    def __init__(self, native, arm, seed):
        super().__init__()
        if arm not in {"MoE", "Dense-AllCell-Mean"}:
            raise ValueError("UNKNOWN_ARCHITECTURE")
        self.native = copy.deepcopy(native)
        self.arm = arm
        del self.native.connection
        del self.native.gate
        for p in self.native.parameters():
            p.requires_grad_(False)
        for p in self.native.cells.parameters():
            p.requires_grad_(True)
        self.depth = 8
        self.last_calls = []
        if arm == "MoE":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 91000)
                self.router = GlobalRouter(
                    native.c["model"]["width"], native.c["model"]["cell_hidden"], 6, 2
                )
            self.router.cells = self.native.cells
            self.router.collect_stats = True

    @property
    def cells(self):
        return self.native.cells

    def forward(self, x):
        h, _ = self.native.initial(x)
        self.last_calls = []
        aux = h.new_zeros(())
        if self.arm == "MoE":
            self.router.reset_stats()
        for _ in range(self.depth):
            z = self.native.norm(h)
            if self.arm == "MoE":
                before = (
                    self.router.counts.clone()
                    if self.router.counts is not None
                    else torch.zeros(6, dtype=torch.long)
                )
                delta, balance = self.router(z)
                self.last_calls.append((self.router.counts - before).tolist())
                aux = aux + balance
            else:
                delta = torch.stack([cell(z) for cell in self.native.cells]).mean(0)
                self.last_calls.append([len(h)] * 6)
            h = h + self.native.alpha_graph * delta
        return SimpleNamespace(
            logits=self.native.head(h.reshape(*x.shape, -1)), balance=aux / self.depth
        )


@torch.no_grad()
def make_models(seed, calibration_inputs):
    """Unit-size only; exact native model constructor, no inherited weights or optimizer."""
    config = copy.deepcopy(substrate())
    config["engineering"].update(vocab_size=4096, sequence_length=8, batch_size=2)
    native, runtime = build(config, seed, unit=True)
    # Unit helper's zero margin is discarded, then the original calibration is applied.
    native.route_switch_margin.fill_(float("inf"))
    native.route_numeric_tolerance.fill_(float("inf"))
    evidence = bind(
        calibrate.__wrapped__, reference=lambda model, config, data: calibration_inputs
    )(native, runtime, None)
    evidence.update(
        source="ZERO_TRAINING_INITIALIZATION_NEW_ENGINEERING_CALIBRATION",
        original_function="minicells_r3f.diagnostics.calibrate",
        formula_changed=False,
    )
    native.margin_evidence = evidence
    native.eval()
    models = {
        "CLM": native,
        "MoE": Reference(native, "MoE", seed),
        "Dense-AllCell-Mean": Reference(native, "Dense-AllCell-Mean", seed),
    }
    return models, runtime, evidence


def sequence_loss(logits, y, valid):
    if (
        logits.shape[:-1] != y.shape
        or y.shape != valid.shape
        or not ((valid == 0) | (valid == 1)).all()
    ):
        raise ValueError("INVALID_TARGET_MASK")
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), y.flatten(), reduction="none"
    ).reshape_as(y)
    if not torch.isfinite(loss).all() or not (valid.sum(-1) > 0).all():
        raise ValueError("NONFINITE_OR_EMPTY_LOSS")
    return {"loss_sum": (loss * valid).sum(-1), "valid_count": valid.sum(-1)}


@torch.no_grad()
def evaluate(model, store, role):
    rows = store.read(role)  # Gate before ANY model call.
    before = fingerprint(model.state_dict())
    result = []
    for row in rows:
        value = sequence_loss(model(row["x"]).logits, row["y"], row["valid"])
        result.append({k: v.tolist() for k, v in value.items()})
    assert fingerprint(model.state_dict()) == before
    return {
        "per_sequence": result,
        "nll": sum(sum(r["loss_sum"]) for r in result) / sum(sum(r["valid_count"]) for r in result),
    }


def initialization_report(models):
    pools, public, storage = {}, {}, {}
    for arm, model in models.items():
        body = model if arm == "CLM" else model.native
        pools[arm] = fingerprint(body.cells.state_dict())
        public[arm] = fingerprint(
            {
                n: p
                for n, p in body.state_dict().items()
                if not n.startswith(("cells.", "connection.", "gate."))
            }
        )
        storage[arm] = {p.data_ptr() for p in body.parameters()}
    independent = all(not storage[a] & storage[b] for a in models for b in models if a != b)
    return {
        "cell_hashes": pools,
        "public_hashes": public,
        "storage_independent": independent,
        "history_updates": 0,
        "source_checkpoint": None,
        "passed": len(set(pools.values())) == len(set(public.values())) == 1 and independent,
    }


def profile(model, row):
    if any(p.is_cuda for p in model.parameters()):
        raise RuntimeError("STAGE1_CPU_ONLY")
    counts, flops = [], [0]
    handles = []
    for module in model.modules():
        if isinstance(module, nn.Linear):

            def hook(layer, args, out):
                flops[0] += 2 * out.numel() * layer.in_features

            handles.append(module.register_forward_hook(hook))
    try:
        start = time.perf_counter()
        with torch.no_grad():
            out = model(row["x"])
        elapsed = time.perf_counter() - start
    finally:
        for handle in handles:
            handle.remove()
    if hasattr(out, "operator_rows"):
        counts = out.operator_rows.tolist()
    else:
        counts = model.last_calls
    # Five CPU forward samples only; no CUDA/decode or steady-state inference claim.
    samples = []
    with torch.no_grad():
        for _ in range(5):
            tick = time.perf_counter()
            model(row["x"])
            samples.append(time.perf_counter() - tick)
    samples.sort()
    routing = sum(
        p.numel()
        for n, p in model.named_parameters()
        if n.startswith(("connection.", "gate.", "router.router."))
    )
    return {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "complete_routing_parameters": routing,
        "cell_calls": sum(map(sum, counts)),
        "unique_active_modules_batch": sum(v > 0 for v in map(sum, zip(*counts))),
        "cell_calls_per_target": sum(map(sum, counts)) / row["y"].numel(),
        "observed_linear_gemm_flops": flops[0],
        "flops_complete": False,
        "flops_excluded": [
            "functional tied LM head",
            "ContextRead QK/AV",
            "norms/softmax/dispatch",
        ],
        "single_cpu_forward_seconds": elapsed,
        "latency_p50_p95": {
            "p50_seconds": statistics.median(samples),
            "p95_seconds": samples[3] + 0.8 * (samples[4] - samples[3]),
            "samples": 5,
            "scope": "CPU_FORWARD_ONLY_NOT_KV_DECODE",
        },
        "peak_device_memory": None,
        "energy": None,
        "coverage": "CPU_FORWARD_ONLY_NO_KV_DECODE_PROFILING; UNIQUE_COUNT_BATCH_NOT_PER_TOKEN",
    }


def neutral_state(model, runtime, root):
    engine = BaseEngine(
        model,
        runtime,
        "BASE",
        {"scope": "CPU_ENGINEERING_ZERO_HISTORY"},
        root / "zero.jsonl",
        unit=True,
    )
    state = engine.state()
    assert state["step"] == 0 and not state["cell_optimizer"]["state"]
    for h in engine._work_handles:
        h.remove()
    return state


class EngineeringReferenceTrainer:
    """At most two CPU fixture updates; never a scientific training launcher."""

    def __init__(self, model, runtime):
        if any(p.is_cuda for p in model.parameters()):
            raise RuntimeError("STAGE1_CPU_ONLY")
        self.model = model
        self.runtime = copy.deepcopy(runtime)
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=runtime["engine"]["training"]["learning_rate"],
            weight_decay=runtime["engine"]["training"]["weight_decay"],
        )
        self.step = 0
        self.uncertain = False
        self.curve = []
        self.wall = 0.0

    def update(self, rows):
        if self.step >= 2 or self.uncertain:
            raise RuntimeError("ENGINEERING_BUDGET_OR_UNCERTAIN_UPDATE")
        if len(rows) != 2 or sum(r["y"].numel() for r in rows) != 16:
            raise RuntimeError("ENGINEERING_TOKEN_CAP")
        tick = time.perf_counter()
        self.uncertain = True
        self.optimizer.zero_grad(set_to_none=True)
        values = []
        for row in rows:
            out = self.model(row["x"])
            loss = sequence_loss(out.logits, row["y"], row["valid"])
            quality = loss["loss_sum"].sum() / loss["valid_count"].sum()
            objective = quality + (0.01 * out.balance if self.model.arm == "MoE" else 0)
            (objective / len(rows)).backward()
            values.append(float(objective.detach()))
        clip = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.runtime["engine"]["training"]["clip"]
        )
        if not torch.isfinite(clip):
            raise RuntimeError("NONFINITE_ENGINEERING_GRADIENT")
        for group in self.optimizer.param_groups:
            group["lr"] = self.runtime["engine"]["training"]["learning_rate"] * min(
                1, (self.step + 1) / self.runtime["engine"]["training"]["warmup_steps"]
            )
        self.optimizer.step()
        self.step += 1
        self.wall += time.perf_counter() - tick
        self.curve.append(
            {
                "step": self.step,
                "objective": sum(values) / len(values),
                "input_sha256": fingerprint([(r["x"], r["y"]) for r in rows]),
            }
        )
        self.uncertain = False

    def state(self):
        if self.uncertain:
            raise RuntimeError("UNCERTAIN_ENGINEERING_UPDATE")
        return copy.deepcopy(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "curve": self.curve,
                "wall": self.wall,
                "runtime_sha256": digest(json.dumps(self.runtime, sort_keys=True).encode()),
            }
        )

    def restore(self, state):
        if state["runtime_sha256"] != self.state()["runtime_sha256"]:
            raise RuntimeError("ENGINEERING_RESTORE_CONFIG_CHANGED")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.step, self.curve, self.wall = (
            state["step"],
            copy.deepcopy(state["curve"]),
            state["wall"],
        )

    def cost(self):
        return {
            "optimizer_calls": self.step,
            "target_tokens": 16 * self.step,
            "wall_seconds": self.wall,
            "gpu_used": False,
            "device": "CPU",
            "uncertain": self.uncertain,
            "random_stream": "NONE_FIXED_ROWS_AND_NO_STOCHASTIC_LAYERS",
        }


def make_clm_trainer(native, runtime, calibration_rows, root):
    """Existing CLM Engine, Q optimizer and full 22-scale admission; CPU fixture cap only."""
    import screening as sc
    from interpolation_audit import Engine

    if any(p.is_cuda for p in native.parameters()):
        raise RuntimeError("STAGE1_CPU_ONLY")
    state = neutral_state(native, runtime, root)
    c = copy.deepcopy(sc.manifest()["runtime_template"])
    c.update(updates=2, batch_size=2, sequence_length=8, targets_per_update=16)
    c["connection"].update(proposal_steps=[2], states=1, policy_steps=1)
    c["stage_b"]["execution_authorized"] = False
    c["diagnostic"]["gate1_execution_authorized"] = False
    runtime = copy.deepcopy(runtime)
    runtime["training"].update(states=1, policy_steps=1)
    refs = [r["x"] for r in calibration_rows]
    c["admission"].update(
        manifest_sha256=fingerprint(refs),
        ordered_input_hashes=[sc.rv.sha(x.squeeze(0).tolist()) for x in refs],
    )
    bundle = {"role": "admission", "manifest_sha256": fingerprint(refs), "inputs": refs}
    identity = {
        "experiment_id": "CLM003-STAGE1-CPU-FIXTURE",
        "seed": native.seed,
        "arm": "CLM",
        "manifest_sha256": sc.rv.sha(c),
    }
    engine = Engine(
        native,
        runtime,
        "B11-H",
        identity,
        root / "clm-cost.jsonl",
        c,
        "B",
        state,
        fingerprint(state),
    )
    engine.r1_cpu_fixture = True
    engine.stage_directory, engine.stage_reports = root / "private", root / "proposals"
    engine.stage_directory.mkdir()
    engine.stage_identity, engine.stage_refs = identity, bundle
    return engine


def clm_cost(engine, candidate_receipt, admission):
    result = dict(engine.journal.summary())
    for key in ("analytic_router_forward_mac", "analytic_cell_forward_mac"):
        result.pop(key, None)
    result.update(
        training_tokens=engine.tokens,
        target_tokens=engine.tokens,
        input_unit="LANGUAGE_TARGET_TOKENS_CPU_ENGINEERING",
        wall_seconds=engine.training_seconds,
        wall_seconds_by_phase=result.pop("cpu_seconds_by_phase"),
        candidate_optimizer_calls=candidate_receipt["actual_optimizer_calls"],
        total_optimizer_calls_main_plus_candidate=result["completed_optimizer_calls"]
        + candidate_receipt["actual_optimizer_calls"],
        candidate_evaluations=len(admission["attempts"]),
        candidate_receipt=candidate_receipt,
        scope="CPU_ENGINEERING_MAIN_AND_SEPARATE_CANDIDATE_JOURNAL",
    )
    return result
