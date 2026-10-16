"""Infrastructure adapter over the existing R3 Trainer/Checkpoint/Event machinery.

No new loss, sampler, model or admission rule. Real scientific execution remains
closed in this engineering release; synthetic fixtures exercise the same state path.
"""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import random
import re
import time
import numpy as np
import torch
from .store import (
    ContinuationError,
    LocalOutputStore,
    MountedInputStore,
    atomic,
    canonical,
    digest,
    read,
)
from clm001_r3 import checkpoint as cp
from clm001_r3.common import BASE, source_sha, canonical_sha256
from clm001_r3.model import Model
from clm001_r3.training import Trainer
from clm001_r3.data import load_assets, batches
from clm001_r3.shadows import Nursery
from clm001_r3.events import pending, process
from clm001_r3.monitoring import bank, record
from clm001_r3.evaluation import evaluate, q_diagnostics

ENGINEERING_ID = "MINICELLS-KAGGLE-CONTINUATION-SMOKE-V1"
PATTERN = re.compile(r"checkpoint-step-(\d{6})(?:-event-(\d{6}))?\.pt$")


def require_source(source):
    if not re.fullmatch(r"[0-9a-f]{40}", source or "") or source != source_sha():
        raise ContinuationError("SOURCE_SHA_MISMATCH", "checkout must equal frozen full SHA")


def check_events(c, step, nursery, identity):
    for key, event in nursery.events.items():
        expected = canonical_sha256(
            {
                "identity": identity,
                "step": event["step"],
                "event": event["event"],
                "base_registry": event["base_registry"],
            }
        )
        if key != expected or event["step"] > step:
            raise ContinuationError("EVENT_STATE_INCONSISTENT", key)
    complete = {v["event"] for v in nursery.events.values() if v["step"] == step}
    if not set(pending(c, step)).issubset(complete):
        raise ContinuationError("EVENT_STATE_INCONSISTENT", "uncommitted lifecycle boundary")


def new_run(c, assets, output, *, source, run_id, experiment_id=ENGINEERING_ID, gates=None):
    require_source(source)
    if not c.get("synthetic"):
        from clm001_r3.common import authorize

        authorize(c, "train-dev", "cuda:0")
        if (
            not gates
            or not gates.get("passed")
            or gates.get("KAGGLE_CROSS_SESSION_RESUME") != "PASS"
        ):
            raise ContinuationError("ENGINEERING_RELEASE_NOT_VERIFIED")
        if gates.get("source_sha") != source or gates.get(
            "runtime_config_sha256"
        ) != canonical_sha256(c):
            raise ContinuationError("PREFLIGHT_EVIDENCE_MISMATCH")
    output = Path(output)
    if output.exists():
        raise ContinuationError("OUTPUT_ALREADY_EXISTS", output)
    plan = c["evaluation"].get("snapshots", [])
    if (
        not c["evaluation"].get("periodic_diagnostics_enabled")
        or not plan
        or plan != sorted(set(plan))
        or plan[0] != 0
        or plan[-1] != c["training"]["steps"]
    ):
        raise ContinuationError("DIAGNOSTIC_PLAN_NOT_FROZEN")
    load_assets(c, assets)
    output.mkdir(parents=True)
    from shutil import copytree, copyfile

    copytree(assets, output / "assets")
    candidate = deepcopy(c)
    runtime = deepcopy(c)
    # Envelope identity is separate and explicit; do not rename scientific R3 arms.
    runtime["engineering_run_id"] = run_id
    if c.get("synthetic"):
        runtime["execution"] = {
            "source_sha": source,
            "synthetic_engineering_only": True,
            "formal_training_authorized": False,
        }
    else:
        # Preserve beta and every original gate. Only engineering metadata is
        # appended before the immutable runtime hash is computed.
        runtime["execution"]["configuration_sha256"] = canonical_sha256(
            {k: v for k, v in runtime.items() if k != "execution"}
        )
    atomic(output / "candidate-config.json", candidate)
    atomic(output / "runtime-config.json", runtime)
    copyfile(BASE / "SPEC.md", output / "protocol.md")
    atomic(
        output / "preflight.json", gates or {"scope": "SYNTHETIC_CPU_ENGINEERING", "passed": True}
    )
    lock = {"experiment_id": experiment_id, "run_id": run_id, "source_sha": source}
    for name, key in (
        ("candidate-config.json", "candidate_config_sha256"),
        ("runtime-config.json", "runtime_config_sha256"),
        ("protocol.md", "protocol_sha256"),
        ("assets/manifest.json", "assets_manifest_sha256"),
        ("assets/tokenizer.json", "tokenizer_sha256"),
    ):
        lock[key] = digest(output / name)
    lock["preflight_sha256"] = digest(output / "preflight.json")
    atomic(output / "experiment.lock.json", lock)
    state(output, "PREFLIGHT_PASSED")
    return runtime


def state(root, status, **extra):
    atomic(
        Path(root) / "run-status.json",
        {"status": status, "scientific_decision": False, "formal_confirmation": False, **extra},
    )


def full_identity(c, arm, source, data):
    result = cp.identity(c, arm, source, data)
    if "engineering_run_id" in c:
        result["engineering_run_id"] = c["engineering_run_id"]
    return result


def load_state(folder, c, arm, data, source, device="cpu", selector=None):
    model = Model(c, arm).to(device)
    trainer = Trainer(model)
    rng = np.random.default_rng(c["seed"] + 10)
    identity = read(Path(folder) / "identity.json")
    if identity != full_identity(c, arm, source, data):
        raise ContinuationError("IDENTITY_MISMATCH", arm)
    # R3 restore uses latest.json. A private selection directory contains only
    # immutable checkpoint bytes/sidecar and a synthesized exact selector.
    import tempfile
    import shutil

    if selector is None:
        nursery = cp.restore(folder, trainer, rng, identity)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(folder) / Path(selector["checkpoint"]).name
            target = Path(tmp) / path.name
            shutil.copyfile(path, target)
            atomic(
                Path(tmp) / "latest.json",
                {"file": path.name, "sha256": selector["checkpoint_sha256"], "identity": identity},
            )
            nursery = cp.restore(tmp, trainer, rng, identity)
    return trainer, nursery, rng, identity


def diagnostics(model, data, c, frozen, step, folder, trainer, *, crash=None):
    plan = c["evaluation"].get("snapshots", [])
    if not c["evaluation"].get("periodic_diagnostics_enabled", True) or step not in plan:
        return
    if crash:
        crash("snapshot-before")
    start = time.perf_counter()
    rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    try:
        high = step in c["evaluation"].get("counterfactual_snapshots", [])
        if high:
            record(model, data, c, frozen, step, folder)
        location = folder / "diagnostics" / ("snapshot-step-%06d.json" % step)
        if not location.exists():
            last = trainer.journal[-1] if trainer.journal else {}
            norms = {k: last.get(k) for k in ("q_grad_norm", "lcg_grad_norm", "lcg_q_cosine")}
            q = norms["q_grad_norm"]
            lcg = norms["lcg_grad_norm"]
            norms["scaled_lcg_q_ratio"] = (
                c["competition"]["beta"] * lcg / q if q and lcg is not None else None
            )
            latest = read(folder / "latest.json")
            # Expensive counterfactual diagnostics have a separately frozen plan.
            high = step in c["evaluation"].get("counterfactual_snapshots", [])
            from .diagnostics import router_scores

            result = {
                "router_scores": router_scores(model, frozen),
                "step": step,
                "checkpoint_sha256": latest["sha256"],
                "configuration_sha256": canonical_sha256(c),
                "probe_bank_sha256": digest(folder / "frozen-probe-bank.pt"),
                "generated_at_unix": time.time(),
                "elapsed_seconds": time.perf_counter() - start,
                "sample_count": len(frozen),
                "gradients": norms,
                "quality": {
                    p: evaluate(model, data, c, "calibration", p)
                    for p in ("greedy", "sampled", "uniform")
                },
                "q_calibration": q_diagnostics(model, data, c)
                if high
                else {"status": "NOT_SCHEDULED"},
                "scope": "ENGINEERING_DIAGNOSTICS_NO_SCIENTIFIC_DECISION",
            }
            result["elapsed_seconds"] = time.perf_counter() - start
            from clm001_r3.report import seal

            seal(location, result)
    finally:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
    if crash:
        crash("snapshot-after")


def segment_arm(root, arm, stop, *, device="cpu", crash=None, committed=None):
    root = Path(root)
    c = read(root / "runtime-config.json")
    if c.get("synthetic"):
        if device != "cpu":
            raise ContinuationError("SYNTHETIC_CPU_ONLY")
    else:
        from clm001_r3.common import authorize

        authorize(c, "train-dev", device)
        gate = read(root / "preflight.json")
        if gate.get("KAGGLE_CROSS_SESSION_RESUME") != "PASS":
            raise ContinuationError("ENGINEERING_RELEASE_NOT_VERIFIED")
    source = read(root / "experiment.lock.json")["source_sha"]
    require_source(source)
    data = load_assets(c, root / "assets")
    folder = root / "arms" / arm
    folder.mkdir(parents=True, exist_ok=True)
    was_restored = (folder / "latest.json").exists()
    if was_restored:
        trainer, nursery, rng, identity = load_state(folder, c, arm, data, source, device)
        if not (folder / "frozen-probe-bank.json").exists():
            raise ContinuationError("FROZEN_PROBE_MISSING", arm)
    else:
        if (folder / "identity.json").exists():
            raise ContinuationError(
                "CHECKPOINT_FILE_MISSING", "partial run cannot silently initialize"
            )
        random.seed(c["seed"])
        np.random.seed(c["seed"])
        torch.manual_seed(c["seed"])
        trainer = Trainer(Model(c, arm).to(device))
        nursery = Nursery()
        rng = np.random.default_rng(c["seed"] + 10)
        identity = full_identity(c, arm, source, data)
        atomic(folder / "identity.json", identity)
    start = trainer.step
    print(
        json.dumps(
            {"arm": arm, "RESTORED_FROM_STEP": start, "STARTED_FROM_RANDOM_INIT": not was_restored}
        ),
        flush=True,
    )
    frozen = bank(trainer.model, data, c, folder, identity)
    if start == 0 and not (folder / "latest.json").exists():
        cp.save(folder, trainer, nursery, rng, identity)
    # If the prior process died after an update/event save but before diagnostics,
    # finish that same boundary without replaying the update.
    if pending(c, trainer.step):
        process(c, trainer, nursery, data, rng, folder, identity)
    diagnostics(trainer.model, data, c, frozen, trainer.step, folder, trainer, crash=crash)
    if committed and (folder / "latest.json").exists():
        committed(root)
    end = min(stop, c["training"]["steps"])
    while trainer.step < end:
        row, snapshot, labels = trainer.update(batches(data, c, rng, device))
        step = trainer.step
        atomic(
            folder / "diagnostics" / ("oracle-step-%06d.json" % step),
            {
                "step": step,
                "snapshot_sha": snapshot.sha,
                "labels": [asdict(label) for _, _, label in labels],
                "scope": "PRIVATE_ONLY",
            },
        )
        atomic(
            folder / "diagnostics" / ("competition-step-%06d.json" % step),
            {"step": step, "records": trainer.last_credit, "scope": "PRIVATE_ONLY"},
        )
        if step % c["training"]["checkpoint_interval"] == 0 or pending(c, step) or step == end:
            if crash:
                crash("checkpoint-before")
            cp.save(folder, trainer, nursery, rng, identity)
            if crash:
                crash("checkpoint-after")
        if pending(c, step):
            process(c, trainer, nursery, data, rng, folder, identity)
        if (
            c["evaluation"].get("periodic_diagnostics_enabled", True)
            and step in c["evaluation"].get("snapshots", [])
            and not read(folder / "latest.json")["file"].startswith("checkpoint-step-%06d" % step)
        ):
            cp.save(folder, trainer, nursery, rng, identity)
        if (folder / "latest.json").exists() and read(folder / "latest.json")["file"].startswith(
            "checkpoint-step-%06d" % step
        ):
            diagnostics(trainer.model, data, c, frozen, step, folder, trainer, crash=crash)
            if committed:
                committed(root)
        print(json.dumps({"arm": arm, "step": step, "restored_from_step": start}), flush=True)
    check_events(c, trainer.step, nursery, identity)
    if trainer.step == c["training"]["steps"] and not (folder / "final-evaluation.json").exists():
        atomic(
            folder / "final-evaluation.json",
            {
                "checkpoint": read(folder / "latest.json"),
                "scientific_decision": False,
                "quality": {
                    p: evaluate(trainer.model, data, c, "development", p)
                    for p in ("greedy", "sampled", "uniform", "constant")
                },
            },
        )
    return {
        "arm": arm,
        "restored_from_step": start,
        "last_committed_step": trainer.step,
        "started_from_random_init": not was_restored,
        "status": "COMPLETE" if trainer.step == c["training"]["steps"] else "PAUSED",
    }


def package(root, *, crash=None, limit_bytes=None):
    root = Path(root)
    c = read(root / "runtime-config.json")
    lock = read(root / "experiment.lock.json")
    data = load_assets(c, root / "assets")
    arms = {}
    for arm in c["arms"]:
        folder = root / "arms" / arm
        if not folder.exists():
            continue
        trainer, nursery, _, identity = load_state(folder, c, arm, data, lock["source_sha"])
        check_events(c, trainer.step, nursery, identity)
        if not (folder / "frozen-probe-bank.json").exists():
            raise ContinuationError("FROZEN_PROBE_MISSING", arm)
        latest = read(folder / "latest.json")
        records = []
        for path in sorted(folder.glob("checkpoint-step-*.pt")):
            match = PATTERN.fullmatch(path.name)
            if match is None:
                raise ContinuationError("UNSAFE_PATH", path.name)
            side = path.with_suffix(".sha256.json")
            if not side.exists():
                # Interrupted checkpoint without completed hash is not a commit.
                continue
            witness = read(side)
            if witness["sha256"] != digest(path) or witness["identity"] != identity:
                raise ContinuationError("CHECKPOINT_HASH_MISMATCH", path.name)
            checkpoint_events = torch.load(path, map_location="cpu", weights_only=False)[
                "extra"
            ].get("events", {})
            records.append(
                {
                    "event_cursor": {
                        k: {"step": v["step"], "event": v["event"]}
                        for k, v in checkpoint_events.items()
                    },
                    "step": int(match[1]),
                    "revision": int(match[2] or 0),
                    "checkpoint": path.relative_to(root).as_posix(),
                    "checkpoint_sha256": witness["sha256"],
                    "sidecar": side.relative_to(root).as_posix(),
                }
            )
        chosen = next(v for v in records if Path(v["checkpoint"]).name == latest["file"])
        arms[arm] = dict(
            chosen,
            checkpoints=records,
            last_committed_step=trainer.step,
            probe_bank_sha256=digest(folder / "frozen-probe-bank.pt"),
            event_cursor={
                k: {"step": v["step"], "event": v["event"]} for k, v in nursery.events.items()
            },
            event_state_sha256=canonical(nursery.events),
            training_state_sha256=state_fingerprint(folder / latest["file"]),
        )
    if not arms:
        raise ContinuationError("NO_VALID_CHECKPOINTS")
    complete = len(arms) == len(c["arms"]) and all(
        v["last_committed_step"] == c["training"]["steps"] for v in arms.values()
    )
    status = "COMPLETE" if complete else "PAUSED"
    state(root, status, arms={a: v["last_committed_step"] for a, v in arms.items()})
    from .diagnostics import resource_estimate

    atomic(root / "resource-estimate.json", resource_estimate(root, c, arms))
    kwargs = {"crash": crash}
    if limit_bytes is not None:
        kwargs["limit_bytes"] = limit_bytes
    return LocalOutputStore(root).commit(
        dict(lock, status=status, complete=complete, arms=arms), **kwargs
    )


def state_fingerprint(path):
    """Exact semantic state hash excludes wall-clock journal timings, never tensors/RNG/events."""
    value = torch.load(path, map_location="cpu", weights_only=False)
    h = __import__("hashlib").sha256()

    def token(value):
        content = value if isinstance(value, bytes) else str(value).encode()
        h.update(str(len(content)).encode() + b":" + content)

    def visit(v, key=""):
        token(type(v).__name__)
        if isinstance(v, torch.Tensor):
            token(v.dtype)
            token(tuple(v.shape))
            token(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, np.ndarray):
            token(v.dtype)
            token(v.shape)
            token(v.tobytes())
        elif isinstance(v, dict):
            keys = [k for k in sorted(v, key=str) if "seconds" not in str(k)]
            token(len(keys))
            for k in keys:
                visit(k)
                visit(v[k], str(k))
        elif isinstance(v, (tuple, list)):
            token(len(v))
            for item in v:
                visit(item)
        else:
            token(repr(v))

    # Identity differs only when explicitly comparing distinct control run IDs.
    for k in (
        "model",
        "optimizer",
        "step",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "cuda_rng",
        "extra",
    ):
        visit(value[k], k)
    return h.hexdigest()


def verify(root, expected_source=None):
    store = MountedInputStore(root)
    meta = store.verify(expected_source)
    c = read(store.root / "runtime-config.json")
    lock = read(store.root / "experiment.lock.json")
    if digest(store.root / "preflight.json") != lock["preflight_sha256"] or not read(
        store.root / "preflight.json"
    ).get("passed"):
        raise ContinuationError("PREFLIGHT_EVIDENCE_MISMATCH")
    data = load_assets(c, store.root / "assets")
    for arm, progress in meta["arms"].items():
        trainer, nursery, _, _ = load_state(
            store.root / "arms" / arm, c, arm, data, meta["source_sha"], selector=progress
        )
        check_events(c, trainer.step, nursery, full_identity(c, arm, meta["source_sha"], data))
        if (
            trainer.step != progress["last_committed_step"]
            or canonical(nursery.events) != progress["event_state_sha256"]
        ):
            raise ContinuationError("EVENT_STATE_INCONSISTENT", arm)
        if (
            state_fingerprint(store.root / progress["checkpoint"])
            != progress["training_state_sha256"]
        ):
            raise ContinuationError("FULL_STATE_MISMATCH", arm)
    complete = len(meta["arms"]) == len(c["arms"]) and all(
        p["last_committed_step"] == c["training"]["steps"] for p in meta["arms"].values()
    )
    if meta["complete"] != complete or meta["status"] != ("COMPLETE" if complete else "PAUSED"):
        raise ContinuationError("RUN_STATUS_INCONSISTENT")
    return meta


def resume(checkpoint_input, output, stop, *, require_resume=True, source=None, arms=None):
    if not require_resume:
        raise ContinuationError(
            "RESUME_NOT_REQUESTED", "input exists: refuse implicit overwrite/retrain"
        )
    meta = verify(checkpoint_input, source)
    require_source(meta["source_sha"])
    store = MountedInputStore(checkpoint_input)
    output = Path(output)
    if output.exists():
        existing = verify(output, meta["source_sha"])
        parent = read(output / "parent-input.json")
        if existing["run_id"] != meta["run_id"] or parent["continuation_sha256"] != digest(
            Path(checkpoint_input) / "continuation.json"
        ):
            raise ContinuationError("OUTPUT_ALREADY_EXISTS", "different parent identity")
        config = read(output / "runtime-config.json")
        selected = arms or config["arms"]
        if all(
            a in existing["arms"]
            and existing["arms"][a]["last_committed_step"] >= min(stop, config["training"]["steps"])
            for a in selected
        ):
            return existing
    else:
        store.materialize(output, meta["source_sha"])
    atomic(
        Path(output) / "parent-input.json",
        {
            "path": str(checkpoint_input),
            "continuation_sha256": digest(Path(checkpoint_input) / "continuation.json"),
            "parent_run_id": meta["run_id"],
            "parent_source_sha": meta["source_sha"],
        },
    )
    state(output, "RESUMING")
    results = segments(output, stop, arms)
    result = package(output)
    atomic(
        Path(output) / "segment-summary.json",
        {
            "arms": results,
            "status": result["status"],
            "preflight_reused": True,
            "beta_recalibrated": False,
            "scientific_decision": False,
        },
    )
    # The summary must also be covered in the final package, not appended uncommitted.
    return package(output)


def evaluate_checkpoint(root, output, arm, step, evaluation_config, *, revision=None):
    meta = verify(root)
    store = MountedInputStore(root)
    chosen = store.select(arm, step, revision)
    output = Path(output)
    if (
        output.exists()
        or output.resolve() == Path(root).resolve()
        or Path(root).resolve() in output.resolve().parents
    ):
        raise ContinuationError("EVALUATION_OUTPUT_MUST_BE_SEPARATE")
    allowed = {"role", "policies", "seed", "retrospective", "counterfactual"}
    if set(evaluation_config) - allowed or evaluation_config.get("role", "development") not in (
        "calibration",
        "development",
    ):
        raise ContinuationError("EVALUATION_CONFIG_REJECTED")
    policies = evaluation_config.get("policies", ["greedy", "sampled", "uniform"])
    if not policies or set(policies) - {"greedy", "sampled", "uniform", "constant"}:
        raise ContinuationError("EVALUATION_CONFIG_REJECTED")
    before = digest(Path(root) / "continuation.json")
    c = read(Path(root) / "runtime-config.json")
    data = load_assets(c, Path(root) / "assets")
    trainer, _, _, identity = load_state(
        Path(root) / "arms" / arm, c, arm, data, meta["source_sha"], selector=chosen
    )
    result = {
        "status": "EVALUATED",
        "retrospective": True,
        "exploratory": True,
        "scientific_decision": False,
        "source_run_id": meta["run_id"],
        "source_identity": identity,
        "checkpoint": chosen,
        "evaluation_source_sha": source_sha(),
        "evaluation_config_sha256": canonical(evaluation_config),
        "evaluation_config": evaluation_config,
        "assets_manifest_sha256": meta["assets_manifest_sha256"],
        "tokenizer_sha256": meta["tokenizer_sha256"],
        "probe_bank_sha256": meta["arms"][arm]["probe_bank_sha256"],
        "step": trainer.step,
        "arm": arm,
        "quality": {
            p: evaluate(
                trainer.model,
                data,
                c,
                evaluation_config.get("role", "development"),
                p,
                seed=evaluation_config.get("seed", c["evaluation"]["seed"]),
            )
            for p in policies
        },
    }
    if evaluation_config.get("counterfactual"):
        result["q_diagnostics"] = q_diagnostics(trainer.model, data, c)
    if digest(Path(root) / "continuation.json") != before:
        raise ContinuationError("INPUT_MUTATED")
    store.verify()
    output.mkdir(parents=True)
    atomic(output / "evaluation.json", result)
    return result


def fork_experiment(
    root, output, arm, step, new_config, *, run_id, optimizer_policy, revision=None
):
    """Create a distinct child identity. This command never takes a training update."""
    meta = verify(root)
    store = MountedInputStore(root)
    chosen = store.select(arm, step, revision)
    if run_id == meta["run_id"] or not run_id:
        raise ContinuationError("FORK_IDENTITY_REQUIRED")
    if optimizer_policy not in ("inherit", "reset"):
        raise ContinuationError("FORK_OPTIMIZER_POLICY_REQUIRED")
    c = read(Path(root) / "runtime-config.json")
    data = load_assets(c, Path(root) / "assets")
    trainer, nursery, rng, _ = load_state(
        Path(root) / "arms" / arm, c, arm, data, meta["source_sha"], selector=chosen
    )
    # Shape/data changes require a separately reviewed migration. Never weaken an
    # existing data identity just to turn an arbitrary config into a valid fork.
    if (
        new_config["model"] != c["model"]
        or new_config["data"] != c["data"]
        or new_config["evaluation"] != c["evaluation"]
    ):
        raise ContinuationError("FORK_MIGRATION_REQUIRED", "model/data/probe contract changed")
    active = new_run(
        new_config,
        Path(root) / "assets",
        output,
        source=source_sha(),
        run_id=run_id,
        experiment_id=meta["experiment_id"] + "-FORK",
    )
    trainer.model.experiment_config = deepcopy(active)
    for candidate in nursery.candidates:
        candidate.model.experiment_config = deepcopy(active)
    if optimizer_policy == "reset":
        old_step, old_position, journal = trainer.step, trainer.data_position, trainer.journal
        trainer = Trainer(trainer.model)
        trainer.step, trainer.data_position, trainer.journal = old_step, old_position, journal
        for candidate in nursery.candidates:
            previous = candidate.trainer
            candidate.trainer = Trainer(candidate.model)
            candidate.trainer.step = previous.step
            candidate.trainer.data_position = previous.data_position
            candidate.trainer.journal = previous.journal
    run_identity = full_identity(active, arm, source_sha(), data)
    nursery.events = {
        canonical_sha256(
            {
                "identity": run_identity,
                "step": v["step"],
                "event": v["event"],
                "base_registry": v["base_registry"],
            }
        ): v
        for v in nursery.events.values()
    }
    folder = Path(output) / "arms" / arm
    folder.mkdir(parents=True)
    atomic(folder / "identity.json", run_identity)
    import shutil

    shutil.copyfile(
        Path(root) / "arms" / arm / "frozen-probe-bank.pt", folder / "frozen-probe-bank.pt"
    )
    atomic(
        folder / "frozen-probe-bank.json",
        {"identity": run_identity, "sha256": digest(folder / "frozen-probe-bank.pt")},
    )
    cp.save(folder, trainer, nursery, rng, run_identity, event_revision=len(nursery.events) or None)
    atomic(
        Path(output) / "parent-fork.json",
        {
            "parent_experiment_id": meta["experiment_id"],
            "parent_run_id": meta["run_id"],
            "parent_arm": arm,
            "parent_step": step,
            "parent_checkpoint_sha256": chosen["checkpoint_sha256"],
            "new_source_sha": source_sha(),
            "new_runtime_config_sha256": digest(Path(output) / "runtime-config.json"),
            "new_protocol_sha256": digest(Path(output) / "protocol.md"),
            "optimizer_policy": optimizer_policy,
            "inherited_frozen_probe_bank": True,
            "training_started": False,
        },
    )
    return package(output)


def import_legacy(r3_output, assets, output, *, expected_source, candidate_config):
    """Read-only adoption of trusted historical R3 Output; no identity rewriting.

    It is an evaluation/import path, not an automatic source migration or training
    authorization. The original source/protocol/config identities must match exactly.
    """
    from shutil import copytree, copyfile
    from clm001_r3.common import load
    from clm001_r3.report import verified
    from .store import inventory

    r3_output, output, assets = Path(r3_output), Path(output), Path(assets)
    if output.exists():
        raise ContinuationError("OUTPUT_ALREADY_EXISTS")
    inventory(r3_output)
    inventory(assets)
    c = load(r3_output / "runtime-config.json")
    candidate = read(candidate_config)
    summary = verified(r3_output / "gpu-pilot-summary.json")
    if c["execution"]["source_sha"] != expected_source:
        raise ContinuationError("SOURCE_SHA_MISMATCH")
    if c["execution"]["candidate_configuration_sha256"] != canonical_sha256(candidate) or c[
        "execution"
    ]["pilot_hash"] != canonical_sha256(summary):
        raise ContinuationError("PREFLIGHT_EVIDENCE_MISMATCH")
    if not all(summary["gates"].values()) or c["execution"][
        "configuration_sha256"
    ] != canonical_sha256({k: v for k, v in c.items() if k != "execution"}):
        raise ContinuationError("IDENTITY_MISMATCH", "historical runtime")
    data = load_assets(c, assets)
    output.mkdir(parents=True)
    copytree(assets, output / "assets")
    copyfile(r3_output / "runtime-config.json", output / "runtime-config.json")
    copyfile(candidate_config, output / "candidate-config.json")
    copyfile(BASE / "SPEC.md", output / "protocol.md")
    atomic(
        output / "preflight.json",
        {
            "passed": True,
            "historical_gate_summary": summary,
            "historical_runtime_execution": c["execution"],
            "scope": "VERIFIED_LEGACY_GATES",
        },
    )
    for arm in c["arms"]:
        original = r3_output / "arms" / arm
        if not original.exists():
            continue
        expected = cp.identity(c, arm, expected_source, data)
        if read(original / "identity.json") != expected:
            raise ContinuationError("IDENTITY_MISMATCH", arm)
        dest = output / "arms" / arm
        dest.mkdir(parents=True)
        for path in original.iterdir():
            if path.name == "portable":
                continue
            if path.is_dir():
                if path.name in ("snapshots", "diagnostics"):
                    copytree(path, dest / path.name)
            elif path.suffix in (".pt", ".json", ".log", ".txt"):
                copyfile(path, dest / path.name)
    lock = {
        "experiment_id": "CLM-001-R3-LEGACY-READONLY",
        "run_id": "legacy-" + expected_source[:12] + "-" + canonical_sha256(c)[:12],
        "source_sha": expected_source,
        "legacy_read_only": True,
        "preflight_sha256": digest(output / "preflight.json"),
    }
    for name, key in (
        ("candidate-config.json", "candidate_config_sha256"),
        ("runtime-config.json", "runtime_config_sha256"),
        ("protocol.md", "protocol_sha256"),
        ("assets/manifest.json", "assets_manifest_sha256"),
        ("assets/tokenizer.json", "tokenizer_sha256"),
    ):
        lock[key] = digest(output / name)
    atomic(output / "experiment.lock.json", lock)
    return package(output)


def segments(root, stop, arms=None):
    """Same two T4 queues as R3; independent worker processes own GPU RNG state."""
    root = Path(root)
    c = read(root / "runtime-config.json")
    arms = list(arms or c["arms"])
    if set(arms) - set(c["arms"]):
        raise ContinuationError("ARM_NOT_FOUND")
    state(root, "TRAINING")
    if c.get("synthetic"):
        return [segment_arm(root, arm, stop) for arm in arms]
    from concurrent.futures import ThreadPoolExecutor
    from clm001_r3.automatic import device_queues
    from clm001_r3.common import ROOT
    import subprocess
    import sys

    script = (
        ROOT / "research/projects/cellular-language-models/engineering/kaggle-continuation/run.py"
    )

    def queue(index):
        codes = []
        for arm in device_queues(c)[index]:
            if arm not in arms:
                continue
            command = [
                sys.executable,
                str(script),
                "--mode",
                "segment-worker",
                "--output",
                str(root),
                "--arm",
                arm,
                "--device",
                "cuda:" + str(index),
                "--stop-after-step",
                str(stop),
            ]
            code = subprocess.run(command).returncode
            codes.append({"arm": arm, "exit_code": code})
        return codes

    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = [v for batch in executor.map(queue, range(2)) for v in batch]
    if any(v["exit_code"] for v in rows):
        raise ContinuationError("SEGMENT_WORKER_FAILED", json.dumps(rows))
    return rows
