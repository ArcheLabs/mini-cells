"""Auditable native-CLM update adapter. Scientific access is separately blocked at Gate 0."""

import copy
import os
import random
from pathlib import Path
import tempfile
import numpy as np
import torch
from minicells_r3e import policy_proposal as source
from minicells_r3e.common import fingerprint
from minicells_r3k.reuse import bind
from minicells_jct001.cost import Journal
from minicells_jct001.connection import TorchAdapter
from .native import masks, treatment, public_fingerprint
from .measurement import nll


class Engine:
    def __init__(self, model, runtime, arm, identity, ledger, base_state=None, unit=False):
        self.model = model
        self.c = copy.deepcopy(runtime)
        self.arm = arm
        self.identity = dict(identity, arm=arm)
        self.unit = unit
        self.step = 0
        self.parent_step = 0
        self.tokens = 0
        self.curve = []
        self.policy_optimizer_state = None
        self.route_rng = torch.Generator().manual_seed(
            model.seed + self.c["training"]["route_seed_offset"]
        )
        self.cf_rng = torch.Generator().manual_seed(
            model.seed + self.c["training"]["cf_seed_offset"]
        )
        self.data_rng = np.random.default_rng(model.seed + self.c["training"]["data_seed_offset"])
        q = self.c["engine"]["training"]
        self.cell_optimizer = torch.optim.AdamW(
            model.cells.parameters(), lr=q["learning_rate"], weight_decay=q["weight_decay"]
        )
        self.journal = Journal(ledger, self.identity)
        self.journal.allow_zero_signal = True
        self.parent_sha = None
        if base_state is not None:
            if base_state["step"] != 0 or base_state["arm"] != "BASE":
                raise RuntimeError("CLM002_ARMS_REQUIRE_SAME_INITIAL_CHECKPOINT")
            self.parent_sha = fingerprint(base_state)
            self._values(base_state)
        masks(model, arm) if arm != "BASE" else [
            p.requires_grad_(False) for p in model.parameters()
        ]
        self.public_sha = public_fingerprint(model)
        self.frozen_sha = fingerprint(
            {n: p for n, p in model.named_parameters() if not p.requires_grad}
        )
        self.initial_mask = {n: p.requires_grad for n, p in model.named_parameters()}
        # Independent of native work counters (the production Q optimizer resets those).
        self._work_handles = []
        modules = [(cell, "cell_rows") for cell in model.cells]
        modules += [
            (model.init, "input_rows"),
            (model.context.query, "context_rows"),
            (model.gate, "gate_rows"),
            (model.connection, "policy_rows"),
            (model.final, "head_rows"),
        ]
        for module, kind in modules:

            def count(module, args, result, kind=kind, meter=self.journal.meter):
                x = args[0]
                meter.add(kind, x.numel() // x.shape[-1])

            self._work_handles.append(module.register_forward_hook(count))

    def epsilon(self):
        q = self.c["training"]
        fraction = min(self.step, q["steps"] - 1) / max(1, q["steps"] - 1)
        return q["epsilon_initial"] + (q["epsilon_final"] - q["epsilon_initial"]) * fraction

    def state(self):
        return copy.deepcopy(
            dict(
                arm=self.arm,
                parent_sha=self.parent_sha,
                model=self.model.state_dict(),
                cell_optimizer=self.cell_optimizer.state_dict(),
                connection_optimizer=dict(
                    lifecycle="FRESH_ADAMW_PER_PROPOSAL",
                    last_completed=self.policy_optimizer_state,
                    pending=None,
                ),
                step=self.step,
                parent_step=self.parent_step,
                tokens=self.tokens,
                curve=self.curve,
                data_rng=self.data_rng.bit_generator.state,
                route_rng=self.route_rng.get_state(),
                cf_rng=self.cf_rng.get_state(),
                cpu_rng=torch.get_rng_state(),
                python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
                schedule=dict(
                    epsilon=self.epsilon(), temperature=self.model.temperature, rho=self.model.rho
                ),
                work=self.model.work,
                parameter_mask={n: p.requires_grad for n, p in self.model.named_parameters()},
            )
        )

    def _values(self, state):
        self.model.load_state_dict(state["model"], strict=True)
        self.cell_optimizer.load_state_dict(state["cell_optimizer"])
        if (
            state["connection_optimizer"]["lifecycle"] != "FRESH_ADAMW_PER_PROPOSAL"
            or state["connection_optimizer"]["pending"] is not None
        ):
            raise RuntimeError("CLM002_UNCERTAIN_CONNECTION_OPTIMIZER")
        self.policy_optimizer_state = copy.deepcopy(state["connection_optimizer"]["last_completed"])
        for key in ("step", "parent_step", "tokens", "curve"):
            setattr(self, key, copy.deepcopy(state[key]))
        self.data_rng.bit_generator.state = copy.deepcopy(state["data_rng"])
        self.route_rng.set_state(state["route_rng"])
        self.cf_rng.set_state(state["cf_rng"])
        torch.set_rng_state(state["cpu_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        self.model.temperature = state["schedule"]["temperature"]
        self.model.rho = state["schedule"]["rho"]
        self.model.work = copy.deepcopy(state["work"])
        if self.epsilon() != state["schedule"]["epsilon"]:
            raise RuntimeError("CLM002_SCHEDULE_CHANGED")

    def save(self, path, stream):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            identity=self.identity,
            state=self.state(),
            stream=stream.state(),
            public_sha=self.public_sha,
            watermark=self.journal.watermark(),
        )
        payload["full_state_sha256"] = fingerprint((payload["state"], payload["stream"]))
        fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        os.close(fd)
        try:
            torch.save(payload, name)
            with open(name, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        return dict(
            sha256=__import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            bytes=path.stat().st_size,
            step=self.step,
            local_full_state_sha256=payload["full_state_sha256"],
            cross_session_verified=False,
        )

    def restore(self, path, stream):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (
            saved["identity"] != self.identity
            or saved["public_sha"] != self.public_sha
            or saved["full_state_sha256"] != fingerprint((saved["state"], saved["stream"]))
        ):
            raise RuntimeError("CLM002_CHECKPOINT_IDENTITY_OR_VALUES_CHANGED")
        if saved["state"]["parent_sha"] != self.parent_sha or saved["state"]["parameter_mask"] != {
            n: p.requires_grad for n, p in self.model.named_parameters()
        }:
            raise RuntimeError("CLM002_PARAMETER_BOUNDARY_CHANGED")
        self.journal.verify_prefix(saved["watermark"])
        self._values(saved["state"])
        stream.restore(saved["stream"])
        if fingerprint((self.state(), stream.state())) != saved["full_state_sha256"]:
            raise RuntimeError("CLM002_RESTORED_STATE_NOT_EXACT")

    def update(self, stream, checkpoint, references, failure_hook=None):
        if not self.unit:
            raise RuntimeError("CLM002_GATE0_SCIENTIFIC_REGISTRATION_BLOCKED")
        if self.step >= self.c["training"]["steps"]:
            raise RuntimeError("CLM002_FROZEN_UNIT_BUDGET_EXHAUSTED")
        if any(p.device.type != "cpu" for p in self.model.parameters()):
            raise RuntimeError("CLM002_UNIT_ENGINE_CPU_ONLY")
        if self.initial_mask != {
            n: p.requires_grad for n, p in self.model.named_parameters()
        } or self.frozen_sha != fingerprint(
            {n: p for n, p in self.model.named_parameters() if not p.requires_grad}
        ):
            raise RuntimeError("CLM002_FROZEN_STATE_CHANGED_BEFORE_UPDATE")
        summary = self.journal.summary()
        if summary["pending_attempts"] or summary["uncertain_optimizer_calls"]:
            raise RuntimeError("CLM002_UNCERTAIN_ATTEMPT_QUARANTINED")
        cells, connection = treatment(self.arm)
        attempt = "attempt-" + str(len(self.journal.events))
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self.journal.append(
            "ATTEMPT_BEGIN",
            dict(
                id=attempt,
                logical_step=self.step + 1,
                examples=self.c["engine"]["training"]["effective_batch"],
                target_tokens=stream.target_tokens,
            ),
        )
        try:
            rows = stream.next()
            valid_tokens = sum(int(row["valid"].sum()) for row in rows)
            if valid_tokens != self.c["training"]["tokens_per_step"]:
                raise RuntimeError("CLM002_EFFECTIVE_TOKEN_BUDGET_CHANGED")
            q = self.c["engine"]["training"]
            if len(rows) != q["effective_batch"] // q["microbatch"] or any(
                tuple(row["x"].shape) != (q["microbatch"], q["sequence_length"])
                or row["y"].shape != row["x"].shape
                or row["valid"].shape != row["x"].shape
                for row in rows
            ):
                raise RuntimeError("CLM002_FROZEN_BATCH_SHAPE_CHANGED")
            # Production Q collector requires unpadded uniform CE windows; never silently train padding.
            if any(not bool(row["valid"].all()) for row in rows):
                raise RuntimeError("CLM002_NATIVE_Q_REQUIRES_UNPADDED_REGISTERED_WINDOWS")
            self.cell_optimizer.zero_grad(set_to_none=True)
            losses = []
            exposure = torch.zeros(len(self.model.cells))
            grad_norms = []
            task_exposure = {
                task: torch.zeros(len(self.model.cells)) for task in ("Story", "Arithmetic")
            }
            with self.journal.phase("main_native_clm_update"):
                self.model.active_step = self.parent_step + self.step
                self.model.microbatch_index = 0
                try:
                    for row in rows:
                        out = self.model(row["x"], epsilon=self.epsilon(), route_rng=self.route_rng)
                        loss = nll(out.logits, row["y"], row["valid"])
                        losses.append(float(loss.detach()))
                        current_exposure = out.operator_rows.sum(0).cpu()
                        exposure += current_exposure
                        task_exposure[row["task"]] += current_exposure
                        if cells:
                            (loss * int(row["valid"].sum()) / valid_tokens).backward()
                finally:
                    self.model.active_step = None
                    self.model.training_context = None
                for cell in self.model.cells:
                    norms = [p.grad.square().sum() for p in cell.parameters() if p.grad is not None]
                    grad_norms.append(float(torch.stack(norms).sum().sqrt()) if norms else 0.0)
                if cells:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.cells.parameters(), self.c["engine"]["training"]["clip"]
                    )
                    for group in self.cell_optimizer.param_groups:
                        group["lr"] = self.c["engine"]["training"]["learning_rate"] * min(
                            1,
                            (self.parent_step + self.step + 1)
                            / self.c["engine"]["training"]["warmup_steps"],
                        )
                    self.journal.optimizer_call(
                        "cell", self.cell_optimizer.step, list(self.model.cells.parameters())
                    )
            proposal = None
            if connection:

                def train_batches(data, engine, rng, device):
                    return [(r["x"], r["y"]) for r in rows]

                collector = torch.no_grad()(
                    bind(source.collect.__wrapped__, train_batches=train_batches)
                )
                optimizer = bind(
                    source.optimize,
                    collect=collector,
                    reference=lambda m, c, d: references,
                    torch=TorchAdapter(self.journal),
                )
                with self.journal.phase("native_connection_all_action_q"):
                    candidate, proposal = optimizer(self, {})
                if candidate is not None:
                    self.model.connection.load_state_dict(
                        candidate.connection.state_dict(), strict=True
                    )
                self.policy_optimizer_state = proposal.pop("proposal_optimizer_final_state")
                self.journal.append(
                    "AUXILIARY_LABEL_WORK",
                    dict(
                        target_evaluations=proposal["counterfactual_states"]
                        * proposal["actions_per_state"]
                    ),
                )
            if failure_hook:
                failure_hook(self)
            for name, p in self.model.named_parameters():
                if not p.requires_grad and (not torch.equal(before[name], p) or p.grad is not None):
                    raise RuntimeError("CLM002_FORBIDDEN_PARAMETER_OR_GRADIENT:" + name)
            if public_fingerprint(self.model) != self.public_sha:
                raise RuntimeError("CLM002_PUBLIC_PARAMETER_CHANGED")
            self.step += 1
            self.tokens += valid_tokens
            self.curve.append(
                dict(
                    step=self.step,
                    effective_targets=valid_tokens,
                    loss=float(np.mean(losses)),
                    input_sha256=fingerprint([(r["x"], r["y"], r["valid"]) for r in rows]),
                    cell_call_counts=exposure.tolist(),
                    cell_gradient_norms=grad_norms,
                    task_cell_call_counts={
                        task: value.tolist() for task, value in task_exposure.items()
                    },
                    main_gradient_norm_scope="ACCUMULATED_MAIN_BACKWARD; TASKWISE_GRADIENTS_ARE_OFFLINE_PROBES",
                    zero_gradient_cells=[i for i, g in enumerate(grad_norms) if g == 0],
                    proposal=proposal,
                    parameter_delta={
                        g: float(
                            sum(
                                (p.detach() - before[n]).square().sum()
                                for n, p in self.model.named_parameters()
                                if n.startswith(g + ".")
                            ).sqrt()
                        )
                        for g in ("cells", "connection")
                    },
                )
            )
            with self.journal.phase("atomic_checkpoint"):
                receipt = self.save(checkpoint, stream)
            self.journal.append("LOGICAL_COMMIT", dict(step=self.step))
            self.journal.append("ATTEMPT_END", dict(id=attempt))
            return receipt
        except Exception as error:
            self.journal.append("ATTEMPT_ERROR", dict(id=attempt, error_type=type(error).__name__))
            raise
