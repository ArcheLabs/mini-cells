"""Native CLM math and R3e Q proposals, with the FAST phase/cadence contract."""

import copy
import random
import time

import numpy as np
import torch
from minicells_clm002.engine import Engine as ParentEngine
from minicells_clm002.measurement import nll
from minicells_clm002.native import treatment
from minicells_jct001.connection import TorchAdapter
from minicells_r3e import policy_proposal
from minicells_r3e.common import fingerprint
from minicells_r3k.reuse import bind


class Engine(ParentEngine):
    def __init__(self, model, runtime, arm, ident, ledger, config, phase, parent_state, parent_sha):
        historical = "r3h" in parent_state
        model.load_state_dict(parent_state["model"], strict=True)
        schedule = parent_state["r3h"] if historical else parent_state["schedule"]
        model.temperature, model.rho = schedule["temperature"], schedule["rho"]
        super().__init__(model, runtime, arm, ident, ledger, unit=False)
        self.cell_optimizer.load_state_dict(copy.deepcopy(parent_state["cell_optimizer"]))
        self.parent_step = (
            parent_state["step"]
            if historical
            else parent_state["parent_step"] + parent_state["step"]
        )
        self.parent_sha = parent_sha
        self.parent_origin = {
            "state_sha256": fingerprint(parent_state),
            "historical_or_phase_parent_step": self.parent_step,
            "phase": phase,
            "initialization": "EXACT_VALUES_AND_CELL_ADAMW; NEW_PAIRED_PHASE_STREAMS",
            "source_parent_sha256": parent_sha,
        }
        self.fast, self.phase = config, phase
        self.training_seconds = 0.0
        self.proposal_calls = 0
        self.route_rng.manual_seed(model.seed + config["phase_seed_offsets"][phase] + 43001)
        self.cf_rng.manual_seed(model.seed + config["phase_seed_offsets"][phase] + 42001)
        self.data_rng = np.random.default_rng(
            model.seed + config["phase_seed_offsets"][phase] + 41001
        )
        random.seed(model.seed + config["phase_seed_offsets"][phase])
        np.random.seed(model.seed + config["phase_seed_offsets"][phase])
        torch.manual_seed(model.seed + config["phase_seed_offsets"][phase])
        if any(p.is_cuda for p in model.parameters()):
            torch.cuda.manual_seed_all(model.seed + config["phase_seed_offsets"][phase])

    def state(self):
        result = super().state()
        result["fast"] = {
            "phase": self.phase,
            "training_seconds": self.training_seconds,
            "proposal_calls": self.proposal_calls,
            "cuda_rng": [torch.cuda.get_rng_state(next(self.model.parameters()).device)]
            if any(p.is_cuda for p in self.model.parameters())
            else [],
        }
        return result

    def _values(self, state):
        super()._values(state)
        if state["fast"]["phase"] != self.phase:
            raise RuntimeError("FAST_RESTORE_PHASE_CHANGED")
        self.training_seconds = state["fast"]["training_seconds"]
        self.proposal_calls = state["fast"]["proposal_calls"]
        rng = state["fast"]["cuda_rng"]
        if rng:
            if not next(self.model.parameters()).is_cuda or len(rng) != 1:
                raise RuntimeError("FAST_CUDA_RESUME_LAYOUT_CHANGED")
            torch.cuda.set_rng_state(rng[0], next(self.model.parameters()).device)

    def restore(self, path, stream):
        super().restore(path, stream)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        tail = self.journal.events[saved["watermark"]["events"] :]
        if any(
            event["kind"] not in ("PHASE_BEGIN", "PHASE_END", "PHASE_ERROR")
            or not (
                event["data"].get("phase", "").startswith("confirm_")
                or event["data"].get("phase") == "inference_profile"
            )
            for event in tail
        ):
            raise RuntimeError("FAST_INFLIGHT_WORK_AFTER_CHECKPOINT_NO_AUTOMATIC_RETRAIN")

    def epsilon(self):
        q = self.c["training"]
        t = min(self.step, self.fast["updates"] - 1) / max(1, self.fast["updates"] - 1)
        return q["epsilon_initial"] + (q["epsilon_final"] - q["epsilon_initial"]) * t

    def update(self, stream, checkpoint, references, failure_hook=None):
        if self.step >= self.fast["updates"] or self.phase not in ("A", "B"):
            raise RuntimeError("FAST_PHASE_BUDGET_EXHAUSTED")
        summary = self.journal.summary()
        if summary["pending_attempts"] or summary["uncertain_optimizer_calls"]:
            raise RuntimeError("FAST_UNCERTAIN_CHECKPOINT_REQUIRES_RECONCILIATION")
        if self.initial_mask != {n: p.requires_grad for n, p in self.model.named_parameters()}:
            raise RuntimeError("FAST_PARAMETER_MASK_CHANGED")
        cells, connection = treatment(self.arm)
        attempt = "update-" + str(self.step + 1)
        self.journal.append(
            "ATTEMPT_BEGIN",
            {
                "id": attempt,
                "logical_step": self.step + 1,
                "examples": self.fast["batch_size"],
                "target_tokens": self.fast["targets_per_update"],
            },
        )
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        try:
            rows = stream.next()
            if (
                len(rows) != self.fast["batch_size"]
                or sum(r["y"].numel() for r in rows) != self.fast["targets_per_update"]
            ):
                raise RuntimeError("FAST_MAIN_TOKEN_BUDGET_CHANGED")
            self.cell_optimizer.zero_grad(set_to_none=True)
            losses, counts = [], torch.zeros(len(self.model.cells), device=device)
            task_counts = {task: torch.zeros_like(counts) for task in ("Story", "Arithmetic")}
            with self.journal.phase("main_update"):
                self.model.active_step, self.model.microbatch_index = (
                    self.parent_step + self.step,
                    0,
                )
                try:
                    for row in rows:
                        out = self.model(row["x"], epsilon=self.epsilon(), route_rng=self.route_rng)
                        loss = nll(out.logits, row["y"], row["valid"])
                        if cells:
                            (loss / len(rows)).backward()
                        losses.append(loss.detach())
                        exposure = out.operator_rows.sum(0)
                        counts += exposure
                        task_counts[row["task"]] += exposure
                finally:
                    self.model.active_step = self.model.training_context = None
                cell_gradient_norms = []
                for cell in self.model.cells:
                    values = [
                        p.grad.square().sum() for p in cell.parameters() if p.grad is not None
                    ]
                    cell_gradient_norms.append(
                        torch.stack(values).sum().sqrt()
                        if values
                        else torch.zeros((), device=device)
                    )
                gradients = [p.grad for p in self.model.cells.parameters() if p.grad is not None]
                if cells:
                    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                        raise FloatingPointError("FAST_INVALID_CELL_GRADIENT")
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.cells.parameters(), self.c["engine"]["training"]["clip"]
                    )
                    if not torch.isfinite(gradient_norm):
                        raise FloatingPointError("FAST_NONFINITE_CELL_GRADIENT_NORM")
                    q = self.c["engine"]["training"]
                    for group in self.cell_optimizer.param_groups:
                        group["lr"] = q["learning_rate"] * min(
                            1, (self.parent_step + self.step + 1) / q["warmup_steps"]
                        )
                    self.journal.optimizer_call(
                        "cell", self.cell_optimizer.step, list(self.model.cells.parameters())
                    )
            proposal = None
            if connection and self.step + 1 in self.fast["connection"]["proposal_steps"]:

                def train_batches(data, config, rng, target_device):
                    return [(r["x"], r["y"]) for r in rows]

                collector = torch.no_grad()(
                    bind(policy_proposal.collect.__wrapped__, train_batches=train_batches)
                )
                optimize = bind(
                    policy_proposal.optimize,
                    collect=collector,
                    reference=lambda m, c, d: references,
                    torch=TorchAdapter(self.journal),
                )
                with self.journal.phase("connection_candidate_search"):
                    candidate, proposal = optimize(self, {})
                if candidate is not None:
                    self.model.connection.load_state_dict(
                        candidate.connection.state_dict(), strict=True
                    )
                self.policy_optimizer_state = proposal.pop("proposal_optimizer_final_state")
                self.proposal_calls += 1
                self.journal.append(
                    "AUXILIARY_LABEL_WORK",
                    {
                        "target_evaluations": proposal["counterfactual_states"]
                        * proposal["actions_per_state"]
                    },
                )
            if (
                fingerprint({n: p for n, p in self.model.named_parameters() if not p.requires_grad})
                != self.frozen_sha
            ):
                raise RuntimeError("FAST_FROZEN_PARAMETER_CHANGED")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.training_seconds += time.perf_counter() - start
            self.step += 1
            self.tokens += self.fast["targets_per_update"]
            self.curve.append(
                {
                    "step": self.step,
                    "ce": float(torch.stack(losses).mean()),
                    "input_sha256": fingerprint([(r["x"], r["y"]) for r in rows]),
                    "cell_counts": counts.cpu().tolist(),
                    "cell_gradient_norms": torch.stack(cell_gradient_norms).detach().cpu().tolist(),
                    "task_cell_counts": {k: v.cpu().tolist() for k, v in task_counts.items()},
                    "proposal": proposal,
                }
            )
            if failure_hook:
                failure_hook(self)
            self.journal.append("LOGICAL_COMMIT", {"step": self.step})
            self.journal.append("ATTEMPT_END", {"id": attempt})
            return self.save(checkpoint, stream)
        except Exception as error:
            self.journal.append(
                "ATTEMPT_ERROR", {"id": attempt, "error_type": type(error).__name__}
            )
            raise
