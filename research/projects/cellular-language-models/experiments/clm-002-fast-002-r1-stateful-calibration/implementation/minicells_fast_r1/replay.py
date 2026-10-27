"""Pause the immutable FAST001 update at the actual pre-proposal boundary, without rerunning it."""

import copy
import os
import time
from types import SimpleNamespace

import torch
from minicells_clm002.measurement import nll
from minicells_fast001.engine import Engine as FrozenEngine
from minicells_fast001.parent import make_engine as original_make_engine
from minicells_r3e import policy_proposal
from minicells_r3e.common import fingerprint
from minicells_r3k.reuse import bind

from .common import atomic, digest, read

collect = policy_proposal.collect


class ProposalBoundary(BaseException):
    """Intentional suspension, not an ATTEMPT_ERROR or an optimizer retry."""


def pause(tr, data):
    # FrozenEngine injects the exact main batch via the original collector binding.
    batch_source = collect.__wrapped__.__globals__["train_batches"]
    tr.pending_batch = batch_source(None, None, None, next(tr.model.parameters()).device)
    raise ProposalBoundary()


class ReplayEngine(FrozenEngine):
    def save(self, path, stream):
        start = time.perf_counter()
        result = super().save(path, stream)
        self.checkpoint_write_seconds = (
            getattr(self, "checkpoint_write_seconds", 0.0) + time.perf_counter() - start
        )
        return result

    def restore(self, path, stream):
        start = time.perf_counter()
        result = super().restore(path, stream)
        self.checkpoint_read_seconds = (
            getattr(self, "checkpoint_read_seconds", 0.0) + time.perf_counter() - start
        )
        return result

    def update(self, stream, checkpoint, references, failure_hook=None):
        self.pending_losses = []

        def observe(logits, targets, valid):
            loss = nll(logits, targets, valid)
            self.pending_losses.append(loss.detach())
            return loss

        return bind(
            FrozenEngine.update,
            nll=observe,
            policy_proposal=SimpleNamespace(collect=collect, optimize=pause),
        )(self, stream, checkpoint, references, failure_hook)


def make_engine(
    c,
    state,
    seed,
    arm,
    identity,
    ledger,
    device="cpu",
    *,
    unit=False,
    parent_sha=None,
    parent_identity=None,
):
    if not unit and (
        seed not in c["seeds"]
        or arm not in ("B01-S", "B01-H", "B11-S", "B11-H")
        or state.get("arm") != "A-" + arm[-1]
        or state.get("step") != 500
        or state.get("fast", {}).get("phase") != "A"
        or not parent_identity
        or parent_identity.get("seed") != seed
        or parent_identity.get("arm") != "A-" + arm[-1]
        or parent_identity.get("source_sha") != c["historical"]["source_sha"]
        or parent_identity.get("run_id") != c["historical"]["run_id"]
        or not parent_sha
    ):
        raise RuntimeError("R1_EXACT_SEED_CONDITION_A500_PARENT_IDENTITY_REQUIRED")
    engine = bind(original_make_engine, Engine=ReplayEngine)(
        c, state, seed, arm, identity, ledger, device, unit=unit, parent_sha=parent_sha
    )
    engine.r1_cpu_fixture = unit
    return engine


def scientific_fingerprint(state):
    state = copy.deepcopy(state)
    state["fast"].pop("training_seconds", None)
    return fingerprint(state)


def pending_values(engine):
    return {
        "gradients": {
            n: None if p.grad is None else p.grad.detach().clone()
            for n, p in engine.model.named_parameters()
        },
        "module_training": {n: m.training for n, m in engine.model.named_modules()},
        "microbatch_index": engine.model.microbatch_index,
        "rows": [(x.detach().clone(), y.detach().clone()) for x, y in engine.pending_batch],
    }


def validate_pending(engine, stream, c):
    summary = engine.journal.summary()
    target = c["diagnostic"]["pause_main_step"]
    cells = engine.arm.startswith("B11")
    pending = [r["data"] for r in engine.journal.events if r["kind"] == "ATTEMPT_BEGIN"][-1]
    if (
        engine.step != target - 1
        or stream.step != target
        or pending["logical_step"] != target
        or summary["uncertain_optimizer_calls"] != 0
        or summary["pending_attempts"] != 1
        or summary["optimizer_calls_by_group"].get("cell", 0) != (target if cells else 0)
        or summary["optimizer_calls_by_group"].get("connection", 0) != 0
        or engine.proposal_calls != 0
    ):
        raise RuntimeError("R1_PREPROPOSAL_BOUNDARY_OR_OPTIMIZER_ACCOUNTING_INVALID")
    if (
        fingerprint({n: p for n, p in engine.model.named_parameters() if not p.requires_grad})
        != engine.frozen_sha
    ):
        raise RuntimeError("R1_FROZEN_PARAMETER_CHANGED")


def restore(engine, stream, directory, c, identity):
    proof = read(directory / "PAUSE.json")
    if proof["identity"] != identity:
        raise RuntimeError("R1_PAUSE_SOURCE_CONFIG_OR_ARM_IDENTITY_CHANGED")
    for name, row in proof["files"].items():
        if digest(directory / name) != row["sha256"]:
            raise RuntimeError("R1_PAUSE_BYTES_CHANGED")
    engine.restore(directory / "preproposal.pt", stream)
    values = torch.load(
        directory / "pending.pt",
        map_location=next(engine.model.parameters()).device,
        weights_only=False,
    )
    for name, p in engine.model.named_parameters():
        p.grad = values["gradients"][name]
    for name, module in engine.model.named_modules():
        module.training = values["module_training"][name]
    engine.model.microbatch_index = values["microbatch_index"]
    engine.pending_batch = values["rows"]
    if fingerprint(values) != proof["pending_values_sha256"]:
        raise RuntimeError("R1_PENDING_GRADIENTS_BATCH_OR_MODE_CHANGED")
    validate_pending(engine, stream, c)
    return proof


def run(engine, stream, directory, c, identity, historical_curve=None):
    if (
        not getattr(engine, "r1_cpu_fixture", False)
        and not c["diagnostic"]["gate1_execution_authorized"]
    ):
        raise RuntimeError("R1_REAL_REPLAY_REQUIRES_EXPLICIT_GATE1_APPROVAL")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "PAUSE.json").exists():
        return restore(engine, stream, directory, c, identity)
    if (directory / "latest.pt").exists():
        engine.restore(directory / "latest.pt", stream)
    elif len(engine.journal.events) != 1:
        raise RuntimeError("R1_UNCERTAIN_WORK_WITHOUT_CHECKPOINT_NO_AUTOMATIC_REPLAY")
    if engine.step >= c["diagnostic"]["pause_main_step"]:
        raise RuntimeError("R1_CANNOT_REVERSE_FINAL_CHECKPOINT_TO_STEP50")
    paused = False
    try:
        while engine.step < c["diagnostic"]["pause_main_step"]:
            engine.update(stream, directory / "latest.pt", [])
            if not engine.r1_cpu_fixture and engine.step % 10 == 0:
                print(
                    "[R1 Gate1]", identity["seed"], identity["arm"], "main", engine.step, flush=True
                )
            if historical_curve:
                actual = engine.curve[-1]
                old = historical_curve[engine.step - 1]
                check_curve(actual, old, c)
    except ProposalBoundary:
        paused = True
    if not paused:
        raise RuntimeError("R1_PREPROPOSAL_HOOK_WAS_NOT_REACHED")
    validate_pending(engine, stream, c)
    current = {
        "step": engine.step + 1,
        "input_sha256": fingerprint(engine.pending_batch),
        "ce": float(torch.stack(engine.pending_losses).mean()),
    }
    if historical_curve:
        check_curve(current, historical_curve[engine.step], c)
    engine.save(directory / "preproposal.pt", stream)
    values = pending_values(engine)
    torch.save(values, directory / "pending.tmp")
    os.replace(directory / "pending.tmp", directory / "pending.pt")
    proof = {
        "status": "MAIN50_COMPLETE_BEFORE_FIRST_CONNECTION_SEARCH",
        "identity": identity,
        "engine_step_before_logical_commit": engine.step,
        "executed_main_updates": engine.step + 1,
        "effective_replay_target_tokens": (engine.step + 1) * c["targets_per_update"],
        "data_cursor_batches": stream.step,
        "counterfactual_optimizer_calls": 0,
        "formal_connection_commits": 0,
        "pending_values_sha256": fingerprint(values),
        "scientific_state_sha256": scientific_fingerprint(engine.state()),
        "pending_main_curve_row": current,
        "files": {
            name: {"sha256": digest(directory / name), "bytes": (directory / name).stat().st_size}
            for name in ("preproposal.pt", "pending.pt")
        },
        "timing_excluded_from_repeat_scientific_fingerprint": ["fast.training_seconds"],
        "checkpoint_write_seconds": getattr(engine, "checkpoint_write_seconds", 0.0),
        "checkpoint_read_seconds": getattr(engine, "checkpoint_read_seconds", 0.0),
        "historical_runtime_bitwise_certification": "NOT_CLAIMED_FROM_INPUT_HASH_OR_CE_MATCHES",
    }
    atomic(directory / "PAUSE.json", proof)
    return proof


def check_curve(actual, expected, c):
    if actual["input_sha256"] != expected["input_sha256"]:
        raise RuntimeError("R1_HISTORICAL_TRAIN_BATCH_MISMATCH")
    tolerance = c["diagnostic"]["historical_ce_atol"] + c["diagnostic"]["historical_ce_rtol"] * abs(
        expected["ce"]
    )
    if abs(actual["ce"] - expected["ce"]) > tolerance:
        raise RuntimeError("R1_HISTORICAL_CE_TOLERANCE_EXCEEDED_NO_ADAPTIVE_RELAXATION")
