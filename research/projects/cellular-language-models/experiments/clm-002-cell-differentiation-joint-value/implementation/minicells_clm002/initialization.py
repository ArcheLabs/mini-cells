"""Zero-update parent fan-out audit; this is not a scientific execution release."""

import copy

import torch
from minicells_r3e.common import fingerprint

from .common import atomic, digest, identity, sha
from .engine import Engine
from .native import build, classify, treatment


def fork_for_cpu_audit(c, state, seed, arm, ident, ledger, parent_sha, *, unit=False):
    """Keep parent values and counters distinct from a fresh experiment's empty stream.

    The policy below is a review candidate, not registered production semantics.
    No source data is opened and Engine.update remains blocked (unit=False).
    """
    before = fingerprint(state)
    model, runtime = build(c, seed, unit=unit)
    model.load_state_dict(state["model"], strict=True)
    model.temperature = state["r3h"]["temperature"]
    model.rho = state["r3h"]["rho"]
    if model.routing_mode != state["r3h"]["routing_mode"]:
        raise RuntimeError("CLM002_PARENT_ROUTER_MISMATCH")
    engine = Engine(model, runtime, arm, ident, ledger, unit=False)
    engine.cell_optimizer.load_state_dict(copy.deepcopy(state["cell_optimizer"]))
    engine.parent_step = state["step"]
    engine.parent_sha = parent_sha
    engine.parent_origin = {
        "checkpoint_sha256": parent_sha,
        "state_fingerprint": before,
        "historical_step": state["step"],
        "historical_data_cursor": state["data_cursor"],
        "historical_rng_sha256": fingerprint(
            {
                k: state[k]
                for k in (
                    "data_rng",
                    "route_rng",
                    "cf_rng",
                    "global_cpu_rng",
                    "global_cuda_rng",
                    "python_rng",
                    "numpy_rng",
                )
            }
        ),
        "historical_work_sha256": fingerprint(state["work"]),
        "initialization_policy": "REVIEW_CANDIDATE_PARENT_VALUES_AND_CELL_MOMENTS_NEW_SEED_STREAMS",
        "scientific_policy_registered": False,
    }
    cells, connection = treatment(arm)
    checks = {
        "model_values_exact": fingerprint(model.state_dict()) == fingerprint(state["model"]),
        "cell_optimizer_values_exact": fingerprint(engine.cell_optimizer.state_dict())
        == fingerprint(state["cell_optimizer"]),
        "mask_exact": all(
            p.requires_grad
            == {"cells": cells, "connection": connection, "public": False}[classify(n)]
            for n, p in model.named_parameters()
        ),
        "no_gradients": all(p.grad is None for p in model.parameters()),
        "new_experiment_counters_zero": engine.step == engine.tokens == 0 and engine.curve == [],
        "historical_counters_retained": engine.parent_step == state["step"]
        and engine.parent_origin["historical_data_cursor"] == state["data_cursor"],
        "no_optimizer_calls": engine.journal.summary()["actual_optimizer_attempts"] == 0,
        "source_state_unchanged": before == fingerprint(state),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "CLM002_PARENT_FORK_VALUES_OR_BOUNDARIES_CHANGED:"
            + ",".join(k for k, passed in checks.items() if not passed)
        )
    return engine, checks


def inspect_initialization(c, checkpoint, output):
    """Inspect the already identified original parent without forward/backward/update."""
    from pathlib import Path

    from .audit import inspect_parent

    torch.set_num_threads(1)
    output = Path(output)
    ident = identity(c)
    ident.update(scope="GATE0_CPU_PARENT_FORK_ONLY", operation="PARENT_FORK_AUDIT")
    ident.pop("run_id")
    ident["run_id"] = sha(ident)[:32]
    root = output / ident["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    parent = inspect_parent(c, checkpoint, root / "private")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state"]
    arms = c["stage_a"]["arms"] + c["stage_b"]["arms"]
    rows = {}
    # Engineering Seed only: do not consume the reserved scientific Seeds.
    for arm in arms:
        engine, checks = fork_for_cpu_audit(
            c,
            state,
            c["engineering"]["seed"],
            arm,
            ident,
            root / "private" / (arm + ".jsonl"),
            parent["parent_sha256"],
        )
        rows[arm] = dict(
            checks,
            model_sha256=fingerprint(engine.model.state_dict()),
            cell_optimizer_sha256=fingerprint(engine.cell_optimizer.state_dict()),
            parent_origin=engine.parent_origin,
        )
        for hook in engine._work_handles:
            hook.remove()
        del engine
    report = {
        "identity": ident,
        "status": "CPU_PARENT_FANOUT_PASS_PRODUCTION_INITIALIZATION_NOT_RELEASED",
        "parent_inspection": parent,
        "arms": rows,
        "scientific_policy_registered": False,
        "scientific_optimizer_updates": 0,
        "actual_optimizer_calls": 0,
        "model_forward_calls": 0,
        "gpu_used": False,
        "confirmation_materialized": False,
        "gate0": "BLOCKED",
        "limitations": [
            "New experiment RNG streams and continued parent Cell warmup are review candidates, not frozen scientific policy.",
            "CUDA RNG is retained as historical provenance only; no GPU resume certified.",
            "This checks an engineering Seed across Arm masks, not released production data streams or scientific Seeds.",
        ],
    }
    atomic(root / "PARENT_FORK_AUDIT.json", report)
    manifest = {
        "identity": ident,
        "experiment_id": c["experiment_id"],
        "run_id": ident["run_id"],
        "scope": ident["scope"],
        "source_sha": ident["source_sha"],
        "config_sha256": ident["config_sha256"],
        "scientific_optimizer_updates": 0,
        "confirmation_materialized": False,
        "run_status": report["status"],
        "artifact_files": {
            "PARENT_FORK_AUDIT.json": {
                "bytes": (root / "PARENT_FORK_AUDIT.json").stat().st_size,
                "sha256": digest(root / "PARENT_FORK_AUDIT.json"),
            }
        },
    }
    atomic(root / "RUN_MANIFEST.json", manifest)
    return root, report
