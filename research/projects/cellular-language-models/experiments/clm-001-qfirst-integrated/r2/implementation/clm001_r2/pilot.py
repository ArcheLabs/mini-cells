"""Real GPU private rehearsal, not a scientific training result."""

from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import torch
from .common import authorize, canonical_sha256, source_sha
from .model import Model
from .training import Trainer, NumericalFailure
from .shadows import Nursery
from .checkpoint import identity, save, restore
from .data import load_assets, batches
from .report import seal


def run(c, assets, output, arm, device):
    authorize(c, "gpu-pilot", device)
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("GPU Pilot requires actual CUDA; never fabricate CPU GPU timings")
    output = Path(output)
    data = load_assets(c, assets)
    m = Model(c, arm).to(device)
    t = Trainer(m)
    nursery = Nursery()
    rng = np.random.default_rng(c["seed"] + 100)
    rows = []
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(c["resources"]["pilot_steps"]):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        row, _, labels = t.update(batches(data, c, rng, device))
        torch.cuda.synchronize(device)
        row["synchronized_seconds"] = time.perf_counter() - start
        if not labels:
            raise RuntimeError("Pilot failed to exercise Oracle")
        rows.append(row)
    private = batches(data, c, rng, device)
    shadow_start = time.perf_counter()
    candidate_results = []
    for kind in ("cell", "connection", "path"):
        candidate = nursery.propose(m, kind, t.opt)
        if not nursery.probation(
            candidate, m, private, c["shadow"]["private_train_steps"], accepted_trainer=t
        ):
            raise RuntimeError("GPU candidate probation failed: " + kind)
        if kind == "cell" and not any(
            p.grad is not None and bool(torch.isfinite(p.grad).all())
            for p in candidate.model.cells[candidate.model.registry.cells].parameters()
        ):
            raise RuntimeError("GPU Shadow reserve slot received no finite gradient")
        candidate_results.append({"kind": kind, "state": candidate.state, "backward_pass": True})
    # Actual backward hooks, not a fabricated exception from a stubbed optimizer.
    fault_results = []
    for kind in ("cell", "connection", "path"):
        injection = Nursery()
        candidate = injection.propose(m, kind, t.opt)
        param = next(
            p
            for name, p in candidate.model.named_parameters()
            if p.requires_grad and name.startswith("controller.scorer.")
        )
        hook = param.register_hook(lambda grad: torch.full_like(grad, float("nan")))
        try:
            passed = injection.probation(
                candidate, m, private, c["shadow"]["private_train_steps"], accepted_trainer=t
            )
        finally:
            hook.remove()
        if passed or candidate.state != "FAILED":
            raise RuntimeError("GPU private fault was not isolated")
        fault_results.append(
            {"kind": kind, "failure_code": candidate.metrics["failure"]["failure_code"]}
        )
    # Accepted faults must fail closed and restore its exact private rehearsal bundle.
    from .integrity import Witness
    from .oracle import Snapshot, OracleUntrusted

    witness = Witness(m, t)
    param = next(p for p in m.parameters() if p.requires_grad)
    hook = param.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    try:
        try:
            t.update(private)
        except NumericalFailure:
            pass
        else:
            raise RuntimeError("accepted numerical failure was swallowed")
    finally:
        hook.remove()
    witness.assert_unchanged()
    broken = Snapshot(m)
    with torch.no_grad():
        next(broken.model.parameters()).add_(1.0)
    try:
        broken.assert_integrity()
    except OracleUntrusted:
        pass
    else:
        raise RuntimeError("Oracle corruption was swallowed")
    torch.cuda.synchronize(device)
    shadow_seconds = time.perf_counter() - shadow_start
    run_id = identity(c, arm, source_sha(), data)
    save(output / "restart", t, nursery, rng, run_id)
    # Compare continuation from identical model, optimizers and RNG, not just tensor loads.
    expected = t.update(batches(data, c, rng, device))[0]
    expected_weights = deepcopy(m.state_dict())
    restored = Trainer(Model(c, arm).to(device))
    restored_rng = np.random.default_rng(1)
    restore(output / "restart", restored, restored_rng, run_id)
    actual = restored.update(batches(data, c, restored_rng, device))[0]
    equal = all(
        torch.allclose(v, restored.model.state_dict()[k], rtol=1e-5, atol=1e-7)
        for k, v in expected_weights.items()
    )
    if not equal or abs(expected["live_ce"] - actual["live_ce"]) > 1e-5:
        raise RuntimeError("GPU restart trajectory differs")
    values = [r["synchronized_seconds"] for r in rows]
    p90 = float(np.quantile(values, 0.9))
    projection = c["resources"]["projection_safety_factor"] * (
        p90 * c["training"]["steps"]
        + shadow_seconds
        + 2 * p90 * c["evaluation"]["development_windows"]
    )
    result = {
        "status": "GPU_PILOT_PASS"
        if projection <= c["resources"]["target_makespan_seconds"]
        else "GPU_BUDGET_PROJECTION_FAILED",
        "arm": arm,
        "source_sha": source_sha(),
        "config_hash": canonical_sha256(c),
        "asset_sha256": data["asset_sha256"],
        "tokenizer_sha256": data["tokenizer_sha256"],
        "p50_step_seconds": float(np.median(values)),
        "p90_step_seconds": p90,
        "projected_arm_seconds": projection,
        "projection_includes": "500 updates + measured Shadow + conservative evaluation allowance; excludes initial downloads/publication",
        "shadow_candidates": candidate_results,
        "local_restore_pass": equal,
        "private_failure_isolation_pass": True,
        "fault_injection": fault_results,
        "accepted_and_oracle_fail_closed": True,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "device_name": torch.cuda.get_device_name(device),
        "scientific_decision": False,
        "training_status": "NOT_TRAINED",
    }
    seal(output / "gpu-pilot.json", result)
    return result
