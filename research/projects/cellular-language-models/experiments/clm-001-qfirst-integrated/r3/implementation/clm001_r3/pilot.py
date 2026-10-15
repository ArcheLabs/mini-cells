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


def _run(c, assets, output, arm, device):
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
    fault_results = [private_fault(m, t, private, kind) for kind in ("cell", "connection", "path")]
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
    # Isolated accounting only: no TorchDispatch mode in ordinary training or timing samples.
    from pd001.accounting import TrainingGemmCounter

    accounting_rng = np.random.default_rng(c["seed"] + 900)
    with TrainingGemmCounter() as counter:
        restored.update(batches(data, c, accounting_rng, device))
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
        "gradient_rows": [
            {
                k: row.get(k)
                for k in (
                    "q_grad_norm",
                    "lcg_grad_norm",
                    "lcg_q_cosine",
                    "lcg_candidate_cell_calls",
                    "lcg_events",
                )
            }
            for row in rows
        ],
        "source_sha": source_sha(),
        "config_hash": canonical_sha256(c),
        "asset_sha256": data["asset_sha256"],
        "tokenizer_sha256": data["tokenizer_sha256"],
        "p50_step_seconds": float(np.median(values)),
        "p90_step_seconds": p90,
        "projected_arm_seconds": projection,
        "shadow_rehearsal_seconds": shadow_seconds,
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
        "isolated_training_gemm_flops": counter.flops,
        "training_flops_scope": "single private update: live forward/backward + Oracle + candidate credit; GEMM lower bound, not full-run total",
    }
    seal(output / "gpu-pilot.json", result)
    return result


def private_fault(model, trainer, batches, kind):
    """Inject only in rehearsal copies; assert an actual backward hook fired.

    A fixed J0 path has no scorer gradient. Hook every trainable private tensor
    instead of assuming scorer participation; J0's critic remains trainable.
    """
    nursery = Nursery()
    candidate = nursery.propose(model, kind, trainer.opt)
    fired = []

    def inject(name):
        def hook(gradient):
            fired.append(name)
            return torch.full_like(gradient, float("nan"))

        return hook

    handles = [
        parameter.register_hook(inject(name))
        for name, parameter in candidate.model.named_parameters()
        if parameter.requires_grad
    ]
    try:
        passed = nursery.probation(
            candidate,
            model,
            batches,
            model.experiment_config["shadow"]["private_train_steps"],
            accepted_trainer=trainer,
        )
    finally:
        for handle in handles:
            handle.remove()
    if not fired:
        raise RuntimeError(
            "GPU private fault injection did not reach any backward gradient: "
            + model.arm
            + "/"
            + kind
        )
    if (
        passed
        or candidate.state != "FAILED"
        or candidate.metrics["failure"]["failure_code"] != "CANDIDATE_NUMERICAL_FAILURE"
    ):
        raise RuntimeError("GPU private fault was not isolated: " + model.arm + "/" + kind)
    return {
        "kind": kind,
        "failure_code": candidate.metrics["failure"]["failure_code"],
        "backward_hook_fired": True,
        "injected_parameters": sorted(set(fired)),
        "accepted_witness_ok": candidate.metrics["failure"]["accepted_witness_ok"],
    }


def run(c, assets, output, arm, device):
    try:
        return _run(c, assets, output, arm, device)
    except Exception as error:
        seal(
            Path(output) / "gpu-pilot-failure.json",
            {
                "arm": arm,
                "source_sha": source_sha(),
                "config_hash": canonical_sha256(c),
                "status": "GPU_PILOT_FAILED",
                "error_type": type(error).__name__,
                "message": str(error),
                "training_status": "NOT_TRAINED",
                "scientific_decision": False,
            },
        )
        raise
