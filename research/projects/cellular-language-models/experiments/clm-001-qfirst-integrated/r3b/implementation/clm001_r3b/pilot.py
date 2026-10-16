"""Three original private Train updates per arm, not 500-step science.

Two queues use the same immutable-handoff/host-publisher implementation as the
real runner. Original R3 beta selection and direction audit are retained. No GPU
fault-injection suite, TorchDispatch or full research run is launched here.
"""

from copy import deepcopy
from pathlib import Path
import os
import time
import numpy as np
from minicells_runtime.store import atomic, canonical, read, digest, ContinuationError


def worker_entry(root, arms, end, physical_device, messages, acknowledgements, stop, *, cpu=False):
    if cpu:
        raise ContinuationError("GPU_PILOT_REQUIRES_CUDA")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    import torch
    from .common import ENGINE, identity, safe_error
    from .runner import handoff
    from .instrumentation import banks, snapshot, isolated
    from clm001_r3.common import authorize
    from clm001_r3.model import Model
    from clm001_r3.training import Trainer
    from clm001_r3.data import load_assets, batches
    from clm001_r3.shadows import Nursery
    from clm001_r3 import checkpoint as cp
    from clm001_r3.audit import capture, measure

    root = Path(root)
    c = read(root / "runtime-config.json")
    source = read(root / "experiment.lock.json")["source_sha"]
    try:
        if not torch.cuda.is_available():
            raise ContinuationError("GPU_PILOT_REQUIRES_CUDA")
        authorize(c, "gpu-pilot", "cuda:0")
        data = load_assets(c, root / "assets")
        for arm in arms:
            if stop.is_set():
                break
            folder = root / "arms" / arm
            folder.mkdir(parents=True, exist_ok=True)
            torch.cuda.reset_peak_memory_stats()
            model = Model(c, ENGINE[arm]).to("cuda:0")
            trainer, nursery = Trainer(model), Nursery()
            rng = np.random.default_rng(c["seed"] + 100)
            ident = identity(c, arm, source, data)
            atomic(folder / "identity.json", ident)
            frozen = banks(model, data, c, folder, ident, pilot=True)
            cp.save(folder, trainer, nursery, rng, ident)
            # Calibration-only observation; snapshot timing is retained separately.
            observed = snapshot(
                trainer, data, c, frozen, folder, "post_event", quality_role="calibration"
            )
            rows = []
            for _ in range(c["resources"]["pilot_steps"]):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                row, _, _ = trainer.update(batches(data, c, rng, "cuda:0"))
                torch.cuda.synchronize()
                rows.append(
                    {
                        "seconds": time.perf_counter() - begin,
                        **{
                            k: row.get(k)
                            for k in (
                                "q_grad_norm",
                                "lcg_grad_norm",
                                "lcg_q_cosine",
                                "lcg_candidate_cell_calls",
                                "lcg_events",
                            )
                        },
                    }
                )
            # Full production-sized optimizer/Registry plus three private candidate
            # states, without scientific proposal/admission or a GPU fault suite.
            for kind in ("cell", "connection", "path"):
                nursery.propose(model, kind, trainer.opt)
            cp.save(folder, trainer, nursery, rng, ident)
            chosen = read(folder / "latest.json")
            # A single continuation comparison on the private rehearsal only.
            expected = trainer.update(batches(data, c, rng, "cuda:0"))[0]
            expected_model = deepcopy(model.state_dict())
            expected_opt = deepcopy(trainer.opt.state_dict())
            expected_cpu_rng = torch.get_rng_state().clone()
            expected_cuda_rng = torch.cuda.get_rng_state().clone()
            expected_generators = {k: g.get_state().clone() for k, g in trainer.rng.items()}
            expected_data_rng = deepcopy(rng.bit_generator.state)
            restored = Trainer(Model(c, ENGINE[arm]).to("cuda:0"))
            rrng = np.random.default_rng(0)
            cp.restore(folder, restored, rrng, ident)
            actual = restored.update(batches(data, c, rrng, "cuda:0"))[0]

            def equivalent(a, b):
                if isinstance(a, torch.Tensor):
                    return (
                        torch.allclose(a, b, rtol=1e-5, atol=1e-7)
                        if a.is_floating_point()
                        else torch.equal(a, b)
                    )
                if isinstance(a, dict):
                    return set(a) == set(b) and all(equivalent(a[k], b[k]) for k in a)
                if isinstance(a, (list, tuple)):
                    return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
                return a == b

            restored_ok = (
                equivalent(expected_model, restored.model.state_dict())
                and equivalent(expected_opt, restored.opt.state_dict())
                and abs(expected["live_ce"] - actual["live_ce"]) <= 1e-5
                and trainer.data_position == restored.data_position
            )
            if not restored_ok:
                raise ContinuationError("GPU_RESTORE_TRAJECTORY_MISMATCH")
            all_rng_equal = (
                torch.equal(expected_cpu_rng, torch.get_rng_state())
                and torch.equal(expected_cuda_rng, torch.cuda.get_rng_state())
                and all(
                    torch.equal(value, restored.rng[k].get_state())
                    for k, value in expected_generators.items()
                )
                and canonical(expected_data_rng) == canonical(rrng.bit_generator.state)
            )
            if not all_rng_equal:
                raise ContinuationError("GPU_RESTORE_RNG_MISMATCH")
            alignment = None
            if arm == "QC-R3b":
                # Unchanged R3 Train-gradient beta rule; held-out Calibration is a
                # separate direction gate, never Development tuning.
                with isolated(restored.model, restored):
                    fixed, missing = capture(restored.model, data, c)
                    audit = measure(restored.model, fixed, c, resampled=True)
                alignment = {"summary": audit["summary"], "coverage_missing": len(missing)}
            report = {
                "status": "PRIVATE_GPU_REHEARSAL_COMPLETE",
                "arm": arm,
                "pilot_steps": len(rows),
                "gradient_rows": rows,
                "p90_step_seconds": float(np.quantile([r["seconds"] for r in rows], 0.9)),
                "checkpoint_bytes": (folder / chosen["file"]).stat().st_size,
                "calibration_snapshot_seconds": observed["elapsed_seconds"] if observed else None,
                "restore_pass": restored_ok,
                "restore_rtol": 1e-5,
                "restore_atol": 1e-7,
                "model_and_optimizer_compared": True,
                "cuda_rng_compared": all_rng_equal,
                "alignment": alignment,
                "torch_version": str(torch.__version__),
                "numpy_version": str(np.__version__),
                "cuda_runtime_version": str(torch.version.cuda),
                "device_name": torch.cuda.get_device_name(),
                "physical_gpu": str(physical_device),
                "device_uuid": str(
                    getattr(torch.cuda.get_device_properties(0), "uuid", "UNAVAILABLE")
                ),
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                "formal_training_started": False,
                "calibration_only": True,
                "parameters": sum(p.numel() for p in model.parameters()),
                "private_candidate_count": len(nursery.candidates),
            }
            atomic(folder / "gpu-pilot.json", report)
            frozen_path = handoff(folder, root.parent / "handoffs" / arm, phase="pilot")
            token = arm + ":pilot"
            messages.put(
                {
                    "kind": "checkpoint",
                    "queue": physical_device,
                    "arm": arm,
                    "path": str(frozen_path),
                    "token": token,
                }
            )
            ack = acknowledgements.get()
            if ack.get("token") != token or not ack.get("ok"):
                raise ContinuationError("STORAGE_BLOCKED")
            del trainer, model, nursery, restored, frozen, expected_model, expected_opt
            messages.put(
                {
                    "kind": "arm-done",
                    "queue": physical_device,
                    "result": {"arm": arm, "step": c["resources"]["pilot_steps"]},
                }
            )
        messages.put({"kind": "queue-done", "queue": physical_device})
    except Exception as error:
        messages.put({"kind": "worker-failed", "queue": physical_device, **safe_error(error)})


def summarize(candidate, root, source):
    from .common import ARMS
    from clm001_r3.automatic import select_beta

    root = Path(root)
    records = {arm: read(root / "arms" / arm / "gpu-pilot.json") for arm in ARMS}
    beta, selection = select_beta(candidate["engine"], records["QC-R3b"])
    alignment = records["QC-R3b"]["alignment"]["summary"]["status"] == "LOCAL_ALIGNMENT_PASS"
    uploads = read(root.parent / (root.name + "-evidence") / "resource-ledger.json")["uploads"]
    # Calibration snapshots undercount real Development and independent-bank
    # measurements. Explicit conservative extrapolation, never fabricated timing.
    ratio = (
        candidate["engine"]["evaluation"]["development_windows"]
        / candidate["engine"]["evaluation"]["calibration_windows"]
    )
    costs = {
        arm: candidate["engine"]["resources"]["projection_safety_factor"]
        * (
            r["p90_step_seconds"] * 500
            + (r["calibration_snapshot_seconds"] or 0)
            * max(2, ratio)
            * len(candidate["diagnostics"]["steps"])
        )
        for arm, r in records.items()
    }
    projection = max(costs["Q-R3b"] + costs["QS-R3b"], costs["QC-R3b"]) + sum(
        r["seconds"] for r in uploads
    )
    t4 = (
        all("T4" in r["device_name"] for r in records.values())
        and records["QC-R3b"]["physical_gpu"] != records["Q-R3b"]["physical_gpu"]
    )
    result = {
        "status": "GPU_PILOT_MEASURED_PENDING_REVIEW",
        "source_sha": source,
        "candidate_sha256": canonical(candidate),
        "assets_manifest_sha256": digest(root / "assets/manifest.json"),
        "tokenizer_sha256": digest(root / "assets/tokenizer.json"),
        "beta": beta,
        "beta_selection": selection,
        "arms": records,
        "gradient_and_alignment_pass": alignment,
        "diagnostic_plan_sha256": canonical(candidate["diagnostics"]),
        "snapshot_and_event_budget_reviewed": False,
        "two_t4_hardware": t4,
        "checkpoint_total_bytes": sum(r["checkpoint_bytes"] for r in records.values()),
        "projected_total_seconds": projection,
        "resource_target_met": projection <= candidate["execution"]["target_makespan_seconds"],
        "projection_scope": "measured synchronized updates + conservative calibration-to-development snapshot extrapolation; event cost/bandwidth interval requires review",
        "storage": uploads,
        "formal_training_authorized": False,
        "scientific_decision": False,
        "missing_before_gate_D_pass": [
            "review production snapshot/Event overhead and freeze upload budget",
            "review private GPU resume evidence and actual HF readback",
            "authorize frozen scientific runtime separately",
        ],
    }
    atomic(root.parent / (root.name + "-evidence") / "gpu-pilot.json", result)
    return result
