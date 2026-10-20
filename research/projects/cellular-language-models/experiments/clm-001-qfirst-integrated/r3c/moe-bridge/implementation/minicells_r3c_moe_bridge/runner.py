"""Bounded synthetic CPU engineering, and an authorization-guarded future GPU smoke."""

from pathlib import Path
import time
import random
import numpy as np
import torch
from .common import ARMS, ID, canonical, atomic, fingerprint, synthetic_configuration
from .model import BridgeModel
from .training import Trainer
from .checkpoint import Store
from .data import synthetic_batches, train_batches, fixed_windows
from .instrumentation import parameters, MatmulAudit
from .provenance import provenance
from .release import require_gpu_smoke, choose_balance


def identity(c, arm, seed, source):
    return {
        "experiment_id": ID,
        "arm": arm,
        "seed": seed,
        "source_sha": source,
        "config_sha256": canonical(c),
        "data_sha256": "SYNTHETIC_TWO_FAMILY_V1"
        if c.get("synthetic")
        else c["input"]["assets_manifest_sha256"],
        "tokenizer_sha256": "SYNTHETIC_INTEGER_VOCAB"
        if c.get("synthetic")
        else c["input"]["tokenizer_sha256"],
        "confirmation": "SEALED",
        "scientific_decision": False,
    }


def engineering(output):
    c = synthetic_configuration()
    meta = provenance(c)
    output = Path(output)
    rows = []
    parity = []
    torch.set_num_threads(1)
    for arm in ARMS:
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        trainer = Trainer(BridgeModel(c, arm, 123))
        ident = identity(c, arm, 123, meta["source_sha"])
        store = Store(output / "private-checkpoints" / arm, ident)
        logs = []
        for _ in range(2):
            logs.append({**ident, **trainer.update(synthetic_batches(c, trainer.data_rng))})
        receipt = store.save(trainer)
        # Run one next update, then restore into a new Trainer and repeat exactly.
        expected = trainer.update(synthetic_batches(c, trainer.data_rng))
        state = fingerprint(trainer.state())
        restored = Trainer(BridgeModel(c, arm, 123))
        store.restore(restored, receipt)
        actual = restored.update(synthetic_batches(c, restored.data_rng))
        same = fingerprint(restored.state()) == state
        if not same:
            raise RuntimeError("exact synthetic CPU resume failed: " + arm)
        x, y = synthetic_batches(c, np.random.default_rng(99))[0]
        with torch.no_grad(), MatmulAudit(restored.model) as audit:
            out = restored.model(x, route_rng=torch.Generator().manual_seed(99))
        rows.append(
            {
                "arm": arm,
                "parameters": parameters(restored.model),
                "cost": audit.finish(out),
                "updates": logs,
                "checkpoint_sha256": receipt["sha256"],
            }
        )
        parity.append(
            {
                "arm": arm,
                "exact_full_state": same,
                "next_ce_exact": expected["ce"] == actual["ce"],
                "cross_session_verified": False,
                "scope": "LOCAL_CPU_RESTORE_NOT_KAGGLE_PERSISTENCE",
            }
        )
    report = {
        "schema": "minicells.r3c.engineering.v1",
        "status": "CPU_ENGINEERING_COMPLETE_GPU_PENDING",
        "scope": "SYNTHETIC_CPU_ONLY",
        "provenance": meta,
        "arms": rows,
        "resume": parity,
        "gpu_pilot_completed": False,
        "actual_kaggle_persistence_verified": False,
        "full_engineering_pass": False,
        "scientific_decision": False,
    }
    atomic(output / "reports/engineering.json", report)
    from .reports import write_reports

    write_reports(output, report)
    return report


def gpu_smoke(c, output, assets, authorization, device="cuda:0"):
    require_gpu_smoke(c, authorization)  # Must precede device allocation and data loading.
    if c.get("synthetic"):
        raise ValueError("real asset pilot requires proposed real configuration")
    from .data import load_assets

    data = load_assets(c, assets)
    if not torch.cuda.is_available():
        raise RuntimeError("authorized smoke requires CUDA")
    output = Path(output)
    measurements = {}
    source = provenance(c)["source_sha"]
    seed = c["candidate_training"]["seeds"][0]
    for coefficient in c["balance"]["candidates"]:
        trainer = Trainer(BridgeModel(c, "C", seed).to(device), coefficient)
        for _ in range(c["balance"]["pilot_steps_per_candidate"]):
            trainer.update(train_batches(data, c, trainer.data_rng, device))
        shares = []
        with torch.no_grad():
            for story, x, y in fixed_windows(data, c, "calibration", device):
                out = trainer.model(x)
                shares.append(
                    (out.operator_rows / out.operator_rows.sum(-1, keepdim=True)).cpu().tolist()
                )
                if len(shares) == c["balance"]["measurement_windows"]:
                    break
        measurements[coefficient] = {"role": "calibration", "cell_shares": shares}
    selected = choose_balance(c, measurements)
    if selected["status"] == "NO_GO":
        atomic(output / "reports/pilot.json", selected)
        return selected
    results = []
    for arm in ARMS:
        trainer = Trainer(BridgeModel(c, arm, seed).to(device), selected["coefficient"])
        logs = []
        ident = {
            **identity(c, arm, seed, source),
            "pilot_balance_coefficient": selected["coefficient"],
        }
        for _ in range(c["pilot"]["steps_per_arm"]):
            batches = train_batches(data, c, trainer.data_rng, device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.reset_peak_memory_stats()
            start.record()
            begin = time.perf_counter()
            row = trainer.update(batches)
            end.record()
            end.synchronize()
            wall = time.perf_counter() - begin
            row.update(
                cuda_time_ms=start.elapsed_time(end),
                wall_seconds=wall,
                tokens_per_second=c["training"]["effective_batch"]
                * c["training"]["sequence_length"]
                / wall,
            )
            logs.append({**ident, **row})
        with torch.no_grad(), MatmulAudit(trainer.model) as audit:
            out = trainer.model(batches[0][0], route_rng=torch.Generator().manual_seed(seed))
        checkpoint = Store(output / "private-checkpoints" / arm, ident).save(trainer)
        results.append(
            {
                "arm": arm,
                "logs": logs,
                "inference_matmul_cost": audit.finish(out),
                "parameters": parameters(trainer.model),
                "checkpoint": checkpoint,
                "projected_500_update_seconds": 500
                * float(np.median([r["wall_seconds"] for r in logs])),
                "projection_scope": "short synthetic-shape Train updates only; excludes assets, evaluation, saving and HF upload",
            }
        )
    report = {
        "status": "GPU_SMOKE_COMPLETED_NOT_FORMALLY_RELEASED",
        "arms": results,
        "selection": selected,
        "measurements": {str(k): v for k, v in measurements.items()},
        "optimizer_updates": c["pilot"]["total_optimizer_updates"],
        "cross_session_verified": False,
        "scientific_decision": False,
    }
    atomic(output / "reports/pilot.json", report)
    return report
