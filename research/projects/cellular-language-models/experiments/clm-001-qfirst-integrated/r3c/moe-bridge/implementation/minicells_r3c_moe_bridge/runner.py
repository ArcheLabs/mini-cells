"""Bounded synthetic CPU engineering, and an authorization-guarded future GPU smoke."""

from pathlib import Path
import random
import numpy as np
import torch
from .common import ARMS, ID, canonical, atomic, fingerprint, synthetic_configuration
from .model import BridgeModel
from .training import Trainer
from .checkpoint import Store
from .data import synthetic_batches
from .instrumentation import parameters, MatmulAudit
from .provenance import provenance
from .release import require_gpu_smoke


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
    """Compatibility adapter to the same resumable bounded pilot engine."""
    require_gpu_smoke(c, authorization)
    if c.get("synthetic"):
        raise ValueError("real asset pilot requires proposed real configuration")
    from .data import load_assets
    from .pilot import execute_pilot

    if not torch.cuda.is_available():
        raise RuntimeError("authorized smoke requires CUDA")
    data = load_assets(c, assets)
    return execute_pilot(c, data, output, provenance(c)["source_sha"], device)
