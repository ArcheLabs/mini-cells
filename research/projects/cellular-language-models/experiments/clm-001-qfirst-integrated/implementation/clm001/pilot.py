"""Explicitly authorized, bounded frozen real-model Oracle and isolated component timings."""

from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import torch
from .common import authorize, write_json, canonical_sha256, hash_accepted_state
from .model import Model
from .training import Trainer
from .data import load_assets, batches
from .evaluation import evaluate, q_diagnostics
from .accounting import profile
from .checkpoint import save, identity
from .shadows import Nursery


def run(c, assets, output, device, source):
    authorize(c, "gpu-pilot", device)
    if not str(device).startswith("cuda"):
        raise ValueError("authorized GPU Pilot needs CUDA")
    data = load_assets(c, assets)
    output = Path(output)
    results = {}

    def timed(call):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        value = call()
        torch.cuda.synchronize()
        return value, time.perf_counter() - begin

    # Two fresh 1M models. Frozen-snapshot labels use Q-calibration, never confirmation.
    for arm in c["arms"]:
        model = Model(c, arm).to(device)
        before = hash_accepted_state(model)
        oracle, oracle_wall = timed(lambda: q_diagnostics(model, data, c, "calibration"))
        repeated, repeat_wall = timed(lambda: q_diagnostics(model, data, c, "calibration"))
        if oracle != repeated or hash_accepted_state(model) != before:
            raise RuntimeError("ORACLE_UNTRUSTED: frozen repeatability or immutability failed")
        private = deepcopy(model)
        trainer = Trainer(private)
        rng = np.random.default_rng(c["seed"] + 900)
        updates = []
        for _ in range(3):
            batch = batches(data, c, rng, device)
            (row, _, _), wall = timed(lambda: trainer.update(batch))
            updates.append({**row, "synchronized_total_seconds": wall})
        nursery = Nursery()
        candidate = nursery.propose(private, "cell", trainer.opt)
        shadow_batch = batches(data, c, np.random.default_rng(c["seed"] + 901), device)
        _, shadow_wall = timed(
            lambda: nursery.probation(
                candidate, private, shadow_batch, c["shadow"]["private_train_steps"]
            )
        )
        _, admission_wall = timed(
            lambda: (
                evaluate(private, data, c, "admission", "sampled"),
                evaluate(candidate.model, data, c, "admission", "sampled"),
            )
        )
        run_id = identity(c, arm, source, data)
        _, checkpoint_wall = timed(
            lambda: save(output / arm / "pilot-checkpoints", trainer, nursery, rng, run_id)
        )
        median = float(np.median([v["synchronized_total_seconds"] for v in updates]))
        results[arm] = {
            "oracle": oracle,
            "oracle_repeat_wall_seconds": repeat_wall,
            "oracle_wall_seconds": oracle_wall,
            "update_rows": updates,
            "shadow_wall_seconds": shadow_wall,
            "admission_pair_wall_seconds": admission_wall,
            "checkpoint_wall_seconds": checkpoint_wall,
            "profile": profile(private, batch[0][0]),
            "projected_500_update_seconds": median * 500,
            "projection_excludes_final_analysis_and_cloud_setup": True,
            "frozen_snapshot_sha": before,
            "scientific_decision": False,
        }
    result = {
        "experiment_id": c["experiment_id"],
        "status": "GPU_PILOT_REVIEW_REQUIRED",
        "source_commit": source,
        "asset_sha256": data["asset_sha256"],
        "tokenizer_sha256": data["tokenizer_sha256"],
        "config_sha256": canonical_sha256(c),
        "arms": results,
        "scientific_decision": False,
        "formal_confirmation": False,
    }
    write_json(output / "pilot.json", result)
    return result
