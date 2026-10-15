"""Independent R2 worker with pre-event training commit and recoverable lifecycle cursor."""

from pathlib import Path
import json
import time
import traceback
import numpy as np
from .common import authorize, source_sha, write_json
from .model import Model
from .training import Trainer
from .shadows import Nursery
from .data import load_assets, batches
from .checkpoint import identity, save, restore, export_portable
from .events import process, pending
from .evaluation import evaluate, q_diagnostics
from .accounting import parameters
from dataclasses import asdict
from .report import seal, verified


def development(c, arm, output, assets, device):
    authorize(c, "train-dev", device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_assets(c, assets)
    model = Model(c, arm).to(device)
    trainer = Trainer(model)
    nursery = Nursery()
    rng = np.random.default_rng(c["seed"] + 10)
    run_id = identity(c, arm, source_sha(), data)
    write_json(output / "identity.json", run_id)
    try:
        if (output / "latest.json").exists():
            nursery = restore(output, trainer, rng, run_id)
        if (output / "metrics.json").exists():
            value = verified(output / "metrics.json")
            if value["identity"] != run_id:
                raise ValueError("completed identity mismatch")
            return value
        begin = time.perf_counter()
        # Complete only unfinished events after a committed training step.
        if pending(c, trainer.step):
            process(c, trainer, nursery, data, rng, output, run_id)
        while trainer.step < c["training"]["steps"]:
            rng_before = rng.bit_generator.state
            try:
                row, snapshot, labels = trainer.update(batches(data, c, rng, device))
            except Exception:
                rng.bit_generator.state = rng_before
                raise
            write_json(
                output / "diagnostics" / ("oracle-step-%06d.json" % trainer.step),
                {
                    "identity": run_id,
                    "step": trainer.step,
                    "snapshot_sha": snapshot.sha,
                    "labels": [asdict(label) for state, target, label in labels],
                    "scope": "LOCAL_TRAIN_LABELS_NEVER_PUBLISHED",
                },
            )
            print(json.dumps(row), flush=True)
            step = trainer.step
            if (
                pending(c, step)
                or step % c["training"]["checkpoint_interval"] == 0
                or step == c["training"]["steps"]
            ):
                save(output, trainer, nursery, rng, run_id)
                seal(
                    output / "progress.json",
                    {
                        "identity": run_id,
                        "last_committed_step": step,
                        "live_training_tokens": trainer.data_position,
                        "last_update": row,
                        "counterfactual_operator_calls": sum(
                            r["counterfactual_operator_calls"] for r in trainer.journal
                        ),
                        "committed_update_seconds": sum(
                            r["total_seconds"] for r in trainer.journal
                        ),
                    },
                )
            if pending(c, step):
                process(c, trainer, nursery, data, rng, output, run_id)
            if step % c["training"]["evaluation_interval"] == 0:
                seal(
                    output / ("development-step-%06d.json" % step),
                    {"step": step, "quality": evaluate(model, data, c, "development")},
                )
        pre = next((v for v in nursery.events.values() if v["event"] == "pre-admission"), None)
        failures = [v for v in nursery.candidates if v.state == "FAILED"]
        result = {
            "identity": run_id,
            "arm": arm,
            "status": "DEVELOPMENT_COMPLETE",
            "training_status": "COMPLETE",
            "scientific_decision": False,
            "formal_confirmation": False,
            "pre_admission": pre,
            "quality": {
                p: evaluate(model, data, c, "development", p)
                for p in ("greedy", "sampled", "uniform", "constant")
            },
            "interventions": {
                p: evaluate(model, data, c, "development", intervention=p)
                for p in ("context_zero", "state_permuted", "previous_permuted", "cell_swapped")
            },
            "q_calibration": q_diagnostics(model, data, c),
            "shadow_status": "PARTIAL_FAILURE"
            if failures
            else (
                "PROMOTED"
                if any(v.state == "PROMOTED" for v in nursery.candidates)
                else "ALL_REJECTED"
            ),
            "shadow_summary": [
                {
                    "kind": v.kind,
                    "state": v.state,
                    "passed": v.metrics.get("passed", False),
                    "failure_code": v.metrics.get("failure", {}).get("failure_code"),
                    "proposed_parent_sha": v.metrics.get("proposed_parent_sha"),
                    "rebased_parent_sha": v.metrics.get("rebased_parent_sha"),
                }
                for v in nursery.candidates
            ],
            "parameters": parameters(model),
            "cost_profile": {"status": "DEFERRED_UNTIL_ALL_WORKERS_FINISH"},
            "resource_ledger": {
                "live_training_tokens": trainer.data_position,
                "counterfactual_operator_calls": sum(
                    r["counterfactual_operator_calls"] for r in trainer.journal
                ),
                "shadow_private_training_tokens": sum(
                    v.trainer.data_position for v in nursery.candidates
                ),
                "oracle_seconds": sum(r["oracle_seconds"] for r in trainer.journal),
                "elapsed_seconds_current_attempt": time.perf_counter() - begin,
                "committed_update_seconds_all_attempts": sum(
                    r["total_seconds"] for r in trainer.journal
                ),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            },
        }
        seal(output / "metrics.json", result)
        if not (output / "portable").exists():
            export_portable(output, assets, output / "portable", run_id, c)
        return result
    except Exception as error:
        seal(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "last_in_memory_step": trainer.step,
                "source_sha": run_id["source_commit"],
                "config_hash": run_id["config_sha256"],
                "arm": arm,
                "scope": "WORKER_FAILURE",
                "complete": False,
            },
        )
        # Native traceback remains local; only safe failure fields enter publication summaries.
        (output / "failure-traceback.txt").write_text(traceback.format_exc())
        raise
