"""Reproducible CPU engineering entry and explicitly gated development/Pilot orchestration."""

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from .common import BASE, ROOT, load, write_json, canonical_sha256, source_sha, authorize
from .model import Model
from .training import Trainer
from .shadows import Nursery
from .data import prepare, load_assets, batches
from .checkpoint import identity, save, restore, export_portable, verify_portable
from .evaluation import evaluate, q_diagnostics
from .admission import assess, promote
from .accounting import parameters, profile
from .report import summarize


def cpu_gate(c, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests/research/clm001_qfirst")]
    import os

    env = {
        **os.environ,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "GIT_ALLOW_PROTOCOL": "file",
        "GIT_TERMINAL_PROMPT": "0",
    }
    result = subprocess.run(
        command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    print(result.stdout, flush=True)
    report = {
        "experiment_id": c["experiment_id"],
        "source_commit": source_sha(),
        "config_sha256": canonical_sha256(c),
        "status": "ENGINEERING_READY" if result.returncode == 0 else "ORACLE_UNTRUSTED",
        "test_output": result.stdout,
        "scope": "SYNTHETIC_CPU_ONLY",
        "real_asset_hashes_frozen": False,
        "gpu_pilot_completed": False,
        "scientific_decision": False,
    }
    write_json(output / "cpu-gate.json", report)
    inventory = {arm: parameters(Model(c, arm)) for arm in c["arms"]}
    if inventory[c["arms"][0]] != inventory[c["arms"][1]]:
        raise RuntimeError("matched parameter audit failed")
    write_json(
        output / "parameter-audit.json", {"experiment_id": c["experiment_id"], "arms": inventory}
    )
    if result.returncode:
        raise RuntimeError("CPU Oracle gate failed; no next phase")
    return report


def development(c, arm, output, asset_folder, device, phase):
    authorize(c, phase, device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_assets(c, asset_folder)
    if not c.get("synthetic") and (
        c.get("frozen_assets_sha256") != data["asset_sha256"]
        or c.get("frozen_tokenizer_sha256") != data["tokenizer_sha256"]
    ):
        raise RuntimeError("training assets/tokenizer must be frozen after approved Pilot")
    model = Model(c, arm).to(device)
    trainer = Trainer(model)
    nursery = Nursery()
    rng = np.random.default_rng(c["seed"] + 10)
    run_id = identity(c, arm, source_sha(), data)
    write_json(output / "identity.json", run_id)
    if (output / "latest.json").exists():
        nursery = restore(output, trainer, rng, run_id)
    report_file = output / "metrics.json"
    if trainer.step == c["training"]["steps"] and report_file.exists():
        return json.loads(report_file.read_text())
    attempt = time.time_ns()
    total_begin = time.perf_counter()
    evaluations = []
    shadow_tokens = 0
    while trainer.step < c["training"]["steps"]:
        row, _, labels = trainer.update(batches(data, c, rng, device))
        write_json(
            output / "oracle-labels" / ("attempt-%d" % attempt) / ("step-%06d.json" % trainer.step),
            {
                "step": trainer.step,
                "labels": [
                    {
                        **asdict(label),
                        "candidate_id": label.record_sha,
                        "candidate_type": "STOP"
                        if label.action == model.registry.stop
                        else ("CTX_READ" if label.action == model.registry.ctx else "CELL"),
                        "max_steps": model.config["budget"],
                        "source_state_hash": label.state_sha,
                        "proposal_probability": label.inclusion,
                        "source_action": state.chosen,
                        "alternative_action": label.action,
                        "source_model_sha": label.snapshot_sha,
                        "source_policy_sha": label.policy_sha,
                        "target_token_hash": label.target_sha,
                        "completion_trace_hash": canonical_sha256(label.traces),
                    }
                    for state, target, label in labels
                ],
            },
        )
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in ("step", "arm", "live_ce", "path_cost", "q_loss", "total_seconds")
                }
            ),
            flush=True,
        )
        step = trainer.step
        # Proposal blueprints have independent branches. Admission opens against the current parent below.
        if step == c["shadow"]["cell_proposal_step"]:
            nursery.propose(model, "cell", trainer.opt)
        if step == c["shadow"]["connection_path_step"]:
            nursery.propose(model, "connection", trainer.opt)
            nursery.propose(model, "path", trainer.opt)
        if step == c["shadow"]["admission_step"]:
            baseline = evaluate(model, data, c, "admission", policy="sampled")
            passing = []
            for candidate in nursery.candidates:
                if candidate.state != "PROPOSED":
                    continue
                nursery.rebase(candidate, model, trainer.opt)
                private_seed = int(
                    torch.randint(2**31 - 1, (), generator=trainer.rng["shadow"], device=device)
                )
                private = batches(data, c, np.random.default_rng(private_seed), device)
                nursery.probation(candidate, model, private, c["shadow"]["private_train_steps"])
                shadow_tokens += (
                    sum(x.numel() for x, y in private) * c["shadow"]["private_train_steps"]
                )
                challenger = evaluate(candidate.model, data, c, "admission", policy="sampled")
                if assess(candidate, model, baseline, challenger, baseline["groups"], c):
                    passing.append(candidate)
            if passing:
                winner = min(passing, key=lambda x: x.metrics["cost_delta_upper95"])
                promote(winner, model, trainer)
                nursery.expire_stale(model)
        if step % c["training"]["evaluation_interval"] == 0:
            evaluations.append(
                {"step": step, "development": evaluate(model, data, c, "development")}
            )
        if step % c["training"]["checkpoint_interval"] == 0 or step == c["training"]["steps"]:
            digest = save(output, trainer, nursery, rng, run_id)
            row["checkpoint_sha256"] = digest
            write_json(
                output / "timings.json",
                {
                    "experiment_id": c["experiment_id"],
                    "steps": trainer.journal,
                    "shadow_training_tokens": shadow_tokens,
                    "elapsed_seconds": time.perf_counter() - total_begin,
                },
            )
    # No quality/label-generation reads from confirmation; Q-calibration cannot produce gradients.
    result = {
        "status": "DEVELOPMENT_COMPLETE",
        "arm": arm,
        "quality": {
            p: evaluate(model, data, c, "development", p)
            for p in ("greedy", "sampled", "soft", "uniform", "constant")
        },
        "interventions": {
            kind: evaluate(model, data, c, "development", intervention=kind)
            for kind in ("context_zero", "state_permuted", "previous_permuted", "cell_swapped")
        },
        "q_train": q_diagnostics(model, data, c, "train"),
        "q_calibration": q_diagnostics(model, data, c, "calibration"),
        "parameters": parameters(model),
        "cost_profile": profile(
            model,
            batches(data, c, np.random.default_rng(c["evaluation"]["seed"] + 17), device)[0][0],
            repeats=3,
        ),
        "development_history": evaluations,
        "live_tokens": trainer.data_position,
        "shadow_training_tokens": shadow_tokens,
        "timings": {"elapsed_seconds": time.perf_counter() - total_begin},
        "exploration_wall_seconds": sum(
            r["oracle_seconds"] + r["snapshot_seconds"] for r in trainer.journal
        ),
        "shadow_counts": {
            "shadow_proposed": len(nursery.candidates),
            "shadow_promoted": sum(v.state == "PROMOTED" for v in nursery.candidates),
            "shadow_rolled_back": sum(
                any(e["details"].get("reason") == "rollback" for e in v.journal)
                for v in nursery.candidates
            ),
        },
        "seed": c["seed"],
        "identity": run_id,
        "scientific_decision": False,
    }
    write_json(report_file, result)
    write_json(
        output / "shadow-journal.json",
        {
            "experiment_id": c["experiment_id"],
            "candidates": [
                {"kind": v.kind, "state": v.state, "journal": v.journal, "metrics": v.metrics}
                for v in nursery.candidates
            ],
        },
    )
    portable = output / "portable"
    if not portable.exists():
        export_portable(output, asset_folder, portable, run_id, c)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        default="auto",
        choices=(
            "auto",
            "cpu-preflight",
            "prepare",
            "gpu-pilot",
            "train-smoke",
            "train-dev",
            "analyze",
            "publish",
            "publish-preflight",
            "verify-portable",
        ),
    )
    p.add_argument("--config", default=str(BASE / "config/proposed.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets", type=Path)
    p.add_argument("--arm", choices=("QF-FULL", "J0-FULL"))
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    c = load(a.config)
    a.output.mkdir(parents=True, exist_ok=True)
    if a.mode == "auto":
        from .automatic import run

        if a.assets is None:
            a.assets = a.output.parent / "clm001-qfirst-assets"
        result = run(c, a.output, a.assets)
    elif a.mode == "cpu-preflight":
        result = cpu_gate(c, a.output)
    elif a.mode == "prepare":
        authorize(c, "gpu-pilot", a.device)
        result = prepare(c, a.assets)
    elif a.mode in ("train-smoke", "train-dev"):
        if not a.arm:
            raise ValueError("one of the two registered arms required")
        if a.mode == "train-smoke" and c["training"]["steps"] != 50:
            raise ValueError("smoke requires the frozen 50-step config")
        result = development(c, a.arm, a.output, a.assets, a.device, a.mode)
    elif a.mode == "gpu-pilot":
        from .pilot import run

        result = run(c, a.assets, a.output, a.device, source_sha())
    elif a.mode == "analyze":
        results = {
            arm: json.loads((a.output / arm / "metrics.json").read_text())
            for arm in c["arms"]
            if (a.output / arm / "metrics.json").exists()
        }
        result = summarize(c, results, oracle_pass=c["gates"]["oracle_cpu_pass"])
        write_json(a.output / "report.json", result)
    elif a.mode == "verify-portable":
        meta = json.loads((a.output / "portable.json").read_text())
        result = verify_portable(a.output, meta["identity"])
        # Transport checks alone do not set portable_pass: the saved-output restart trajectory must be reviewed.
    else:
        from .publication import publish

        result = publish(
            a.output,
            a.output.parent / "clm001-result-checkout",
            source_sha(),
            {"config": canonical_sha256(c), "seed": c["seed"]},
            preflight_only=a.mode == "publish-preflight",
        )
    print(json.dumps(result, indent=2), flush=True)
