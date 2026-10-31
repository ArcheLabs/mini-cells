"""One-click, GPU-profiled A0-v0.2 screening workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
ROOT = HERE.parents[5]
CONFIG_PATH = EXPERIMENT_DIR / "config/screening-v0.2.json"
PROTOCOL_PATH = ROOT / "research/governance/validations/epclm-002-a0-persistent-state-recall/protocol.json"
sys.path[:0] = [str(ROOT), str(ROOT / "shared/python")]

from experiments.common.development import occupied_seeds, record, seed_for  # noqa: E402
from experiments.common.train import write_json  # noqa: E402
from scripts.research.publish_experiment_results import (  # noqa: E402
    load_github_token,
    remote_branch_sha,
    validate_origin,
)
from scripts.research.epclm_002_a0.runner import run_condition  # noqa: E402
from scripts.research.epclm_002_a0.train import train  # noqa: E402
from scripts.research.epclm_002_a0.validate import validate_protocol  # noqa: E402


EXPERIMENT_ID = "epclm-002-a0-persistent-state-recall"
PLAN_ID = "A0-v0.2-screening"
MAX_LAUNCH_SECONDS = 6000
EXECUTION_BUDGET_SECONDS = 5400
PUBLICATION_RESERVE_SECONDS = 600
NOTEBOOK_HARD_LIMIT_SECONDS = 7200
ARCHITECTURES = ("flat-recurrent", "learned-cells")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)


def _checkpoint_metadata(path: Path) -> dict[str, Any] | None:
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("checkpoint_sha256") != _sha256(path):
        raise ValueError(f"completed checkpoint hash mismatch: {path}")
    return metadata


def _validate_config(config: dict[str, Any], protocol: dict[str, Any]) -> None:
    if (
        config.get("format") != "minicells.epclm-002-a0-screening-config.v1"
        or config.get("plan_id") != PLAN_ID
        or config.get("role") != "development_screening"
        or config.get("formal_run") is not False
    ):
        raise ValueError("screening config identity is invalid or attempts a formal run")
    runtime = config["runtime"]
    expected_runtime = {
        "execution_budget_seconds": EXECUTION_BUDGET_SECONDS,
        "publication_reserve_seconds": PUBLICATION_RESERVE_SECONDS,
        "notebook_hard_limit_seconds": NOTEBOOK_HARD_LIMIT_SECONDS,
        "safety_margin_seconds": 1200,
    }
    if runtime != expected_runtime:
        raise ValueError("screening runtime budgets differ from the preregistered 5,400/600/1,200-second plan")
    training = config["training"]
    if training.get("device") != "cuda":
        raise ValueError("A0-v0.2 requires a Kaggle GPU; enable a GPU accelerator and rerun")
    if int(training.get("steps", 0)) != int(protocol["training"]["development_trainer_default_steps"]):
        raise ValueError("screening training steps differ from the frozen development trainer")
    if int(training.get("batch_size", 0)) != int(protocol["training"]["batch_size"]):
        raise ValueError("screening batch size differs from the frozen development trainer")
    matrix = config["matrix"]
    limits = protocol["task"]
    if matrix.get("facts") != [1, 8, 32] or matrix.get("distractors") != [0, 64]:
        raise ValueError("screening matrix differs from the registered six-condition plan")
    if matrix.get("architectures") != list(ARCHITECTURES):
        raise ValueError("screening architectures differ from the registered Flat/Cellular comparison")
    if not set(matrix["facts"]).issubset(set(limits["facts_per_episode"])):
        raise ValueError("screening fact load is outside the frozen task")
    if not set(matrix["distractors"]).issubset(set(limits["distractors_per_episode"])):
        raise ValueError("screening distractor load is outside the frozen task")
    if matrix.get("overwrites") != 0:
        raise ValueError("A0-v0.2 fixes overwrites at zero; write/update belongs to 002-B")
    if int(matrix.get("target_episodes_per_cell", 0)) != 128:
        raise ValueError("screening target must remain 128 episodes per cell")
    if int(matrix.get("pilot_episodes_per_architecture", 0)) < 20:
        raise ValueError("the runtime pilot needs at least 20 episodes per architecture for a usable P95 estimate")
    if matrix.get("pilot_condition") != {"facts": 32, "distractors": 64}:
        raise ValueError("runtime pilot must use the preregistered highest-load condition")


def _seed_manifest(
    run_root: Path,
    *,
    source_commit: str,
    protocol_sha: str,
    config_sha: str,
    protocol: dict[str, Any],
) -> dict[str, int]:
    path = run_root / "seed-manifest.json"
    existing = _read_json(path)
    identity = {
        "source_commit": source_commit,
        "protocol_sha256": protocol_sha,
        "config_sha256": config_sha,
    }
    if existing is not None:
        if any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError("saved seed manifest belongs to a different source, protocol, or plan")
        return existing["seeds"]

    occupied = occupied_seeds(ROOT)
    occupied.update(int(value) for value in protocol["training"]["formal_seeds"])
    for name in ("formal_evaluation_seeds", "validation_seeds"):
        occupied.update(int(value) for value in protocol["task"]["splits"][name])
    occupied.add(int(protocol["task"]["splits"]["development_seed"]))
    training_seed = seed_for(EXPERIMENT_ID + ":screening-training", source_commit, occupied)
    occupied.add(training_seed)
    evaluation_seed = seed_for(EXPERIMENT_ID + ":screening-evaluation", source_commit, occupied)
    occupied.add(evaluation_seed)
    pilot_seed = seed_for(EXPERIMENT_ID + ":screening-pilot-evaluation", source_commit, occupied)
    manifest = {
        **identity,
        "seeds": {
            "training": training_seed,
            "evaluation": evaluation_seed,
            "pilot_evaluation": pilot_seed,
        },
    }
    _write_json(path, manifest)
    return manifest["seeds"]


def _condition_key(architecture: str, facts: int, distractors: int) -> str:
    return f"{architecture}-f{facts:02d}-d{distractors:02d}"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty timing sample")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _condition_result(output: Path) -> dict[str, Any] | None:
    path = output / "result.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_training(
    *,
    architecture: str,
    seed: int,
    protocol_sha: str,
    steps: int,
    batch_size: int,
    run_root: Path,
    deadline: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    checkpoint = run_root / "checkpoints" / f"{architecture}.pt"
    progress = checkpoint.with_suffix(checkpoint.suffix + ".progress.pt")
    metadata = _checkpoint_metadata(checkpoint)
    if metadata is not None:
        expected = (architecture, seed, protocol_sha, steps)
        observed = (
            metadata.get("model_name"),
            metadata.get("training_seed"),
            metadata.get("protocol_sha256"),
            metadata.get("training_steps"),
        )
        if observed != expected:
            raise ValueError("completed checkpoint differs from the screening identity")
        runtime = metadata.get("runtime_metrics", {})
        return metadata, {
            "training_elapsed_seconds": float(runtime.get("training_elapsed_seconds", 0.0)),
            "training_compute_seconds": float(runtime.get("training_compute_seconds", 0.0)),
            "checkpoint_save_seconds": float(runtime.get("checkpoint_save_seconds", 0.0)),
            "resume_restore_seconds": float(runtime.get("resume_restore_seconds", 0.0)),
            "reused_completed_checkpoint": 1.0,
        }

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    probe_wall_seconds = 0.0
    if not progress.is_file():
        remaining = int(deadline - time.monotonic())
        if remaining < 1:
            raise TimeoutError("execution budget expired before the GPU resume probe")
        probe_started = time.monotonic()
        try:
            metadata = train(
                model_name=architecture,
                seed=seed,
                steps=steps,
                batch_size=batch_size,
                output=checkpoint,
                device="cuda",
                formal_run=False,
                max_runtime_seconds=min(1, remaining),
                resume=False,
            )
        except TimeoutError:
            probe_wall_seconds = time.monotonic() - probe_started
            if not progress.is_file():
                raise RuntimeError("GPU resume probe timed out without a resumable progress checkpoint")
        else:
            probe_wall_seconds = time.monotonic() - probe_started
            runtime = metadata.get("runtime_metrics", {})
            return metadata, {
                "training_elapsed_seconds": probe_wall_seconds,
                "training_compute_seconds": float(runtime.get("training_compute_seconds", probe_wall_seconds)),
                "checkpoint_save_seconds": float(runtime.get("checkpoint_save_seconds", 0.0)),
                "resume_restore_seconds": 0.0,
                "reused_completed_checkpoint": 0.0,
            }

    remaining = int(deadline - time.monotonic())
    if remaining < 1:
        raise TimeoutError("execution budget expired before resuming GPU training")
    resumed_started = time.monotonic()
    metadata = train(
        model_name=architecture,
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        output=checkpoint,
        device="cuda",
        formal_run=False,
        max_runtime_seconds=min(7200, remaining),
        resume=progress.is_file(),
    )
    resumed_wall_seconds = time.monotonic() - resumed_started
    runtime = metadata.get("runtime_metrics", {})
    training_elapsed = probe_wall_seconds + resumed_wall_seconds
    checkpoint_seconds = float(runtime.get("checkpoint_save_seconds", 0.0))
    return metadata, {
        "training_elapsed_seconds": training_elapsed,
        "training_compute_seconds": max(0.0, training_elapsed - checkpoint_seconds),
        "checkpoint_save_seconds": checkpoint_seconds,
        "resume_restore_seconds": float(runtime.get("resume_restore_seconds", 0.0)),
        "reused_completed_checkpoint": 0.0,
    }


def _run_measured_condition(
    *,
    architecture: str,
    facts: int,
    distractors: int,
    episodes: int,
    evaluation_seed: int,
    checkpoint: Path,
    negative_control_count: int,
    output: Path,
    deadline: float,
    timing_path: Path | None = None,
) -> dict[str, Any]:
    existing = _condition_result(output)
    if existing is not None:
        return existing
    remaining = int(deadline - time.monotonic())
    if remaining < 1:
        raise TimeoutError("execution budget expired before the next condition")
    previous_timing = _read_json(timing_path, {"attempt_elapsed_seconds": []}) if timing_path else {}
    attempts = list(previous_timing.get("attempt_elapsed_seconds", []))
    started = time.monotonic()
    try:
        result = run_condition(
            backend=architecture,
            episodes=episodes,
            seed=evaluation_seed,
            fact_count=facts,
            distractor_count=distractors,
            overwrite_count=0,
            negative_control_count=negative_control_count,
            checkpoint=checkpoint,
            formal_run=False,
            max_runtime_seconds=min(7200, remaining),
            output=output,
        )
    except TimeoutError:
        attempts.append(time.monotonic() - started)
        if timing_path:
            _write_json(timing_path, {"attempt_elapsed_seconds": attempts, "status": "INCOMPLETE"})
        raise
    elapsed = time.monotonic() - started
    attempts.append(elapsed)
    if timing_path:
        _write_json(timing_path, {"attempt_elapsed_seconds": attempts, "status": "COMPLETE"})
    return result


def _paired_analysis(
    conditions: dict[str, dict[str, Any]],
    facts_values: list[int],
    distractor_values: list[int],
) -> dict[str, Any]:
    paired: dict[str, Any] = {}
    for facts in facts_values:
        for distractors in distractor_values:
            flat_key = _condition_key("flat-recurrent", facts, distractors)
            cell_key = _condition_key("learned-cells", facts, distractors)
            flat = conditions.get(flat_key, {})
            cellular = conditions.get(cell_key, {})
            flat_result = flat.get("result")
            cell_result = cellular.get("result")
            label = f"facts-{facts}-distractors-{distractors}"
            if not flat_result or not cell_result:
                paired[label] = {"status": "INCOMPLETE"}
                continue
            flat_records = flat_result.get("records", [])
            cell_records = cell_result.get("records", [])
            flat_by_id = {record["episode_id"]: record for record in flat_records}
            cell_by_id = {record["episode_id"]: record for record in cell_records}
            shared_ids = sorted(set(flat_by_id).intersection(cell_by_id))
            exact_same = len(shared_ids) == len(flat_records) == len(cell_records) and all(
                flat_by_id[episode_id].get("query_keys") == cell_by_id[episode_id].get("query_keys")
                and flat_by_id[episode_id].get("targets") == cell_by_id[episode_id].get("targets")
                for episode_id in shared_ids
            )
            if not exact_same or not shared_ids:
                paired[label] = {
                    "status": "INVALID_PAIRING",
                    "shared_episode_count": len(shared_ids),
                    "same_queries_and_targets": exact_same,
                }
                continue
            episode_deltas = [
                float(flat_by_id[episode_id]["exact_accuracy"])
                - float(cell_by_id[episode_id]["exact_accuracy"])
                for episode_id in shared_ids
            ]
            paired[label] = {
                "status": "PAIRED_SCREENING_DIAGNOSTIC",
                "paired_episode_count": len(shared_ids),
                "same_queries_and_targets": True,
                "mean_flat_minus_cellular_episode_accuracy": sum(episode_deltas) / len(episode_deltas),
                "flat_query_weighted_accuracy": flat_result["metrics"]["exact_match_accuracy"],
                "cellular_query_weighted_accuracy": cell_result["metrics"]["exact_match_accuracy"],
                "query_weighted_flat_minus_cellular": (
                    flat_result["metrics"]["exact_match_accuracy"]
                    - cell_result["metrics"]["exact_match_accuracy"]
                ),
            }
    return paired


def _interpret_screen(
    conditions: dict[str, dict[str, Any]],
    paired: dict[str, Any],
    facts_values: list[int],
    distractor_values: list[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    rules = config["stop_rules"]
    flags: list[dict[str, str]] = []
    valid_results = [
        entry["result"]
        for entry in conditions.values()
        if entry.get("result") is not None
    ]

    for result in valid_results:
        metrics = result.get("metrics", {})
        if not metrics.get("isolation_audit_pass", False):
            flags.append({
                "rule": "CONTEXT_REMOVAL_OR_ISOLATION_AUDIT_FAILED",
                "action": "STOP_INTERPRETATION_AND_INSPECT_READER_ISOLATION",
            })
            break
        if float(metrics.get("full_context_oracle_accuracy", 0.0)) < float(
            rules["full_context_oracle_min_accuracy"]
        ):
            flags.append({
                "rule": "FULL_CONTEXT_ORACLE_FAILED",
                "action": "STOP_AND_INSPECT_TASK_GENERATION_AND_EVALUATION",
            })
            break
        if float(metrics.get("no_memory_accuracy", 1.0)) > float(
            rules["no_memory_max_accuracy_for_diagnostic"]
        ):
            flags.append({
                "rule": "NO_MEMORY_BASELINE_HIGH",
                "action": "STOP_AND_AUDIT_INFORMATION_LEAKAGE_OR_TASK_DIFFICULTY",
            })
            break

    low = conditions.get(_condition_key("flat-recurrent", 8, 0), {}).get("result")
    cell_low = conditions.get(_condition_key("learned-cells", 8, 0), {}).get("result")
    low_threshold = float(rules["low_load_condition"]["both_architectures_below_accuracy"])
    if low and cell_low:
        if (
            float(low["metrics"]["exact_match_accuracy"]) < low_threshold
            and float(cell_low["metrics"]["exact_match_accuracy"]) < low_threshold
        ):
            flags.append({
                "rule": "BOTH_ARCHITECTURES_FAIL_LOW_LOAD",
                "action": "STOP_SAMPLE_EXPANSION_AND_INSPECT_STATE_WRITE_READ",
            })

    all_cells = [
        conditions.get(_condition_key(architecture, facts, distractors), {}).get("result")
        for architecture in ARCHITECTURES
        for facts in facts_values
        for distractors in distractor_values
    ]
    if all(all_cells):
        ceiling = float(rules["ceiling_accuracy"])
        if all(
            float(result["metrics"]["exact_match_accuracy"]) >= ceiling
            for result in all_cells
        ):
            flags.append({
                "rule": "ALL_CELLS_NEAR_CEILING",
                "action": "DO_NOT_ADAPT_LOAD_IN_THIS_RUN; REVIEW_A_SEPARATE_FOLLOWUP",
            })

    high_pair = paired.get("facts-32-distractors-64", {})
    practical_margin = float(rules["cellular_worse_than_flat_diagnostic_margin"])
    if (
        high_pair.get("status") == "PAIRED_SCREENING_DIAGNOSTIC"
        and float(high_pair["query_weighted_flat_minus_cellular"]) > practical_margin
    ):
        flags.append({
            "rule": "CELLULAR_TRAILS_FLAT_AT_HIGH_LOAD",
            "action": "INSPECT_STATE_CAPACITY_AND_UPDATE_BEFORE_CONFIRMATION",
        })

    complete_pairs = [
        item for item in paired.values()
        if item.get("status") == "PAIRED_SCREENING_DIAGNOSTIC"
    ]
    if any(item.get("status") == "INVALID_PAIRING" for item in paired.values()):
        flags.append({
            "rule": "PAIRED_SAMPLE_IDENTITY_FAILED",
            "action": "STOP_INTERPRETATION_AND_INSPECT_SAMPLE_GENERATION",
        })
    if len(complete_pairs) == len(facts_values) * len(distractor_values):
        margin = float(rules["minimum_practical_difference_for_claim"])
        if all(
            abs(float(item["query_weighted_flat_minus_cellular"])) < margin
            for item in complete_pairs
        ):
            flags.append({
                "rule": "NO_PRACTICAL_CELLULAR_ADVANTAGE_IN_SCREEN",
                "action": "DO_NOT_CLAIM_CELLULAR_ADVANTAGE",
            })

    return {
        "flags": flags,
        "scientific_decision": False,
        "interpretation": (
            "Development screening diagnostics only. Small differences require paired confirmatory "
            "analysis across multiple training seeds."
        ),
    }


def _publish(
    identity: dict[str, Any],
    decision: dict[str, Any],
    metrics: dict[str, Any],
    *,
    deadline: float,
) -> tuple[str, str]:
    publish_dir = ROOT / "results/epclm-002-a0/publish"
    publish_dir.mkdir(parents=True, exist_ok=True)
    write_json(publish_dir / "decision.json", decision)
    write_json(publish_dir / "metrics.json", metrics)
    write_json(publish_dir / "execution-identity.json", identity)
    summary = [
        "# EPCLM-002-A0 A0-v0.2 screening run",
        "",
        f"Status: **{decision['status']}**",
        "",
        "Development screening only. This is not a formal result and supports no scientific decision.",
        "",
        f"Source commit: {identity['source_commit']}",
        f"Training seed: {identity['seeds']['training']}; evaluation seed: {identity['seeds']['evaluation']}",
        f"Execution budget: {EXECUTION_BUDGET_SECONDS} seconds; publication reserve: {PUBLICATION_RESERVE_SECONDS} seconds; notebook hard limit: {NOTEBOOK_HARD_LIMIT_SECONDS} seconds.",
        "",
        "Per-condition diagnostics and resumable checkpoints remain in Kaggle Output. Model weights are not published to GitHub.",
        "",
    ]
    (publish_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    branch = f"kaggle/epclm-002-a0-screen-v02-{identity['source_commit'][:12]}"
    remaining = int(deadline - time.monotonic())
    if remaining < 1:
        raise TimeoutError("the 7,200-second notebook budget expired before publication; Kaggle Output is retained")
    remaining = min(PUBLICATION_RESERVE_SECONDS, remaining)
    command = [
        sys.executable,
        str(ROOT / "scripts/research/publish_experiment_results.py"),
        "epclm-002-a0",
        "--push",
        "--branch",
        branch,
        "--base-ref",
        "origin/main",
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=(os.name == "posix"),
    )
    try:
        output, _ = process.communicate(timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            process.communicate()
        raise TimeoutError("result publication exceeded its 600-second reserve; Kaggle Output is retained") from exc
    if output:
        print(output, end="", flush=True)
    if process.returncode != 0:
        raise RuntimeError(
            "result publication failed; Kaggle Output is retained. "
            f"Retrying this notebook reuses completed checkpoints/results.\n{output[-4000:]}"
        )
    published_sha = remote_branch_sha(ROOT, branch)
    if not published_sha:
        raise RuntimeError("result publisher returned successfully but the remote result branch could not be verified")
    return branch, published_sha


def launch(max_runtime_seconds: int = MAX_LAUNCH_SECONDS) -> dict[str, Any]:
    if max_runtime_seconds != MAX_LAUNCH_SECONDS:
        raise ValueError(f"A0-v0.2 fixes the launcher budget at {MAX_LAUNCH_SECONDS} seconds")
    started = time.monotonic()
    deadline = started + max_runtime_seconds
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol_errors = validate_protocol(PROTOCOL_PATH)
    if protocol_errors:
        raise ValueError("frozen protocol preflight failed: " + "; ".join(protocol_errors))
    protocol_payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    _validate_config(config, protocol_payload)
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU preflight failed. Select a GPU accelerator, restart the session, and run all cells again.")
    gpu_name = torch.cuda.get_device_name(0)

    source_commit = _git("rev-parse", "HEAD")
    source_branch = _git("branch", "--show-current")
    working_tree = _git("status", "--porcelain", "--untracked-files=all").splitlines()
    allowed_runtime_prefixes = (
        "results/epclm-002-a0/",
        "artifacts/experiments/epclm-002-a0-persistent-state-recall/development/",
    )
    unrelated_changes = [
        line for line in working_tree
        if not any(line[3:].startswith(prefix) for prefix in allowed_runtime_prefixes)
    ]
    if unrelated_changes:
        raise RuntimeError(
            "tracked source changes or unrelated untracked files prevent an auditable launch: "
            + "; ".join(unrelated_changes[:10])
        )
    validate_origin(ROOT)
    token = load_github_token("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN preflight failed")
    token = ""

    protocol_sha = _sha256(PROTOCOL_PATH)
    config_sha = _sha256(CONFIG_PATH)
    implementation_digest = hashlib.sha256()
    for path in sorted((ROOT / "scripts/research/epclm_002_a0").glob("*.py")):
        implementation_digest.update(path.name.encode("utf-8"))
        implementation_digest.update(path.read_bytes())
    implementation_digest.update(Path(__file__).read_bytes())
    run_id = source_commit[:12] + "-screening-v0.2"
    run_root = ROOT / "results/epclm-002-a0/runs" / run_id
    seeds = _seed_manifest(
        run_root,
        source_commit=source_commit,
        protocol_sha=protocol_sha,
        config_sha=config_sha,
        protocol=protocol_payload,
    )
    identity = {
        "format": "minicells.epclm-002-a0.execution-identity.v1",
        "experiment_id": EXPERIMENT_ID,
        "plan_id": PLAN_ID,
        "run_id": run_id,
        "source_commit": source_commit,
        "source_branch": source_branch,
        "protocol_sha256": protocol_sha,
        "config_sha256": config_sha,
        "implementation_sha256": implementation_digest.hexdigest(),
        "seeds": seeds,
        "configuration": config,
        "runtime": {
            "launcher_max_seconds": max_runtime_seconds,
            "execution_budget_seconds": EXECUTION_BUDGET_SECONDS,
            "publication_reserve_seconds": PUBLICATION_RESERVE_SECONDS,
            "notebook_hard_limit_seconds": NOTEBOOK_HARD_LIMIT_SECONDS,
            "training_device": "cuda",
            "evaluation_device": "cpu (frozen runner path)",
            "python": sys.version,
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
        },
    }
    record(run_root, "execution-identity.json", identity)
    analysis_deadline = min(deadline - PUBLICATION_RESERVE_SECONDS, started + EXECUTION_BUDGET_SECONDS)
    stages: dict[str, Any] = {
        "preflight": "PASS",
        "training": {architecture: "PENDING" for architecture in ARCHITECTURES},
        "gpu_pilot": "PENDING",
        "screening": {},
        "paired_analysis": "PENDING",
        "publication": "PENDING",
    }
    training_runtime: dict[str, dict[str, float]] = {}
    pilot_conditions: dict[str, dict[str, Any]] = {}
    screening_conditions: dict[str, dict[str, Any]] = {}
    pilot_path = run_root / "pilot-runtime.json"
    saved_pilot = _read_json(pilot_path)
    timed_out = False
    diagnostic_stop = False
    selected_episodes = 0
    target_path = run_root / "screening-target.json"
    capacity_path = run_root / "capacity-estimate.json"
    training_phase_started = time.monotonic()
    training_current_checkpoint_seconds = 0.0
    training_current_elapsed_seconds = 0.0
    pilot_elapsed_this_run = 0.0

    try:
        for architecture in ARCHITECTURES:
            remaining = int(analysis_deadline - time.monotonic())
            if remaining < 1:
                raise TimeoutError("execution budget expired before both architecture checkpoints completed")
            checkpoint = run_root / "checkpoints" / f"{architecture}.pt"
            metadata, timing = _ensure_training(
                architecture=architecture,
                seed=int(seeds["training"]),
                protocol_sha=protocol_sha,
                steps=int(config["training"]["steps"]),
                batch_size=int(config["training"]["batch_size"]),
                run_root=run_root,
                deadline=analysis_deadline,
            )
            training_runtime[architecture] = timing
            stages["training"][architecture] = (
                "REUSED_COMPLETED_CHECKPOINT"
                if timing.get("reused_completed_checkpoint")
                else "COMPLETE"
            )
            if metadata.get("model_name") != architecture:
                raise ValueError("trainer returned a checkpoint for the wrong architecture")
        training_current_elapsed_seconds = time.monotonic() - training_phase_started
        training_current_checkpoint_seconds = sum(
            float(timing.get("checkpoint_save_seconds", 0.0))
            for timing in training_runtime.values()
            if not timing.get("reused_completed_checkpoint")
        )

        if saved_pilot is not None:
            pilot_conditions = saved_pilot["pilot_conditions"]
            training_runtime = saved_pilot.get("training", training_runtime)
            stages["gpu_pilot"] = "REUSED_COMPLETED_PILOT"
        else:
            pilot_started = time.monotonic()
            pilot_condition = config["matrix"]["pilot_condition"]
            pilot_episodes = int(config["matrix"]["pilot_episodes_per_architecture"])
            for architecture in ARCHITECTURES:
                key = _condition_key(
                    architecture,
                    int(pilot_condition["facts"]),
                    int(pilot_condition["distractors"]),
                )
                output = run_root / "pilot" / key
                timing_path = output / "runtime-attempts.json"
                checkpoint = run_root / "checkpoints" / f"{architecture}.pt"
                result = _run_measured_condition(
                    architecture=architecture,
                    facts=int(pilot_condition["facts"]),
                    distractors=int(pilot_condition["distractors"]),
                    episodes=pilot_episodes,
                    evaluation_seed=int(seeds["pilot_evaluation"]),
                    checkpoint=checkpoint,
                    negative_control_count=int(config["matrix"]["negative_control_count"]),
                    output=output,
                    deadline=analysis_deadline,
                    timing_path=timing_path,
                )
                records = result.get("records", [])
                episode_ms = [float(row["episode_wall_time_ms"]) for row in records]
                attempts = _read_json(timing_path, {}).get("attempt_elapsed_seconds", [])
                total_elapsed = sum(float(value) for value in attempts)
                episode_seconds = sum(episode_ms) / 1000.0
                measured_overhead = max(0.0, total_elapsed - episode_seconds)
                pilot_conditions[key] = {
                    "status": "COMPLETE",
                    "episodes": len(records),
                    "metrics": result.get("metrics", {}),
                    "p50_episode_seconds": _percentile(episode_ms, 0.50) / 1000.0,
                    "p95_episode_seconds": _percentile(episode_ms, 0.95) / 1000.0,
                    "condition_wall_seconds": total_elapsed,
                    "cell_overhead_seconds": measured_overhead if attempts else max(30.0, measured_overhead),
                    "negative_controls": result.get("negative_controls", {}),
                }
                _write_json(
                    run_root / "pilot-progress.json",
                    {
                        "training": training_runtime,
                        "pilot_conditions": pilot_conditions,
                    },
                )
            pilot_elapsed_this_run = time.monotonic() - pilot_started
            training_runtime = {
                architecture: {
                    **training_runtime.get(architecture, {}),
                    "training_gpu_name": gpu_name,
                }
                for architecture in ARCHITECTURES
            }
            pilot_summary = {
                "format": "minicells.epclm-002-a0-runtime-pilot.v1",
                "plan_id": PLAN_ID,
                "training": training_runtime,
                "pilot_conditions": pilot_conditions,
                "pilot_elapsed_seconds_this_run": pilot_elapsed_this_run,
                "screening_execution_budget_seconds": EXECUTION_BUDGET_SECONDS,
                "publication_reserve_seconds": PUBLICATION_RESERVE_SECONDS,
                "notebook_hard_limit_seconds": NOTEBOOK_HARD_LIMIT_SECONDS,
                "publication_elapsed_seconds": None,
                "training_gpu_name": gpu_name,
                "evaluation_device": "cpu (frozen runner path)",
            }
            _write_json(pilot_path, pilot_summary)
            stages["gpu_pilot"] = "COMPLETE"

        # A pilot baseline failure is a stop-and-diagnose result, not a reason to spend the matrix budget.
        for pilot_result in pilot_conditions.values():
            pilot_metrics = pilot_result.get("metrics", {})
            if not pilot_metrics.get("isolation_audit_pass", False):
                diagnostic_stop = True
                stages["gpu_pilot"] = "STOPPED_ISOLATION_AUDIT_FAILURE"
            if float(pilot_metrics.get("full_context_oracle_accuracy", 1.0)) < float(
                config["stop_rules"]["full_context_oracle_min_accuracy"]
            ):
                diagnostic_stop = True
                stages["gpu_pilot"] = "STOPPED_FULL_CONTEXT_DIAGNOSTIC"
            if float(pilot_metrics.get("no_memory_accuracy", 0.0)) > float(
                config["stop_rules"]["no_memory_max_accuracy_for_diagnostic"]
            ):
                diagnostic_stop = True
                stages["gpu_pilot"] = "STOPPED_NO_MEMORY_DIAGNOSTIC"
            controls = pilot_result.get("negative_controls", {})
            if controls and float(controls.get("wrong_episode_state_accuracy", 0.0)) > 0.05:
                diagnostic_stop = True
                stages["gpu_pilot"] = "STOPPED_WRONG_EPISODE_CONTROL_DIAGNOSTIC"
        if diagnostic_stop:
            selected_episodes = 0
        elif target_path.is_file():
            selected_episodes = int(_read_json(target_path)["selected_episodes_per_cell"])
        else:
            p95_seconds = max(
                float(item["p95_episode_seconds"])
                for item in pilot_conditions.values()
            )
            cell_overhead = max(
                float(item["cell_overhead_seconds"])
                for item in pilot_conditions.values()
            )
            elapsed_before_capacity = time.monotonic() - started
            training_compute_current = max(
                0.0,
                training_current_elapsed_seconds - training_current_checkpoint_seconds,
            )
            preflight_setup_current = max(
                0.0,
                elapsed_before_capacity
                - training_current_elapsed_seconds
                - pilot_elapsed_this_run,
            )
            measured_remainder = (
                EXECUTION_BUDGET_SECONDS
                - preflight_setup_current
                - training_compute_current
                - training_current_checkpoint_seconds
                - pilot_elapsed_this_run
            )
            remaining = max(
                0.0,
                min(measured_remainder, analysis_deadline - time.monotonic()),
            )
            planned_cells = (
                len(ARCHITECTURES)
                * len(config["matrix"]["facts"])
                * len(config["matrix"]["distractors"])
            )
            numerator = remaining - planned_cells * cell_overhead
            n_max = max(0, math.floor(numerator / (planned_cells * p95_seconds))) if p95_seconds > 0 else 0
            selected_episodes = min(int(config["matrix"]["target_episodes_per_cell"]), n_max)
            capacity_payload = {
                "format": "minicells.epclm-002-a0-capacity-estimate.v1",
                "plan_id": PLAN_ID,
                "execution_budget_seconds": EXECUTION_BUDGET_SECONDS,
                "preflight_setup_seconds": preflight_setup_current,
                "training_compute_seconds_current_run": training_compute_current,
                "checkpoint_save_seconds_current_run": training_current_checkpoint_seconds,
                "pilot_elapsed_seconds_current_run": pilot_elapsed_this_run,
                "remaining_execution_seconds": remaining,
                "planned_cells": planned_cells,
                "max_p95_episode_seconds": p95_seconds,
                "conservative_cell_overhead_seconds": cell_overhead,
                "estimated_episode_capacity_per_cell": n_max,
                "selected_episodes_per_cell": selected_episodes,
                "target_episodes_per_cell": int(config["matrix"]["target_episodes_per_cell"]),
                "capacity_numerator_seconds": max(0.0, numerator),
                "capacity_formula": "floor((remaining_execution_seconds - planned_cells*cell_overhead)/(planned_cells*worst_p95_episode_seconds))",
            }
            _write_json(capacity_path, capacity_payload)
            if selected_episodes > 0:
                _write_json(target_path, capacity_payload)

        if selected_episodes == 0 and not diagnostic_stop:
            stages["screening"]["matrix"] = "NOT_STARTED_PILOT_CAPACITY_INSUFFICIENT"
        elif diagnostic_stop:
            stages["screening"]["matrix"] = "NOT_STARTED_STOP_RULE"
        else:
            facts_values = [int(value) for value in config["matrix"]["facts"]]
            distractor_values = [int(value) for value in config["matrix"]["distractors"]]
            for facts in facts_values:
                for distractors in distractor_values:
                    for architecture in ARCHITECTURES:
                        key = _condition_key(architecture, facts, distractors)
                        output = run_root / "screening" / key
                        checkpoint = run_root / "checkpoints" / f"{architecture}.pt"
                        remaining = int(analysis_deadline - time.monotonic())
                        if remaining < 1:
                            stages["screening"][key] = "INCOMPLETE_BUDGET_NOT_STARTED"
                            raise TimeoutError("execution budget expired before all screening cells completed")
                        try:
                            result = _run_measured_condition(
                                architecture=architecture,
                                facts=facts,
                                distractors=distractors,
                                episodes=selected_episodes,
                                evaluation_seed=int(seeds["evaluation"]),
                                checkpoint=checkpoint,
                                negative_control_count=int(config["matrix"]["negative_control_count"]),
                                output=output,
                                deadline=analysis_deadline,
                            )
                        except TimeoutError:
                            stages["screening"][key] = "INCOMPLETE_TIMEOUT"
                            raise
                        screening_conditions[key] = {
                            "status": "COMPLETE",
                            "episodes": int(result["episodes"]),
                            "result": result,
                        }
                        stages["screening"][key] = "COMPLETE"
                        result_metrics = result.get("metrics", {})
                        if not result_metrics.get("isolation_audit_pass", False):
                            diagnostic_stop = True
                            stages["screening"]["stop_reason"] = "STOPPED_ISOLATION_AUDIT_FAILURE"
                        elif float(result_metrics.get("full_context_oracle_accuracy", 1.0)) < float(
                            config["stop_rules"]["full_context_oracle_min_accuracy"]
                        ):
                            diagnostic_stop = True
                            stages["screening"]["stop_reason"] = "STOPPED_FULL_CONTEXT_DIAGNOSTIC"
                        elif float(result_metrics.get("no_memory_accuracy", 0.0)) > float(
                            config["stop_rules"]["no_memory_max_accuracy_for_diagnostic"]
                        ):
                            diagnostic_stop = True
                            stages["screening"]["stop_reason"] = "STOPPED_NO_MEMORY_DIAGNOSTIC"
                        if diagnostic_stop:
                            break
                    if facts == 8 and distractors == 0:
                        flat = screening_conditions.get(_condition_key("flat-recurrent", 8, 0), {}).get("result")
                        cell = screening_conditions.get(_condition_key("learned-cells", 8, 0), {}).get("result")
                        threshold = float(config["stop_rules"]["low_load_condition"]["both_architectures_below_accuracy"])
                        if flat and cell and (
                            float(flat["metrics"]["exact_match_accuracy"]) < threshold
                            and float(cell["metrics"]["exact_match_accuracy"]) < threshold
                        ):
                            diagnostic_stop = True
                            stages["screening"]["remaining_matrix"] = "NOT_RUN_LOW_LOAD_STOP_RULE"
                            break
                if diagnostic_stop:
                    break
    except TimeoutError:
        timed_out = True
        stages["execution"] = "INCOMPLETE_TIMEOUT"
        for architecture in ARCHITECTURES:
            if stages["training"].get(architecture) == "PENDING":
                stages["training"][architecture] = "INCOMPLETE_TIMEOUT"
        if stages["gpu_pilot"] == "PENDING":
            stages["gpu_pilot"] = "INCOMPLETE_TIMEOUT"

    # Collect completed and partial cells from their auditable runner outputs.
    for architecture in ARCHITECTURES:
        for facts in config["matrix"]["facts"]:
            for distractors in config["matrix"]["distractors"]:
                key = _condition_key(architecture, int(facts), int(distractors))
                output = run_root / "screening" / key
                result = _condition_result(output)
                if result is not None:
                    screening_conditions[key] = {
                        "status": "COMPLETE",
                        "episodes": int(result["episodes"]),
                        "result": result,
                    }
                    stages["screening"].setdefault(key, "COMPLETE")
                elif key not in stages["screening"]:
                    partial = _read_json(output / "partial-status.json")
                    if partial:
                        stages["screening"][key] = "INCOMPLETE"
                        screening_conditions[key] = {"status": "INCOMPLETE", "partial": partial}
                    else:
                        existing_status = stages["screening"].get(key)
                        if existing_status and existing_status.startswith("INCOMPLETE"):
                            reason = existing_status
                        elif timed_out:
                            reason = "INCOMPLETE_NOT_RUN_TIMEOUT"
                        elif diagnostic_stop:
                            reason = "INCOMPLETE_STOP_RULE"
                        elif selected_episodes == 0:
                            reason = "INCOMPLETE_PILOT_CAPACITY"
                        else:
                            reason = "INCOMPLETE_NOT_RUN"
                        stages["screening"][key] = reason
                        screening_conditions[key] = {"status": reason}
                else:
                    screening_conditions[key] = {
                        "status": stages["screening"][key],
                    }

    facts_values = [int(value) for value in config["matrix"]["facts"]]
    distractor_values = [int(value) for value in config["matrix"]["distractors"]]
    paired = _paired_analysis(screening_conditions, facts_values, distractor_values)
    stages["paired_analysis"] = (
        "INVALID_SAMPLE_PAIRING"
        if any(item.get("status") == "INVALID_PAIRING" for item in paired.values())
        else "COMPLETE_WITH_AVAILABLE_CELLS"
    )
    interpretation = _interpret_screen(
        screening_conditions,
        paired,
        facts_values,
        distractor_values,
        config,
    )
    completed_cells = sum(
        entry.get("status") == "COMPLETE" for entry in screening_conditions.values()
    )
    planned_cells = len(ARCHITECTURES) * len(facts_values) * len(distractor_values)
    all_screen_cells_complete = completed_cells == planned_cells
    if timed_out:
        status = "DEVELOPMENT_INCOMPLETE_TIMEOUT"
    elif diagnostic_stop:
        status = "DEVELOPMENT_SCREENING_STOPPED_FOR_DIAGNOSTICS"
    elif selected_episodes == 0:
        status = "DEVELOPMENT_PILOT_ONLY_CAPACITY_INSUFFICIENT"
    elif all_screen_cells_complete and selected_episodes < int(config["matrix"]["target_episodes_per_cell"]):
        status = "DEVELOPMENT_SCREENING_COMPLETE_CAPACITY_ADAPTED"
    elif all_screen_cells_complete:
        status = "DEVELOPMENT_SCREENING_COMPLETE"
    else:
        status = "DEVELOPMENT_INCOMPLETE"

    decision = {
        "format": "minicells.epclm-002-a0-development.v1",
        "experiment_id": EXPERIMENT_ID,
        "plan_id": PLAN_ID,
        "status": status,
        "formal_run": False,
        "scientific_decision": False,
        "h1": "INCONCLUSIVE_DEVELOPMENT_ONLY",
        "h2": "INCONCLUSIVE_DEVELOPMENT_ONLY",
        "h3": "OUT_OF_SCOPE_FOR_A0_V0_2",
        "stages": stages,
        "timed_out": timed_out,
        "selected_episodes_per_cell": selected_episodes,
        "target_episodes_per_cell": int(config["matrix"]["target_episodes_per_cell"]),
        "completed_cells": completed_cells,
        "planned_cells": planned_cells,
        "protocol_sha256": protocol_sha,
        "source_commit": source_commit,
        "diagnostic_interpretation": interpretation,
    }
    condition_metrics: dict[str, Any] = {}
    for key, entry in screening_conditions.items():
        condition_metrics[key] = {
            "status": entry.get("status"),
            "episodes": entry.get("episodes"),
            "metrics": (entry.get("result") or {}).get("metrics"),
            "partial": entry.get("partial"),
        }
    runtime_report = _read_json(pilot_path, {})
    capacity_report = _read_json(capacity_path, _read_json(target_path, {
        "selected_episodes_per_cell": 0,
        "target_episodes_per_cell": int(config["matrix"]["target_episodes_per_cell"]),
        "reason": "Pilot did not leave enough budget for one complete pass over the matrix.",
    }))
    runtime_report.update({
        "training": training_runtime or runtime_report.get("training", {}),
        "selected_episodes_per_cell": selected_episodes,
        "target_episodes_per_cell": int(config["matrix"]["target_episodes_per_cell"]),
        "pilot_conditions": pilot_conditions or runtime_report.get("pilot_conditions", {}),
        "capacity_estimate": capacity_report,
        "publication_elapsed_seconds": None,
    })
    metrics = {
        "format": "minicells.epclm-002-a0-development-metrics.v1",
        "plan_id": PLAN_ID,
        "training": {
            "status": stages["training"],
            "models": training_runtime,
            "checkpoint_sha256": {
                architecture: (
                    _sha256(run_root / "checkpoints" / f"{architecture}.pt")
                    if (run_root / "checkpoints" / f"{architecture}.pt").is_file()
                    else None
                )
                for architecture in ARCHITECTURES
            },
        },
        "runtime_pilot": runtime_report,
        "screening": {
            "target_episodes_per_cell": int(config["matrix"]["target_episodes_per_cell"]),
            "selected_episodes_per_cell": selected_episodes,
            "planned_cells": planned_cells,
            "completed_cells": completed_cells,
            "conditions": condition_metrics,
            "paired_architecture_analysis": paired,
            "stop_rule_interpretation": interpretation,
        },
        "interpretation": (
            "Mechanism-screening and runtime-pilot evidence only. No formal threshold, hypothesis, "
            "or scientific decision is evaluated."
        ),
    }
    write_json(run_root / "decision.json", decision)
    write_json(run_root / "metrics.json", metrics)
    _write_json(run_root / "runtime-pilot.json", runtime_report)

    decision["stages"]["publication"] = "AUTOMATIC_PUSH_REQUESTED"
    publication_started = time.monotonic()
    try:
        branch, remote_sha = _publish(identity, decision, metrics, deadline=deadline)
        publication_elapsed = time.monotonic() - publication_started
        runtime_report["publication_elapsed_seconds"] = publication_elapsed
        runtime_report["publication_branch"] = branch
        runtime_report["publication_remote_sha"] = remote_sha
        runtime_report["publication_status"] = "VERIFIED"
        _write_json(run_root / "runtime-pilot.json", runtime_report)
        stages["publication"] = {
            "status": "VERIFIED",
            "branch": branch,
            "remote_sha": remote_sha,
            "elapsed_seconds": publication_elapsed,
        }
        write_json(
            run_root / "publication-receipt.json",
            {
                "status": "VERIFIED",
                "branch": branch,
                "remote_sha": remote_sha,
                "source_commit": source_commit,
                "verified_by_publisher": True,
            },
        )
    except Exception as exc:
        stages["publication"] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
        write_json(run_root / "publication-failure.json", stages["publication"])
        write_json(run_root / "decision.json", decision)
        raise
    decision["stages"] = stages
    write_json(run_root / "decision.json", decision)
    return {
        "status": decision["status"],
        "stages": stages,
        "run_dir": str(run_root),
        "selected_episodes_per_cell": selected_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-runtime-seconds", type=int, default=MAX_LAUNCH_SECONDS)
    args = parser.parse_args()
    result = launch(args.max_runtime_seconds)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
