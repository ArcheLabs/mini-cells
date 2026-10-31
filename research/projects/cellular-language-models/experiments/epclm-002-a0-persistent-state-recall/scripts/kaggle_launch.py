"""Single-launch, bounded development workflow for EPCLM-002-A0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
ROOT = HERE.parents[5]
CONFIG_PATH = EXPERIMENT_DIR / "config/development.json"
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
MAX_LAUNCH_SECONDS = 6600


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


def _checkpoint_metadata(path: Path) -> dict[str, Any] | None:
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("checkpoint_sha256") != _sha256(path):
        raise ValueError(f"completed checkpoint hash mismatch: {path}")
    return metadata


def validate_development_config(config: dict[str, Any], protocol: dict[str, Any]) -> None:
    runtime = config.get("runtime", {})
    training = config.get("training", {})
    limits = protocol["task"]
    if config.get("format") != "minicells.epclm-002-a0-development-config.v1":
        raise ValueError("unexpected development config format")
    if config.get("role") != "development_only" or config.get("formal_run") is not False:
        raise ValueError("the Kaggle launcher must remain development-only")
    launcher_limit = int(runtime.get("launcher_max_seconds", 0))
    publish_reserve = int(runtime.get("publication_reserve_seconds", -1))
    notebook_limit = int(runtime.get("notebook_hard_limit_seconds", 0))
    if not 1 <= launcher_limit <= MAX_LAUNCH_SECONDS:
        raise ValueError("development launcher limit exceeds the two-hour policy")
    if publish_reserve < 60 or launcher_limit + publish_reserve > notebook_limit or notebook_limit > 7200:
        raise ValueError("publication reserve and notebook hard limit are inconsistent")
    if training.get("model") != "learned-cells" or training.get("device") != "cpu":
        raise ValueError("development config must use the registered learned-cells CPU trainer")
    if int(training.get("steps", 0)) != int(protocol["training"]["development_trainer_default_steps"]):
        raise ValueError("development training steps differ from the frozen protocol default")
    if int(training.get("batch_size", 0)) != int(protocol["training"]["batch_size"]):
        raise ValueError("development batch size differs from the frozen protocol")
    registered_facts = set(int(value) for value in limits["facts_per_episode"])
    registered_distractors = set(int(value) for value in limits["distractors_per_episode"])
    formal_episodes = int(limits["splits"]["formal_episodes_per_condition_per_model_seed"])
    seen_backends: set[str] = set()
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("development config must declare at least one condition")
    for condition in conditions:
        backend = str(condition.get("backend"))
        if backend not in {"direct-address", "learned-cells"} or backend in seen_backends:
            raise ValueError(f"unsupported or duplicated development backend: {backend}")
        seen_backends.add(backend)
        if int(condition.get("episodes", 0)) < 1 or int(condition["episodes"]) >= formal_episodes:
            raise ValueError("development episode count must be positive and below the formal count")
        if int(condition.get("facts", -1)) not in registered_facts:
            raise ValueError("development fact count is outside the frozen task")
        if int(condition.get("distractors", -1)) not in registered_distractors:
            raise ValueError("development distractor count is outside the frozen task")
        if int(condition.get("overwrites", -1)) not in {0, 2}:
            raise ValueError("development overwrite count is outside the frozen task")
        if int(condition.get("negative_control_count", 0)) < 1:
            raise ValueError("every development condition requires negative controls")


def _publish(
    run_dir: Path,
    identity: dict[str, Any],
    decision: dict[str, Any],
    metrics: dict[str, Any],
    *,
    deadline: float,
) -> str:
    publish_dir = ROOT / "results/epclm-002-a0/publish"
    publish_dir.mkdir(parents=True, exist_ok=True)
    write_json(publish_dir / "decision.json", decision)
    write_json(publish_dir / "metrics.json", metrics)
    write_json(publish_dir / "execution-identity.json", identity)
    summary = [
        "# EPCLM-002-A0 development run",
        "",
        f"Status: **{decision['status']}**",
        "",
        "This is a development-only engineering run. It is not a formal result and supports no scientific decision.",
        "",
        f"Source commit: `{identity['source_commit']}`",
        f"Training seed: `{identity['seeds']['training']}`; evaluation seed: `{identity['seeds']['evaluation']}`",
        f"Runtime limit: {identity['runtime']['launcher_max_seconds']} seconds (hard notebook limit: 7200 seconds).",
        "",
        "Per-condition diagnostics and resumable checkpoints remain in Kaggle Output. Model weights are not published to GitHub.",
        "",
    ]
    (publish_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    branch = f"kaggle/epclm-002-a0-dev-{identity['source_commit'][:12]}"
    remaining = int(deadline - time.monotonic())
    if remaining < 1:
        raise TimeoutError("the two-hour launch budget expired before publication; Kaggle Output is retained")
    remaining = min(600, remaining)
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
        raise TimeoutError("result publication exceeded its reserved time; Kaggle Output is retained") from exc
    if output:
        print(output, end="", flush=True)
    if process.returncode != 0:
        raise RuntimeError(
            "result publication failed; Kaggle Output is retained. "
            f"Retrying this launcher will reuse completed checkpoints/results.\n{output[-4000:]}"
        )
    return branch


def launch(max_runtime_seconds: int = MAX_LAUNCH_SECONDS) -> dict[str, Any]:
    if not 60 <= max_runtime_seconds <= MAX_LAUNCH_SECONDS:
        raise ValueError(f"max_runtime_seconds must be between 60 and {MAX_LAUNCH_SECONDS}")
    started = time.monotonic()
    deadline = started + max_runtime_seconds
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol_errors = validate_protocol(PROTOCOL_PATH)
    if protocol_errors:
        raise ValueError("protocol preflight failed: " + "; ".join(protocol_errors))
    protocol_payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_development_config(config, protocol_payload)
    source_commit = _git("rev-parse", "HEAD")
    source_branch = _git("branch", "--show-current")
    dirty = _git("status", "--porcelain")
    if dirty:
        raise RuntimeError("repository must be clean before launch; source SHA cannot represent local edits")
    validate_origin(ROOT)
    token = load_github_token("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN preflight failed")
    token = ""  # Never persist or print the secret.
    result_branch = f"kaggle/epclm-002-a0-dev-{source_commit[:12]}"
    remote_branch_sha(ROOT, result_branch)
    protocol_sha = _sha256(PROTOCOL_PATH)
    config_sha = _sha256(CONFIG_PATH)
    implementation_digest = hashlib.sha256()
    for path in sorted((ROOT / "scripts/research/epclm_002_a0").glob("*.py")):
        implementation_digest.update(path.name.encode("utf-8"))
        implementation_digest.update(path.read_bytes())
    implementation_sha = implementation_digest.hexdigest()

    occupied = occupied_seeds(ROOT)
    legacy_seed_registry = ROOT / "research/governance/formal_seed_registry.json"
    if legacy_seed_registry.is_file():
        legacy_registry = json.loads(legacy_seed_registry.read_text(encoding="utf-8"))
        occupied.update(
            int(row["seed"])
            for row in legacy_registry.get("seeds", [])
            if isinstance(row, dict) and str(row.get("seed", "")).isdecimal()
        )
    occupied.update(int(value) for value in protocol_payload["training"]["formal_seeds"])
    occupied.update(int(value) for value in protocol_payload["task"]["splits"]["formal_evaluation_seeds"])
    occupied.update(int(value) for value in protocol_payload["task"]["splits"]["validation_seeds"])
    occupied.add(int(protocol_payload["task"]["splits"]["development_seed"]))
    training_seed = seed_for(EXPERIMENT_ID + ":development-training", source_commit, occupied)
    occupied.add(training_seed)
    evaluation_seed = seed_for(EXPERIMENT_ID + ":development-evaluation", source_commit, occupied)
    run_id = source_commit[:12]
    run_dir = ROOT / "results/epclm-002-a0/runs" / run_id
    runtime = config["runtime"]
    reserve = int(runtime["publication_reserve_seconds"])
    if max_runtime_seconds <= reserve:
        raise ValueError("launch budget must exceed the publication reserve")
    identity = {
        "format": "minicells.epclm-002-a0.execution-identity.v1",
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "source_commit": source_commit,
        "source_branch": source_branch,
        "protocol_sha256": protocol_sha,
        "config_sha256": config_sha,
        "implementation_sha256": implementation_sha,
        "seeds": {"training": training_seed, "evaluation": evaluation_seed},
        "configuration": config,
        "runtime": {
            "launcher_max_seconds": max_runtime_seconds,
            "publication_reserve_seconds": reserve,
            "notebook_hard_limit_seconds": int(runtime["notebook_hard_limit_seconds"]),
            "python": sys.version,
            "platform": platform.platform(),
            "device": "cpu",
        },
    }
    record(run_dir, "execution-identity.json", identity)
    for record_name in ("execution-identity.json",):
        target = run_dir / record_name
        if not target.is_file():
            raise RuntimeError(f"preflight failed to persist execution identity: {target}")
    analysis_deadline = deadline - reserve
    stages: dict[str, Any] = {
        "preflight": "PASS",
        "identity": "RECORDED",
        "training": "PENDING",
        "development_conditions": {},
        "post_run_analysis": "PENDING",
        "publication": "PENDING",
    }
    training_complete = False
    timed_out = False
    checkpoint = run_dir / "checkpoints/learned-cells.pt"
    try:
        metadata = _checkpoint_metadata(checkpoint)
        if metadata is not None:
            if (
                metadata.get("model_name") != "learned-cells"
                or metadata.get("training_seed") != training_seed
                or metadata.get("protocol_sha256") != protocol_sha
                or metadata.get("training_steps") != int(config["training"]["steps"])
            ):
                raise ValueError("existing completed checkpoint differs from the immutable development identity")
            stages["training"] = "REUSED_COMPLETED_CHECKPOINT"
        else:
            remaining = int(analysis_deadline - time.monotonic())
            if remaining < 1:
                raise TimeoutError("development budget expired before training")
            metadata = train(
                model_name="learned-cells",
                seed=training_seed,
                steps=int(config["training"]["steps"]),
                batch_size=int(config["training"]["batch_size"]),
                output=checkpoint,
                device="cpu",
                formal_run=False,
                max_runtime_seconds=min(7200, remaining),
                resume=checkpoint.with_suffix(checkpoint.suffix + ".progress.pt").exists(),
            )
            stages["training"] = "COMPLETE"
        training_complete = True

        for condition in config["conditions"]:
            condition_key = condition["backend"]
            condition_output = run_dir / "conditions" / condition_key
            remaining = int(analysis_deadline - time.monotonic())
            if remaining < 1:
                raise TimeoutError("development budget expired before all evaluation conditions completed")
            try:
                result = run_condition(
                    backend=condition_key,
                    episodes=int(condition["episodes"]),
                    seed=evaluation_seed,
                    fact_count=int(condition["facts"]),
                    distractor_count=int(condition["distractors"]),
                    overwrite_count=int(condition["overwrites"]),
                    negative_control_count=int(condition["negative_control_count"]),
                    checkpoint=checkpoint if condition_key != "direct-address" else None,
                    formal_run=False,
                    max_runtime_seconds=min(7200, remaining),
                    output=condition_output,
                )
            except TimeoutError:
                stages["development_conditions"][condition_key] = "INCOMPLETE_TIMEOUT"
                raise
            stages["development_conditions"][condition_key] = "COMPLETE"
            print(
                f"Completed {condition_key}: {result['metrics']['exact_match_accuracy']:.4f} exact-match; "
                "development only",
                flush=True,
            )
        stages["post_run_analysis"] = "COMPLETE"
    except TimeoutError:
        timed_out = True
        if not training_complete and stages["training"] == "PENDING":
            stages["training"] = "INCOMPLETE_TIMEOUT"

    complete = training_complete and stages["post_run_analysis"] == "COMPLETE"
    decision = {
        "format": "minicells.epclm-002-a0-development.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "DEVELOPMENT_COMPLETE" if complete else "DEVELOPMENT_INCOMPLETE_TIMEOUT",
        "formal_run": False,
        "scientific_decision": False,
        "h1": "INCONCLUSIVE_DEVELOPMENT_ONLY",
        "h2": "INCONCLUSIVE_DEVELOPMENT_ONLY",
        "h3": "INCONCLUSIVE_DEVELOPMENT_ONLY",
        "stages": stages,
        "timed_out": timed_out,
        "protocol_sha256": protocol_sha,
        "source_commit": source_commit,
    }
    condition_metrics: dict[str, Any] = {}
    for condition in config["conditions"]:
        name = condition["backend"]
        result_path = run_dir / "conditions" / name / "result.json"
        status_path = run_dir / "conditions" / name / "partial-status.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            condition_metrics[name] = {
                "status": "COMPLETE_DEVELOPMENT_CONDITION",
                "metrics": result.get("metrics", {}),
                "episodes": result.get("episodes"),
            }
        elif status_path.is_file():
            condition_metrics[name] = json.loads(status_path.read_text(encoding="utf-8"))
        else:
            condition_metrics[name] = {"status": stages["development_conditions"].get(name, "NOT_RUN")}
    metrics = {
        "format": "minicells.epclm-002-a0-development-metrics.v1",
        "training": {
            "status": stages["training"],
            "checkpoint_sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
        },
        "conditions": condition_metrics,
        "interpretation": "Small CPU development diagnostics only; no formal threshold or scientific claim is evaluated.",
    }
    write_json(run_dir / "decision.json", decision)
    write_json(run_dir / "metrics.json", metrics)

    decision["stages"]["publication"] = "AUTOMATIC_PUSH_REQUESTED"
    try:
        branch = _publish(run_dir, identity, decision, metrics, deadline=deadline)
        stages["publication"] = {"status": "VERIFIED", "branch": branch}
        write_json(
            run_dir / "publication-receipt.json",
            {"status": "VERIFIED", "branch": branch, "source_commit": source_commit, "verified_by_publisher": True},
        )
    except Exception as exc:
        stages["publication"] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
        write_json(run_dir / "publication-failure.json", stages["publication"])
        write_json(run_dir / "decision.json", decision)
        raise
    decision["stages"] = stages
    write_json(run_dir / "decision.json", decision)
    return {"status": decision["status"], "stages": stages, "run_dir": str(run_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-runtime-seconds", type=int, default=MAX_LAUNCH_SECONDS)
    args = parser.parse_args()
    result = launch(args.max_runtime_seconds)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
