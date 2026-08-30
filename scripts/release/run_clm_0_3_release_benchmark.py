#!/usr/bin/env python3
"""Run and aggregate the CLM-0.3 public release benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from minicells.clm_release_benchmark import (  # noqa: E402
    BRIDGE_ARMS,
    BRIDGE_BATCH_SIZE,
    BRIDGE_BUDGET_TOKENS,
    BRIDGE_CHECKPOINT_TOKENS,
    BRIDGE_MATERIALIZED_TRAIN_TOKENS,
    BRIDGE_SEQUENCE_LENGTH,
    BRIDGE_SUMMARY_FORMAT,
    CAPABILITY_ARTIFACT_ROOT,
    CAPABILITY_RESULTS_REF,
    SOURCE_006_CHECKPOINT_SHA256,
    SOURCE_006_DECISION,
    SOURCE_007_DECISION,
    make_release_decision,
    normalize_capability_evidence,
    validate_historical_evidence,
    verify_source_checkpoint,
)
from minicells.clm_release_reporting import write_public_release_summary  # noqa: E402
from minicells.clm_release_visualization import save_release_figures  # noqa: E402
from minicells.language_scaling import prepare_scaling_corpus  # noqa: E402


OUTPUT_ROOT = Path("results/clm-0.3-release-benchmark")
SOURCE_005 = Path("artifacts/experiments/005-consumer-language-bridge")
WORKER = Path("scripts/release/run_clm_0_3_release_bridge_worker.py")
CAPABILITY_REMOTE_REF = f"refs/remotes/origin/{CAPABILITY_RESULTS_REF}"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the CLM-0.3 public release benchmark")
    result.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    result.add_argument("--execute", action="store_true")
    result.add_argument("--restart-existing", action="store_true")
    return result


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=check
    )


def _git_text(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _git_json(ref: str, path: str) -> Any:
    raw = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT, text=True)
    return json.loads(raw)


def _fetch_capability_results() -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    subprocess.run(
        [
            "git",
            "fetch",
            "origin",
            f"{CAPABILITY_RESULTS_REF}:{CAPABILITY_REMOTE_REF}",
        ],
        cwd=ROOT,
        check=True,
    )
    source_commit = _git_text("rev-parse", CAPABILITY_REMOTE_REF)
    decision = _git_json(
        CAPABILITY_REMOTE_REF, f"{CAPABILITY_ARTIFACT_ROOT}/decision.json"
    )
    replicate_summary = _git_json(
        CAPABILITY_REMOTE_REF, f"{CAPABILITY_ARTIFACT_ROOT}/replicate-summary.json"
    )
    return decision, replicate_summary, source_commit


def _worker_dir(output_root: Path, arm: str) -> Path:
    return output_root / "bridge" / arm


def _worker_command(arm: str, cache_dir: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / WORKER),
        "--arm",
        arm,
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(output_dir),
        "--execute",
    ]


def _run_workers(output_root: Path, cache_dir: Path) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("formal CLM-0.3 release bridge requires CUDA")
    used = min(2, available)
    if used == 1:
        for arm in BRIDGE_ARMS:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            output_dir = _worker_dir(output_root, arm)
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "worker.log"
            with log_path.open("a", encoding="utf-8") as handle:
                result = subprocess.run(
                    _worker_command(arm, cache_dir, output_dir),
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            print(f"--- {arm} / GPU0 ---")
            print(log_path.read_text(encoding="utf-8")[-12000:])
            if result.returncode != 0:
                raise RuntimeError(f"{arm} bridge worker failed; see {log_path}")
        return 1

    active: list[tuple[str, int, subprocess.Popen[str], Path, Any]] = []
    for gpu_index, arm in enumerate(BRIDGE_ARMS):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        output_dir = _worker_dir(output_root, arm)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "worker.log"
        handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            _worker_command(arm, cache_dir, output_dir),
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        active.append((arm, gpu_index, process, log_path, handle))
        print(f"started {arm:22s} on physical GPU {gpu_index}", flush=True)

    failures: list[str] = []
    for arm, gpu_index, process, log_path, handle in active:
        code = process.wait()
        handle.close()
        print(f"--- {arm} / GPU{gpu_index} ---")
        print(log_path.read_text(encoding="utf-8")[-12000:])
        if code != 0:
            failures.append(f"{arm} exited {code}; see {log_path}")
    if failures:
        raise RuntimeError("; ".join(failures))
    return used


def _load_worker(output_root: Path, arm: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = _worker_dir(output_root, arm)
    result_path = directory / "worker-result.json"
    checkpoints_path = directory / "bridge-checkpoints.json"
    if not result_path.is_file() or not checkpoints_path.is_file():
        raise FileNotFoundError(f"bridge worker evidence incomplete for {arm}")
    return (
        json.loads(result_path.read_text(encoding="utf-8")),
        json.loads(checkpoints_path.read_text(encoding="utf-8")),
    )


def _aggregate_bridge(output_root: Path, *, gpu_count_used: int) -> dict[str, Any]:
    current_commit = _git_text("rev-parse", "HEAD")
    current_tree = _git_text("rev-parse", "HEAD^{tree}")
    arms: dict[str, Any] = {}
    commits: set[str] = set()
    trees: set[str] = set()
    for arm in BRIDGE_ARMS:
        result, checkpoints = _load_worker(output_root, arm)
        if result.get("formal_gpu_experiment_run") is not True:
            raise RuntimeError(f"{arm} was not a formal GPU worker")
        if result.get("arm") != arm:
            raise RuntimeError(f"bridge worker arm mismatch for {arm}")
        if result.get("source_checkpoint_sha256") != SOURCE_006_CHECKPOINT_SHA256:
            raise RuntimeError(f"{arm} source checkpoint provenance mismatch")
        ages = [int(row["consumed_tokens"]) for row in checkpoints]
        if ages != list(BRIDGE_CHECKPOINT_TOKENS):
            raise RuntimeError(f"{arm} checkpoint ages are incomplete: {ages}")
        commit = str(result.get("code_commit"))
        tree = str(result.get("code_tree_sha"))
        commits.add(commit)
        trees.add(tree)
        arms[arm] = {
            "final_ppl": float(result["final_ppl"]),
            "final_nll": float(result["final_nll"]),
            "parameters": result["parameters"],
            "runtime": result["runtime"],
            "checkpoints": checkpoints,
        }
    if commits != {current_commit} or trees != {current_tree}:
        raise RuntimeError(
            f"mixed/stale release bridge provenance: commits={sorted(commits)}, trees={sorted(trees)}"
        )
    clm_result, _ = _load_worker(output_root, "clm_fixed4")
    equivalence = clm_result.get("age_zero_equivalence")
    if equivalence is None or equivalence.get("status") != "CLM_RELEASE_BRIDGE_EQUIVALENCE":
        raise RuntimeError("CLM fixed4 failed age-zero release-bridge equivalence")
    return {
        "format": BRIDGE_SUMMARY_FORMAT,
        "formal_gpu_experiment_run": True,
        "training_commit": current_commit,
        "training_tree_sha": current_tree,
        "source_checkpoint_sha256": SOURCE_006_CHECKPOINT_SHA256,
        "budget_tokens_per_arm": BRIDGE_BUDGET_TOKENS,
        "checkpoint_ages": list(BRIDGE_CHECKPOINT_TOKENS),
        "age_zero_equivalence": equivalence,
        "runtime": {
            "gpu_count_used": gpu_count_used,
            "physical_gpus": [
                torch.cuda.get_device_name(index)
                for index in range(min(gpu_count_used, torch.cuda.device_count()))
            ],
        },
        "arms": arms,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_plan(output_root: Path) -> None:
    print("CLM-0.3 PUBLIC RELEASE BENCHMARK — PLAN")
    print(f"branch commit: {_git_text('rev-parse', 'HEAD')}")
    print(f"output: {output_root}")
    print("historical foundation: Experiment 006 + Experiment 007 (no retraining)")
    print("new GPU bridge: same trained 10M TextNCA -> TextNCA continuation vs CLM fixed4")
    print(f"bridge budget: {BRIDGE_BUDGET_TOKENS:,} tokens per arm")
    print(f"bridge checkpoints: {BRIDGE_CHECKPOINT_TOKENS}")
    print("capability evidence: immutable CLM-0.3d result branch")
    print("visualizations: 4 PNG + 4 SVG public figures")


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    if not args.execute:
        _print_plan(output_root)
        return 0

    current_dirty = _git_text("status", "--porcelain", "--untracked-files=no")
    if current_dirty:
        raise RuntimeError("formal release benchmark requires a clean tracked Git tree")
    if args.restart_existing and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_checkpoint = ROOT / "artifacts/experiments/006-consumer-language-scaling/minicells-v2-10m.pt"
    observed_source_sha = verify_source_checkpoint(source_checkpoint)
    print(f"verified source TextNCA checkpoint: {observed_source_sha}")

    historical_006 = json.loads((ROOT / SOURCE_006_DECISION).read_text(encoding="utf-8"))
    historical_007 = json.loads((ROOT / SOURCE_007_DECISION).read_text(encoding="utf-8"))
    historical = validate_historical_evidence(historical_006, historical_007)
    _write_json(output_root / "historical-evidence.json", historical)

    capability_decision, capability_replicates, capability_source_commit = _fetch_capability_results()
    capability = normalize_capability_evidence(
        capability_decision,
        capability_replicates,
        source_ref=CAPABILITY_RESULTS_REF,
        source_commit=capability_source_commit,
    )
    _write_json(output_root / "capability-evidence.json", capability)

    train, validation, tokenizer_path, corpus_manifest = prepare_scaling_corpus(
        ROOT,
        source_005_dir=ROOT / SOURCE_005,
        train_stream_tokens=BRIDGE_MATERIALIZED_TRAIN_TOKENS,
        validation_stream_tokens=200_000,
    )
    cache_dir = tokenizer_path.parent
    print(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "materialized_train_tokens": int(train.numel()),
            "validation_tokens": int(validation.numel()),
            "corpus_manifest": corpus_manifest.get("format"),
        }
    )
    del train, validation

    gpu_count_used = _run_workers(output_root, cache_dir)
    bridge = _aggregate_bridge(output_root, gpu_count_used=gpu_count_used)
    _write_json(output_root / "bridge-summary.json", bridge)

    decision = make_release_decision(
        historical=historical,
        bridge=bridge,
        capability=capability,
    )
    decision["formal_gpu_experiment_run"] = True
    decision["runtime_environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    _write_json(output_root / "decision.json", decision)

    figures_dir = output_root / "figures"
    generated = save_release_figures(
        historical=historical,
        bridge=bridge,
        capability=capability,
        decision=decision,
        output_dir=figures_dir,
    )
    decision["figures"] = generated
    _write_json(output_root / "decision.json", decision)
    write_public_release_summary(
        output_root / "PUBLIC-RELEASE-SUMMARY.md",
        historical=historical,
        bridge=bridge,
        capability=capability,
        decision=decision,
    )

    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"Public summary: {output_root / 'PUBLIC-RELEASE-SUMMARY.md'}")
    return 0


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
