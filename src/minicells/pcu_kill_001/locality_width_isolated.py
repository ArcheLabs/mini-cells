"""Process-isolated execution wrapper for PCU-LOCALITY-WIDTH-001.

The scientific diagnostic implementation remains in ``locality_width.py`` and
is deliberately pinned byte-for-byte to the original implementation commit.
This wrapper repairs only CUDA execution isolation: K=16 and K=32 run in
separate Python processes so each GPU owns an independent CUDA context.

Completed width evidence from the original threaded run remains resumable
because width-result scientific identity stays pinned to the unchanged core
implementation.  The current execution-repair commit/tree is recorded
separately in RUN_IDENTITY/DECISION.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch

from .governance import git_provenance, write_json
from .locality_width import (
    BASELINE_K,
    BATCH_SIZE,
    CALIBRATION_BATCH_SIZE,
    CALIBRATION_ROWS,
    DEFAULT_OUTPUT,
    DIRECT_CAPABILITY_FLOOR,
    ENGINEERING_SEED,
    EXPERIMENT_ID,
    FALLBACK_WIDTH,
    LAYER_BASELINE_ROOT,
    LEARNING_RATE,
    MAX_OPTIMIZER_STEPS,
    MAX_TRAINING_TOKENS,
    PRIMARY_WIDTHS,
    TARGET_LAYER,
    _assert_nested_widths,
    _classify,
    _load_layer_baseline,
    _load_resumable_width_result,
    _run_one_width,
    should_run_fallback,
)


EXECUTION_MODE = "spawned_python_process_per_cuda_device"
SCIENTIFIC_SOURCE_COMMIT = "a567c3d386ebbbcc1b5707be4af69fedd27fb455"
SCIENTIFIC_SOURCE_TREE = "16426efe9c1000ee5c9064bdf6a3c4a6cc96e73b"
SCIENTIFIC_CORE_PATH = "src/minicells/pcu_kill_001/locality_width.py"
SCIENTIFIC_CORE_BLOB_SHA = "3e0528380baa4b9dba0d5fe51871f1f98a578264"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=_repo_root(),
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _assert_scientific_core_unchanged() -> None:
    """Fail closed unless the locality-width scientific worker is byte-identical."""
    blob = _git_output("rev-parse", f"HEAD:{SCIENTIFIC_CORE_PATH}")
    if blob != SCIENTIFIC_CORE_BLOB_SHA:
        raise RuntimeError(
            "LOCALITY_SCIENTIFIC_CORE_CHANGED: CUDA repair must not modify locality_width.py"
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", SCIENTIFIC_SOURCE_COMMIT, "HEAD"],
        cwd=_repo_root(),
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("locality-width execution repair is not a descendant of the scientific source")


def _scientific_source() -> dict[str, Any]:
    return {
        "source_commit": SCIENTIFIC_SOURCE_COMMIT,
        "source_tree": SCIENTIFIC_SOURCE_TREE,
        "source_dirty": False,
        "source_ref": "codex/pcu-composability-kill-001",
        "status_porcelain": "",
        "execution_repair_separated": True,
    }


def _execution_source() -> dict[str, Any]:
    source = git_provenance(_repo_root())
    if source.get("source_dirty") is not False:
        raise RuntimeError("locality-width process-isolated execution requires a clean source tree")
    return source


def _validate_device(device: str) -> int:
    parsed = torch.device(str(device))
    if parsed.type != "cuda" or parsed.index is None:
        raise ValueError(f"locality-width worker requires explicit CUDA device, got {device}")
    if not torch.cuda.is_available() or parsed.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device is unavailable: {device}")
    return int(parsed.index)


def run_width_worker(
    *,
    width: int,
    device: str,
    output: Path = DEFAULT_OUTPUT,
    baseline_root: Path = LAYER_BASELINE_ROOT,
    seed: int = ENGINEERING_SEED,
) -> dict[str, Any]:
    """Run exactly one width in its own process and persist the complete result."""
    if int(seed) != ENGINEERING_SEED:
        raise ValueError("PCU-LOCALITY-WIDTH-001 is engineering-seed only")
    if int(width) not in {*PRIMARY_WIDTHS, FALLBACK_WIDTH}:
        raise ValueError(f"unexpected locality width worker request: {width}")
    _assert_scientific_core_unchanged()
    execution_source = _execution_source()
    scientific_source = _scientific_source()
    baseline = _load_layer_baseline(Path(baseline_root))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"WIDTH_{int(width):03d}.json"

    resumed = _load_resumable_width_result(
        path,
        width=int(width),
        baseline=baseline,
        source=scientific_source,
    )
    if resumed is not None:
        print(f"[pcu-locality-width] isolated resume K={width} from {path}", flush=True)
        return resumed

    device_index = _validate_device(device)
    torch.cuda.set_device(device_index)
    print(
        f"[pcu-locality-width] isolated worker pid={os.getpid()} K={width} device={device}",
        flush=True,
    )
    result = _run_one_width(
        width=int(width),
        device=str(device),
        baseline=baseline,
        source=scientific_source,
    )
    result["execution_isolation"] = {
        "mode": EXECUTION_MODE,
        "pid": os.getpid(),
        "execution_source_commit": execution_source.get("source_commit"),
        "execution_source_tree": execution_source.get("source_tree"),
        "scientific_core_blob_sha": SCIENTIFIC_CORE_BLOB_SHA,
    }
    write_json(path, result)
    return result


def _worker_command(
    *,
    width: int,
    device: str,
    output: Path,
    baseline_root: Path,
    seed: int,
) -> list[str]:
    script = _repo_root() / "scripts/research/run_pcu_locality_width_worker.py"
    return [
        sys.executable,
        str(script),
        "--width",
        str(int(width)),
        "--device",
        str(device),
        "--seed",
        str(int(seed)),
        "--baseline",
        str(Path(baseline_root)),
        "--out",
        str(Path(output)),
    ]


def _spawn_primary_workers(
    *,
    missing: Mapping[int, str],
    output: Path,
    baseline_root: Path,
    seed: int,
) -> None:
    processes: dict[int, subprocess.Popen[str]] = {}
    for width, device in missing.items():
        command = _worker_command(
            width=int(width),
            device=str(device),
            output=output,
            baseline_root=baseline_root,
            seed=seed,
        )
        print("+ " + " ".join(command), flush=True)
        processes[int(width)] = subprocess.Popen(
            command,
            cwd=_repo_root(),
            text=True,
            env=os.environ.copy(),
        )
    failures: list[str] = []
    for width, process in processes.items():
        code = process.wait()
        if code != 0:
            failures.append(f"K={width}:exit={code}")
    if failures:
        raise RuntimeError("LOCALITY_WIDTH_WORKER_FAILED: " + ", ".join(failures))


def _write_design_and_identity(
    *,
    output: Path,
    baseline: Mapping[str, Any],
    execution_source: Mapping[str, Any],
    scientific_source: Mapping[str, Any],
) -> None:
    design = {
        "schema": "minicells.pcu-locality-width-001.design.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "causal_variable": "selected_cell_width_k_only",
        "fixed": {
            "task": "A_only_U_to_V",
            "target_layer": TARGET_LAYER,
            "target_path": baseline["target_path"],
            "dataset_manifest_sha256": baseline["dataset_manifest_sha256"],
            "loss": "answer-token-causal-cross-entropy",
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "max_training_tokens": MAX_TRAINING_TOKENS,
            "batch_size": BATCH_SIZE,
            "allocation": "task-conditioned-gradient-l2-per-parameter:first_64_A_train",
            "allocation_execution": f"weighted_microbatch_{CALIBRATION_BATCH_SIZE}",
            "routing": "inherited_parent_router",
            "evaluation": "A_eval_greedy_exact",
            "capability_floor": DIRECT_CAPABILITY_FLOOR,
        },
        "widths": {
            "baseline_reused": BASELINE_K,
            "primary_parallel": list(PRIMARY_WIDTHS),
            "fallback": FALLBACK_WIDTH,
            "fallback_rule": "run only if max(K16,K32) direct accuracy < 0.80",
        },
        "baseline": {
            "artifact": baseline["artifact_source"],
            "selected_k": BASELINE_K,
            "direct_accuracy": baseline["direct_accuracy"],
            "selected": list(baseline["selected"]),
            "effective_count": baseline["effective_count"],
            "gradient_mass_at_k": baseline["topk_mass"].get(str(BASELINE_K)),
        },
        "execution": {
            "mode": EXECUTION_MODE,
            "reason": "isolate CUDA contexts after threaded illegal-memory-access failure",
            "scientific_core_blob_sha": SCIENTIFIC_CORE_BLOB_SHA,
            "scientific_semantics_changed": False,
        },
        "formal_execution_not_started": True,
        "scientific_evidence": False,
    }
    write_json(output / "DESIGN.json", design)
    write_json(
        output / "RUN_IDENTITY.json",
        {
            "schema": "minicells.pcu-locality-width-001.run-identity.v1",
            "experiment": EXPERIMENT_ID,
            "phase": "engineering_diagnostic",
            "seed": ENGINEERING_SEED,
            "run_id": output.name,
            "source": dict(execution_source),
            "scientific_source": dict(scientific_source),
            "baseline_source": baseline["baseline_source"],
            "formal_execution_not_started": True,
        },
    )


def run_locality_width_diagnostic_isolated(
    *,
    output: Path = DEFAULT_OUTPUT,
    baseline_root: Path = LAYER_BASELINE_ROOT,
    devices: tuple[str, str] = ("cuda:0", "cuda:1"),
    seed: int = ENGINEERING_SEED,
) -> dict[str, Any]:
    """Canonical process-isolated locality-width engineering execution."""
    if int(seed) != ENGINEERING_SEED:
        raise ValueError("PCU-LOCALITY-WIDTH-001 is engineering-seed only")
    if len(devices) != 2 or devices[0] == devices[1]:
        raise ValueError("two distinct CUDA devices are required")
    _assert_scientific_core_unchanged()
    for device in devices:
        _validate_device(device)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline_root = Path(baseline_root)
    baseline = _load_layer_baseline(baseline_root)
    execution_source = _execution_source()
    scientific_source = _scientific_source()
    _write_design_and_identity(
        output=output,
        baseline=baseline,
        execution_source=execution_source,
        scientific_source=scientific_source,
    )

    results: dict[int, dict[str, Any]] = {}
    missing: dict[int, str] = {}
    for width, device in zip(PRIMARY_WIDTHS, devices):
        path = output / f"WIDTH_{width:03d}.json"
        resumed = _load_resumable_width_result(
            path,
            width=width,
            baseline=baseline,
            source=scientific_source,
        )
        if resumed is None:
            missing[width] = device
        else:
            print(f"[pcu-locality-width] resume K={width} from {path}", flush=True)
            results[width] = resumed

    if missing:
        _spawn_primary_workers(
            missing=missing,
            output=output,
            baseline_root=baseline_root,
            seed=seed,
        )
    for width in PRIMARY_WIDTHS:
        path = output / f"WIDTH_{width:03d}.json"
        loaded = _load_resumable_width_result(
            path,
            width=width,
            baseline=baseline,
            source=scientific_source,
        )
        if loaded is None:
            raise RuntimeError(f"isolated locality-width worker produced no result for K={width}")
        results[width] = loaded

    primary = {width: results[width] for width in PRIMARY_WIDTHS}
    fallback_required = should_run_fallback(primary)
    if fallback_required:
        width = FALLBACK_WIDTH
        path = output / f"WIDTH_{width:03d}.json"
        resumed = _load_resumable_width_result(
            path,
            width=width,
            baseline=baseline,
            source=scientific_source,
        )
        if resumed is None:
            command = _worker_command(
                width=width,
                device=devices[0],
                output=output,
                baseline_root=baseline_root,
                seed=seed,
            )
            print("+ " + " ".join(command), flush=True)
            subprocess.run(command, cwd=_repo_root(), check=True, text=True, env=os.environ.copy())
            resumed = _load_resumable_width_result(
                path,
                width=width,
                baseline=baseline,
                source=scientific_source,
            )
        if resumed is None:
            raise RuntimeError("isolated locality-width K=64 worker produced no result")
        results[width] = resumed

    _assert_nested_widths(results, baseline)
    status, comparison = _classify(results, baseline)
    decision = {
        "schema": "minicells.pcu-locality-width-001.decision.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "status": status,
        "valid_run": True,
        "scientific_evidence": False,
        "formal_execution_not_started": True,
        "fallback_k64_required": fallback_required,
        "comparison": comparison["rows"],
        "best": comparison["best"],
        "rescued": comparison["rescued"],
        "improved": comparison["improved"],
        "capability_floor": DIRECT_CAPABILITY_FLOOR,
        "execution": {
            "mode": EXECUTION_MODE,
            "scientific_core_blob_sha": SCIENTIFIC_CORE_BLOB_SHA,
            "scientific_semantics_changed": False,
        },
        "interpretation": (
            "increasing only L7 mutation width reached the inherited direct-capability floor"
            if comparison["rescued"]
            else "increasing only L7 mutation width did not reach the inherited direct-capability floor"
        ),
        "source": dict(execution_source),
        "scientific_source": dict(scientific_source),
    }
    write_json(output / "DECISION.json", decision)
    return decision


__all__ = [
    "EXECUTION_MODE",
    "SCIENTIFIC_SOURCE_COMMIT",
    "SCIENTIFIC_CORE_BLOB_SHA",
    "run_width_worker",
    "run_locality_width_diagnostic_isolated",
]
