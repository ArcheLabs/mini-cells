from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "scripts"))

import run_proposal_utility_discovery as base  # noqa: E402
import run_proposal_utility_discovery_resumable as resumable  # noqa: E402


OUT = ROOT / "results" / "proposal-utility-discovery-stable-v1"
WORKER = ROOT / "scripts" / "run_proposal_utility_discovery_worker_stable.py"
CHECKPOINT_DIR = OUT / "checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corrected, checkpointed Experiment 019.")
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Reuse complete stable worker CSV/JSON outputs without launching GPU workers.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore existing Phase-1/donor checkpoints and overwrite them after retraining.",
    )
    return parser.parse_args()


def _write_checkpoint_manifest() -> dict[str, object]:
    files = []
    if CHECKPOINT_DIR.is_dir():
        for path in sorted(CHECKPOINT_DIR.glob("*.pt")):
            files.append({"name": path.name, "bytes": path.stat().st_size})
    manifest = {
        "format": "minicells.proposal-utility-checkpoint-manifest.v1",
        "experiment": "019-stable",
        "checkpoint_dir": str(CHECKPOINT_DIR.relative_to(ROOT)),
        "files": files,
        "file_count": len(files),
        "expected_file_count": base.N_REPLICATES * (1 + len(base.SKILL_FAMILIES) + 1),
        "published_model_checkpoints": False,
        "purpose": "Kaggle-local recovery and oracle/feature remeasurement without donor retraining",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoint-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _annotate_decision(decision: dict[str, object], manifest: dict[str, object]) -> None:
    validation = decision.setdefault("numerical_validation", {})
    assert isinstance(validation, dict)
    validation.update({
        "gradient_oracle_numerics": "forward-equivalent stable gated replicator",
        "variance_denominator": "sqrt(max(weighted_variance, 1e-8))",
        "growth": "exp(min(ACTIVITY_RATE * fitness, log(20)))",
        "original_failed_run_preserved": "results/proposal-utility-discovery-v1",
        "stable_run": "results/proposal-utility-discovery-stable-v1",
        "checkpoint_protocol": "minicells.proposal-utility-checkpoint.v1",
        "checkpoint_file_count": int(manifest["file_count"]),
        "checkpoint_expected_file_count": int(manifest["expected_file_count"]),
        "checkpoints_published": False,
    })
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    # Keep the original failed 019 worker artifacts untouched. The stable rerun
    # has its own result directory so gradient-oracle comparisons remain auditable.
    base.OUT = OUT
    base.WORKER = WORKER
    os.environ["MINICELLS_019_CHECKPOINT_DIR"] = str(CHECKPOINT_DIR)
    if args.force_retrain:
        os.environ["MINICELLS_019_FORCE_RETRAIN"] = "1"
    else:
        os.environ.pop("MINICELLS_019_FORCE_RETRAIN", None)

    cache, _ = base.prepare_corpus()
    if args.postprocess_only:
        missing = [
            replicate
            for replicate in range(base.N_REPLICATES)
            if not resumable._replicate_complete(replicate)
        ]
        if missing:
            raise RuntimeError(f"stable worker outputs incomplete for replicates {missing}; cannot postprocess-only")
        gpu_count = min(2, max(1, torch.cuda.device_count()))
        print("postprocess-only: reusing complete stable worker CSV/JSON outputs")
    else:
        print(f"checkpoint cache: {CHECKPOINT_DIR}")
        if args.force_retrain:
            print("force-retrain enabled: existing checkpoints will be ignored and replaced")
        else:
            print("existing Phase-1/donor checkpoints will be reused; oracle/features will be remeasured")
        gpu_count = base.run_workers(cache)

    decision = resumable._postprocess(gpu_count)
    manifest = _write_checkpoint_manifest()
    _annotate_decision(decision, manifest)
    print(
        json.dumps(
            {
                "stable_status": decision["status"],
                "checkpoint_files": manifest["file_count"],
                "expected_checkpoint_files": manifest["expected_file_count"],
                "rerun_hint": "rerun this command to remeasure from checkpoints; use --postprocess-only for CSV-only postprocessing",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
