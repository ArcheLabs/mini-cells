from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

from minicells.cow_clm import COWRuntime, apply_cell_artifact, load_cell_artifact

ROOT = Path(__file__).resolve().parents[3]
LOCAL_ROOT = Path(__file__).resolve().parent
if str(LOCAL_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ROOT))

from dataset import track_candidates, track_rows  # noqa: E402
from run import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    _candidate_choice,
    _compatibility_probe,
    _load_model,
    _load_protocol,
    _write_json,
)


def verify_track(
    *,
    protocol: dict[str, Any],
    track: str,
    device: str,
    result_root: Path,
) -> dict[str, Any]:
    result = json.loads((result_root / track / "result.json").read_text(encoding="utf-8"))
    minimum = result.get("minimum_supported_capacity")
    if minimum is None:
        payload = {
            "track": track,
            "status": "SKIPPED_NO_PASSING_CAPACITY",
            "verified": False,
        }
        _write_json(result_root / track / "reload.json", payload)
        return payload

    capacity = int(minimum)
    row = next(
        item for item in result["capacity_results"] if int(item["capacity_sites"]) == capacity
    )
    artifact_path = result_root / row["artifact_path"]
    artifact = load_cell_artifact(artifact_path)
    if artifact.digest() != row["artifact_digest"]:
        raise RuntimeError("artifact digest differs from in-memory result")

    model, tokenizer = _load_model(device)
    runtime = COWRuntime(
        model,
        foundation_model_id=MODEL_ID,
        foundation_revision=MODEL_REVISION,
    )
    apply_cell_artifact(runtime, artifact)
    rows = track_rows(track, facts=int(protocol["tracks"][track].get("facts", 8)))
    candidates = track_candidates(track)
    root_before = _compatibility_probe(model, tokenizer, device)
    choice = _candidate_choice(
        model,
        tokenizer,
        rows["evaluation"],
        candidates,
        device,
        runtime=runtime,
        cell_id=artifact.cell_id,
    )
    root_after = _compatibility_probe(model, tokenizer, device)
    runtime.assert_foundation_unchanged()
    rollback_exact = torch.equal(root_before, root_after)
    threshold = float(protocol["tracks"][track]["minimum_choice_accuracy"])
    verified = (
        float(choice["strict_choice_accuracy"]) >= threshold
        and rollback_exact
        and artifact.parent_digest == runtime.root_digest
    )
    payload = {
        "track": track,
        "status": "VERIFIED" if verified else "FAILED",
        "verified": verified,
        "capacity_sites": capacity,
        "artifact_digest": artifact.digest(),
        "choice": choice,
        "rollback_exact": rollback_exact,
    }
    _write_json(result_root / track / "reload.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Fresh-runtime verifier for COW-CLM-001")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--track", choices=("knowledge", "capability"), required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "research" / "validations" / "cow-clm-001" / "protocol.json",
    )
    parser.add_argument("--result-root", type=Path)
    args = parser.parse_args()
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for fresh COW-CLM reload verification")
    protocol = _load_protocol(args.protocol)
    root = args.result_root or ROOT / "results" / "cow-clm-001" / f"seed-{protocol['seed']}"
    result = verify_track(
        protocol=protocol,
        track=args.track,
        device=args.device,
        result_root=root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
