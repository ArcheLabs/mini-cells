from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from minicells.hybrid_clm import HybridManifest, load_cell_artifact

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = ROOT / "research" / "validations" / "granite-hybrid-clm-v0.1" / "protocol.json"
DEFAULT_RESULT_DIR = ROOT / "results" / "granite-hybrid-clm-v0.1"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_payload(
    result: dict[str, Any],
    protocol: dict[str, Any],
    *,
    result_dir: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    milestone = protocol["milestone"]

    if result.get("experiment") != protocol["experiment"]:
        errors.append("experiment identity mismatch")
    if result.get("foundation", {}).get("trainable") is not False:
        errors.append("foundation must remain frozen")
    if float(result.get("compatibility_max_abs_logit_delta", 1.0)) != 0.0:
        errors.append("zero-commit compatibility must be exact")
    if int(result.get("committed_facts", 0)) < int(milestone["minimum_committed_facts"]):
        errors.append("not all required controlled facts were committed")
    if float(result.get("retention_choice_accuracy", 0.0)) < float(
        milestone["minimum_final_semantic_choice_accuracy"]
    ):
        errors.append("final semantic-choice retention is below threshold")

    cells = list(result.get("cells", []))
    if len(cells) != int(milestone["controlled_facts"]):
        errors.append("controlled fact result count does not match protocol")
    for cell in cells:
        if cell.get("status") != "COMMITTED":
            errors.append(f"{cell.get('cell_id', 'unknown')} was not committed")
            continue
        if float(cell.get("production_choice_accuracy", 0.0)) != 1.0:
            errors.append(f"{cell.get('cell_id', 'unknown')} lacks exact semantic choice")

    child = result.get("contextual_child", {})
    if milestone.get("contextual_child_required", False):
        if child.get("status") != "COMMITTED":
            errors.append("contextual child was not committed")
        if float(child.get("old_choice_accuracy_with_child", 0.0)) != 1.0:
            errors.append("contextual child damages parent v1 semantics")
        if float(child.get("new_choice_accuracy", 0.0)) != 1.0:
            errors.append("contextual child did not acquire v2 semantics")
        if float(child.get("rollback_old_choice_accuracy", 0.0)) != 1.0:
            errors.append("contextual child rollback did not restore v1 semantics")

    manifest_payload = result.get("manifest", {})
    manifest_cells = manifest_payload.get("cells", [])
    expected_manifest_cells = int(result.get("committed_facts", 0))
    if child.get("status") == "COMMITTED":
        expected_manifest_cells += 1
    if len(manifest_cells) != expected_manifest_cells:
        errors.append("manifest does not contain every committed Cell artifact")

    if result_dir is not None and manifest_cells:
        artifacts = []
        for entry in manifest_cells:
            cell_id = str(entry["cell_id"])
            path = result_dir / "cells" / f"{cell_id}.pt"
            if not path.exists():
                errors.append(f"missing Cell artifact: {cell_id}")
                continue
            artifact = load_cell_artifact(path)
            if artifact.digest() != str(entry["digest"]):
                errors.append(f"artifact digest mismatch: {cell_id}")
            artifacts.append(artifact)

        if milestone.get("manifest_fork_merge_required", False) and len(artifacts) >= 2:
            base = HybridManifest(
                foundation_model_id=str(manifest_payload["foundation_model_id"]),
                foundation_revision=str(manifest_payload["foundation_revision"]),
            )
            for artifact in artifacts[:-2]:
                base = base.add(artifact)
            branch_a = base.add(artifacts[-2])
            branch_b = base.add(artifacts[-1])
            merged = branch_a.merge(branch_b)
            expected = dict(base.cells)
            expected[artifacts[-2].cell_id] = artifacts[-2].digest()
            expected[artifacts[-1].cell_id] = artifacts[-1].digest()
            if dict(merged.cells) != expected:
                errors.append("manifest fork/merge union failed")
            if merged.remove(artifacts[-2].cell_id) != branch_b:
                errors.append("manifest rollback after merge failed")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Granite Hybrid CLM v0.1 result")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    result = _load_json(args.result_dir / "result.json")
    protocol = _load_json(args.protocol)
    errors = validate_payload(result, protocol, result_dir=args.result_dir)
    payload = {
        "status": "GRANITE_HYBRID_CLM_V01_MILESTONE_ACCEPTED" if not errors else "GRANITE_HYBRID_CLM_V01_MILESTONE_REJECTED",
        "errors": errors,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
