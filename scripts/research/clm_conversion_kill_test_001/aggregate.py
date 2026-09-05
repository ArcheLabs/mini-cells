from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = (
    ROOT / "research" / "validations" / "clm-conversion-kill-test-001" / "protocol.json"
)
ARTIFACTS = ROOT / "artifacts" / "experiments" / "clm-conversion-kill-test-001"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_seed_summary(
    summary: dict[str, Any],
    *,
    seed: int,
    protocol: dict[str, Any],
    protocol_sha256: str,
) -> None:
    if summary.get("experiment") != protocol["experiment"]:
        raise RuntimeError(f"seed {seed} experiment identity mismatch")
    if int(summary.get("seed", -1)) != seed:
        raise RuntimeError(f"seed {seed} seed identity mismatch")
    if summary.get("status") not in {"PASS", "FAIL"}:
        raise RuntimeError(f"seed {seed} is not terminal")
    if summary.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError(
            f"seed {seed} protocol identity mismatch: "
            f"{summary.get('protocol_sha256')} != {protocol_sha256}"
        )
    expected_dataset = protocol["dataset"]["generator_git_blob_sha"]
    if summary.get("dataset_generator_git_blob_sha") != expected_dataset:
        raise RuntimeError(
            f"seed {seed} dataset generator identity mismatch: "
            f"{summary.get('dataset_generator_git_blob_sha')} != {expected_dataset}"
        )
    expected_implementation = protocol.get("implementation_git_blobs")
    if not isinstance(expected_implementation, dict) or not expected_implementation:
        raise RuntimeError("frozen protocol has no implementation_git_blobs map")
    observed_implementation = summary.get("implementation_git_blobs")
    if observed_implementation != expected_implementation:
        raise RuntimeError(
            f"seed {seed} implementation identity mismatch; refusing mixed formal evidence"
        )


def aggregate() -> dict[str, Any]:
    protocol = _load(PROTOCOL_PATH)
    protocol_sha256 = _sha256(PROTOCOL_PATH)
    formal = [int(value) for value in protocol["formal_seeds"]]
    completed: list[int] = []
    passed: list[int] = []
    failed: list[int] = []
    summaries: dict[str, dict[str, Any]] = {}
    for seed in formal:
        path = ARTIFACTS / f"seed-{seed}" / "seed_summary.json"
        if not path.is_file():
            continue
        summary = _load(path)
        _validate_seed_summary(
            summary,
            seed=seed,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
        )
        completed.append(seed)
        summaries[str(seed)] = summary
        if summary["status"] == "PASS":
            passed.append(seed)
        else:
            failed.append(seed)

    if len(completed) < len(formal):
        status = protocol["decision"]["status_if_incomplete"]
        scientific_decision = False
    else:
        scientific_decision = True
        status = (
            protocol["decision"]["status_if_supported"]
            if len(passed) >= 2
            else protocol["decision"]["status_if_not_supported"]
        )

    decision = {
        "experiment": protocol["experiment"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_sha256,
        "dataset_generator_git_blob_sha": protocol["dataset"]["generator_git_blob_sha"],
        "implementation_git_blobs": protocol["implementation_git_blobs"],
        "status": status,
        "scientific_decision": scientific_decision,
        "formal_seeds": formal,
        "completed_seeds": completed,
        "passed_seeds": passed,
        "failed_seeds": failed,
        "required_passing_seeds": 2,
        "seed_summaries": summaries,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return decision


if __name__ == "__main__":
    print(json.dumps(aggregate(), indent=2, sort_keys=True))
