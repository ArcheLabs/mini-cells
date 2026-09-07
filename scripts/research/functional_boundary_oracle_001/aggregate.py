from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = (
    ROOT
    / "research"
    / "validations"
    / "functional-boundary-oracle-001"
    / "protocol.json"
)
ARTIFACTS = ROOT / "artifacts" / "experiments" / "functional-boundary-oracle-001"
ENVIRONMENT_KEYS = (
    "torch",
    "transformers",
    "huggingface_hub",
    "safetensors",
    "cuda_device_name",
    "dtype",
)


def _environment_signature(record: dict[str, Any]) -> dict[str, Any]:
    environment = record.get("environment") or {}
    return {key: environment.get(key) for key in ENVIRONMENT_KEYS}


def aggregate() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    formal = [int(seed) for seed in protocol["formal_seeds"]]
    records: list[dict[str, Any]] = []
    for seed in formal:
        path = ARTIFACTS / f"seed-{seed}" / "result.json"
        if path.is_file():
            records.append(json.loads(path.read_text(encoding="utf-8")))

    completed = [int(record["seed"]) for record in records]
    passed = [int(record["seed"]) for record in records if record.get("status") == "PASS"]
    failed = [int(record["seed"]) for record in records if record.get("status") == "FAIL"]
    signatures = {
        str(record["seed"]): _environment_signature(record)
        for record in records
    }
    encoded = {
        json.dumps(signature, sort_keys=True, separators=(",", ":"))
        for signature in signatures.values()
    }
    environment_consistent = len(encoded) <= 1
    protocol_hashes = {str(record.get("protocol_sha256")) for record in records}
    protocol_consistent = len(protocol_hashes) <= 1

    if not environment_consistent:
        status = "ENVIRONMENT_MISMATCH"
        scientific_decision: bool | None = None
    elif not protocol_consistent:
        status = "PROTOCOL_MISMATCH"
        scientific_decision = None
    elif len(passed) >= 2:
        status = "FUNCTIONAL_BOUNDARY_ORACLE_SUPPORTED"
        scientific_decision = True
    elif len(failed) >= 2:
        status = "FUNCTIONAL_BOUNDARY_ORACLE_REJECTED"
        scientific_decision = False
    else:
        status = "FORMAL_SEEDS_INCOMPLETE"
        scientific_decision = None

    decision = {
        "experiment": protocol["experiment"],
        "status": status,
        "scientific_decision": scientific_decision,
        "formal_seeds": formal,
        "completed_seeds": completed,
        "passed_seeds": passed,
        "failed_seeds": failed,
        "missing_seeds": [seed for seed in formal if seed not in completed],
        "environment_consistent": environment_consistent,
        "environment_signatures": signatures,
        "protocol_consistent": protocol_consistent,
        "observed_protocol_hashes": sorted(protocol_hashes),
        "claim_if_supported": protocol["decision"]["claim_if_supported"],
        "claim_if_rejected": protocol["decision"]["claim_if_rejected"],
        "not_claimed": protocol["decision"]["not_claimed"],
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return decision


if __name__ == "__main__":
    print(json.dumps(aggregate(), indent=2, sort_keys=True))
