#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "research" / "validations" / "moe-mutation-001" / "protocol.json"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "moe-mutation-001"
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
    encoded_signatures = {
        json.dumps(signature, sort_keys=True, separators=(",", ":"))
        for signature in signatures.values()
    }
    environment_consistent = len(encoded_signatures) <= 1

    if not environment_consistent:
        status = "ENVIRONMENT_MISMATCH"
        scientific_decision: bool | None = None
    elif len(passed) >= 2:
        status = "MOE_MUTATION_SUPPORTED"
        scientific_decision = True
    elif len(failed) >= 2:
        status = "MOE_MUTATION_REJECTED"
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
        "claim_if_supported": protocol["decision"]["claim_if_supported"],
        "not_claimed": protocol["decision"]["not_claimed"],
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return decision


if __name__ == "__main__":
    print(json.dumps(aggregate(), indent=2, sort_keys=True))
