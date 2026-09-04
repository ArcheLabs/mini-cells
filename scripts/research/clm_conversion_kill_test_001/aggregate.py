from __future__ import annotations

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


def aggregate() -> dict[str, Any]:
    protocol = _load(PROTOCOL_PATH)
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
        if summary.get("experiment") != protocol["experiment"] or int(summary.get("seed", -1)) != seed:
            raise RuntimeError(f"seed identity mismatch for {path}")
        if summary.get("status") not in {"PASS", "FAIL"}:
            raise RuntimeError(f"seed is not terminal: {path}")
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
