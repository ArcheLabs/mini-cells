from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "research" / "validations" / "jam-knowledge-mutation-001" / "protocol.json"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "jam-knowledge-mutation-001"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(artifacts: Path = ARTIFACTS) -> dict[str, Any]:
    protocol = _load(PROTOCOL)
    formal = [int(value) for value in protocol["formal_seeds"]]
    summaries: dict[int, dict[str, Any]] = {}
    malformed: list[int] = []
    for seed in formal:
        path = artifacts / f"seed-{seed}" / "seed_summary.json"
        if not path.is_file():
            continue
        try:
            row = _load(path)
            if row.get("experiment") != protocol["experiment"] or int(row.get("seed", -1)) != seed:
                malformed.append(seed)
                continue
            summaries[seed] = row
        except Exception:
            malformed.append(seed)

    completed = sorted(summaries)
    missing = [seed for seed in formal if seed not in summaries]
    passed = [seed for seed, row in summaries.items() if row.get("status") == "PASS"]
    selected_capacities = {
        str(seed): row.get("selected_capacity") for seed, row in summaries.items()
    }
    complete = not missing and not malformed
    supported = complete and len(passed) >= 2
    if not complete:
        status = protocol["decision"]["status_if_incomplete"]
        scientific = False
    elif supported:
        status = protocol["decision"]["status_if_supported"]
        scientific = True
    else:
        status = protocol["decision"]["status_if_not_supported"]
        scientific = True

    decision = {
        "experiment": protocol["experiment"],
        "status": status,
        "scientific_decision": scientific,
        "formal_seeds": formal,
        "completed_seeds": completed,
        "missing_seeds": missing,
        "malformed_seeds": malformed,
        "passed_seeds": passed,
        "support_threshold": 2,
        "selected_capacity_by_seed": selected_capacities,
        "minimum_passing_capacity_observed": min(
            [int(value) for value in selected_capacities.values() if value is not None],
            default=None,
        ),
        "not_claimed": protocol["decision"]["not_claimed"],
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return decision


def main() -> None:
    print(json.dumps(aggregate(), sort_keys=True))


if __name__ == "__main__":
    main()
