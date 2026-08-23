from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from .reproducibility import write_json


def _first_step(path: Path, threshold: float):
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if float(row["token_accuracy"]) >= threshold:
                return int(row["step"])
    return None


def aggregate_seed_runs(root: str | Path, seeds=(1, 2, 3)) -> dict[str, object]:
    root = Path(root); reports = []
    for seed in seeds:
        path = root / f"seed-{seed}" / "report.json"
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    token = [r["metrics"]["final"]["token_accuracy"] for r in reports]
    exact = [r["metrics"]["final"]["exact_sequence_accuracy"] for r in reports]
    steps95 = [_first_step(root / f"seed-{seed}" / "metrics.csv", .95) for seed in seeds]
    steps99 = [_first_step(root / f"seed-{seed}" / "metrics.csv", .99) for seed in seeds]
    passed = all(value >= .99 for value in token) and statistics.mean(exact) >= .95
    result = {"format": "minicells.echo.multiseed.v1", "status": "PASS" if passed else "FAIL",
              "seeds": list(seeds), "final_token_accuracy": token,
              "mean_final_token_accuracy": statistics.mean(token), "std_final_token_accuracy": statistics.pstdev(token),
              "final_exact_sequence_accuracy": exact,
              "mean_final_exact_sequence_accuracy": statistics.mean(exact), "std_final_exact_sequence_accuracy": statistics.pstdev(exact),
              "steps_to_95_token_accuracy": steps95, "steps_to_99_token_accuracy": steps99}
    write_json(root / "three-seed-report.json", result)
    return result
