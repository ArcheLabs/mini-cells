"""Compute P1.1 scientific gates from independently paired NLL measurements."""
from protocol import SEEDS, paired_summary


def classify(results, *, engineering_pass, smoke=False, replicates=10000):
    if not engineering_pass:
        return {"status": "RUN_FAILED", "engineering_status": "FAIL",
                "scientific_status": "NOT_EVALUATED"}
    if smoke:
        return {"status": "INCONCLUSIVE", "engineering_status": "PASS",
                "scientific_status": "NOT_EVALUATED"}
    if set(results) != set(SEEDS) or any(len(rows) != 64 for rows in results.values()):
        raise ValueError("Scientific classification requires three seeds and 64 paired documents")
    summary = {}
    for name, control in (("memory", "empty"), ("specificity", "donor_mean"),
                          ("content", "zero_read")):
        differences = [[row[control] - row["correct"] for row in results[seed]]
                       for seed in SEEDS]
        summary[name] = paired_summary(differences, replicates)
    memory, specific, content = (summary[name] for name in ("memory", "specificity", "content"))
    fraction = specific["mean"] / memory["mean"] if memory["mean"] > 0 else None
    gates = {
        "memory_effectiveness": memory["ci95"][0] > 0,
        "content_specificity": specific["mean"] >= .01 and specific["ci95"][0] > 0,
        "zero_read_separation": content["mean"] >= .01 and content["ci95"][0] > 0,
        "specificity_fraction": fraction is not None and fraction >= .10,
        "seed_consistency": all(metric["win_rate"] >= .60 and
                                sum(value > 0 for value in metric["per_seed"]) >= 2
                                for metric in (specific, content)),
    }
    status = ("PASS" if all(gates.values()) else
              "FAIL" if any(metric["ci95"][1] <= 0 for metric in summary.values()) else
              "INCONCLUSIVE")
    return {"status": status, "engineering_status": "PASS", "scientific_status": status,
            "gates": gates, "specificity_fraction": fraction, "statistics": summary}
