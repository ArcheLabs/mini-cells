"""Summarize measured outcomes without treating run completion as causality."""
from collections import defaultdict
import json
from statistics import mean, median


def summarize(output):
    groups = defaultdict(list)
    comparisons = []
    for line in (output / "test.jsonl").read_text().splitlines():
        batch = json.loads(line)
        paired = defaultdict(dict)
        for row in batch["scores"]:
            task = "unknown" if row["unknown"] else f"positive_template_{row['template']}"
            groups[row["condition"], task].append(row)
            paired[row["entity"]][row["condition"]] = row
        for conditions in paired.values():
            if "E4b" in conditions:
                correct, donor = conditions["E2"], conditions["E4b"]
                comparisons.append((correct, donor))
    lines = ["# Test results", "", "| Condition | Task | Exact match | Mean answer NLL | Median answer NLL | Mean token F1 |",
             "|---|---|---:|---:|---:|---:|"]
    for (condition, task), rows in sorted(groups.items()):
        nll = [row["answer_nll"] for row in rows]
        matches = sum(row["greedy_exact_match"] for row in rows)
        lines.append(f"| {condition} | {task} | {matches}/{len(rows)} | "
                     f"{mean(nll):.4f} | {median(nll):.4f} | "
                     f"{mean(row['token_f1'] for row in rows):.4f} |")
    (output / "results.md").write_text("\n".join(lines) + "\n")
    switches = sum(a["greedy_exact_match"] and b["greedy_exact_match"]
                   for a, b in comparisons)
    changed = sum(a["generated_ids"] != b["generated_ids"] for a, b in comparisons)
    lines = ["# Counterfactual diagnosis", "",
             f"Generated tokens changed in {changed}/{len(comparisons)} paired queries.",
             f"Both generated answers exactly matched their respective Cell values in {switches}/{len(comparisons)} pairs.",
             ""]
    for key in ("read_rms_delta", "projected_read_rms_delta", "logit_rms_delta"):
        values = [b["diagnostics"][key] for _, b in comparisons]
        lines.append(f"- {key}: median {median(values):.6g}")
    zero_writes = sum(not any(b["diagnostics"]["fast_weight_rms_delta"])
                      for _, b in comparisons)
    lines.append(f"- Identical final fast weights: {zero_writes}/{len(comparisons)} pairs.")
    (output / "diagnosis.md").write_text("\n".join(lines) + "\n")
