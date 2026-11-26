"""Pairwise statistics and a reproducible decision from evaluation rows."""
from collections import defaultdict
import csv
import json
import math
from random import Random
from statistics import fmean, median, pstdev

COMPARISONS = (("B", "A"), ("C", "A"), ("D", "A"), ("B", "D"))
FIELDS = ("mean_target_nll", "target_perplexity", "exact_match", "token_f1", "constraint_satisfied")
TASKS = ("single_cell", "cross_cell", "irrelevant")
MIN_AUTO_EM_GAIN = 0.05
FORMAT = "minicells.mvp-quality.001.v1"


def describe(values):
    return dict(n=len(values), mean=fmean(values), median=median(values),
                sd=pstdev(values)) if values else dict(n=0, mean=None, median=None, sd=None)


def paired(values, seed=57219, replicates=2000):
    if not values:
        return describe(values) | {"ci95": None}
    rng = Random(seed)
    means = sorted(fmean(rng.choices(values, k=len(values))) for _ in range(replicates))
    return describe(values) | {"ci95": [means[int(.025 * replicates)], means[int(.975 * replicates)]]}


def routing(trace, relevant):
    got, ranked = trace["selected"], trace["ranked"]
    gold = set(relevant)
    if not gold:
        return dict(false_selection=bool(got), false_selection_count=len(got))
    matches = len(gold.intersection(got))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(2, len(gold))))
    return dict(recall_at_2=matches / len(gold), precision_at_2=matches / 2,
                hit_at_2=bool(matches), mrr=next((1 / (i + 1) for i, k in enumerate(ranked)
                                                if k in gold), 0.0),
                ndcg_at_2=sum(1 / math.log2(i + 2) for i, k in enumerate(got) if k in gold) / ideal,
                all_relevant_selected=matches == len(gold))


def calculate(samples, rows, arms, evaluation):
    indexed = defaultdict(dict)
    for row in rows:
        indexed[row["sample_id"]][row["arm"]] = row
    groups = {"all": samples, "relevant": [s for s in samples if s["relevant_cell_ids"]]}
    groups.update({task: [s for s in samples if s["task_type"] == task] for task in TASKS})
    summary = {}
    tables = []
    for group, members in groups.items():
        arm_rows = [(s, indexed[s["sample_id"]]) for s in members]
        scores, pairs = {}, {}
        for arm in arms:
            items = [m[arm] for _, m in arm_rows if arm in m and m[arm]["status"] == "OK"]
            scores[arm] = dict(samples=len(items), fallback=sum(r["fallback"] for r in items))
            for field in FIELDS:
                result = describe([r[field] for r in items if field in r])
                scores[arm][field] = result
                tables.append({"group": group, "metric": field, "comparison": arm, **result})
            tokens = [value for item in items for value in item.get("target_token_nll", [])]
            if tokens:
                scores[arm]["micro_target_nll"] = fmean(tokens)
                scores[arm]["micro_target_perplexity"] = math.exp(fmean(tokens))
                sums = scores[arm]["mean_target_nll"]["mean"]
                scores[arm]["macro_target_perplexity"] = math.exp(sums)
        for left, right in (*COMPARISONS, ("B", "best_single")):
            if left not in arms or (right not in arms and right != "best_single"):
                continue
            name = f"{left}-{right}"
            pairs[name] = {}
            for field in FIELDS:
                values = []
                for _, result in arm_rows:
                    other = [right] if right != "best_single" else ["single_1", "single_2"]
                    needed = [left, *other]
                    if any(x not in result or result[x]["status"] != "OK" or field not in result[x]
                           for x in needed):
                        continue
                    best = min if field in ("mean_target_nll", "target_perplexity") else max
                    values.append(result[left][field] - best(result[x][field] for x in other))
                result = paired(values)
                pairs[name][field] = result
                tables.append({"group": group, "metric": field, "comparison": name, **result})
        auto = [(s, m["D"]) for s, m in arm_rows if "D" in m and m["D"]["status"] == "OK"]
        selection = defaultdict(list)
        by_domain = defaultdict(list)
        misses = 0
        for sample, row in auto:
            trace = row["selection"]
            relevant = row["relevant_cell_ids"]
            for key, value in routing(trace, relevant).items():
                selection[key].append(value)
            for domain, cell in zip(sample["relevant_cell_ids"], relevant, strict=True):
                by_domain[domain].append(cell in trace["selected"])
            misses += bool(relevant and not set(relevant).issubset(trace["selected"]))
        by_coverage = defaultdict(list)
        for _, m in arm_rows:
            if any(arm not in m or m[arm]["status"] != "OK" for arm in "ABCD"):
                continue
            relevant = set(m["D"]["relevant_cell_ids"])
            coverage = ("not_applicable" if not relevant else
                        "all_support" if relevant.issubset(m["D"]["selection"]["selected"]) else
                        "missing_support")
            for arm in "ABCD":
                if "exact_match" in m[arm]:
                    by_coverage[f"{coverage}_{arm}_exact_match"].append(m[arm]["exact_match"])
                if "mean_target_nll" in m[arm]:
                    by_coverage[f"{coverage}_{arm}_mean_nll"].append(m[arm]["mean_target_nll"])
        summary[group] = dict(groups=scores, contrasts=pairs,
            performance_by_supported_selection={k: describe(v) for k, v in by_coverage.items()},
            selection={key: describe(values) for key, values in selection.items()},
            domain_hit={key: describe(values) for key, values in by_domain.items()},
            automatic_missing_support=misses)
    return summary, tables


def decision(split, samples, rows, summary, arms, evaluation):
    if split != "held_out":
        return "INCONCLUSIVE", "Development diagnostics only."
    needed = sum(len(arms) + (2 if s["task_type"] == "cross_cell" and "B" in arms else 0) for s in samples)
    if set(arms) != set("ABCD") or evaluation != "both":
        return "INCONCLUSIVE", "Partial evaluation configuration."
    if len(rows) != needed or any(row["status"] != "OK" or row.get("execution_mismatch") for row in rows):
        return "INVALID", "Incomplete or inconsistent paired execution."
    ci = summary["relevant"]["contrasts"]["D-A"]["exact_match"]["ci95"]
    if ci is None:
        return "INCONCLUSIVE", "Missing held-out estimate."
    if ci[0] > MIN_AUTO_EM_GAIN:
        return "PASS", "The lower paired CI for automatic EM gain exceeds five percentage points."
    if ci[1] <= MIN_AUTO_EM_GAIN:
        return "FAIL", "The upper paired CI for automatic EM gain does not exceed five percentage points."
    return "INCONCLUSIVE", "The paired CI spans the frozen five-point gain threshold."


def publish_tables(out, tables):
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, ["split", "group", "metric", "comparison", "n",
                                        "mean", "median", "sd", "ci95"])
        writer.writeheader()
        for entry in tables:
            writer.writerow(entry)
