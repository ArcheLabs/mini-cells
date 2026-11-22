"""Fact-group statistics for residual-based Cell selection."""
from collections import defaultdict
import math
import random


def pairwise(scores, gold):
    hits = [s for cell, s in scores.items() if cell in gold]
    misses = [s for cell, s in scores.items() if cell not in gold]
    return [(p > n) + 0.5 * (p == n) for p in hits for n in misses]


def curve(items):
    """Tie-aware pooled ROC AUC and threshold-based average precision."""
    totals = defaultdict(lambda: [0, 0])
    for score, positive in items:
        totals[score][int(positive)] += 1
    pos = sum(p for _, p in totals.values())
    neg = sum(n for n, _ in totals.values())
    tp = fp = auc = ap = 0
    for _, (n, p) in sorted(totals.items(), reverse=True):
        auc += p * (neg - fp - 0.5 * n)
        tp, fp = tp + p, fp + n
        ap += p * tp / (tp + fp)
    return auc / (pos * neg), ap / pos


def ranking(scores):
    return sorted(scores, key=lambda cell: (-scores[cell], cell))


def metrics(rows, key):
    scored = [(row[key], set(row["relevant"])) for row in rows]
    ranked = [(ranking(scores), gold) for scores, gold in scored if gold]
    auc, ap = curve((score, cell in gold) for scores, gold in scored for cell, score in scores.items())
    pairs = [x for scores, gold in scored if gold for x in pairwise(scores, gold)]
    n = len(ranked)
    return dict(auroc=auc, auprc=ap, pairwise=sum(pairs) / len(pairs),
                recall_at_1=sum(rank[0] in gold for rank, gold in ranked) / n,
                recall_at_2=sum(len(gold.intersection(rank[:2])) / len(gold) for rank, gold in ranked) / n,
                full_coverage_at_2=sum(gold.issubset(rank[:2]) for rank, gold in ranked) / n,
                mrr=sum(1 / next(i for i, c in enumerate(rank, 1) if c in gold) for rank, gold in ranked) / n,
                queries=n, no_answer_queries=len(scored) - n,
                no_answer_forced_selection_rate=1.0 if n != len(scored) else None)


def group_bootstrap(rows, key, repeats=2000):
    rng = random.Random(10117)
    blocks = defaultdict(list)
    for row in rows:
        blocks[row["group"]].append(row)
    # Preserve the number of fact groups per seed.
    strata = defaultdict(list)
    for group in blocks.values():
        strata[group[0]["seed"]].append(group)
    records = defaultdict(list)
    for _ in range(repeats):
        picked = [r for groups in strata.values() for group in rng.choices(groups, k=len(groups)) for r in group]
        summary = metrics(picked, key)
        for metric in ("auroc", "auprc", "pairwise", "recall_at_1", "recall_at_2", "full_coverage_at_2", "mrr"):
            records[metric].append(summary[metric])
    return {key: [sorted(values)[math.floor((repeats - 1) * q)] for q in (0.025, 0.975)]
            for key, values in records.items()}


def rank_signal(result, seed_metrics, excluded, controls, random_auc):
    ci = result["ci95"]["auroc"]
    if (result["auroc"] >= .70 and ci[0] > .60 and result["pairwise"] >= .65
            and sum(m["pairwise"] > .50 for m in seed_metrics.values()) >= 2
            and min(excluded) > .60 and controls and result["auroc"] > random_auc):
        return "RESEARCH_PASS"
    if ci[1] > .50 and (result["pairwise"] > .5 or result["auroc"] > .5):
        return "UNSTABLE_SIGNAL"
    return "NO_POSITIVE_SIGNAL"
