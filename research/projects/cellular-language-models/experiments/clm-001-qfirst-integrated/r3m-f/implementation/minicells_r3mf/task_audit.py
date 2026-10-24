"""Fact, exact-text and role checks with disclosed numerical control mismatch."""

from collections import Counter
from minicells_r3me.common import canonical


def distribution(rows, weighted=False):
    result = Counter()
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], r)
    for r in groups.values():
        result[int(r["answer"])] += r.get("sampling_weight", 1) if weighted else 1
    total = sum(result.values())
    return {str(k): v / total for k, v in sorted(result.items())}


def audit(c, r):
    facts = {}
    for task in ("a1", "copy"):
        tr, dv = r[task]["train"], r[task]["dev"]
        tg, dg = {x["group"] for x in tr}, {x["group"] for x in dv}
        if tg & dg or {x["sample_sha256"] for x in tr} & {x["sample_sha256"] for x in dv}:
            raise ValueError("FACT_OR_TEXT_LEAKAGE")
        facts[task] = dict(
            train=len(tg), dev=len(dg), overlap=0, train_hash=canonical(tr), dev_hash=canonical(dv)
        )
    if facts["a1"]["train"] != 44 or facts["a1"]["dev"] != 11:
        raise ValueError("FACT_COUNT_CHANGED")
    for role in ("train", "dev"):
        groups = {x["group"]: x for x in r["a1"][role]}
        rows = list(groups.values())
        if not all(
            (
                any(int(x["answer"]) < 10 for x in rows),
                any(int(x["answer"]) >= 10 for x in rows),
                any(0 in x["operands"] for x in rows),
                any(x["operands"][0] == x["operands"][1] for x in rows),
            )
        ):
            raise ValueError("FACT_STRATIFICATION_FAILED")
        facts["a1"][role + "_strata"] = {
            "below10": sum(int(x["answer"]) < 10 for x in rows),
            "at_least10": sum(int(x["answer"]) >= 10 for x in rows),
            "zero": sum(0 in x["operands"] for x in rows),
            "equal": sum(x["operands"][0] == x["operands"][1] for x in rows),
        }
    a, cp = distribution(r["a1"]["train"]), distribution(r["copy"]["train"], True)
    tv = 0.5 * sum(abs(a.get(str(i), 0) - cp.get(str(i), 0)) for i in range(19))
    counts = Counter(int(x["answer"]) for x in {x["group"]: x for x in r["a1"]["train"]}.values())
    majority = min(counts, key=lambda k: (-counts[k], k))
    dev = list({x["group"]: x for x in r["a1"]["dev"]}.values())
    return dict(
        status="GENERATED_TASK_AUDIT_PASS",
        facts=facts,
        tiny_facts=[x["group"] for x in r["tiny"]],
        a1_train_distribution=a,
        copy_train_distribution=cp,
        total_variation_distance=tv,
        one_digit_mass={
            "a1": sum(v for k, v in a.items() if int(k) < 10),
            "copy": sum(v for k, v in cp.items() if int(k) < 10),
        },
        frequency_baseline={
            "answer": str(majority),
            "dev_fact_accuracy": sum(int(x["answer"]) == majority for x in dev) / len(dev),
        },
        mismatch="Disjoint Copy numbers exclude some A1 answers; not an exactly frequency-matched control",
        a2_a3_materialized=False,
        confirmation="SEALED",
        r3m_main_candidates_accessed=False,
        historical_global_isolation="NOT_CLAIMED_EXPLORATORY_REUSED_DEVELOPMENT",
    )
