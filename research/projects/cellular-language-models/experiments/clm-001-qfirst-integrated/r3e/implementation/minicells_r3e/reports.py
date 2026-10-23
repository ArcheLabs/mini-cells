"""Separate exploratory hypotheses; Story resampling never substitutes for three seeds."""

import numpy as np
from minicells_r3d.reports import paired_story_interval
from .common import ARMS, ID, canonical, atomic


def summary(c, results, source, failures):
    complete = all(set(results.get(str(s), {})) == set(ARMS) for s in c["seeds"])
    nll = {s: {a: r["quality"]["nll"] for a, r in per.items()} for s, per in results.items()}
    paired = {}
    comparisons = {}
    for a, b in c["statistics"]["comparisons"]:
        key = a + " vs " + b
        rows = []
        deltas = []
        for seed in c["seeds"]:
            per = results.get(str(seed), {})
            if a not in per or b not in per:
                continue
            x = per[a]["quality"]["development"]
            y = per[b]["quality"]["development"]
            r = paired_story_interval(x, y, seed, c["statistics"]["bootstrap_replicates"])
            r["seed"] = seed
            rows.append(r)
            deltas.append(nll[str(seed)][a] - nll[str(seed)][b])
        paired[key] = rows
        comparisons[key] = {
            "paired_seed_deltas": deltas,
            "mean": float(np.mean(deltas)) if deltas else None,
            "seed_sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
            "supported": complete
            and all(v < 0 for v in deltas)
            and float(np.mean(deltas)) <= -c["statistics"]["minimum_gain"],
        }
    for seed in c["seeds"]:
        entries = [v for rows in paired.values() for v in rows if v["seed"] == seed]
        previous = 0.0
        for i, r in enumerate(sorted(entries, key=lambda x: x.get("exploratory_bootstrap_p", 1))):
            previous = max(
                previous, min(1.0, (len(entries) - i) * r.get("exploratory_bootstrap_p", 1))
            )
            r["holm_exploratory_p"] = previous
    hypotheses = {h: "INCONCLUSIVE" for h in ("H1", "H2", "H3", "H4")}
    if complete:
        for h, keys in [
            ("H1", ["A-C vs A-F"]),
            ("H2", ["A-C vs A-T"]),
            ("H3", ["B-C vs B-F", "B-C vs B-R"]),
        ]:
            hypotheses[h] = (
                "SUPPORTED_EXPLORATORY"
                if all(comparisons[k]["supported"] for k in keys)
                else "NOT_SUPPORTED"
            )
        # H4 describes organization only. It is explicitly independent of task quality H3.
        g = [results[str(seed)]["B-C"]["quality"]["graph"] for seed in c["seeds"]]
        tv = c["statistics"]["h4_minimum_tv"]
        structural = all(
            row["predecessor_dependence_tv"] >= tv and row["input_group_joint_edge_tv"] >= tv
            for row in g
        )
        hypotheses["H4"] = "SUPPORTED_EXPLORATORY" if structural else "NOT_SUPPORTED"
    if failures:
        hypotheses = {h: "INVALID_RUN" for h in hypotheses}
    return {
        "experiment_id": ID,
        "source_sha": source,
        "config_sha256": canonical(c),
        "status": "COMPLETED"
        if complete and not failures
        else "PARTIAL_INVALID_RUN"
        if failures
        else "PARTIAL",
        "nll": nll,
        "mean_nll": {
            a: float(np.mean([per[a] for per in nll.values() if a in per]))
            for a in ARMS
            if any(a in per for per in nll.values())
        },
        "comparisons": comparisons,
        "story_paired_bootstrap": paired,
        "hypotheses": hypotheses,
        "independent_seed_count": len(c["seeds"]),
        "confirmation": "SEALED",
        "development": "HISTORICALLY_USED_EXPLORATORY_ONLY",
        "calibration": "TRAINING_ORGANIZATION_SELECTION_NOT_INDEPENDENT_VALIDATION",
        "failures": failures,
        "budget_comparison": "EQUAL_MAIN_TOKENS_NOT_EQUAL_TOTAL_TRAINING_COMPUTE",
        "h4_scope": "Conditional nonuniform connection distributions only; no specialization or small-world evidence",
        "actual_kaggle_persistence_verified": False,
    }


def write(c, results, source, output, failures, manifest):
    r = summary(c, results, source, failures)
    root = output / "reports"
    atomic(root / "METRICS.json", r)
    atomic(root / "RUN_MANIFEST.json", manifest)
    policy = {
        s: {
            a: {
                "proposals": v["proposals"],
                "fixed_weight_route_drift": v["fixed_weight_route_drift"],
            }
            for a, v in per.items()
        }
        for s, per in results.items()
    }
    atomic(root / "POLICY_STABILITY.json", policy)
    atomic(
        root / "ADAPTATION_RESULTS.json",
        {
            s: {
                a: {
                    "accepted": sum(p["accepted"] for p in v["proposals"]),
                    "proposed": len(v["proposals"]),
                    "events": v["proposals"],
                }
                for a, v in per.items()
            }
            for s, per in results.items()
        },
    )
    atomic(
        root / "GRAPH_SUMMARY.json",
        {
            s: {
                a: {
                    "graph": v["quality"]["graph"],
                    "same_cells_route_replacement_nll": v["quality"][
                        "same_cells_route_replacement_nll"
                    ],
                }
                for a, v in per.items()
                if a.startswith("B-")
            }
            for s, per in results.items()
        },
    )
    atomic(
        root / "TRAINING_COST.json",
        {
            s: {
                a: {
                    "training": v["training_cost"],
                    "inference": v["quality"]["inference_cost"],
                    "gradient_coverage": v["gradient_coverage"],
                }
                for a, v in per.items()
            }
            for s, per in results.items()
        },
    )
    lines = [
        "# R3e exploratory results",
        "",
        f"Status: {r['status']}",
        f"Source: {source}",
        f"Config: {r['config_sha256']}",
        "",
        "## Development NLL",
        "",
        "| Seed | " + " | ".join(ARMS) + " |",
        "|---|" + "---:|" * len(ARMS),
    ]
    for s, per in r["nll"].items():
        lines.append(
            "| "
            + s
            + " | "
            + " | ".join("%.6f" % per[a] if a in per else "MISSING" for a in ARMS)
            + " |"
        )
    lines += ["", "## Independent hypotheses", ""] + [
        f"- {h}: {state}" for h, state in r["hypotheses"].items()
    ]
    lines += [
        "",
        "Three training seeds only. Story intervals and Holm adjustments are corroborative; no strong confirmation claim.",
        "Calibration selected training organizations. Development was used historically. Confirmation remains SEALED.",
        "A-C/B-C include discarded adaptation and proposal costs. Equal main tokens do not imply equal total compute.",
        "H4 describes conditional edge distributions only; it does not prove functional specialization or small-world topology.",
    ]
    (root / "SCIENTIFIC-FINDINGS.md").write_text("\n".join(lines) + "\n")
    return r
