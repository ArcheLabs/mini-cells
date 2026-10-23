"""Conservative observations and explicit limits, never a single-cause diagnosis."""

import numpy as np
from .common import ID


def decision(c, jobs, complete):
    cases = []
    evidence = {}
    st = c["statistics"]
    p = st["practical_nll"]
    seeds = [str(s) for s in c["seeds"]]
    if not complete:
        return {
            "cases": [],
            "recommendation": "INCONCLUSIVE",
            "Q1": "INCONCLUSIVE",
            "Q2": "INCONCLUSIVE",
            "Q3": "INCONCLUSIVE",
            "Q4": "NOT_IDENTIFIABLE_FROM_SAVED_CHECKPOINTS",
            "reason": "All 18 required checkpoint analyses must finish before cross-seed mechanism interpretation.",
        }

    def metric(seed, arm, name):
        return jobs[seed][arm]["policy_ranking"]["development"]["overall"][name]

    rows = {}
    for arm in c["scaled_arms"]:
        rows[arm] = []
        for seed in seeds:
            interpolation = jobs[seed][arm]["interpolation"]["scales"]
            rows[arm].append(
                {
                    "seed": int(seed),
                    "route_delta": interpolation["1.0"]["mean_nll"]
                    - interpolation["0.0"]["mean_nll"],
                    "scale_deltas": {
                        k: v["mean_nll"] - interpolation["0.0"]["mean_nll"]
                        for k, v in interpolation.items()
                    },
                    "q_accuracy": metric(seed, arm, "q_accuracy"),
                    "q_spearman": metric(seed, arm, "q_spearman"),
                    "base_accuracy": metric(seed, arm, "base_accuracy"),
                    "ranking_change": metric(seed, arm, "q_minus_base_accuracy"),
                    "range": metric(seed, arm, "range"),
                }
            )
    evidence["per_arm_seed"] = rows
    material = all(
        metric(seed, "L", "range")["mean"] is not None and metric(seed, "L", "range")["mean"] >= p
        for seed in seeds
    )
    negligible = all(
        metric(seed, arm, "range")["ci95"] is not None and metric(seed, arm, "range")["ci95"][1] < p
        for seed in seeds
        for arm in c["counterfactual_arms"]
    )
    if negligible:
        cases.append("A")
    poor = material and all(
        row["q_accuracy"]["mean"] is not None
        and row["q_accuracy"]["mean"] <= 0.5 + st["ranking_chance_band"]
        for row in rows["L"]
    )
    if poor:
        cases.append("B")
    good_harm = all(
        row["q_accuracy"]["mean"] is not None
        and row["q_accuracy"]["mean"] >= st["ranking_good_min"]
        and row["q_accuracy"]["ci95"] is not None
        and row["q_accuracy"]["ci95"][0] > 0.5
        and row["q_spearman"]["mean"] is not None
        and row["q_spearman"]["mean"] > 0
        and row["route_delta"] >= p
        for row in rows["J"]
    )
    if good_harm:
        cases.append("C")
    small_benefits = []
    for arm in c["scaled_arms"]:
        for scale in ("0.1", "0.25"):
            if all(
                row["scale_deltas"][scale] <= -p and row["route_delta"] >= p for row in rows[arm]
            ):
                small_benefits.append({"arm": arm, "fixed_scale": scale})
    if small_benefits:
        cases.append("D")
    evidence["predefined_small_scale_patterns"] = small_benefits
    f_improves = all(
        jobs[s]["F"]["interpolation"]["scales"]["0.0"]["mean_nll"]
        <= jobs[s]["F0"]["interpolation"]["scales"]["0.0"]["mean_nll"] - p
        for s in seeds
    )
    if f_improves and any(
        all(abs(row["route_delta"]) >= p for row in rows[arm]) for arm in ("P", "J")
    ):
        cases.append("E_COMPATIBLE_NOT_CAUSALLY_IDENTIFIED")
    if any(
        min(row["route_delta"] for row in rows[arm])
        < 0
        < max(row["route_delta"] for row in rows[arm])
        for arm in c["scaled_arms"]
    ):
        cases.append("F")
    baseline_predicts = all(
        row["base_accuracy"]["ci95"] is not None
        and row["base_accuracy"]["ci95"][0] > 0.5
        and row["ranking_change"]["ci95"] is not None
        and row["ranking_change"]["ci95"][1] < 0
        for row in rows["L"]
    )
    recommendation = (
        "POLICY_STABILITY_REVISION"
        if ("C" in cases or "D" in cases)
        else "ROUTING_LEARNING_REVISION"
        if poor and baseline_predicts
        else "CELL_ORGANIZATION_REVIEW"
        if negligible
        else "INCONCLUSIVE"
    )
    return {
        "cases": cases,
        "recommendation": recommendation,
        "Q1": {
            "observation": "Material sampled conditional losses"
            if material
            else "No reproducible material sampled difference established",
            "predictability_limit": "Observed-label candidate gaps alone do not establish predictable expected future value or deployable gains.",
        },
        "Q2": {
            "current_correction_near_chance": poor,
            "original_gate_independent_ranking_advantage": baseline_predicts,
            "scope": "held-checkpoint Calibration and Development; historical Train metrics are separate",
        },
        "Q3": {
            "fixed_checkpoint_route_effects": rows,
            "small_scale_pattern": small_benefits,
            "scope": "direct within-checkpoint intervention; no deployment coefficient selected",
        },
        "Q4": "NOT_IDENTIFIABLE_FROM_SAVED_CHECKPOINTS",
        "evidence": evidence,
        "limits": [
            "Cases are descriptive and may coexist; rank association is not mechanism causality.",
            "No new predictor was trained, so absence of predictability beyond these existing scorers is not established.",
            "Different final Cell weights cannot identify the causal effect of every historical Connection update.",
        ],
        "r3e": "Do not advance on this audit when recommendation is INCONCLUSIVE.",
    }


def aggregate(c, jobs, failures, identity, inputs):
    complete = (
        all(all(a in jobs.get(str(seed), {}) for a in c["checkpoint_steps"]) for seed in c["seeds"])
        and not failures
    )
    attribution = {}
    for seed, per in jobs.items():
        attribution[seed] = {
            "route_deltas": {
                a: per[a]["interpolation"]["scales"]["1.0"]["mean_nll"]
                - per[a]["interpolation"]["scales"]["0.0"]["mean_nll"]
                for a in c["scaled_arms"]
                if a in per
            },
            "fixed_original_route_nll": {
                a: per[a]["interpolation"]["scales"]["0.0"]["mean_nll"]
                for a in ("F", "P", "J")
                if a in per
            },
            "historical_causal_identification": "NOT_IDENTIFIABLE_FROM_SAVED_CHECKPOINTS",
        }
    total = {}
    for per in jobs.values():
        for row in per.values():
            for key, value in row["analysis_cost"].items():
                if isinstance(value, (int, float)):
                    total[key] = total.get(key, 0) + value
    if complete:
        expected = {
            "sampled_states": c["sampling"]["estimated_states"],
            "candidate_actions": c["sampling"]["estimated_actions"],
            "candidate_cell_rows": c["sampling"]["estimated_counterfactual_cell_rows"],
        }
        if any(total.get(k) != v for k, v in expected.items()):
            failures.append(
                {"code": "ANALYSIS_BUDGET_VIOLATION", "expected": expected, "observed": total}
            )
            complete = False
        for seed, per in jobs.items():
            components = {a: r["checkpoint"]["component_fingerprints"] for a, r in per.items()}
            same_shared = all(
                len({r[k] for r in components.values()}) == 1 for k in ("shared", "gate")
            )
            frozen_cells = len({components[a]["cells"] for a in ("F0", "L", "M")}) == 1
            inherited = components["P"]["correction"] == components["L"]["correction"]
            if not (same_shared and frozen_cells and inherited):
                failures.append(
                    {"code": "CHECKPOINT_FROZEN_COMPONENT_RELATIONSHIP_MISMATCH", "seed": seed}
                )
                complete = False
    return {
        "experiment_id": ID,
        "identity": identity,
        "status": "READ_ONLY_AUDIT_COMPLETE" if complete else "PARTIAL" if jobs else "BLOCKED",
        "completed_checkpoints": sum(len(v) for v in jobs.values()),
        "required_checkpoints": 18,
        "jobs": jobs,
        "failures": failures,
        "attribution": attribution,
        "conclusions": decision(c, jobs, complete),
        "input_provenance": inputs,
        "analysis_cost": total,
        "training_updates": 0,
        "optimizer_instances": 0,
        "confirmation": "SEALED",
        "scientific_scope": "Exploratory analysis; three training Seeds only. Story/Stage/Token/action samples are not independent training repeats.",
    }


def markdown(summary):
    lines = [
        "# R3d-A — Checkpoint Read-Only Mechanism Audit",
        "",
        "Status: " + summary["status"],
        "Completed Checkpoints: %s / 18" % summary["completed_checkpoints"],
        "",
        "Training updates: 0. Optimizer instances: 0. Confirmation: SEALED.",
        "",
        "## Closed-loop Development NLL",
        "",
        "| Seed | F0 | L | M | F | P | J |",
        "|---|---|---|---|---|---|---|",
    ]
    for seed in map(str, summary["identity"]["seeds"]):
        per = summary["jobs"].get(seed, {})
        lines.append(
            "| "
            + seed
            + " | "
            + " | ".join(
                "—" if a not in per else "%.6f" % per[a]["reproduced_nll"]
                for a in ("F0", "L", "M", "F", "P", "J")
            )
            + " |"
        )
    lines += ["", "## Q1–Q4", ""]
    for key in ("Q1", "Q2", "Q3", "Q4"):
        lines.append("- " + key + ": " + str(summary["conclusions"].get(key)))
    lines += [
        "",
        "## Next research direction",
        "",
        summary["conclusions"]["recommendation"],
        "",
        "Observed values, ranking associations and within-checkpoint interventions are reported separately. The affine correction can only express a shared state projection plus Stage biases; these data do not prove a richer predictor would or would not learn expected future value.",
        "",
        "Hindsight Candidate Opportunity Gap uses true labels; it is neither a deployable Oracle Router nor a guaranteed achievable gain. Last-Stage short/long equality is a mathematical consistency check, not evidence for long-horizon learning.",
        "",
        "Historical Train ranking is pre-update sampled-candidate telemetry. Calibration and Development rankings here enumerate all 15 actions at frozen checkpoints. Quantile levels were fixed in advance; numerical cutoffs use Calibration only.",
        "",
        "Fixed-Checkpoint lambda interventions isolate routing effects at that saved parameter state. Differences between F/P/J final Cell weights do not identify historical training causes. Such claims remain NOT_IDENTIFIABLE_FROM_SAVED_CHECKPOINTS.",
        "",
        "Only three figures are produced. No training, hyperparameter selection, new model, or R3e run is performed. Private raw scores/indices and Checkpoint caches stay outside public GitHub.",
    ]
    if summary["failures"]:
        lines += ["", "## Missing or failed work", "", str(summary["failures"])]
    return "\n".join(lines) + "\n"


def plots(summary, folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    jobs = summary["jobs"]
    arms = ("F0", "L", "M", "P", "J")
    colors = dict(zip(arms, plt.cm.tab10.colors))
    # Figure 1: two roles, three complementary summaries; no per-token decoration.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for ri, role in enumerate(("calibration", "development")):
        for arm in arms:
            rows = [p[arm] for p in jobs.values() if arm in p and p[arm]["policy_ranking"]]
            if not rows:
                continue
            span = np.asarray(
                [
                    [
                        r["policy_ranking"][role]["stages"][t]["metrics"]["range"]["mean"]
                        for t in range(4)
                    ]
                    for r in rows
                ],
                dtype=float,
            )
            rank = np.asarray(
                [
                    [
                        r["policy_ranking"][role]["stages"][t]["metrics"]["q_accuracy"]["mean"]
                        for t in range(4)
                    ]
                    for r in rows
                ],
                dtype=float,
            )
            base = np.asarray(
                [
                    [
                        r["policy_ranking"][role]["stages"][t]["metrics"]["base_accuracy"]["mean"]
                        for t in range(4)
                    ]
                    for r in rows
                ],
                dtype=float,
            )
            gap = np.asarray(
                [
                    [
                        r["policy_ranking"][role]["overall"]["gap_bin_" + str(k)]["mean"]
                        for k in range(4)
                    ]
                    for r in rows
                ],
                dtype=float,
            )
            axes[ri, 0].plot(range(4), span.mean(0), marker="o", label=arm, color=colors[arm])
            axes[ri, 1].plot(
                range(4),
                np.nanmean(rank, axis=0),
                marker="o",
                label=arm + " corrected",
                color=colors[arm],
            )
            axes[ri, 1].plot(
                range(4),
                np.nanmean(base, axis=0),
                linestyle="--",
                color=colors[arm],
                label=arm + " original",
            )
            axes[ri, 2].plot(
                range(4), np.nanmean(gap, axis=0), marker="o", label=arm, color=colors[arm]
            )
        axes[ri, 0].set_title(role + ": sampled 15-action CE span")
        axes[ri, 0].set_xlabel("Stage")
        axes[ri, 1].set_title(role + ": ranking at frozen states")
        axes[ri, 1].set_xlabel("Stage")
        axes[ri, 1].axhline(0.5, color="gray", ls=":")
        axes[ri, 2].set_title(role + ": ranking by Calibration gap bins")
        axes[ri, 2].set_xlabel("Frozen quantile bin")
        axes[ri, 2].axhline(0.5, color="gray", ls=":")
    for ax in axes.flat:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Means across training Seeds; Story-cluster intervals and individual Seeds are in JSON"
    )
    fig.tight_layout()
    fig.savefig(folder / "01-values-and-ranking.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for arm in ("L", "M", "P", "J"):
        per = [p[arm] for p in jobs.values() if arm in p]
        if not per:
            continue
        x = sorted(float(k) for k in per[0]["interpolation"]["scales"])
        for row in per:
            y = [row["interpolation"]["scales"][str(k)]["mean_nll"] for k in x]
            axes[0].plot(x, y, alpha=0.25, color=colors[arm])
        y = np.mean(
            [[r["interpolation"]["scales"][str(k)]["mean_nll"] for k in x] for r in per], axis=0
        )
        change = np.mean(
            [
                [
                    np.mean(
                        [
                            v["trajectory_change"]["mean"]
                            for v in r["interpolation"]["scales"][str(k)]["stages"]
                        ]
                    )
                    for k in x
                ]
                for r in per
            ],
            axis=0,
        )
        axes[0].plot(x, y, marker="o", label=arm, color=colors[arm])
        axes[1].plot(x, change, marker="o", label=arm, color=colors[arm])
    axes[0].set_ylabel("Closed-loop Development NLL")
    axes[1].set_ylabel("Trajectory action change vs lambda=0")
    for ax in axes:
        ax.set_xlabel("Analytical correction scale (no coefficient selected)")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "02-scale-and-closed-loop.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for j, arm in enumerate(("L", "M", "P", "J")):
        v = [
            p[arm]["interpolation"]["scales"]["1.0"]["mean_nll"]
            - p[arm]["interpolation"]["scales"]["0.0"]["mean_nll"]
            for p in jobs.values()
            if arm in p
        ]
        if v:
            axes[0].bar(j, np.mean(v), alpha=0.5, color=colors[arm])
            axes[0].scatter(np.full(len(v), j), v, color=colors[arm])
    axes[0].set_xticks(range(4), ("L", "M", "P", "J"))
    axes[0].axhline(0, color="gray")
    axes[0].set_ylabel("Learned minus original route NLL, fixed own Cells")
    for seed, per in jobs.items():
        if all(a in per for a in ("F", "P", "J")):
            axes[1].plot(
                range(3),
                [per[a]["interpolation"]["scales"]["0.0"]["mean_nll"] for a in ("F", "P", "J")],
                marker="o",
                label=seed,
            )
    axes[1].set_xticks(range(3), ("F Cells", "P Cells", "J Cells"))
    axes[1].set_ylabel("Original-route NLL; different training trajectories")
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(folder / "03-fixed-checkpoint-route-effects.png", dpi=160)
    plt.close(fig)
