"""Plots contain aggregate quantities only; raw states and tokens remain private."""

from pathlib import Path


def render(reports, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for report, prefix in (("cf2-forced-stop.json", "cf2"), ("cf3-repeat-constraint.json", "cf3")):
        rows = reports.get(report, {}).get("comparisons", {})
        if not rows:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, metric in zip(axes, ("delta_nll", "delta_cost", "delta_utility")):
            names = list(rows)
            means = [rows[n][metric]["mean"] for n in names]
            ax.bar(range(len(names)), means)
            ax.axhline(0, color="black", lw=0.7)
            ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
            ax.set_title(metric)
            for i, n in enumerate(names):
                ci = rows[n][metric].get("ci95")
                if ci:
                    ax.plot([i, i], ci, color="black")
        fig.suptitle("Paired Story bootstrap; single-seed exploratory")
        fig.tight_layout()
        fig.savefig(directory / (prefix + "-paired-effects.png"), dpi=140)
        plt.close(fig)
    cf = reports.get("cf1-stop-value-calibration.json", {})
    depths = cf.get("by_depth", {})
    if depths:
        fig, ax = plt.subplots(figsize=(7, 4))
        for kind in ("greedy_consistent", "frozen_sampled_oracle"):
            ax.plot(
                [int(d) for d in depths],
                [depths[d][kind]["misranking_lower_bound"]["estimate"] for d in depths],
                marker="o",
                label=kind,
            )
        ax.set(
            xlabel="Decision depth (zero based)",
            ylabel="Design-weighted STOP misranking lower bound",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / "cf1-stop-misranking.png", dpi=140)
        plt.close(fig)
