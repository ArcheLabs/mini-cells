"""Scientific figures from already recorded aggregates; never new model evaluations."""

from pathlib import Path
from .analysis import aggregate
from minicells_runtime.store import atomic, digest


def render(root):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    root = Path(root)
    values = aggregate(root)
    target = root / "figures"
    target.mkdir(exist_ok=True)
    outputs = []
    for key, title in (
        ("mean_nll", "Development greedy NLL"),
        ("mean_path_length", "Greedy non-STOP path calls"),
        ("forced_end_rate", "Greedy forced-end rate"),
        ("context_call_fraction", "Greedy CTX call fraction"),
    ):
        fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for arm in ("Q-R3b", "QC-R3b", "QS-R3b"):
            rows = sorted(
                (
                    r
                    for r in values["metrics-summary.json"]["rows"]
                    if r["arm"] == arm and r["phase"] == "post_event"
                ),
                key=lambda r: r["step"],
            )
            if rows:
                axis.plot(
                    [r["step"] for r in rows],
                    [r["quality"]["greedy"].get(key) for r in rows],
                    marker="o",
                    label=arm,
                )
        axis.set(
            xlabel="committed training step (post_event)",
            ylabel=title,
            title="Single-seed exploratory; no causal claim",
        )
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend()
        path = target / (key + ".png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(path)
    trajectories = values["specialization-summary.json"]["advantage_route_trajectories"]
    for arm in ("Q-R3b", "QC-R3b", "QS-R3b"):
        pairs = sorted({tuple(r["cells"]) for r in trajectories if r["arm"] == arm})
        for pair in pairs:
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 6), constrained_layout=True)
            for role in ("calibration", "development"):
                for group in (0, 1):
                    rows = sorted(
                        (
                            r
                            for r in trajectories
                            if r["arm"] == arm
                            and tuple(r["cells"]) == pair
                            and r["role"] == role
                            and r["group"] == group
                            and r["phase"] == "post_event"
                        ),
                        key=lambda r: r["step"],
                    )
                    if not rows:
                        continue
                    for axis, field in zip(axes, ("advantage", "route_log_preference")):
                        axis.plot(
                            [r["step"] for r in rows],
                            [r[field] for r in rows],
                            label=role + " group " + str(group),
                        )
            axes[0].set(
                ylabel="D(i,j): L(j)-L(i)", title=arm + " / " + str(pair) + " / descriptive only"
            )
            axes[1].set(xlabel="post-event step", ylabel="log preference i/j")
            for axis in axes:
                axis.axhline(0, color="grey", linewidth=0.5)
                axis.grid(alpha=0.2)
                if axis.lines:
                    axis.legend(fontsize=7)
            path = target / (arm + "-pair-" + "-".join(map(str, pair)) + ".png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            outputs.append(path)
    atomic(
        target / "index.json",
        {
            "files": {p.name: digest(p) for p in outputs},
            "source": "RECORDED_AGGREGATES_ONLY",
            "scientific_decision": False,
        },
    )
    return outputs


def render_analysis(values, destination, plan):
    """Plot already aggregated observations; NaN creates gaps at unobserved steps."""
    import math
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for phase in ("pre_event", "post_event"):
        fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for arm in ("Q-R3b", "QC-R3b", "QS-R3b"):
            rows = {
                r["step"]: r["quality"]["greedy"]["mean_nll"]
                for r in values["metrics-summary.json"]["rows"]
                if r["arm"] == arm and r["phase"] == phase and "greedy" in r["quality"]
            }
            steps = plan["training"]["journal_telemetry_steps"]
            axis.plot(steps, [rows.get(s, math.nan) for s in steps], marker="o", label=arm)
        axis.set(
            xlabel="Historical checkpoint step",
            ylabel="Development greedy NLL",
            title=phase + " — gaps are NOT_EVALUATED",
        )
        axis.legend()
        fig.savefig(destination / ("quality-" + phase + ".png"), dpi=140)
        plt.close(fig)
