"""Saved scientific figures from completed aggregate evidence, never new data reads."""

from pathlib import Path
from .report import verified, seal
from .common import file_sha256


def render(c, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output = Path(output)
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    completed = {
        arm: verified(output / "arms" / arm / "metrics.json")
        for arm in c["arms"]
        if (output / "arms" / arm / "metrics.json").exists()
    }
    for arm, result in completed.items():
        curves = result.get("specialization", {}).get("advantage_route_trajectories", [])
        for pair in c["alignment"]["cell_pairs"]:
            figure, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, constrained_layout=True)
            for group in (0, 1):
                for mode, style in [("frozen_state_probes", "-"), ("live_recomputed_probes", "--")]:
                    rows = sorted(
                        (
                            r
                            for r in curves
                            if r["cells"] == pair and r["group"] == group and r["mode"] == mode
                        ),
                        key=lambda r: r["step"],
                    )
                    if not rows:
                        continue
                    label = (
                        "group "
                        + str(group)
                        + " / "
                        + ("fixed h" if mode.startswith("frozen") else "live h")
                    )
                    for axis, key in zip(axes, ("A_ij", "R_ij")):
                        axis.plot(
                            [r["step"] for r in rows], [r[key] for r in rows], style, label=label
                        )
            axes[0].set_ylabel("paired advantage A(i,j)")
            axes[1].set_ylabel("log preference R(i,j)")
            axes[1].set_xlabel("committed training step")
            axes[0].set_title(arm + " / Cells " + str(pair) + " / exploratory")
            for axis in axes:
                axis.axhline(0, color="grey", linewidth=0.6)
                axis.grid(alpha=0.2)
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                axes[0].legend(fontsize=7)
            path = directory / (arm + "-cells-" + str(pair[0]) + "-" + str(pair[1]) + ".png")
            figure.savefig(path, dpi=160)
            plt.close(figure)
            files.append(path)
    if completed:
        figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for arm, result in completed.items():
            for policy, row in result["quality"].items():
                axis.scatter(sum(row["cost"]) / len(row["cost"]), row["mean_nll"])
                axis.annotate(
                    arm + "/" + policy,
                    (sum(row["cost"]) / len(row["cost"]), row["mean_nll"]),
                    fontsize=7,
                )
        axis.set_xlabel("observed normalized non-STOP calls")
        axis.set_ylabel("Development NLL")
        axis.set_title("Deployment policies / single-seed exploratory evidence")
        axis.grid(alpha=0.2)
        path = directory / "quality-cost.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        files.append(path)
    result = {
        "figures": [
            {"path": str(path.relative_to(output)), "sha256": file_sha256(path)} for path in files
        ],
        "scope": "saved figures are descriptive; no causal or confirmation claims",
        "scientific_decision": False,
    }
    seal(output / "figures-summary.json", result)
    return result
