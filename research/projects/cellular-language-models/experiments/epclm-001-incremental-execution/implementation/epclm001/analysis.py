import csv
import json
from pathlib import Path


def analyze(root):
    root = Path(root)
    raw = json.loads((root / "EVENT_RESULTS.json").read_text())["records"]
    with (root / "EVENT_SUMMARY.csv").open("w", newline="") as f:
        fields = [
            "layout",
            "ratio",
            "implementation",
            "repeat",
            "elapsed_ms",
            "logical_updates",
            "physical_updates",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in raw:
            writer.writerow(
                dict(
                    layout=r["workload"]["layout"],
                    ratio=r["workload"]["ratio"],
                    **{k: r[k] for k in fields[2:]},
                )
            )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for metric in ("elapsed_ms", "physical_updates"):
            fig, ax = plt.subplots()
            for impl in sorted({r["implementation"] for r in raw}):
                points = [r for r in raw if r["implementation"] == impl]
                ax.scatter(
                    [r["workload"]["ratio"] for r in points],
                    [r[metric] for r in points],
                    label=impl,
                    s=8,
                )
            ax.set(
                xlabel="Requested activity ratio (mixed layouts; engineering only)", ylabel=metric
            )
            ax.legend(fontsize=6)
            fig.savefig(root / (metric + ".png"))
            plt.close(fig)
        status = "GENERATED_ENGINEERING_ONLY"
    except ImportError:
        status = "MATPLOTLIB_UNAVAILABLE"
    return {
        "plots": status,
        "scientific_figures": "NOT_AVAILABLE_WITHOUT_FROZEN_HARDWARE_MEASUREMENTS",
    }
