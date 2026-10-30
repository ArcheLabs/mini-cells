"""Paired descriptive uncertainty and nine figures from retained formal samples."""

import csv
import json
from pathlib import Path

import numpy as np

from .common import write


def interval(left, right, config, family, difference=False):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if (
        len(left) != len(right)
        or len(left) < config["repeats"]
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or (right <= 0).any()
    ):
        raise ValueError("PAIRED_SAMPLE_SET_INVALID_OR_INCOMPLETE")
    values = left - right if difference else left / right
    rng = np.random.default_rng(config["bootstrap_seed"])
    indices = rng.integers(0, len(values), size=(config["bootstrap_samples"], len(values)))
    means = values[indices].mean(axis=1)
    alpha = config["interval_alpha"] / family
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)),
        "lower": float(np.quantile(means, alpha / 2)),
        "upper": float(np.quantile(means, 1 - alpha / 2)),
        "confidence": 1 - alpha,
        "paired_repetitions": len(values),
        "scope": "Timing repetitions, not model seeds or quality uncertainty",
    }


def records(root, name):
    path = root / name
    return json.loads(path.read_text())["records"] if path.exists() else []


def vector(rows, **keys):
    selected = sorted(
        (r for r in rows if all(r.get(k) == v for k, v in keys.items())), key=lambda r: r["repeat"]
    )
    if len({r["repeat"] for r in selected}) != len(selected):
        raise ValueError("DUPLICATE_PAIRED_REPEAT")
    return selected


def analyze(directory, config, manifest):
    root = Path(directory)
    stages = manifest["stages"]
    a = records(root, "E1A_TIMING.json")
    b = records(root, "E1B_TIMING.json")
    c = records(root, "E1C_TIMING.json")
    decisions = {
        key: {
            "status": "INCOMPLETE",
            "reason": "Required correctness, freeze or complete formal measurements unavailable",
        }
        for key in ("H1_INCREMENTAL", "H2_EVENT_RUNTIME", "H3_COMPETITIVE")
    }
    summary = []

    def ready(*names):
        return all(stages.get(name, {}).get("status") == "COMPLETE" for name in names)

    if (
        ready("e1a-correctness", "e1a-benchmark")
        and len(a) == 2 * len(config["workloads"]) * 3 * config["repeats"]
    ):
        comparisons = []
        for model in ("X3", "X4"):
            for work in config["workloads"]:
                full = vector(a, model=model, workload=work["id"], mode="full")
                for mode in ("A0", "A1"):
                    candidate = vector(a, model=model, workload=work["id"], mode=mode)
                    metrics = {
                        field: interval(
                            [r[field] for r in full],
                            [r[field] for r in candidate],
                            config,
                            config["model_family_comparisons"],
                        )
                        for field in ("prefill_host_ms", "decode_host_ms", "total_host_ms")
                    }
                    comparisons.append(
                        {
                            "model": model,
                            "workload": work["id"],
                            "mode": mode,
                            "paired_speedups": metrics,
                        }
                    )
        decisions["H1_INCREMENTAL"] = {
            "status": "SUPPORTED"
            if any(r["paired_speedups"]["decode_host_ms"]["lower"] > 1 for r in comparisons)
            else "NOT_SUPPORTED_IN_TESTED_REGIME",
            "H1a": "SUPPORTED",
            "H1b": "SUPPORTED"
            if any(r["paired_speedups"]["decode_host_ms"]["lower"] > 1 for r in comparisons)
            else "NOT_SUPPORTED_IN_TESTED_REGIME",
            "comparisons": comparisons,
            "scope": "FP32 pinned models, tested GPU/lengths; no benefit means not established, not proof of zero effect",
        }
    if (
        ready("e1b-correctness", "e1b-benchmark")
        and len(b) == len(config["event_points"]) * 3 * 4 * config["repeats"]
    ):
        comparisons = []
        for point in config["event_points"]:
            for kernel in config["event_kernels"]:
                for sparse in ("S0_sparse_cell", "S1_sparse_region"):
                    control = vector(
                        b, point=point["id"], kernel=kernel, implementation="D0_dense_reference"
                    )
                    other = vector(b, point=point["id"], kernel=kernel, implementation=sparse)
                    ci = interval(
                        [r["host_ms"] for r in control],
                        [r["host_ms"] for r in other],
                        config,
                        config["event_family_comparisons"],
                    )
                    masked = vector(
                        b, point=point["id"], kernel=kernel, implementation="D1_masked_dense"
                    )
                    comparisons.append(
                        {
                            "point": point["id"],
                            "kernel": kernel,
                            "implementation": sparse,
                            "paired_speedup": ci,
                            "vs_masked": interval(
                                [r["host_ms"] for r in masked],
                                [r["host_ms"] for r in other],
                                config,
                                config["event_family_comparisons"],
                            ),
                        }
                    )
        decisions["H2_EVENT_RUNTIME"] = {
            "status": "SUPPORTED"
            if any(r["paired_speedup"]["lower"] > 1 for r in comparisons)
            else "NOT_SUPPORTED_IN_TESTED_REGIME",
            "H2a": "SUPPORTED",
            "comparisons": comparisons,
            "scope": "External synthetic events; B0 and B1 are not full CLM inference",
        }
    if (
        ready("e1a-correctness", "e1c-benchmark")
        and len(c) == len(config["workloads"]) * 8 * config["repeats"]
        and (root / "E1C_QUALITY.json").exists()
    ):
        comparisons = []
        for point in config["workloads"]:
            anchor = vector(c, model="T1", mode="A0", workload=point["id"])
            for model in ("X3", "X4"):
                full = vector(c, model=model, mode="full", workload=point["id"])
                sparse = vector(c, model=model, mode="A1", workload=point["id"])
                speed = interval(
                    [r["decode_host_ms"] for r in anchor],
                    [r["decode_host_ms"] for r in sparse],
                    config,
                    config["model_family_comparisons"],
                )
                old_gap = interval(
                    [r["decode_host_ms"] for r in full],
                    [r["decode_host_ms"] for r in anchor],
                    config,
                    config["model_family_comparisons"],
                    difference=True,
                )
                reduction = interval(
                    [r["decode_host_ms"] for r in full],
                    [r["decode_host_ms"] for r in sparse],
                    config,
                    config["model_family_comparisons"],
                    difference=True,
                )
                comparisons.append(
                    {
                        "model": model,
                        "workload": point["id"],
                        "paired_speedup_vs_cached_T1": speed,
                        "CLM_faster": speed["lower"] > 1,
                        "gap_reduced": old_gap["lower"] > 0 and reduction["lower"] > 0,
                        "quality": "Lower recorded development PPL, not equal-quality comparison",
                        "lower_memory": float(
                            np.median([r["cache"]["kv_storage_bytes"] for r in sparse])
                        )
                        < float(np.median([r["cache"]["kv_storage_bytes"] for r in anchor])),
                    }
                )
        decisions["H3_COMPETITIVE"] = {
            "status": "SUPPORTED"
            if any(r["CLM_faster"] or r["gap_reduced"] for r in comparisons)
            else "NOT_SUPPORTED_IN_TESTED_REGIME",
            "comparisons": comparisons,
            "scope": "Cached T1 research implementation, not vLLM/SGLang or universal architecture efficiency",
        }
    for phase, rows in (("E1A", a), ("E1B", b), ("E1C", c)):
        for row in rows:
            if phase == "E1B":
                summary.append(
                    {
                        "phase": phase,
                        "model": row["kernel"],
                        "workload": row["point"],
                        "mode": row["implementation"],
                        "repeat": row["repeat"],
                        "host_ms": row["host_ms"],
                    }
                )
            else:
                summary.append(
                    {
                        "phase": phase,
                        "model": row["model"],
                        "workload": row["workload"],
                        "mode": row["mode"],
                        "repeat": row["repeat"],
                        "host_ms": row["decode_host_ms"],
                    }
                )
    aggregates = []
    for phase, rows in [("E1A", a), ("E1B", b), ("E1C", c)]:
        for model, work, mode in sorted(
            {
                (
                    r.get("model", r.get("kernel")),
                    r.get("workload", r.get("point")),
                    r.get("mode", r.get("implementation")),
                )
                for r in rows
            }
        ):
            group = [
                r
                for r in rows
                if r.get("model", r.get("kernel")) == model
                and r.get("workload", r.get("point")) == work
                and r.get("mode", r.get("implementation")) == mode
            ]
            fields = (
                ("host_ms", "device_stream_ms")
                if phase == "E1B"
                else (
                    "prefill_host_ms",
                    "prefill_device_stream_ms",
                    "decode_host_ms",
                    "total_host_ms",
                    "tpot_p50_ms",
                    "tpot_p95_ms",
                    "tpot_p99_ms",
                )
            )
            metrics = {}
            for field in fields:
                values = np.asarray(
                    [r[field] for r in group if r.get(field) is not None], dtype=float
                )
                if len(values):
                    metrics[field] = dict(
                        n=len(values),
                        mean=float(values.mean()),
                        median=float(np.median(values)),
                        std=float(values.std(ddof=1)) if len(values) > 1 else None,
                        **{f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)},
                    )
            aggregates.append(
                {"phase": phase, "model": model, "workload": work, "mode": mode, "metrics": metrics}
            )
    write(
        root / "SAMPLE_STATISTICS.json",
        {
            "records": aggregates,
            "outlier_policy": "Retain all raw samples; no removal",
            "primary_metrics": "H1/H3 decode_host_ms; H2 host_ms versus D0; other metrics descriptive",
        },
    )
    write(
        root / "INFERENCE_COVERAGE.json",
        {
            "primary_model_family": 50,
            "primary_event_family": 96,
            "primary_model_metrics": "20 H1 decode speedups + 10 H3 cached-T1 speedups + 10 old-gap + 10 gap-reduction comparisons",
            "primary_event_metrics": "16 points x 3 kernels x 2 sparse vs D0 comparisons",
            "secondary_metrics": "Prefill/total and vs-D1 intervals descriptive only; not additional support tests",
            "bootstrap_scope": "Paired timing-repetition percentile bootstrap; multiplicity adjusted, approximate and hardware-specific; not seed-level scientific confirmation",
            "TTFT": "NOT_MEASURED",
            "end_to_end": "NOT_MEASURED",
            "training_updates": 0,
        },
    )
    write(root / "DECISIONS.json", decisions)
    with (root / "RAW_SUMMARY.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["phase", "model", "workload", "mode", "repeat", "host_ms"]
        )
        writer.writeheader()
        writer.writerows(summary)
    table = []
    quality = (
        json.loads((root / "E1C_QUALITY.json").read_text())
        if (root / "E1C_QUALITY.json").exists()
        else {}
    )
    for model, work, mode in sorted({(r["model"], r["workload"], r["mode"]) for r in c}):
        group = vector(c, model=model, workload=work, mode=mode)
        table.append(
            {
                "model": model,
                "workload": work,
                "mode": mode,
                "development_ppl": quality.get(model, {}).get("quality", {}).get("validation_ppl"),
                "quality_status": "HASH_BOUND_REUSE_NOT_REEVALUATED"
                if model in quality
                else "UNAVAILABLE",
                "prefill_host_ms": float(np.median([r["prefill_host_ms"] for r in group])),
                "decode_host_ms": float(np.median([r["decode_host_ms"] for r in group])),
                "tpot_p50_ms": float(np.median([r["tpot_p50_ms"] for r in group])),
                "kv_storage_bytes": float(
                    np.median([r["cache"]["kv_storage_bytes"] for r in group])
                ),
                "samples": len(group),
                "status": "COMPLETE_POINT"
                if len(group) == config["repeats"]
                else "INCOMPLETE_POINT",
            }
        )
    with (root / "COMPETITIVE_TABLE.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]) if table else ["status"])
        writer.writeheader()
        writer.writerows(table)
    (root / "SUMMARY.md").write_text(
        "# EPCLM-001 formal evidence\n\n"
        + "\n".join(f"- {k}: {v['status']}" for k, v in decisions.items())
        + "\n\nFP32 only. Model-only timing excludes tokenizer/sampling/return. No equal-quality claim. No new training. See raw samples and independent persistence index.\n"
    )
    (root / "SUMMARY.zh-CN.md").write_text(
        "# EPCLM-001 研究总结\n\n"
        + "\n".join(f"- {k}：{v['status']}" for k, v in decisions.items())
        + "\n\n结果仅适用于固定权重、FP32、当前设备及预先冻结的工作负载。计时不包含完整用户生成路径；较低 PPL 不等于严格等质量比较。B0/B1 外部事件执行不代表真实 CLM 的自适应稀疏推理。未执行或门禁不完整的实验不标记为完成。\n"
    )
    (root / "WHAT_NOT_PROVED.md").write_text(
        "No persistent Cell memory, learned event policy, removal of token attention, 30M scaling, production Transformer superiority, or equal-quality efficiency is established. FP16/BF16 and full user TTFT/E2E were not measured. Partial analytic FLOPs are not measured DRAM traffic or energy.\n"
    )
    (root / "REVIEW.md").write_text(
        "# Review status\n\nIndependent result review required. Decisions are machine-derived and limited to registered workloads. Check failed cases, raw samples, source/device freeze and persistence index before accepting scientific conclusions.\n"
    )
    figures = []
    if a or b or c:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        folder = root / "figures"
        folder.mkdir(exist_ok=True)

        def save(name, draw, xlabel, ylabel):
            fig, ax = plt.subplots(figsize=(16, 9))
            draw(ax)
            ax.set(xlabel=xlabel, ylabel=ylabel)
            ax.legend(fontsize=4, loc="upper left", bbox_to_anchor=(1.01, 1), ncol=2)
            fig.tight_layout()
            fig.savefig(folder / name)
            plt.close(fig)
            figures.append(name)

        def grouped(ax, rows, field):
            for model in sorted({r["model"] for r in rows}):
                for mode in sorted({r["mode"] for r in rows if r["model"] == model}):
                    values = [r for r in rows if r["model"] == model and r["mode"] == mode]
                    xs = sorted(
                        {r["workload"] for r in values},
                        key=lambda w: next(
                            i for i, p in enumerate(config["workloads"]) if p["id"] == w
                        ),
                    )
                    ax.plot(
                        xs,
                        [np.median([r[field] for r in values if r["workload"] == x]) for x in xs],
                        marker="o",
                        label=model + "/" + mode,
                    )

        if (root / "E1A_PARITY.json").exists():

            def parity_plot(ax):
                for key, value in json.loads((root / "E1A_PARITY.json").read_text()).items():
                    if key not in ("X3", "X4"):
                        continue
                    for case in value["cases"]:
                        if case["batch"] == 1 and case["input_role"] == "holdout":
                            ax.plot(
                                [r["prefix_length"] for r in case["report"]["rows"]],
                                [r["max_abs"] for r in case["report"]["rows"]],
                                label=key + "/" + case["adapter"],
                            )

            save("01-prefix-error.png", parity_plot, "Prefix length", "Max absolute logit error")
        if a:
            save(
                "02-full-a0-a1-decode.png",
                lambda ax: grouped(ax, a, "decode_host_ms"),
                "Frozen workload",
                "Decode host ms",
            )
        if b:
            for name, metric, label in [
                ("03-event-speedup.png", "speed", "D0 / candidate host time"),
                ("04-logical-physical.png", "updates", "Physical update slots"),
                ("05-event-work-time.png", "host", "Host ms"),
            ]:

                def event_plot(ax, metric=metric):
                    for cells in (256, 1024):
                        for kind in config["event_kernels"]:
                            for layout in (
                                "uniform-random",
                                "clustered",
                                "all-active",
                                "zero-active",
                                "wavefront",
                                "burst",
                            ):
                                for impl in sorted({r["implementation"] for r in b}):
                                    points = [
                                        p
                                        for p in config["event_points"]
                                        if p["cells"] == cells and p["layout"] == layout
                                    ]
                                    x = []
                                    y = []
                                    for point in points:
                                        rows = vector(
                                            b, point=point["id"], kernel=kind, implementation=impl
                                        )
                                        if not rows:
                                            continue
                                        x.append(
                                            np.median([r["logical_updates"] for r in rows])
                                            if metric == "updates"
                                            else np.median(
                                                [r["partial_matmul_flops"] for r in rows]
                                            )
                                            if metric == "host"
                                            else point["ratio"]
                                        )
                                        if metric == "speed":
                                            ref = vector(
                                                b,
                                                point=point["id"],
                                                kernel=kind,
                                                implementation="D0_dense_reference",
                                            )
                                            y.append(
                                                np.median([r["host_ms"] for r in ref])
                                                / np.median([r["host_ms"] for r in rows])
                                            )
                                        else:
                                            y.append(
                                                np.median(
                                                    [
                                                        r[
                                                            "physical_updates"
                                                            if metric == "updates"
                                                            else "host_ms"
                                                        ]
                                                        for r in rows
                                                    ]
                                                )
                                            )
                                    if x:
                                        ax.scatter(
                                            x, y, s=9, label=f"N{cells}/{kind}/{layout}/{impl}"
                                        )

                save(
                    name,
                    event_plot,
                    "Logical updates"
                    if metric == "updates"
                    else "Partial matmul FLOPs"
                    if metric == "host"
                    else "Requested activity ratio (raw point IDs preserve N/layout)",
                    label,
                )
        if c:
            save(
                "06-prefill.png",
                lambda ax: grouped(ax, c, "prefill_host_ms"),
                "Frozen workload",
                "Prefill host ms",
            )
            save(
                "07-decode-tpot.png",
                lambda ax: grouped(ax, c, "tpot_p50_ms"),
                "Frozen workload",
                "Median TPOT host ms",
            )
            quality = (
                json.loads((root / "E1C_QUALITY.json").read_text())
                if (root / "E1C_QUALITY.json").exists()
                else {}
            )

            def quality_plot(ax):
                for key, mode in (("T1", "A0"), ("X3", "A1"), ("X4", "A1")):
                    rows = vector(c, model=key, mode=mode, workload="p32-d32")
                    if rows and key in quality:
                        ax.scatter(
                            [np.median([r["tpot_p50_ms"] for r in rows])],
                            [quality[key]["quality"]["validation_ppl"]],
                            label=key,
                        )

            if quality:
                save(
                    "08-quality-latency.png",
                    quality_plot,
                    "TPOT host ms; p32-d32",
                    "Bound reused development PPL (FP16 evaluation)",
                )

            def cache_plot(ax):
                for key, mode in (
                    ("T1", "A0"),
                    ("X3", "A0"),
                    ("X3", "A1"),
                    ("X4", "A0"),
                    ("X4", "A1"),
                ):
                    values = []
                    for point in config["workloads"]:
                        rows = vector(c, model=key, mode=mode, workload=point["id"])
                        if rows:
                            values.append(
                                (
                                    point["prefill"] + point["decode"],
                                    np.median([r["cache"]["kv_storage_bytes"] for r in rows]),
                                )
                            )
                    if values:
                        ax.plot(
                            [x for x, y in values],
                            [y for x, y in values],
                            marker="o",
                            label=key + "/" + mode,
                        )

            save(
                "09-cache-context.png", cache_plot, "Consumed positions", "KV backing-storage bytes"
            )
    write(
        root / "FIGURE_INDEX.json",
        {
            "files": figures,
            "status": "GENERATED_FROM_RAW" if figures else "NOT_GENERATED_NO_FORMAL_GPU_DATA",
        },
    )
    return decisions
