from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "src").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))

from minicells.language_data import prepare_tinystories_corpus  # noqa: E402


OUT = ROOT / "results" / "growing-cellular-lm-v1"
WORKER = ROOT / "scripts" / "run_language_growing_organism_variant.py"
VARIANTS = ("T", "F", "G")
N_REPLICATES = 3
BUDGET_TOKENS = 1_024_000


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(max(1e-12, float(value))) for value in values) / len(values))


def prepare_corpus() -> tuple[Path, dict[str, object]]:
    corpus = prepare_tinystories_corpus(ROOT)
    cache = corpus.tokenizer_path.parent
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(corpus.tokenizer_path, OUT / "tokenizer.json")
    manifest_path = cache / "corpus-manifest.json"
    shutil.copy2(manifest_path, OUT / "corpus-manifest.json")
    return cache, corpus.manifest


def worker_command(replicate: int, variant: str, cache_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(WORKER),
        "--replicate", str(replicate),
        "--variant", variant,
        "--cache-dir", str(cache_dir),
        "--output-dir", str(OUT),
    ]


def run_models(cache_dir: Path) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 016 requires CUDA")
    gpu_count = min(2, available)
    groups = [tuple(range(start, min(start + gpu_count, N_REPLICATES))) for start in range(0, N_REPLICATES, gpu_count)]
    for group in groups:
        gpu_for = {replicate: index for index, replicate in enumerate(group)}
        for variant in VARIANTS:
            active = []
            for replicate in group:
                gpu_index = gpu_for[replicate]
                run_name = f"r{replicate}-{variant}"
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
                log_path = OUT / f"{run_name}.log"
                handle = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(worker_command(replicate, variant, cache_dir), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
                active.append((run_name, gpu_index, process, log_path, handle))
                print(f"started {run_name} on physical GPU {gpu_index}")
            failures = []
            for run_name, gpu_index, process, log_path, handle in active:
                exit_code = process.wait()
                handle.close()
                print(f"--- {run_name} / GPU {gpu_index} ---")
                print(log_path.read_text(encoding="utf-8").rstrip())
                if exit_code != 0:
                    failures.append(f"{run_name} exited {exit_code}; see {log_path}")
            if failures:
                raise RuntimeError("; ".join(failures))
    return gpu_count


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def collect_results() -> dict[str, pd.DataFrame]:
    workers = []
    keys = (
        "checkpoints", "structural-events", "cells", "edges", "structural-probes",
        "interventions", "local-learning", "transplantation", "skill-localization",
    )
    frames: dict[str, list[pd.DataFrame]] = {key: [] for key in keys}
    for replicate in range(N_REPLICATES):
        for variant in VARIANTS:
            run = f"r{replicate}-{variant}"
            worker = json.loads((OUT / f"{run}-worker.json").read_text(encoding="utf-8"))
            local = worker.get("local_learning") or {}
            workers.append({
                "run": run,
                "replicate": replicate,
                "variant": variant,
                "parameters": worker["parameters"],
                "final_nll": worker["final_nll"],
                "final_ppl": worker["final_ppl"],
                "final_token_accuracy": worker["final_token_accuracy"],
                "core_seconds_per_million_tokens": worker["seconds_per_million_tokens"],
                "wall_seconds_per_million_tokens": worker.get("wall_seconds_per_million_tokens", worker["seconds_per_million_tokens"]),
                "peak_vram_gib": worker["peak_vram_bytes"] / (1024**3),
                "phase1_alive_cells": worker.get("phase1_alive_cells"),
                "phase1_edges": worker.get("phase1_edges"),
                "language_structural_events": worker.get("language_structural_events", 0),
                "local_skill_improvement": local.get("skill_improvement"),
                "transplant_recovery": local.get("transplant_recovery"),
                "donor_language_ratio": local.get("donor_language_ratio"),
                "selected_cell_count": local.get("selected_cell_count"),
            })
            for key in keys:
                frame = read_csv(OUT / f"{run}-{key}.csv")
                if not frame.empty:
                    frames[key].append(frame)
    result = {"model_summary": pd.DataFrame(workers)}
    for key, parts in frames.items():
        result[key.replace("-", "_")] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    result["model_summary"].to_csv(OUT / "model-summary.csv", index=False)
    for key in keys:
        result[key.replace("-", "_")].to_csv(OUT / f"{key}.csv", index=False)
    return result


def make_paired_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for replicate in range(N_REPLICATES):
        group = summary.loc[summary["replicate"] == replicate].set_index("variant")
        for left, right in (("G", "T"), ("F", "T"), ("G", "F")):
            for metric in ("final_nll", "parameters", "core_seconds_per_million_tokens", "wall_seconds_per_million_tokens", "peak_vram_gib"):
                rows.append({
                    "replicate": replicate,
                    "comparison": f"{left}_over_{right}",
                    "metric": metric,
                    "ratio": float(group.loc[left, metric] / group.loc[right, metric]),
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "paired-comparisons.csv", index=False)
    return frame


def save_task_spec(manifest: dict[str, object]) -> None:
    payload = {
        "format": "minicells.growing-cellular-lm-task.v1",
        "experiment": "016",
        "phase1": {
            "dataset": manifest.get("dataset"),
            "objective": "autoregressive next-token prediction",
            "tokens_per_model": BUDGET_TOKENS,
            "variants": {
                "T": "parameter-matched standard causal Transformer",
                "F": "fixed four-cell organism with one shared time-homogeneous cellular rule",
                "G": "same cellular model with utility-driven connect/prune and pressure-conflict-driven fork",
            },
        },
        "phase2": {
            "skill": "REVERSE_INC synthetic transformation",
            "genome": "frozen",
            "trainable": "cell_memory plus non-parametric structural events",
            "purpose": "test local capability acquisition without global genome update",
        },
        "phase3": {
            "operation": "copy high-change/newborn phenotype cells and their internal subgraph into a phase1-identical recipient",
            "purpose": "first same-genome tissue transplantation probe",
        },
        "structural_principle": {
            "connect": "persistent positive first-order marginal edge utility -dL/dw",
            "prune": "persistent negative marginal edge utility for non-protected edges",
            "fork": "persistent phenotype-gradient pressure + microbatch gradient conflict, only after useful connection is unavailable",
        },
    }
    (OUT / "task-spec.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_training_curves(checkpoints: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in VARIANTS:
        selected = checkpoints.loc[checkpoints["variant"] == variant]
        for replicate in range(N_REPLICATES):
            run = selected.loc[selected["replicate"] == replicate]
            ax.plot(run["tokens"], run["validation_nll"], alpha=0.30)
        mean = selected.groupby("tokens")["validation_nll"].mean()
        ax.plot(mean.index, mean.values, linewidth=2.5, label=variant)
    ax.set_xlabel("consumed language tokens")
    ax.set_ylabel("validation NLL")
    ax.set_title("Experiment 016 language learning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "language-learning-curves.png", dpi=180)
    plt.close(fig)


def plot_quality_cost(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    means = summary.groupby("variant").agg({"wall_seconds_per_million_tokens": "mean", "final_nll": "mean"})
    for variant, row in means.iterrows():
        ax.scatter(row["wall_seconds_per_million_tokens"], row["final_nll"], s=90)
        ax.annotate(variant, (row["wall_seconds_per_million_tokens"], row["final_nll"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("wall seconds / million training tokens")
    ax.set_ylabel("final validation NLL")
    ax.set_title("Quality vs physical training cost")
    fig.tight_layout()
    fig.savefig(OUT / "quality-vs-gpu-cost.png", dpi=180)
    plt.close(fig)


def plot_structure_timeline(checkpoints: pd.DataFrame) -> None:
    g = checkpoints.loc[checkpoints["variant"] == "G"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for replicate in range(N_REPLICATES):
        run = g.loc[g["replicate"] == replicate]
        ax.plot(run["tokens"], run["alive_cells"], marker="o", label=f"cells r{replicate}")
        ax.plot(run["tokens"], run["edges"], linestyle="--", alpha=0.65, label=f"edges r{replicate}")
    ax.set_xlabel("consumed language tokens")
    ax.set_ylabel("count")
    ax.set_title("Growing organism size")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cells-edges-over-training.png", dpi=180)
    plt.close(fig)


def circular_positions(cells: list[int]) -> dict[int, tuple[float, float]]:
    if not cells:
        return {}
    return {cell: (math.cos(2 * math.pi * i / len(cells)), math.sin(2 * math.pi * i / len(cells))) for i, cell in enumerate(cells)}


def plot_final_organisms(cells: pd.DataFrame, edges: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, N_REPLICATES, figsize=(15, 5))
    for replicate, ax in enumerate(np.atleast_1d(axes)):
        c = cells.loc[(cells["variant"] == "G") & (cells["replicate"] == replicate)]
        if c.empty:
            ax.set_axis_off()
            continue
        final_step = c["step"].max()
        c = c.loc[c["step"] == final_step]
        e = edges.loc[(edges["variant"] == "G") & (edges["replicate"] == replicate) & (edges["step"] == final_step)]
        ids = sorted(int(v) for v in c["cell"].unique())
        pos = circular_positions(ids)
        for _, row in e.iterrows():
            receiver, source = int(row["receiver"]), int(row["source"])
            if receiver not in pos or source not in pos:
                continue
            x0, y0 = pos[source]
            x1, y1 = pos[receiver]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "alpha": 0.25 + 0.45 * int(row["protected"]), "linewidth": 1.0})
        by_cell = c.set_index("cell")
        for cell in ids:
            x, y = pos[cell]
            activity = float(by_cell.loc[cell, "activity"])
            ax.scatter([x], [y], s=180 + 500 * activity)
            ax.text(x, y, f"C{cell}", ha="center", va="center", fontsize=8)
        ax.set_title(f"G replicate {replicate}")
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.suptitle("Final growing organisms — node size = activity")
    fig.tight_layout()
    fig.savefig(OUT / "final-organism-atlas.png", dpi=180)
    plt.close(fig)


def plot_lineage(cells: pd.DataFrame) -> None:
    g = cells.loc[cells["variant"] == "G"]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = 0
    for replicate in range(N_REPLICATES):
        c = g.loc[g["replicate"] == replicate]
        if c.empty:
            continue
        final = c.loc[c["step"] == c["step"].max()].drop_duplicates("cell")
        for _, row in final.iterrows():
            cell = int(row["cell"])
            birth = int(row["birth_step"])
            parent = int(row["parent"])
            ax.scatter([birth], [y + cell], s=50)
            ax.text(birth, y + cell, f" r{replicate}:C{cell}", fontsize=7, va="bottom")
            if cell >= 4 and parent >= 0:
                parent_row = final.loc[final["cell"] == parent]
                parent_birth = int(parent_row.iloc[0]["birth_step"]) if not parent_row.empty else 0
                ax.plot([parent_birth, birth], [y + parent, y + cell], alpha=0.5)
        y += 14
    ax.set_xlabel("training step at birth")
    ax.set_ylabel("lineage / cell")
    ax.set_title("Cell lineage tree")
    fig.tight_layout()
    fig.savefig(OUT / "lineage-tree.png", dpi=180)
    plt.close(fig)


def plot_activity_heatmap(cells: pd.DataFrame) -> None:
    g = cells.loc[cells["variant"] == "G"]
    pivot = g.pivot_table(index=["replicate", "cell"], columns="step", values="activity", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(pivot.fillna(0.0).to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"r{r}:C{c}" for r, c in pivot.index], fontsize=7)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(v)) for v in pivot.columns], rotation=45)
    ax.set_xlabel("training step")
    ax.set_title("Cell activity over language training")
    fig.colorbar(image, ax=ax, label="mean activity")
    fig.tight_layout()
    fig.savefig(OUT / "cell-activity-heatmap.png", dpi=180)
    plt.close(fig)


def plot_pressure_conflict(probes: pd.DataFrame) -> None:
    if probes.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(probes["pressure"], probes["conflict"], s=20, alpha=0.6, c=probes["step"])
    ax.set_xlabel("cell pressure")
    ax.set_ylabel("microbatch gradient conflict")
    ax.set_title("Structural pressure and conflict")
    fig.colorbar(scatter, ax=ax, label="training step")
    fig.tight_layout()
    fig.savefig(OUT / "cell-gradient-conflict.png", dpi=180)
    plt.close(fig)


def plot_local_learning(local: pd.DataFrame, transplant: pd.DataFrame) -> None:
    if local.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for replicate in range(N_REPLICATES):
        run = local.loc[local["replicate"] == replicate]
        ax.plot(run["step"], run["skill_nll"], marker="o", label=f"donor r{replicate}")
    if not transplant.empty:
        for _, row in transplant.iterrows():
            ax.scatter([205], [row["recipient_skill_nll"]], marker="x", s=80)
    ax.set_xlabel("local-learning step")
    ax.set_ylabel("held-out skill NLL")
    ax.set_title("Frozen-genome local learning and transplantation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "local-learning-transplantation.png", dpi=180)
    plt.close(fig)


def make_decision(data: dict[str, pd.DataFrame], comparisons: pd.DataFrame, gpu_count: int) -> dict[str, object]:
    summary = data["model_summary"]
    events = data["structural_events"]
    interventions = data["interventions"]
    transplant = data["transplantation"]
    def ratios(comparison: str, metric: str) -> list[float]:
        return comparisons.loc[(comparisons["comparison"] == comparison) & (comparisons["metric"] == metric), "ratio"].tolist()
    g_t = ratios("G_over_T", "final_nll")
    f_t = ratios("F_over_T", "final_nll")
    g_f = ratios("G_over_F", "final_nll")
    cost = ratios("G_over_T", "wall_seconds_per_million_tokens")
    growth_reps = 0
    fork_reps = 0
    connect_reps = 0
    for replicate in range(N_REPLICATES):
        selected = events.loc[(events.get("phase") == "language") & (events.get("replicate") == replicate)] if not events.empty else pd.DataFrame()
        if not selected.empty:
            growth_reps += 1
            fork_reps += int((selected["event"] == "fork").any())
            connect_reps += int((selected["event"] == "connect").any())
    causal_reps = 0
    if not interventions.empty:
        selected = interventions.loc[interventions["intervention"].isin(["learned_edges_off", "initial_organism"])]
        for replicate in range(N_REPLICATES):
            rows = selected.loc[selected["replicate"] == replicate]
            causal_reps += int((rows["delta_nll"] > 0).any())
    local_improvement_reps = int((transplant["skill_improvement"] > 0).sum()) if not transplant.empty else 0
    retention_reps = int((transplant["donor_language_ratio"] <= 1.10).sum()) if not transplant.empty else 0
    transplant_reps = int((transplant["transplant_recovery"] >= 0.50).sum()) if not transplant.empty else 0
    pass_flags = {
        "language_competitiveness": geometric_mean(g_t) <= 1.25 and max(g_t) <= 1.50,
        "structural_growth": growth_reps >= 2,
        "structural_causal_use": causal_reps >= 2,
        "frozen_genome_skill_learning": local_improvement_reps >= 2,
        "language_retention": retention_reps >= 2,
        "tissue_transplantation": transplant_reps >= 2,
    }
    passed = sum(int(v) for v in pass_flags.values())
    if all(pass_flags.values()):
        status = "HOLISTIC_GROWING_CLM_SIGNAL"
    elif passed >= 3:
        status = "PARTIAL_HOLISTIC_GROWING_CLM_SIGNAL"
    else:
        status = "NO_HOLISTIC_GROWING_CLM_SIGNAL"
    g = summary.loc[summary["variant"] == "G"]
    return {
        "format": "minicells.growing-cellular-lm.v1",
        "experiment": "MINI Cells Experiment 016 — Growing Cellular Language Model",
        "status": status,
        "question": "Can a small language organism with one shared cell genome autonomously allocate activity, connect, prune and fork while remaining competitive with a parameter-matched Transformer, then acquire and transplant a skill with the genome frozen?",
        "results": {
            "G_over_T_nll_geomean": geometric_mean(g_t),
            "F_over_T_nll_geomean": geometric_mean(f_t),
            "G_over_F_nll_geomean": geometric_mean(g_f),
            "G_over_T_wall_cost_geomean": geometric_mean(cost),
            "growth_replicates": growth_reps,
            "fork_replicates": fork_reps,
            "connect_replicates": connect_reps,
            "structural_causal_replicates": causal_reps,
            "local_skill_improvement_replicates": local_improvement_reps,
            "language_retention_replicates": retention_reps,
            "transplant_recovery_replicates": transplant_reps,
            "mean_phase1_alive_cells": float(g["phase1_alive_cells"].mean()),
            "mean_phase1_edges": float(g["phase1_edges"].mean()),
            "pass_flags": pass_flags,
        },
        "pre_registered_signal": {
            "G_over_T_nll_geomean_max": 1.25,
            "per_seed_G_over_T_nll_max": 1.50,
            "minimum_growth_replicates": 2,
            "minimum_structural_causal_replicates": 2,
            "minimum_frozen_genome_skill_improvement_replicates": 2,
            "language_retention_ratio_max": 1.10,
            "transplant_recovery_min": 0.50,
            "minimum_signal_replicates": 2,
        },
        "design": {
            "replicates": N_REPLICATES,
            "models_total": N_REPLICATES * len(VARIANTS),
            "tokens_per_model": BUDGET_TOKENS,
            "gpu_count": gpu_count,
            "parameter_matched_transformer": True,
            "cellular_interface": "only cell 0 receives token embedding and supplies logits",
            "cell_rule": "one globally shared time-homogeneous recurrent rule",
            "initial_cells": 4,
            "max_cells": 12,
            "persistent_phenotype": "per-cell memory vector",
            "structure": "directed graph with protected initial/lineage edges plus utility-selected learned edges",
        },
        "scope": {
            "claim": "small end-to-end prototype of a growing cellular language organism",
            "not_claimed": [
                "production-scale language quality",
                "arbitrary cross-genome transplantation",
                "unbounded physical allocation",
                "proof that mature capabilities must form MoE-like communities",
                "optimal structural policy",
            ],
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache, manifest = prepare_corpus()
    save_task_spec(manifest)
    gpu_count = run_models(cache)
    data = collect_results()
    comparisons = make_paired_comparisons(data["model_summary"])
    plot_training_curves(data["checkpoints"])
    plot_quality_cost(data["model_summary"])
    plot_structure_timeline(data["checkpoints"])
    plot_final_organisms(data["cells"], data["edges"])
    plot_lineage(data["cells"])
    plot_activity_heatmap(data["cells"])
    plot_pressure_conflict(data["structural_probes"])
    plot_local_learning(data["local_learning"], data["transplantation"])
    decision = make_decision(data, comparisons, gpu_count)
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
