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
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))

from minicells.language_data import prepare_tinystories_corpus  # noqa: E402


OUT = ROOT / "results" / "localized-cellular-learning-v1"
WORKER = ROOT / "scripts" / "run_language_localized_learning_worker.py"
N_REPLICATES = 3
POLICIES = ("B", "L")


def prepare_corpus() -> tuple[Path, dict[str, object]]:
    corpus = prepare_tinystories_corpus(ROOT)
    cache = corpus.tokenizer_path.parent
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(corpus.tokenizer_path, OUT / "tokenizer.json")
    shutil.copy2(cache / "corpus-manifest.json", OUT / "corpus-manifest.json")
    return cache, corpus.manifest


def worker_command(replicate: int, cache: Path) -> list[str]:
    return [sys.executable, str(WORKER), "--replicate", str(replicate), "--cache-dir", str(cache), "--output-dir", str(OUT)]


def run_workers(cache: Path) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 017 requires CUDA")
    gpu_count = min(2, available)
    for start in range(0, N_REPLICATES, gpu_count):
        group = list(range(start, min(start + gpu_count, N_REPLICATES)))
        active = []
        for local_gpu, replicate in enumerate(group):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(local_gpu)
            log = OUT / f"r{replicate}.log"
            handle = log.open("w", encoding="utf-8")
            process = subprocess.Popen(worker_command(replicate, cache), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            active.append((replicate, local_gpu, process, log, handle))
            print(f"started r{replicate} on physical GPU {local_gpu}")
        failures = []
        for replicate, gpu, process, log, handle in active:
            code = process.wait()
            handle.close()
            print(f"--- r{replicate} / GPU {gpu} ---")
            print(log.read_text(encoding="utf-8").rstrip())
            if code != 0:
                failures.append(f"r{replicate} exited {code}; see {log}")
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


def collect() -> dict[str, object]:
    workers = []
    keys = ("phase1-checkpoints", "structural-events", "local-learning", "transplantation", "localization", "tissue-ablation")
    frames: dict[str, list[pd.DataFrame]] = {key: [] for key in keys}
    summary_rows = []
    for replicate in range(N_REPLICATES):
        worker = json.loads((OUT / f"r{replicate}-worker.json").read_text(encoding="utf-8"))
        workers.append(worker)
        phase1 = worker["phase1"]
        for policy in POLICIES:
            row = {"replicate": replicate, "policy": policy, **phase1, **worker["policies"][policy]}
            summary_rows.append(row)
        for key in keys:
            frame = read_csv(OUT / f"r{replicate}-{key}.csv")
            if not frame.empty:
                frames[key].append(frame)
    result: dict[str, object] = {"workers": workers, "summary": pd.DataFrame(summary_rows)}
    for key, parts in frames.items():
        result[key.replace("-", "_")] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    summary = result["summary"]
    assert isinstance(summary, pd.DataFrame)
    summary.to_csv(OUT / "policy-summary.csv", index=False)
    for key in keys:
        frame = result[key.replace("-", "_")]
        assert isinstance(frame, pd.DataFrame)
        frame.to_csv(OUT / f"{key}.csv", index=False)
    return result


def reconstruct_graph(events: pd.DataFrame, replicate: int, policy: str) -> tuple[set[int], set[tuple[int, int]]]:
    cells = {0, 1, 2, 3}
    edges = {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)}
    phase1 = events.loc[(events["replicate"] == replicate) & (events["policy"] == "phase1")]
    local = events.loc[(events["replicate"] == replicate) & (events["policy"] == policy)]
    ordered = pd.concat([phase1, local], ignore_index=True)
    for _, row in ordered.iterrows():
        event = str(row["event"])
        if "fork" in event and pd.notna(row.get("child")):
            parent = int(row["parent"])
            child = int(row["child"])
            cells.add(child)
            edges.add((parent, child))
            edges.add((child, parent))
        elif "connect" in event and pd.notna(row.get("receiver")):
            edges.add((int(row["receiver"]), int(row["source"])))
        elif "prune" in event and pd.notna(row.get("receiver")):
            edges.discard((int(row["receiver"]), int(row["source"])))
    return cells, edges


def plot_skill_learning(local: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy in POLICIES:
        selected = local.loc[local["policy"] == policy]
        for replicate in range(N_REPLICATES):
            run = selected.loc[selected["replicate"] == replicate]
            ax.plot(run["step"], run["skill_nll"], alpha=0.25)
        mean = selected.groupby("step")["skill_nll"].mean()
        ax.plot(mean.index, mean.values, linewidth=2.5, label=policy)
    ax.set_xlabel("local learning step")
    ax.set_ylabel("REVERSE_INC validation NLL")
    ax.set_title("Experiment 017 skill acquisition")
    ax.legend(title="policy")
    fig.tight_layout()
    fig.savefig(OUT / "skill-learning-policy-comparison.png", dpi=180)
    plt.close(fig)


def plot_retention(local: pd.DataFrame, summary: pd.DataFrame) -> None:
    base = summary.set_index(["replicate", "policy"])["base_language_nll"].to_dict()
    frame = local.copy()
    frame["language_ratio"] = [row.language_nll / base[(int(row.replicate), str(row.policy))] for row in frame.itertuples()]
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy in POLICIES:
        selected = frame.loc[frame["policy"] == policy]
        for replicate in range(N_REPLICATES):
            run = selected.loc[selected["replicate"] == replicate]
            ax.plot(run["step"], run["language_ratio"], alpha=0.25)
        mean = selected.groupby("step")["language_ratio"].mean()
        ax.plot(mean.index, mean.values, linewidth=2.5, label=policy)
    ax.axhline(1.10, linestyle="--", linewidth=1)
    ax.set_xlabel("local learning step")
    ax.set_ylabel("TinyStories NLL / Phase-1 NLL")
    ax.set_title("Old-language retention")
    ax.legend(title="policy")
    fig.tight_layout()
    fig.savefig(OUT / "language-retention-policy-comparison.png", dpi=180)
    plt.close(fig)


def plot_localization(localization: pd.DataFrame) -> None:
    selected = localization.loc[localization["policy"] == "L"]
    matrix = np.zeros((N_REPLICATES, 12), dtype=float)
    newborn = np.zeros((N_REPLICATES, 12), dtype=bool)
    for row in selected.itertuples():
        matrix[int(row.replicate), int(row.cell)] = float(row.memory_delta_norm)
        newborn[int(row.replicate), int(row.cell)] = bool(row.newborn)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    image = ax.imshow(matrix, aspect="auto")
    fig.colorbar(image, ax=ax, label="phenotype update norm")
    for r in range(N_REPLICATES):
        for c in range(12):
            if newborn[r, c]:
                ax.text(c, r, "N", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(12), [f"C{i}" for i in range(12)])
    ax.set_yticks(range(N_REPLICATES), [f"r{i}" for i in range(N_REPLICATES)])
    ax.set_title("Localized policy phenotype changes (N = newborn)")
    fig.tight_layout()
    fig.savefig(OUT / "localized-memory-updates.png", dpi=180)
    plt.close(fig)


def plot_transfer(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.32
    x = np.arange(N_REPLICATES)
    for offset, policy in ((-width / 2, "B"), (width / 2, "L")):
        values = summary.loc[summary["policy"] == policy].sort_values("replicate")["transplant_recovery"].to_numpy()
        ax.bar(x + offset, values, width=width, label=policy)
    ax.axhline(0.50, linestyle="--", linewidth=1)
    ax.set_xticks(x, [f"r{i}" for i in range(N_REPLICATES)])
    ax.set_ylabel("transplant recovery")
    ax.set_title("Skill transfer from selected tissue")
    ax.legend(title="policy")
    fig.tight_layout()
    fig.savefig(OUT / "transplantation-recovery.png", dpi=180)
    plt.close(fig)


def plot_tissue_atlas(events: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(N_REPLICATES, 2, figsize=(10, 13))
    for replicate in range(N_REPLICATES):
        for column, policy in enumerate(POLICIES):
            ax = axes[replicate, column]
            cells, edges = reconstruct_graph(events, replicate, policy)
            ids = sorted(cells)
            positions = {cell: (math.cos(2 * math.pi * i / len(ids)), math.sin(2 * math.pi * i / len(ids))) for i, cell in enumerate(ids)}
            row = summary.loc[(summary["replicate"] == replicate) & (summary["policy"] == policy)].iloc[0]
            newborn = set(json.loads(row["newborn_cells"])) if isinstance(row["newborn_cells"], str) else set()
            for receiver, source in edges:
                if receiver not in positions or source not in positions:
                    continue
                x0, y0 = positions[source]
                x1, y1 = positions[receiver]
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "alpha": 0.35, "lw": 1.0})
            for cell in ids:
                x, y = positions[cell]
                marker = "*" if cell in newborn else "o"
                ax.scatter([x], [y], s=210, marker=marker)
                ax.text(x, y, f"C{cell}", ha="center", va="center", fontsize=8)
            ax.set_title(f"r{replicate} {policy}  (* newborn)")
            ax.set_aspect("equal")
            ax.set_axis_off()
    fig.suptitle("Final adaptation structure: distributed baseline vs localized growth")
    fig.tight_layout()
    fig.savefig(OUT / "localized-tissue-atlas.png", dpi=180)
    plt.close(fig)


def make_decision(summary: pd.DataFrame, ablation: pd.DataFrame, gpu_count: int) -> dict[str, object]:
    paired = []
    for replicate in range(N_REPLICATES):
        group = summary.loc[summary["replicate"] == replicate].set_index("policy")
        b = group.loc["B"]
        l = group.loc["L"]
        paired.append({
            "replicate": replicate,
            "localized_over_baseline_skill_improvement": float(l["skill_improvement"] / max(1e-12, b["skill_improvement"])),
            "baseline_language_ratio": float(b["donor_language_ratio"]),
            "localized_language_ratio": float(l["donor_language_ratio"]),
            "localized_recipient_language_ratio": float(l["recipient_language_ratio"]),
            "localized_transplant_recovery": float(l["transplant_recovery"]),
            "localized_newborn_count": int(l["newborn_count"]),
            "localized_base_memory_drift": float(l["base_memory_drift"]),
            "localized_tissue_causal_fraction": float(l["tissue_causal_fraction"]),
        })
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(OUT / "paired-policy-comparisons.csv", index=False)
    skill_reps = int(((paired_frame["localized_over_baseline_skill_improvement"] >= 0.60)).sum())
    retention_reps = int((paired_frame["localized_language_ratio"] <= 1.10).sum())
    retention_better_reps = int((paired_frame["localized_language_ratio"] < paired_frame["baseline_language_ratio"]).sum())
    stable_reps = int((paired_frame["localized_base_memory_drift"] <= 1e-6).sum())
    compact_reps = int((paired_frame["localized_newborn_count"] <= 3).sum())
    causal_reps = int((paired_frame["localized_tissue_causal_fraction"] >= 0.50).sum())
    transfer_reps = int(((paired_frame["localized_transplant_recovery"] >= 0.50) & (paired_frame["localized_recipient_language_ratio"] <= 1.10)).sum())
    flags = {
        "skill_preserved": skill_reps >= 2,
        "language_retention": retention_reps >= 2,
        "retention_improves_over_distributed": retention_better_reps >= 2,
        "old_tissue_stable": stable_reps == N_REPLICATES,
        "compact_new_tissue": compact_reps == N_REPLICATES,
        "new_tissue_causally_used": causal_reps >= 2,
        "compact_tissue_transplantation": transfer_reps >= 2,
    }
    if all(flags.values()):
        status = "LOCALIZED_CELLULAR_LEARNING_SIGNAL"
    elif flags["skill_preserved"] and flags["old_tissue_stable"] and flags["compact_new_tissue"] and not flags["language_retention"]:
        status = "LOCALIZATION_WITHOUT_RETENTION"
    elif flags["language_retention"] and not flags["compact_tissue_transplantation"]:
        status = "RETENTION_WITHOUT_TRANSFER"
    elif sum(bool(value) for value in flags.values()) >= 4:
        status = "PARTIAL_LOCALIZED_CELLULAR_LEARNING_SIGNAL"
    else:
        status = "NO_LOCALIZED_CELLULAR_LEARNING_SIGNAL"
    decision = {
        "format": "minicells.localized-cellular-learning.v1",
        "experiment": "MINI Cells Experiment 017 — Localized Cellular Learning",
        "question": "Can frozen existing tissue divert a new capability into compact newborn cellular capacity, retain old language behavior, and transfer that capability by grafting only the newborn tissue?",
        "design": {
            "replicates": N_REPLICATES,
            "policies": {
                "B": "016-style distributed phenotype learning: all cell memories trainable",
                "L": "localized gradient diversion: existing phenotype frozen; only newborn cells trainable; only newborn-touching structure can change",
            },
            "same_phase1_checkpoint_within_replicate": True,
            "genome_frozen_in_both_policies": True,
            "skill": "REVERSE_INC",
            "local_steps": 200,
            "max_local_newborns": 3,
            "gpu_count": gpu_count,
        },
        "pre_registered_signal": {
            "localized_skill_improvement_fraction_of_baseline_min": 0.60,
            "donor_language_ratio_max": 1.10,
            "recipient_language_ratio_max": 1.10,
            "base_memory_drift_max": 1e-6,
            "newborn_cell_count_max": 3,
            "newborn_tissue_causal_fraction_min": 0.50,
            "transplant_recovery_min": 0.50,
            "minimum_replicates": 2,
        },
        "results": {
            "skill_preserved_replicates": skill_reps,
            "language_retention_replicates": retention_reps,
            "retention_better_replicates": retention_better_reps,
            "old_tissue_stable_replicates": stable_reps,
            "compact_tissue_replicates": compact_reps,
            "tissue_causal_replicates": causal_reps,
            "compact_transplant_replicates": transfer_reps,
            "mean_baseline_language_ratio": float(paired_frame["baseline_language_ratio"].mean()),
            "mean_localized_language_ratio": float(paired_frame["localized_language_ratio"].mean()),
            "mean_localized_transplant_recovery": float(paired_frame["localized_transplant_recovery"].mean()),
            "mean_localized_tissue_causal_fraction": float(paired_frame["localized_tissue_causal_fraction"].mean()),
            "pass_flags": flags,
        },
        "status": status,
        "scope": {
            "claim": "controlled same-checkpoint test of distributed versus newborn-localized frozen-genome capability acquisition",
            "not_claimed": [
                "arbitrary real-world skill transplantation",
                "optimal birth policy",
                "zero forgetting at larger scales",
                "cross-genome tissue compatibility",
                "production compute efficiency",
            ],
        },
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return decision


def save_task_spec(manifest: dict[str, object]) -> None:
    spec = {
        "format": "minicells.localized-cellular-learning-task.v1",
        "experiment": "017",
        "phase1": {
            "dataset": manifest.get("dataset"),
            "model": "Experiment 016 Growing Cellular LM",
            "tokens": 1_024_000,
            "shared_checkpoint": "B and L branch from the exact same phase1 state per replicate",
        },
        "policy_B": "freeze genome, update all existing/newborn cell_memory as in Experiment 016 local phase",
        "policy_L": {
            "old_phenotype": "bit-stable / optimizer gradient masked to zero",
            "initial_allocation": "conservative fork from the non-interface old cell with maximum counterfactual rewrite pressure",
            "continued_growth": "connect first; conservative fork only when old rewrite pressure persists",
            "trainable": "newborn cell_memory only",
            "structural_changes": "only edges touching newborn tissue",
            "transplant": "newborn phenotype plus its boundary graph; no old-cell phenotype copied",
        },
    }
    (OUT / "task-spec.json").write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    cache, manifest = prepare_corpus()
    gpu_count = run_workers(cache)
    data = collect()
    summary = data["summary"]
    local = data["local_learning"]
    localization = data["localization"]
    events = data["structural_events"]
    ablation = data["tissue_ablation"]
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(local, pd.DataFrame)
    assert isinstance(localization, pd.DataFrame)
    assert isinstance(events, pd.DataFrame)
    assert isinstance(ablation, pd.DataFrame)
    plot_skill_learning(local)
    plot_retention(local, summary)
    plot_localization(localization)
    plot_transfer(summary)
    plot_tissue_atlas(events, summary)
    save_task_spec(manifest)
    decision = make_decision(summary, ablation, gpu_count)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
