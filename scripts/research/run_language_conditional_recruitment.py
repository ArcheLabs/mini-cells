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


OUT = ROOT / "results" / "conditional-tissue-recruitment-v1"
WORKER = ROOT / "scripts" / "run_language_conditional_recruitment_worker.py"
N_REPLICATES = 3
POLICIES = ("S", "C")


def prepare_corpus() -> tuple[Path, dict[str, object]]:
    corpus = prepare_tinystories_corpus(ROOT)
    cache = corpus.tokenizer_path.parent
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(corpus.tokenizer_path, OUT / "tokenizer.json")
    shutil.copy2(cache / "corpus-manifest.json", OUT / "corpus-manifest.json")
    return cache, corpus.manifest


def run_workers(cache: Path) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 018 requires CUDA")
    gpu_count = min(2, available)
    for start in range(0, N_REPLICATES, gpu_count):
        group = list(range(start, min(start + gpu_count, N_REPLICATES)))
        active = []
        for local_gpu, replicate in enumerate(group):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(local_gpu)
            log = OUT / f"r{replicate}.log"
            handle = log.open("w", encoding="utf-8")
            cmd = [sys.executable, str(WORKER), "--replicate", str(replicate), "--cache-dir", str(cache), "--output-dir", str(OUT)]
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
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
    keys = ("phase1-checkpoints", "structural-events", "local-learning", "transplantation", "localization", "recruitment-interventions")
    frames: dict[str, list[pd.DataFrame]] = {key: [] for key in keys}
    workers = []
    summary_rows = []
    for replicate in range(N_REPLICATES):
        worker = json.loads((OUT / f"r{replicate}-worker.json").read_text(encoding="utf-8"))
        workers.append(worker)
        phase1 = worker["phase1"]
        for policy in POLICIES:
            summary_rows.append({"replicate": replicate, "policy": policy, **phase1, **worker["policies"][policy]})
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


def plot_skill(local: pd.DataFrame) -> None:
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
    ax.set_title("Experiment 018 skill acquisition")
    ax.legend(title="S=static, C=conditional")
    fig.tight_layout()
    fig.savefig(OUT / "skill-learning-recruitment-comparison.png", dpi=180)
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
    ax.set_title("Conditional recruitment and retained language")
    ax.legend(title="policy")
    fig.tight_layout()
    fig.savefig(OUT / "language-retention-recruitment-comparison.png", dpi=180)
    plt.close(fig)


def plot_recruitment(local: pd.DataFrame) -> None:
    selected = local.loc[local["policy"] == "C"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ("skill_recruitment", "language_recruitment"):
        for replicate in range(N_REPLICATES):
            run = selected.loc[selected["replicate"] == replicate]
            ax.plot(run["step"], run[metric], alpha=0.22)
        mean = selected.groupby("step")[metric].mean()
        ax.plot(mean.index, mean.values, linewidth=2.5, label=metric)
    ax.set_xlabel("local learning step")
    ax.set_ylabel("mean newborn conductance")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Local homeostatic tissue recruitment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "recruitment-selectivity.png", dpi=180)
    plt.close(fig)


def plot_interventions(interventions: pd.DataFrame) -> None:
    pivot = interventions.pivot(index="replicate", columns="intervention", values="delta_nll")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(N_REPLICATES)
    columns = list(pivot.columns)
    width = 0.8 / max(1, len(columns))
    for index, name in enumerate(columns):
        ax.bar(x + (index - (len(columns) - 1) / 2) * width, pivot[name].to_numpy(), width=width, label=name)
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x, [f"r{i}" for i in range(N_REPLICATES)])
    ax.set_ylabel("altered NLL - normal NLL")
    ax.set_title("Causal recruitment interventions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "recruitment-causal-interventions.png", dpi=180)
    plt.close(fig)


def plot_transfer(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.32
    x = np.arange(N_REPLICATES)
    for offset, policy in ((-width / 2, "S"), (width / 2, "C")):
        values = summary.loc[summary["policy"] == policy].sort_values("replicate")["transplant_recovery"].to_numpy()
        ax.bar(x + offset, values, width=width, label=policy)
    ax.axhline(0.50, linestyle="--", linewidth=1)
    ax.set_xticks(x, [f"r{i}" for i in range(N_REPLICATES)])
    ax.set_ylabel("newborn-only transplant recovery")
    ax.set_title("Transfer of conditionally recruited tissue")
    ax.legend(title="policy")
    fig.tight_layout()
    fig.savefig(OUT / "conditional-transplantation-recovery.png", dpi=180)
    plt.close(fig)


def reconstruct_graph(events: pd.DataFrame, replicate: int, policy: str) -> tuple[set[int], set[tuple[int, int]]]:
    cells = {0, 1, 2, 3}
    edges = {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)}
    selected = events.loc[(events["replicate"] == replicate) & (events["policy"].isin(["phase1", policy]))]
    for _, row in selected.iterrows():
        event = str(row["event"])
        if "fork" in event and pd.notna(row.get("child")):
            parent, child = int(row["parent"]), int(row["child"])
            cells.add(child)
            edges.add((parent, child)); edges.add((child, parent))
        elif "connect" in event and pd.notna(row.get("receiver")):
            edges.add((int(row["receiver"]), int(row["source"])))
        elif "prune" in event and pd.notna(row.get("receiver")):
            edges.discard((int(row["receiver"]), int(row["source"])))
    return cells, edges


def plot_atlas(events: pd.DataFrame, summary: pd.DataFrame) -> None:
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
                x0, y0 = positions[source]; x1, y1 = positions[receiver]
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "alpha": 0.35, "lw": 1.0})
            for cell in ids:
                x, y = positions[cell]
                ax.scatter([x], [y], s=220, marker="*" if cell in newborn else "o")
                ax.text(x, y, f"C{cell}", ha="center", va="center", fontsize=8)
            ax.set_title(f"r{replicate} {policy} (* newborn)")
            ax.set_aspect("equal"); ax.set_axis_off()
    fig.suptitle("Static vs conditionally recruited adaptation tissue")
    fig.tight_layout()
    fig.savefig(OUT / "conditional-tissue-atlas.png", dpi=180)
    plt.close(fig)


def make_decision(summary: pd.DataFrame, gpu_count: int) -> dict[str, object]:
    paired = []
    for replicate in range(N_REPLICATES):
        group = summary.loc[summary["replicate"] == replicate].set_index("policy")
        s, c = group.loc["S"], group.loc["C"]
        paired.append({
            "replicate": replicate,
            "static_skill_improvement": float(s["skill_improvement"]),
            "conditional_skill_improvement": float(c["skill_improvement"]),
            "conditional_over_static_skill_improvement": float(c["skill_improvement"] / max(1e-12, s["skill_improvement"])),
            "static_language_ratio": float(s["donor_language_ratio"]),
            "conditional_language_ratio": float(c["donor_language_ratio"]),
            "conditional_recipient_language_ratio": float(c["recipient_language_ratio"]),
            "conditional_transplant_recovery": float(c["transplant_recovery"]),
            "skill_recruitment": float(c["donor_skill_recruitment"]),
            "language_recruitment": float(c["donor_language_recruitment"]),
            "recruitment_selectivity": float(c["recruitment_selectivity"]),
            "recruitment_gap": float(c["recruitment_gap"]),
            "recruitment_causal_fraction": float(c["recruitment_causal_fraction"]),
            "force_on_language_delta_nll": float(c["force_on_language_delta_nll"]),
            "newborn_count": int(c["newborn_count"]),
            "base_memory_drift": float(c["base_memory_drift"]),
            "tissue_causal_fraction": float(c["tissue_causal_fraction"]),
        })
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(OUT / "paired-policy-comparisons.csv", index=False)

    skill_reps = int(((paired_frame["conditional_skill_improvement"] > 0) & (paired_frame["conditional_over_static_skill_improvement"] >= 0.70)).sum())
    retention_reps = int((paired_frame["conditional_language_ratio"] <= 1.10).sum())
    retention_better_reps = int((paired_frame["conditional_language_ratio"] < paired_frame["static_language_ratio"]).sum())
    stable_reps = int((paired_frame["base_memory_drift"] <= 1e-6).sum())
    compact_reps = int((paired_frame["newborn_count"] <= 3).sum())
    selectivity_reps = int(((paired_frame["recruitment_selectivity"] >= 2.0) & (paired_frame["recruitment_gap"] >= 0.20)).sum())
    recruitment_causal_reps = int((paired_frame["recruitment_causal_fraction"] >= 0.50).sum())
    protection_causal_reps = int((paired_frame["force_on_language_delta_nll"] >= 0.05).sum())
    tissue_causal_reps = int((paired_frame["tissue_causal_fraction"] >= 0.50).sum())
    transfer_reps = int(((paired_frame["conditional_transplant_recovery"] >= 0.50) & (paired_frame["conditional_recipient_language_ratio"] <= 1.10)).sum())

    flags = {
        "skill_preserved": skill_reps >= 2,
        "language_retention": retention_reps >= 2,
        "retention_improves_over_static": retention_better_reps >= 2,
        "old_tissue_stable": stable_reps == N_REPLICATES,
        "compact_new_tissue": compact_reps == N_REPLICATES,
        "recruitment_selective": selectivity_reps >= 2,
        "recruitment_causally_required_for_skill": recruitment_causal_reps >= 2,
        "conditional_closure_causally_protects_language": protection_causal_reps >= 2,
        "new_tissue_causally_used": tissue_causal_reps >= 2,
        "conditional_tissue_transplantation": transfer_reps >= 2,
    }
    if all(flags.values()):
        status = "CONDITIONAL_TISSUE_RECRUITMENT_SIGNAL"
    elif flags["recruitment_selective"] and flags["skill_preserved"] and not flags["language_retention"]:
        status = "CONDITIONAL_RECRUITMENT_WITHOUT_RETENTION"
    elif flags["language_retention"] and not flags["recruitment_causally_required_for_skill"]:
        status = "RETENTION_WITHOUT_RECRUITMENT_CAUSALITY"
    elif sum(bool(value) for value in flags.values()) >= 6:
        status = "PARTIAL_CONDITIONAL_TISSUE_RECRUITMENT_SIGNAL"
    else:
        status = "NO_CONDITIONAL_TISSUE_RECRUITMENT_SIGNAL"

    decision = {
        "format": "minicells.conditional-tissue-recruitment.v1",
        "experiment": "MINI Cells Experiment 018 — Conditional Tissue Recruitment",
        "status": status,
        "question": "Can a localized newborn capability tissue remain mostly quiescent on retained language but locally excite and conduct information when its birth-site dynamics depart from the Phase-1 homeostatic manifold?",
        "design": {
            "replicates": N_REPLICATES,
            "gpu_count": gpu_count,
            "same_phase1_checkpoint_within_replicate": True,
            "policies": {
                "S": "Experiment-017 static localized newborn tissue",
                "C": "same localized learning with parent-local homeostatic excitability gating all newborn incident edges",
            },
            "central_router": False,
            "task_label_used_by_recruitment": False,
            "homeostatic_reference": "per recurrent step and parent cell, calibrated on held-out Phase-1 training-language batches",
        },
        "pre_registered_signal": {
            "conditional_skill_fraction_of_static_min": 0.70,
            "donor_language_ratio_max": 1.10,
            "base_memory_drift_max": 1e-6,
            "newborn_cell_count_max": 3,
            "skill_to_language_recruitment_ratio_min": 2.0,
            "skill_minus_language_recruitment_min": 0.20,
            "recruitment_causal_fraction_min": 0.50,
            "forced_on_language_delta_nll_min": 0.05,
            "transplant_recovery_min": 0.50,
            "recipient_language_ratio_max": 1.10,
            "minimum_replicates": 2,
        },
        "results": {
            "pass_flags": flags,
            "skill_preserved_replicates": skill_reps,
            "language_retention_replicates": retention_reps,
            "retention_better_replicates": retention_better_reps,
            "selective_recruitment_replicates": selectivity_reps,
            "recruitment_causal_replicates": recruitment_causal_reps,
            "language_protection_causal_replicates": protection_causal_reps,
            "transfer_replicates": transfer_reps,
            "mean_static_language_ratio": float(paired_frame["static_language_ratio"].mean()),
            "mean_conditional_language_ratio": float(paired_frame["conditional_language_ratio"].mean()),
            "mean_skill_recruitment": float(paired_frame["skill_recruitment"].mean()),
            "mean_language_recruitment": float(paired_frame["language_recruitment"].mean()),
            "mean_transplant_recovery": float(paired_frame["conditional_transplant_recovery"].mean()),
        },
        "scope": {
            "claim": "same-checkpoint causal test of local state-dependent recruitment for a compact newborn skill tissue",
            "not_claimed": ["general task routing", "arbitrary real-world skills", "cross-genome transplantation", "optimal novelty thresholds", "production compute efficiency"],
        },
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return decision


def save_task_spec(manifest: dict[str, object]) -> None:
    payload = {
        "format": "minicells.conditional-tissue-recruitment-task.v1",
        "experiment": "018",
        "dataset": manifest.get("dataset"),
        "skill": "REVERSE_INC",
        "principle": "newborn edges are conductances gated by local parent-state deviation from a recurrent-step-specific Phase-1 homeostatic manifold",
        "comparison": "static localized tissue versus conditional localized tissue from the same Phase-1 checkpoint",
    }
    (OUT / "task-spec.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    cache, manifest = prepare_corpus()
    gpu_count = run_workers(cache)
    result = collect()
    summary = result["summary"]; local = result["local_learning"]; events = result["structural_events"]; interventions = result["recruitment_interventions"]
    assert isinstance(summary, pd.DataFrame) and isinstance(local, pd.DataFrame)
    assert isinstance(events, pd.DataFrame) and isinstance(interventions, pd.DataFrame)
    plot_skill(local)
    plot_retention(local, summary)
    plot_recruitment(local)
    plot_interventions(interventions)
    plot_transfer(summary)
    plot_atlas(events, summary)
    save_task_spec(manifest)
    decision = make_decision(summary, gpu_count)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
