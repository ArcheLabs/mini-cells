from __future__ import annotations

import json
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
sys.path.insert(0, str(ROOT / "scripts"))

import run_probationary_trait_genesis_worker as worker  # noqa: E402
from minicells.language_conflict_differentiation import prepare_arithmetic_cache  # noqa: E402
from minicells.language_data import load_tokenizer, prepare_tinystories_corpus  # noqa: E402
from minicells.language_probationary_trait_genesis import (  # noqa: E402
    ARMS,
    CONDITIONS,
    GEOMETRY_ADVANTAGE_MIN,
    POSITIVE_REPLICATES_MIN,
    PROBATION_WINDOWS,
    ROUTING_PURITY_MIN,
    STRUCTURAL_COST_FRACTION,
)


OUT = ROOT / "results" / "probationary-trait-genesis-v1"
WORKER = ROOT / "scripts" / "run_probationary_trait_genesis_worker.py"
N_REPLICATES = worker.N_REPLICATES
EXPECTED_CHECKPOINTS = N_REPLICATES * (1 + len(CONDITIONS))


def prepare_corpus() -> Path:
    corpus = prepare_tinystories_corpus(ROOT)
    cache = corpus.tokenizer_path.parent
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(corpus.tokenizer_path, OUT / "tokenizer.json")
    manifest = cache / "corpus-manifest.json"
    if manifest.is_file():
        shutil.copy2(manifest, OUT / "corpus-manifest.json")
    arithmetic = prepare_arithmetic_cache(cache, load_tokenizer(corpus.tokenizer_path))
    shutil.copy2(arithmetic["path"], OUT / "arithmetic-manifest.json")
    for experiment, source_name in (
        ("022-emergent-trait-bifurcation", "source-022-decision.json"),
        ("023-online-nonparametric-trait-genesis", "source-023-decision.json"),
    ):
        source = ROOT / "artifacts" / "experiments" / experiment / "decision.json"
        if source.is_file():
            shutil.copy2(source, OUT / source_name)
    return cache


def _worker_complete(replicate: int) -> bool:
    meta = OUT / f"r{replicate}-worker.json"
    summary = OUT / f"r{replicate}-condition-summary.csv"
    windows = OUT / f"r{replicate}-probation-windows.csv"
    if not meta.is_file() or not summary.is_file() or not windows.is_file():
        return False
    payload = json.loads(meta.read_text(encoding="utf-8"))
    return (
        payload.get("format") == "minicells.probationary-trait-genesis-worker.v1"
        and int(payload.get("replicate", -1)) == replicate
    )


def run_workers(cache: Path) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 023b requires a Kaggle GPU accelerator")
    gpu_count = min(2, available)
    missing = [replicate for replicate in range(N_REPLICATES) if not _worker_complete(replicate)]
    if not missing:
        print("reusing complete Experiment 023b workers")
        return gpu_count
    for start in range(0, len(missing), gpu_count):
        group = missing[start : start + gpu_count]
        active = []
        for local_gpu, replicate in enumerate(group):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(local_gpu)
            log = OUT / f"r{replicate}.log"
            handle = log.open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(WORKER),
                "--replicate", str(replicate),
                "--cache-dir", str(cache),
                "--output-dir", str(OUT),
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append((replicate, local_gpu, process, log, handle))
            print(f"started Experiment 023b r{replicate} on physical GPU {local_gpu}")
        failures = []
        for replicate, gpu, process, log, handle in active:
            code = process.wait()
            handle.close()
            print(f"--- Experiment 023b r{replicate} / GPU {gpu} ---")
            print(log.read_text(encoding="utf-8").rstrip())
            if code != 0:
                failures.append(f"r{replicate} exited {code}; see {log}")
        if failures:
            raise RuntimeError("; ".join(failures))
    return gpu_count


def collect() -> dict[str, object]:
    workers = []
    tables: dict[str, list[pd.DataFrame]] = {
        "proposal": [],
        "condition_summary": [],
        "windows": [],
        "learning": [],
        "routing": [],
        "evaluation": [],
        "pretrain": [],
    }
    suffixes = {
        "proposal": "proposal.csv",
        "condition_summary": "condition-summary.csv",
        "windows": "probation-windows.csv",
        "learning": "learning-curve.csv",
        "routing": "routing.csv",
        "evaluation": "evaluation.csv",
        "pretrain": "pretrain.csv",
    }
    for replicate in range(N_REPLICATES):
        workers.append(json.loads((OUT / f"r{replicate}-worker.json").read_text(encoding="utf-8")))
        for key, suffix in suffixes.items():
            path = OUT / f"r{replicate}-{suffix}"
            if path.is_file() and path.stat().st_size > 0:
                tables[key].append(pd.read_csv(path))
    result: dict[str, object] = {"workers": workers}
    for key, frames in tables.items():
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        result[key] = frame
        frame.to_csv(OUT / suffixes[key], index=False)
    return result


def validate_invariants(data: dict[str, object]) -> dict[str, object]:
    summary: pd.DataFrame = data["condition_summary"]
    windows: pd.DataFrame = data["windows"]
    routing: pd.DataFrame = data["routing"]
    proposal: pd.DataFrame = data["proposal"]
    if len(summary) != N_REPLICATES * len(CONDITIONS):
        raise RuntimeError("Experiment 023b condition summary row count mismatch")
    if len(windows) != N_REPLICATES * len(CONDITIONS) * PROBATION_WINDOWS:
        raise RuntimeError("Experiment 023b probation window row count mismatch")
    if len(proposal) != N_REPLICATES * len(CONDITIONS):
        raise RuntimeError("Experiment 023b proposal row count mismatch")

    numeric = windows[
        [
            "parent_prequential_nll",
            "capacity_prequential_nll",
            "geometry_prequential_nll",
            "capacity_net_utility",
            "geometry_net_utility",
            "geometry_advantage",
        ]
    ].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Experiment 023b probation metrics contain non-finite values")

    capacity = routing.loc[routing["arm"] == "capacity-shadow"].copy()
    exact_balance = True
    for _, group in capacity.groupby(["replicate", "condition", "stream_key_posthoc"]):
        counts = group["branch"].value_counts().to_dict()
        exact_balance &= int(counts.get(0, 0)) == int(counts.get(1, 0))
    if not exact_balance:
        raise RuntimeError("capacity shadow did not provide exact per-stream balanced exposure")

    payload = {
        "format": "minicells.probationary-trait-genesis-invariants.v1",
        "finite_probation_metrics": True,
        "capacity_exact_per_stream_balance": True,
        "proposal_rows": int(len(proposal)),
        "probation_window_rows": int(len(windows)),
        "condition_summary_rows": int(len(summary)),
        "candidate_trigger_uses_task_label": False,
        "geometry_routing_uses_task_label": False,
        "commit_decision_uses_task_label": False,
    }
    (OUT / "invariants.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def write_checkpoint_manifest() -> dict[str, object]:
    checkpoint_dir = OUT / "checkpoints"
    files = [
        {"name": path.name, "bytes": path.stat().st_size}
        for path in sorted(checkpoint_dir.glob("*.pt"))
    ] if checkpoint_dir.is_dir() else []
    manifest = {
        "format": "minicells.probationary-trait-genesis-checkpoint-manifest.v1",
        "experiment": "023b",
        "file_count": len(files),
        "expected_file_count": EXPECTED_CHECKPOINTS,
        "files": files,
        "published_model_checkpoints": False,
        "purpose": "Kaggle-local recovery for parent plus four independent probation conditions per replicate",
    }
    (OUT / "checkpoint-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if len(files) != EXPECTED_CHECKPOINTS:
        raise RuntimeError(f"Experiment 023b checkpoint set incomplete: {len(files)}/{EXPECTED_CHECKPOINTS}")
    return manifest


def plot_utility(windows: pd.DataFrame) -> None:
    selected = windows.loc[windows["condition"].isin(CONDITIONS)].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    for condition in CONDITIONS:
        part = selected.loc[selected["condition"] == condition]
        means = part.groupby("window", as_index=False)["geometry_net_utility"].mean()
        ax.plot(means["window"], means["geometry_net_utility"], marker="o", label=condition)
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Probation window")
    ax.set_ylabel("Geometry fork normalized net utility")
    ax.set_title("Prospective probation utility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "probation-utility.png", dpi=180)
    plt.close(fig)


def plot_advantage(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    rows = summary.sort_values(["condition", "replicate"])
    positions = np.arange(len(rows))
    ax.bar(positions, rows["geometry_advantage_last3"].to_numpy(float))
    ax.axhline(GEOMETRY_ADVANTAGE_MIN, linewidth=1)
    ax.set_xticks(
        positions,
        [f"r{int(row.replicate)}\n{row.condition}" for row in rows.itertuples()],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("Geometry - capacity normalized utility")
    ax.set_title("Does differentiated routing earn its extra structure?")
    fig.tight_layout()
    fig.savefig(OUT / "geometry-advantage.png", dpi=180)
    plt.close(fig)


def decide(data: dict[str, object], gpu_count: int, checkpoint_manifest: dict[str, object]) -> dict[str, object]:
    summary: pd.DataFrame = data["condition_summary"]
    story = summary.loc[summary["condition"] == "STORY_ONLY"].sort_values("replicate")
    duplicate = summary.loc[summary["condition"] == "DUPLICATED_STORY"].sort_values("replicate")
    positive = summary.loc[summary["condition"] == "STORY_ARITHMETIC"].sort_values("replicate")
    weak = summary.loc[summary["condition"] == "WEAK_ARITHMETIC"].sort_values("replicate")

    story_reject = int((story["accepted"] == 0).sum())
    duplicate_reject = int((duplicate["accepted"] == 0).sum())
    positive_accept = int((positive["accepted"] == 1).sum())
    positive_strong = int((positive["strong_positive"] == 1).sum())
    positive_identity = int(positive["geometry_identity_pass"].fillna(0).sum())
    positive_routing = int(positive["routing_purity_pass"].fillna(0).sum())
    positive_advantage = int((positive["geometry_advantage_last3"] >= GEOMETRY_ADVANTAGE_MIN).sum())
    positive_utility = int(
        (
            (positive["geometry_sustained_positive"] == 1)
            & (positive["geometry_cumulative_positive"] == 1)
        ).sum()
    )
    weak_accept = int((weak["accepted"] == 1).sum())

    if story_reject < N_REPLICATES or duplicate_reject < N_REPLICATES:
        status = "FALSE_POSITIVE_PROBATIONARY_BIRTH"
    elif positive_accept >= POSITIVE_REPLICATES_MIN and positive_identity < POSITIVE_REPLICATES_MIN:
        status = "UTILITY_WITHOUT_FUNCTIONAL_IDENTITY"
    elif positive_accept >= POSITIVE_REPLICATES_MIN and positive_routing < POSITIVE_REPLICATES_MIN:
        status = "UTILITY_WITHOUT_STABLE_GEOMETRY_ROUTING"
    elif positive_strong >= POSITIVE_REPLICATES_MIN:
        status = "PROBATIONARY_TRAIT_GENESIS_SIGNAL"
    elif positive_utility >= POSITIVE_REPLICATES_MIN and positive_advantage < POSITIVE_REPLICATES_MIN:
        status = "CAPACITY_EXPLAINS_PROBATION_UTILITY"
    else:
        status = "NO_PROBATIONARY_BIRTH_SIGNAL"

    decision = {
        "format": "minicells.probationary-trait-genesis.v1",
        "experiment": "MINI Cells Experiment 023b — Probationary Trait Genesis",
        "question": "Can a temporary task-label-free fork prove on future online data that it deserves to become a persistent trait?",
        "design": {
            "conditions": list(CONDITIONS),
            "arms": list(ARMS),
            "replicates": N_REPLICATES,
            "gpu_count": gpu_count,
            "probation_windows": PROBATION_WINDOWS,
            "structural_cost_fraction": STRUCTURAL_COST_FRACTION,
            "geometry_advantage_min": GEOMETRY_ADVANTAGE_MIN,
            "routing_purity_min": ROUTING_PURITY_MIN,
            "proposal_always_opened": True,
            "proposal_uses_task_label": False,
            "geometry_routing_uses_task_label": False,
            "commit_uses_task_label": False,
            "utility": "(parent prequential NLL - candidate prequential NLL) / abs(parent NLL) - structural_cost",
        },
        "results": {
            "story_only_reject_replicates": story_reject,
            "duplicated_story_reject_replicates": duplicate_reject,
            "story_arithmetic_accept_replicates": positive_accept,
            "story_arithmetic_strong_positive_replicates": positive_strong,
            "story_arithmetic_identity_replicates": positive_identity,
            "story_arithmetic_routing_replicates": positive_routing,
            "story_arithmetic_geometry_advantage_replicates": positive_advantage,
            "story_arithmetic_positive_utility_replicates": positive_utility,
            "weak_arithmetic_accept_replicates": weak_accept,
        },
        "checkpoint_manifest": {
            "file_count": checkpoint_manifest["file_count"],
            "expected_file_count": checkpoint_manifest["expected_file_count"],
            "published_model_checkpoints": False,
        },
        "interpretation": {
            "success": "Strong positive requires STORY_ONLY and DUPLICATED_STORY rejection in 3/3, and STORY+ARITHMETIC probation acceptance with 2x2 functional identity and routing purity >=0.75 in >=2/3. WEAK_ARITHMETIC is discovery-only.",
            "scope": "023b tests birth valuation, not unknown-K growth, local sensing, rewiring, merging, pruning, or inference-time recruitment.",
        },
        "status": status,
    }
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return decision


def write_task_spec() -> None:
    payload = {
        "format": "minicells.probationary-trait-genesis-task.v1",
        "conditions": {
            "STORY_ONLY": "single TinyStories stream",
            "DUPLICATED_STORY": "two independent schedules over the exact same TinyStories token distribution",
            "STORY_ARITHMETIC": "50/50 TinyStories and deterministic arithmetic",
            "WEAK_ARITHMETIC": "90/10 TinyStories and deterministic arithmetic; discovery-only",
        },
        "task_labels_used_by_proposal": False,
        "task_labels_used_by_geometry_routing": False,
        "task_labels_used_by_commit": False,
        "posthoc_labels_used_for_scientific_validation": True,
        "capacity_control_uses_stream_identity_for_exact_balance": True,
    }
    (OUT / "task-spec.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    cache = prepare_corpus()
    gpu_count = run_workers(cache)
    data = collect()
    validate_invariants(data)
    checkpoint_manifest = write_checkpoint_manifest()
    write_task_spec()
    plot_utility(data["windows"])
    plot_advantage(data["condition_summary"])
    decision = decide(data, gpu_count, checkpoint_manifest)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
