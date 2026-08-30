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
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_conflict_driven_differentiation_worker as worker  # noqa: E402
from minicells.language_conflict_differentiation import (  # noqa: E402
    ARMS,
    CALIBRATION_WINDOWS,
    DIFFERENTIATION_REPLICATES_MIN,
    DOMAINS,
    IDENTITY_NORMALIZED_MARGIN_MIN,
    ROUTING_PURITY_MIN,
    prepare_arithmetic_cache,
)
from minicells.language_data import load_tokenizer, prepare_tinystories_corpus  # noqa: E402


OUT = ROOT / "results" / "conflict-driven-differentiation-v1"
WORKER = ROOT / "scripts" / "run_conflict_driven_differentiation_worker.py"
N_REPLICATES = worker.N_REPLICATES
EXPECTED_CHECKPOINTS = N_REPLICATES * (1 + len(ARMS))


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
    return cache


def _worker_complete(replicate: int) -> bool:
    meta = OUT / f"r{replicate}-worker.json"
    summary = OUT / f"r{replicate}-arm-summary.csv"
    evaluation = OUT / f"r{replicate}-evaluation.csv"
    if not meta.is_file() or not summary.is_file() or not evaluation.is_file():
        return False
    payload = json.loads(meta.read_text(encoding="utf-8"))
    return payload.get("format") == "minicells.conflict-driven-differentiation-worker.v1" and int(payload.get("replicate", -1)) == replicate


def run_workers(cache: Path) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 021 requires a Kaggle GPU accelerator")
    gpu_count = min(2, available)
    missing = [replicate for replicate in range(N_REPLICATES) if not _worker_complete(replicate)]
    if not missing:
        print("reusing complete Experiment 021 workers")
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
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
            active.append((replicate, local_gpu, process, log, handle))
            print(f"started Experiment 021 r{replicate} on physical GPU {local_gpu}")
        failures = []
        for replicate, gpu, process, log, handle in active:
            code = process.wait()
            handle.close()
            print(f"--- Experiment 021 r{replicate} / GPU {gpu} ---")
            print(log.read_text(encoding="utf-8").rstrip())
            if code != 0:
                failures.append(f"r{replicate} exited {code}; see {log}")
        if failures:
            raise RuntimeError("; ".join(failures))
    return gpu_count


def collect() -> dict[str, pd.DataFrame | list[dict[str, object]]]:
    workers = []
    arm_summary = []
    evaluation = []
    learning = []
    routing = []
    conflict = []
    pretrain = []
    for replicate in range(N_REPLICATES):
        workers.append(json.loads((OUT / f"r{replicate}-worker.json").read_text(encoding="utf-8")))
        arm_summary.append(pd.read_csv(OUT / f"r{replicate}-arm-summary.csv"))
        evaluation.append(pd.read_csv(OUT / f"r{replicate}-evaluation.csv"))
        learning.append(pd.read_csv(OUT / f"r{replicate}-learning-curve.csv"))
        conflict.append(pd.read_csv(OUT / f"r{replicate}-conflict-windows.csv"))
        pretrain.append(pd.read_csv(OUT / f"r{replicate}-pretrain.csv"))
        route_path = OUT / f"r{replicate}-routing.csv"
        if route_path.is_file() and route_path.stat().st_size > 0:
            routing.append(pd.read_csv(route_path))
    result = {
        "workers": workers,
        "arm_summary": pd.concat(arm_summary, ignore_index=True),
        "evaluation": pd.concat(evaluation, ignore_index=True),
        "learning": pd.concat(learning, ignore_index=True),
        "routing": pd.concat(routing, ignore_index=True) if routing else pd.DataFrame(),
        "conflict": pd.concat(conflict, ignore_index=True),
        "pretrain": pd.concat(pretrain, ignore_index=True),
    }
    for key, filename in (
        ("arm_summary", "arm-summary.csv"),
        ("evaluation", "evaluation.csv"),
        ("learning", "learning-curve.csv"),
        ("routing", "routing.csv"),
        ("conflict", "conflict-windows.csv"),
        ("pretrain", "pretrain.csv"),
    ):
        result[key].to_csv(OUT / filename, index=False)
    return result


def write_checkpoint_manifest() -> dict[str, object]:
    checkpoint_dir = OUT / "checkpoints"
    files = []
    for path in sorted(checkpoint_dir.glob("*.pt")) if checkpoint_dir.is_dir() else []:
        files.append({"name": path.name, "bytes": path.stat().st_size})
    manifest = {
        "format": "minicells.conflict-driven-differentiation-checkpoint-manifest.v1",
        "experiment": "021",
        "file_count": len(files),
        "expected_file_count": EXPECTED_CHECKPOINTS,
        "files": files,
        "published_model_checkpoints": False,
        "purpose": "Kaggle-local recovery for parent and arm checkpoints",
    }
    (OUT / "checkpoint-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if len(files) != EXPECTED_CHECKPOINTS:
        raise RuntimeError(f"Experiment 021 checkpoint set incomplete: {len(files)}/{EXPECTED_CHECKPOINTS}")
    return manifest


def plot_identity(summary: pd.DataFrame) -> None:
    forked = summary.loc[summary["arm"].isin(["capacity-fork", "differentiation-fork"])].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(forked))
    ax.bar(positions, forked["normalized_identity_margin"].to_numpy(float))
    ax.axhline(IDENTITY_NORMALIZED_MARGIN_MIN, linewidth=1)
    ax.set_xticks(positions, [f"r{int(r.replicate)}\n{r.arm}" for r in forked.itertuples()], rotation=30, ha="right")
    ax.set_ylabel("Normalized identity margin")
    ax.set_title("Experiment 021 functional differentiation")
    fig.tight_layout()
    fig.savefig(OUT / "identity-margin.png", dpi=180)
    plt.close(fig)


def plot_conflict(conflict: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(conflict))
    ax.bar(positions - 0.18, conflict["interference_story_to_math"].to_numpy(float), width=0.36, label="story update -> math")
    ax.bar(positions + 0.18, conflict["interference_math_to_story"].to_numpy(float), width=0.36, label="math update -> story")
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(positions, [f"r{int(r.replicate)}w{int(r.window)}" for r in conflict.itertuples()], rotation=45, ha="right")
    ax.set_ylabel("Counterfactual NLL harm")
    ax.set_title("Dual-ability cross-interference")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "conflict-interference.png", dpi=180)
    plt.close(fig)


def plot_utility_matrix(evaluation: pd.DataFrame, arm: str, filename: str) -> None:
    part = evaluation.loc[evaluation["arm"] == arm]
    matrix = part.groupby(["domain", "branch"], as_index=False)["utility"].mean().pivot(index="domain", columns="branch", values="utility")
    matrix = matrix.reindex(index=DOMAINS)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix.to_numpy(float), aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), [f"branch {int(value)}" for value in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Forked cellular population")
    ax.set_ylabel("Ability")
    ax.set_title(f"{arm}: parent NLL improvement")
    fig.colorbar(image, ax=ax, label="Utility")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def plot_routing(routing: pd.DataFrame) -> None:
    if routing.empty:
        return
    selected = routing.loc[routing["arm"] == "differentiation-fork"].copy()
    selected = selected.loc[np.isfinite(pd.to_numeric(selected["projection_score"], errors="coerce"))]
    if selected.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for domain in DOMAINS:
        values = selected.loc[selected["domain_posthoc"] == domain, "projection_score"].to_numpy(float)
        ax.hist(values, bins=30, alpha=0.5, label=domain)
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Unlabeled gradient projection score")
    ax.set_ylabel("Microbatches")
    ax.set_title("Differentiation routing discovered from gradient geometry")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "routing-projections.png", dpi=180)
    plt.close(fig)


def decide(data: dict[str, pd.DataFrame | list[dict[str, object]]], gpu_count: int, checkpoint_manifest: dict[str, object]) -> dict[str, object]:
    summary = data["arm_summary"]
    conflict = data["conflict"]
    workers = data["workers"]
    conflict_replicates = sum(int(worker_payload["conflict"]["conflict_confirmed"]) for worker_payload in workers)
    differentiation = summary.loc[summary["arm"] == "differentiation-fork"]
    capacity = summary.loc[summary["arm"] == "capacity-fork"]
    diff_identity = int(differentiation["identity_pass"].fillna(0).sum())
    capacity_identity = int(capacity["identity_pass"].fillna(0).sum())
    diff_routing = int(differentiation["routing_purity_pass"].fillna(0).sum())
    if capacity_identity >= DIFFERENTIATION_REPLICATES_MIN:
        status = "CAPACITY_ALONE_SPECIALIZES"
    elif conflict_replicates >= DIFFERENTIATION_REPLICATES_MIN and diff_identity >= DIFFERENTIATION_REPLICATES_MIN and diff_routing >= DIFFERENTIATION_REPLICATES_MIN:
        status = "CONFLICT_DRIVEN_DIFFERENTIATION_SIGNAL"
    elif conflict_replicates >= DIFFERENTIATION_REPLICATES_MIN:
        status = "CONFLICT_WITHOUT_DIFFERENTIATION"
    elif diff_identity >= DIFFERENTIATION_REPLICATES_MIN:
        status = "DIFFERENTIATION_WITHOUT_CONFIRMED_CONFLICT"
    else:
        status = "NO_DUAL_ABILITY_CONFLICT"
    decision = {
        "format": "minicells.conflict-driven-differentiation.v1",
        "experiment": "MINI Cells Experiment 021 — Conflict-Driven Differentiation",
        "question": "Can persistent story/arithmetic learning conflict cause two shared-genome TextNCA populations to become functionally distinct when learning pressure is split by unlabeled phenotype-gradient geometry?",
        "design": {
            "domains": list(DOMAINS),
            "arms": list(ARMS),
            "replicates": N_REPLICATES,
            "gpu_count": gpu_count,
            "story_pretraining_steps": worker.PRETRAIN_STEPS,
            "postfork_steps": worker.POSTFORK_STEPS,
            "fork_epsilon": worker.FORK_EPSILON,
            "calibration_windows": CALIBRATION_WINDOWS,
            "routing_uses_task_label": False,
            "conflict_axis_uses_task_label": False,
            "task_labels_used_for_posthoc_validation": True,
            "shared_genome_after_fork": True,
            "capacity_and_differentiation_same_symmetry_break": True,
            "capacity_routing": "task-agnostic deterministic 50/50",
            "one_child_update_per_microbatch_in_both_fork_controls": True,
            "identity_normalized_margin_min": IDENTITY_NORMALIZED_MARGIN_MIN,
            "routing_purity_min": ROUTING_PURITY_MIN,
            "positive_replicates_min": DIFFERENTIATION_REPLICATES_MIN,
        },
        "results": {
            "conflict_confirmed_replicates": conflict_replicates,
            "capacity_identity_pass_replicates": capacity_identity,
            "differentiation_identity_pass_replicates": diff_identity,
            "differentiation_routing_pass_replicates": diff_routing,
            "mean_capacity_identity_margin": float(capacity["normalized_identity_margin"].mean()),
            "mean_differentiation_identity_margin": float(differentiation["normalized_identity_margin"].mean()),
            "mean_differentiation_routing_purity": float(differentiation["routing_purity_posthoc"].mean()),
            "conflict_windows_passed": int(conflict["window_conflict_pass"].sum()),
            "conflict_windows_total": len(conflict),
        },
        "checkpoint_manifest": {
            "file_count": checkpoint_manifest["file_count"],
            "expected_file_count": checkpoint_manifest["expected_file_count"],
            "published_model_checkpoints": False,
        },
        "interpretation": {
            "success": "A positive result requires conflict, posthoc evidence that unlabeled gradient geometry separates the abilities, functional branch identity in >=2/3 replicates, and failure of the capacity-only control to explain the effect.",
            "capacity_control": "Capacity fork uses the same symmetry break and the same per-child update budget as differentiation fork, but routes by a deterministic task-agnostic 50/50 schedule rather than conflict geometry.",
            "scope": "021 proves or falsifies conflict->division->differentiation in a minimal fixed-topology TextNCA substrate. It does not yet test autonomous topology rewiring or inference-time recruitment.",
        },
        "status": status,
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return decision


def write_task_spec() -> None:
    payload = {
        "format": "minicells.conflict-driven-differentiation-task.v1",
        "story_domain": "TinyStories token stream",
        "arithmetic_domain": "deterministic synthetic ADD/SUB/MUL/solve-x text encoded by the same TinyStories byte-level BPE",
        "pre_fork": "one TextNCA parent trained on story only",
        "fork_site": "population phenotype injected before the final shared NCA stage",
        "capacity_fork": "two symmetry-broken child phenotypes; each microbatch updates one child selected by a deterministic task-agnostic 50/50 schedule",
        "differentiation_fork": "same child initialization and update budget; each microbatch updates only the child selected by the sign of its unlabeled gradient projection onto the fixed conflict axis",
        "forbidden_routing_inputs": ["task label", "domain id", "expert id", "learned central router"],
    }
    (OUT / "task-spec.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    cache = prepare_corpus()
    gpu_count = run_workers(cache)
    data = collect()
    checkpoint_manifest = write_checkpoint_manifest()
    write_task_spec()
    plot_identity(data["arm_summary"])
    plot_conflict(data["conflict"])
    plot_utility_matrix(data["evaluation"], "capacity-fork", "capacity-utility-matrix.png")
    plot_utility_matrix(data["evaluation"], "differentiation-fork", "differentiation-utility-matrix.png")
    plot_routing(data["routing"])
    decision = decide(data, gpu_count, checkpoint_manifest)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
