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

import run_emergent_trait_bifurcation_worker as worker  # noqa: E402
from minicells.language_conflict_differentiation import (  # noqa: E402
    CALIBRATION_WINDOWS,
    DOMAINS,
    FORK_EPSILON,
    POSTFORK_STEPS,
    PRETRAIN_STEPS,
    prepare_arithmetic_cache,
)
from minicells.language_data import load_tokenizer, prepare_tinystories_corpus  # noqa: E402
from minicells.language_trait_bifurcation import (  # noqa: E402
    ARMS,
    BIFURCATION_AXIS_STABILITY_MIN,
    BIFURCATION_GAIN_MIN,
    BIFURCATION_SPLIT_BALANCE_MIN,
    GEOMETRY_MARGIN_ADVANTAGE_MIN,
    IDENTITY_NORMALIZED_MARGIN_MIN,
    POSITIVE_REPLICATES_MIN,
    ROUTING_PURITY_MIN,
)


OUT = ROOT / "results" / "emergent-trait-bifurcation-v1"
WORKER = ROOT / "scripts" / "run_emergent_trait_bifurcation_worker.py"
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
    source_021 = ROOT / "artifacts" / "experiments" / "021-conflict-driven-differentiation" / "decision.json"
    if source_021.is_file():
        shutil.copy2(source_021, OUT / "source-021-decision.json")
    return cache


def _worker_complete(replicate: int) -> bool:
    meta = OUT / f"r{replicate}-worker.json"
    summary = OUT / f"r{replicate}-arm-summary.csv"
    evaluation = OUT / f"r{replicate}-evaluation.csv"
    windows = OUT / f"r{replicate}-bifurcation-windows.csv"
    if not meta.is_file() or not summary.is_file() or not evaluation.is_file() or not windows.is_file():
        return False
    payload = json.loads(meta.read_text(encoding="utf-8"))
    return payload.get("format") == "minicells.emergent-trait-bifurcation-worker.v1" and int(payload.get("replicate", -1)) == replicate


def run_workers(cache: Path) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 022 requires a Kaggle GPU accelerator")
    gpu_count = min(2, available)
    missing = [replicate for replicate in range(N_REPLICATES) if not _worker_complete(replicate)]
    if not missing:
        print("reusing complete Experiment 022 workers")
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
            print(f"started Experiment 022 r{replicate} on physical GPU {local_gpu}")
        failures = []
        for replicate, gpu, process, log, handle in active:
            code = process.wait()
            handle.close()
            print(f"--- Experiment 022 r{replicate} / GPU {gpu} ---")
            print(log.read_text(encoding="utf-8").rstrip())
            if code != 0:
                failures.append(f"r{replicate} exited {code}; see {log}")
        if failures:
            raise RuntimeError("; ".join(failures))
    return gpu_count


def collect() -> dict[str, object]:
    workers = []
    arm_summary = []
    evaluation = []
    learning = []
    routing = []
    windows = []
    pretrain = []
    for replicate in range(N_REPLICATES):
        workers.append(json.loads((OUT / f"r{replicate}-worker.json").read_text(encoding="utf-8")))
        arm_summary.append(pd.read_csv(OUT / f"r{replicate}-arm-summary.csv"))
        evaluation.append(pd.read_csv(OUT / f"r{replicate}-evaluation.csv"))
        learning.append(pd.read_csv(OUT / f"r{replicate}-learning-curve.csv"))
        windows.append(pd.read_csv(OUT / f"r{replicate}-bifurcation-windows.csv"))
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
        "windows": pd.concat(windows, ignore_index=True),
        "pretrain": pd.concat(pretrain, ignore_index=True),
    }
    for key, filename in (
        ("arm_summary", "arm-summary.csv"),
        ("evaluation", "evaluation.csv"),
        ("learning", "learning-curve.csv"),
        ("routing", "routing.csv"),
        ("windows", "bifurcation-windows.csv"),
        ("pretrain", "pretrain.csv"),
    ):
        result[key].to_csv(OUT / filename, index=False)
    return result


def validate_invariants(data: dict[str, object]) -> dict[str, object]:
    summary: pd.DataFrame = data["arm_summary"]
    windows: pd.DataFrame = data["windows"]
    numeric_windows = windows[["residual_k1", "residual_k2", "bifurcation_gain", "split_balance", "centroid_separation"]].to_numpy(float)
    if not np.isfinite(numeric_windows).all():
        raise RuntimeError("Experiment 022 bifurcation metrics contain non-finite values")

    capacity = summary.loc[summary["arm"] == "stratified-capacity-fork"].copy()
    exact_balance = True
    for row in capacity.itertuples():
        exact_balance &= int(row.story_branch0_updates) == int(row.story_branch1_updates)
        exact_balance &= int(row.arithmetic_branch0_updates) == int(row.arithmetic_branch1_updates)
    if not exact_balance:
        raise RuntimeError("stratified capacity control did not provide exactly matched domain exposure")

    payload = {
        "format": "minicells.emergent-trait-bifurcation-invariants.v1",
        "finite_bifurcation_metrics": True,
        "stratified_capacity_exact_domain_balance": True,
        "expected_calibration_rows": N_REPLICATES * CALIBRATION_WINDOWS,
        "calibration_rows": int(len(windows)),
    }
    if len(windows) != N_REPLICATES * CALIBRATION_WINDOWS:
        raise RuntimeError("Experiment 022 calibration window count mismatch")
    (OUT / "invariants.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_checkpoint_manifest() -> dict[str, object]:
    checkpoint_dir = OUT / "checkpoints"
    files = []
    for path in sorted(checkpoint_dir.glob("*.pt")) if checkpoint_dir.is_dir() else []:
        files.append({"name": path.name, "bytes": path.stat().st_size})
    manifest = {
        "format": "minicells.emergent-trait-bifurcation-checkpoint-manifest.v1",
        "experiment": "022",
        "file_count": len(files),
        "expected_file_count": EXPECTED_CHECKPOINTS,
        "files": files,
        "published_model_checkpoints": False,
        "purpose": "Kaggle-local recovery for parent and arm checkpoints",
    }
    (OUT / "checkpoint-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if len(files) != EXPECTED_CHECKPOINTS:
        raise RuntimeError(f"Experiment 022 checkpoint set incomplete: {len(files)}/{EXPECTED_CHECKPOINTS}")
    return manifest


def plot_bifurcation_windows(windows: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(windows))
    ax.bar(positions, windows["bifurcation_gain"].to_numpy(float))
    ax.axhline(BIFURCATION_GAIN_MIN, linewidth=1)
    ax.set_xticks(positions, [f"r{int(r.replicate)}w{int(r.window)}" for r in windows.itertuples()], rotation=45, ha="right")
    ax.set_ylabel("K=2 residual-fit gain")
    ax.set_title("Experiment 022 gradient-field bifurcation")
    fig.tight_layout()
    fig.savefig(OUT / "bifurcation-gain.png", dpi=180)
    plt.close(fig)


def plot_identity(summary: pd.DataFrame) -> None:
    forked = summary.loc[summary["arm"].isin(["stratified-capacity-fork", "geometry-bifurcation-fork"])].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(forked))
    ax.bar(positions, forked["normalized_identity_margin"].to_numpy(float))
    ax.axhline(IDENTITY_NORMALIZED_MARGIN_MIN, linewidth=1)
    ax.set_xticks(positions, [f"r{int(r.replicate)}\n{r.arm}" for r in forked.itertuples()], rotation=30, ha="right")
    ax.set_ylabel("Normalized identity margin")
    ax.set_title("Functional identity: stratified capacity vs geometry")
    fig.tight_layout()
    fig.savefig(OUT / "identity-margin.png", dpi=180)
    plt.close(fig)


def plot_utility_matrix(evaluation: pd.DataFrame, arm: str, filename: str) -> None:
    part = evaluation.loc[evaluation["arm"] == arm]
    matrix = part.groupby(["domain", "branch"], as_index=False)["utility"].mean().pivot(index="domain", columns="branch", values="utility")
    matrix = matrix.reindex(index=DOMAINS)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix.to_numpy(float), aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), [f"branch {int(value)}" for value in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Forked phenotype")
    ax.set_ylabel("Ability")
    ax.set_title(f"{arm}: parent NLL improvement")
    fig.colorbar(image, ax=ax, label="Utility")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def plot_routing(routing: pd.DataFrame) -> None:
    if routing.empty:
        return
    selected = routing.loc[routing["arm"] == "geometry-bifurcation-fork"].copy()
    score = pd.to_numeric(selected["cluster_distance_score"], errors="coerce")
    selected = selected.loc[np.isfinite(score)].copy()
    if selected.empty:
        return
    selected["score"] = pd.to_numeric(selected["cluster_distance_score"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for domain in DOMAINS:
        values = selected.loc[selected["domain_posthoc"] == domain, "score"].to_numpy(float)
        ax.hist(values, bins=30, alpha=0.5, label=domain)
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("K=2 centroid distance score")
    ax.set_ylabel("Microbatches")
    ax.set_title("Task-label-free geometry routing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "routing-clusters.png", dpi=180)
    plt.close(fig)


def decide(data: dict[str, object], gpu_count: int, checkpoint_manifest: dict[str, object]) -> dict[str, object]:
    summary: pd.DataFrame = data["arm_summary"]
    windows: pd.DataFrame = data["windows"]
    workers: list[dict[str, object]] = data["workers"]

    persistent_replicates = sum(int(payload["bifurcation"]["persistent_multimodality"]) for payload in workers)
    capacity = summary.loc[summary["arm"] == "stratified-capacity-fork"].sort_values("replicate")
    geometry = summary.loc[summary["arm"] == "geometry-bifurcation-fork"].sort_values("replicate")
    capacity_identity = int(capacity["identity_pass"].fillna(0).sum())
    geometry_identity = int(geometry["identity_pass"].fillna(0).sum())
    geometry_routing = int(geometry["routing_purity_pass"].fillna(0).sum())
    advantages = geometry["normalized_identity_margin"].to_numpy(float) - capacity["normalized_identity_margin"].to_numpy(float)
    advantage_replicates = int((advantages >= GEOMETRY_MARGIN_ADVANTAGE_MIN).sum())

    if capacity_identity >= POSITIVE_REPLICATES_MIN:
        status = "STRATIFIED_CAPACITY_ALONE_SPECIALIZES"
    elif (
        persistent_replicates >= POSITIVE_REPLICATES_MIN
        and geometry_identity >= POSITIVE_REPLICATES_MIN
        and geometry_routing >= POSITIVE_REPLICATES_MIN
        and advantage_replicates >= POSITIVE_REPLICATES_MIN
    ):
        status = "EMERGENT_TRAIT_BIFURCATION_SIGNAL"
    elif persistent_replicates >= POSITIVE_REPLICATES_MIN:
        status = "MULTIMODALITY_WITHOUT_FUNCTIONAL_BIFURCATION"
    elif geometry_identity >= POSITIVE_REPLICATES_MIN:
        status = "FUNCTIONAL_BIFURCATION_WITHOUT_PERSISTENT_MULTIMODALITY"
    else:
        status = "NO_PERSISTENT_GRADIENT_MULTIMODALITY"

    decision = {
        "format": "minicells.emergent-trait-bifurcation.v1",
        "experiment": "MINI Cells Experiment 022 — Emergent Trait Bifurcation",
        "question": "Can a persistent task-label-free two-mode phenotype-gradient field cause functional trait bifurcation beyond a strictly stratified capacity control?",
        "design": {
            "domains": list(DOMAINS),
            "arms": list(ARMS),
            "replicates": N_REPLICATES,
            "gpu_count": gpu_count,
            "story_pretraining_steps": PRETRAIN_STEPS,
            "postfork_steps": POSTFORK_STEPS,
            "fork_epsilon": FORK_EPSILON,
            "calibration_windows": CALIBRATION_WINDOWS,
            "trigger_uses_task_label": False,
            "geometry_routing_uses_task_label": False,
            "capacity_control_uses_labels_only_to_enforce_exact_domain_balance": True,
            "bifurcation_gain_min": BIFURCATION_GAIN_MIN,
            "bifurcation_split_balance_min": BIFURCATION_SPLIT_BALANCE_MIN,
            "bifurcation_axis_stability_min": BIFURCATION_AXIS_STABILITY_MIN,
            "identity_normalized_margin_min": IDENTITY_NORMALIZED_MARGIN_MIN,
            "routing_purity_min": ROUTING_PURITY_MIN,
            "geometry_margin_advantage_min": GEOMETRY_MARGIN_ADVANTAGE_MIN,
            "positive_replicates_min": POSITIVE_REPLICATES_MIN,
        },
        "results": {
            "persistent_multimodality_replicates": persistent_replicates,
            "stratified_capacity_identity_pass_replicates": capacity_identity,
            "geometry_identity_pass_replicates": geometry_identity,
            "geometry_routing_pass_replicates": geometry_routing,
            "geometry_margin_advantage_replicates": advantage_replicates,
            "mean_capacity_identity_margin": float(capacity["normalized_identity_margin"].mean()),
            "mean_geometry_identity_margin": float(geometry["normalized_identity_margin"].mean()),
            "mean_geometry_minus_capacity_margin": float(np.mean(advantages)),
            "mean_geometry_routing_purity": float(geometry["routing_purity_posthoc"].mean()),
            "mean_unified_combined_nll": float(summary.loc[summary["arm"] == "unified", "oracle_combined_nll"].mean()),
            "mean_geometry_oracle_combined_nll": float(geometry["oracle_combined_nll"].mean()),
            "bifurcation_windows_passed": int(windows["window_bifurcation_pass"].sum()),
            "bifurcation_windows_total": int(len(windows)),
        },
        "checkpoint_manifest": {
            "file_count": checkpoint_manifest["file_count"],
            "expected_file_count": checkpoint_manifest["expected_file_count"],
            "published_model_checkpoints": False,
        },
        "interpretation": {
            "success": "A positive result requires persistent K=2 gradient-field structure, functional identity and posthoc label purity in >=2/3 replicates, a >=0.05 identity-margin advantage over stratified capacity in >=2/3, and failure of the strictly balanced capacity control to explain specialization.",
            "capacity_control": "The negative control deliberately uses benchmark labels only to guarantee that each child receives exactly the same count of STORY and ARITHMETIC updates. It is not a proposed routing mechanism.",
            "scope": "022 tests developmental bifurcation in a minimal fixed-topology TextNCA phenotype substrate. It does not test inference-time recruitment or autonomous rewiring.",
        },
        "status": status,
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return decision


def write_task_spec() -> None:
    payload = {
        "format": "minicells.emergent-trait-bifurcation-task.v1",
        "gradient_fit": "normalized phenotype gradients; deterministic K=1 vs K=2 residual comparison",
        "persistent_multimodality": "at least two of three windows pass gain/balance and pairwise axis stability >= threshold; combined field also passes",
        "geometry_routing": "nearest unlabeled K=2 gradient centroid",
        "stratified_capacity_control": "each child receives exactly matched counts of STORY and ARITHMETIC updates",
        "task_labels": "used only to construct the benchmark, enforce the negative-control stratification, and perform posthoc validation",
    }
    (OUT / "task-spec.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    cache = prepare_corpus()
    gpu_count = run_workers(cache)
    data = collect()
    validate_invariants(data)
    checkpoint_manifest = write_checkpoint_manifest()
    write_task_spec()
    plot_bifurcation_windows(data["windows"])
    plot_identity(data["arm_summary"])
    plot_utility_matrix(data["evaluation"], "stratified-capacity-fork", "stratified-capacity-utility-matrix.png")
    plot_utility_matrix(data["evaluation"], "geometry-bifurcation-fork", "geometry-bifurcation-utility-matrix.png")
    plot_routing(data["routing"])
    decision = decide(data, gpu_count, checkpoint_manifest)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
