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

import run_sequential_probationary_genesis_worker as worker  # noqa: E402
from minicells.language_conflict_differentiation import prepare_arithmetic_cache  # noqa: E402
from minicells.language_data import load_tokenizer, prepare_tinystories_corpus  # noqa: E402
from minicells.language_online_trait_genesis import prepare_transform_cache  # noqa: E402
from minicells.language_probationary_trait_genesis import (  # noqa: E402
    GEOMETRY_ADVANTAGE_MIN,
    ROUTING_PURITY_MIN,
    STRUCTURAL_COST_FRACTION,
)
from minicells.language_sequential_probationary_genesis import (  # noqa: E402
    MAX_TRAITS,
    POSITIVE_REPLICATES_MIN,
    PROBATION_WINDOWS,
    STAGES,
    aggregate_status,
    classify_replicate,
    expected_trajectory,
    stage_spec,
)


OUT = ROOT / "results" / "sequential-probationary-genesis-v1"
WORKER = ROOT / "scripts" / "run_sequential_probationary_genesis_worker.py"
N_REPLICATES = worker.N_REPLICATES
EXPECTED_CHECKPOINTS = N_REPLICATES * (1 + len(STAGES))


def prepare_corpus() -> Path:
    corpus = prepare_tinystories_corpus(ROOT)
    cache = corpus.tokenizer_path.parent
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(corpus.tokenizer_path, OUT / "tokenizer.json")
    manifest = cache / "corpus-manifest.json"
    if manifest.is_file():
        shutil.copy2(manifest, OUT / "corpus-manifest.json")
    tokenizer = load_tokenizer(corpus.tokenizer_path)
    arithmetic = prepare_arithmetic_cache(cache, tokenizer)
    transform = prepare_transform_cache(cache, tokenizer)
    shutil.copy2(arithmetic["path"], OUT / "arithmetic-manifest.json")
    shutil.copy2(transform["path"], OUT / "transform-manifest.json")

    provenance = (
        ("022-emergent-trait-bifurcation", "source-022-decision.json", "EMERGENT_TRAIT_BIFURCATION_SIGNAL"),
        ("023-online-nonparametric-trait-genesis", "source-023-decision.json", "NO_ONLINE_TRAIT_GENESIS"),
        ("023b-probationary-trait-genesis", "source-023b-decision.json", "PROBATIONARY_TRAIT_GENESIS_SIGNAL"),
    )
    for experiment, destination, expected_status in provenance:
        source = ROOT / "artifacts" / "experiments" / experiment / "decision.json"
        if not source.is_file():
            raise FileNotFoundError(f"required provenance missing: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("status") != expected_status:
            raise RuntimeError(
                f"unexpected provenance status for {experiment}: {payload.get('status')!r}; expected {expected_status!r}"
            )
        shutil.copy2(source, OUT / destination)
    return cache


def _worker_complete(replicate: int) -> bool:
    meta = OUT / f"r{replicate}-worker.json"
    summary = OUT / f"r{replicate}-stage-summary.csv"
    trajectory = OUT / f"r{replicate}-trajectory.csv"
    if not meta.is_file() or not summary.is_file() or not trajectory.is_file():
        return False
    payload = json.loads(meta.read_text(encoding="utf-8"))
    return (
        payload.get("format") == "minicells.sequential-probationary-genesis-worker.v1"
        and int(payload.get("replicate", -1)) == replicate
    )


def run_workers(cache: Path) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 024 requires a Kaggle GPU accelerator")
    gpu_count = min(2, available)
    missing = [replicate for replicate in range(N_REPLICATES) if not _worker_complete(replicate)]
    if not missing:
        print("reusing complete Experiment 024 workers")
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
            print(f"started Experiment 024 r{replicate} on physical GPU {local_gpu}")
        failures = []
        for replicate, gpu, process, log, handle in active:
            code = process.wait()
            handle.close()
            print(f"--- Experiment 024 r{replicate} / GPU {gpu} ---")
            print(log.read_text(encoding="utf-8").rstrip())
            if code != 0:
                failures.append(f"r{replicate} exited {code}; see {log}")
        if failures:
            raise RuntimeError("; ".join(failures))
    return gpu_count


def collect() -> dict[str, object]:
    workers = []
    suffixes = {
        "proposal": "proposal.csv",
        "stage_summary": "stage-summary.csv",
        "windows": "probation-windows.csv",
        "learning": "learning-curve.csv",
        "routing": "routing.csv",
        "evaluation": "evaluation.csv",
        "trajectory": "trajectory.csv",
        "pretrain": "pretrain.csv",
    }
    tables: dict[str, list[pd.DataFrame]] = {key: [] for key in suffixes}
    for replicate in range(N_REPLICATES):
        workers.append(json.loads((OUT / f"r{replicate}-worker.json").read_text(encoding="utf-8")))
        for key, suffix in suffixes.items():
            path = OUT / f"r{replicate}-{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
            tables[key].append(pd.read_csv(path))
    result: dict[str, object] = {"workers": workers}
    for key, frames in tables.items():
        frame = pd.concat(frames, ignore_index=True)
        result[key] = frame
        frame.to_csv(OUT / suffixes[key], index=False)
    return result


def validate_invariants(data: dict[str, object]) -> dict[str, object]:
    summary: pd.DataFrame = data["stage_summary"]
    windows: pd.DataFrame = data["windows"]
    proposal: pd.DataFrame = data["proposal"]
    routing: pd.DataFrame = data["routing"]
    trajectory: pd.DataFrame = data["trajectory"]
    if len(summary) != N_REPLICATES * len(STAGES):
        raise RuntimeError("Experiment 024 stage summary row count mismatch")
    if len(proposal) != N_REPLICATES * len(STAGES):
        raise RuntimeError("Experiment 024 proposal row count mismatch")
    if len(windows) != N_REPLICATES * len(STAGES) * PROBATION_WINDOWS:
        raise RuntimeError("Experiment 024 probation window row count mismatch")
    if len(trajectory) != N_REPLICATES * (len(STAGES) + 1):
        raise RuntimeError("Experiment 024 trajectory row count mismatch")

    numeric = windows[
        [
            "incumbent_prequential_nll",
            "capacity_prequential_nll",
            "geometry_prequential_nll",
            "capacity_net_utility",
            "geometry_net_utility",
            "geometry_advantage",
        ]
    ].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Experiment 024 probation metrics contain non-finite values")

    incumbent = routing.loc[routing["arm"] == "incumbent", [
        "replicate", "stage", "step", "stream_key_posthoc", "branch", "candidate_parent_branch"
    ]].rename(columns={"branch": "incumbent_branch"})
    capacity = routing.loc[routing["arm"] == "capacity-shadow", [
        "replicate", "stage", "step", "stream_key_posthoc", "branch"
    ]].rename(columns={"branch": "capacity_branch"})
    paired = incumbent.merge(
        capacity,
        on=["replicate", "stage", "step", "stream_key_posthoc"],
        validate="one_to_one",
    )
    wrong_unsplit = paired.loc[
        (paired["incumbent_branch"] != paired["candidate_parent_branch"])
        & (paired["capacity_branch"] != paired["incumbent_branch"])
    ]
    if not wrong_unsplit.empty:
        raise RuntimeError("capacity shadow changed a non-parent incumbent route")

    split = paired.loc[paired["incumbent_branch"] == paired["candidate_parent_branch"]].copy()
    split_balance_ok = True
    for _, group in split.groupby(["replicate", "stage", "stream_key_posthoc"]):
        parent = int(group["candidate_parent_branch"].iloc[0])
        branches = sorted(int(value) for value in group["capacity_branch"].unique())
        if any(value not in (parent, parent + 1, 2, 3) for value in branches):
            split_balance_ok = False
        counts = group["capacity_branch"].value_counts().to_dict()
        # The newborn branch is active_k at proposal time and can be inferred as
        # the largest branch used only by the split control. Alternation permits
        # one extra sample when the routed subset has odd cardinality.
        values = sorted((int(v) for v in counts.values()), reverse=True)
        if len(values) >= 2 and values[0] - values[1] > 1:
            split_balance_ok = False
    if not split_balance_ok:
        raise RuntimeError("capacity shadow split exposure was not locally balanced")

    payload = {
        "format": "minicells.sequential-probationary-genesis-invariants.v1",
        "finite_probation_metrics": True,
        "capacity_preserves_nonparent_routes": True,
        "capacity_parent_split_local_balance": True,
        "proposal_rows": int(len(proposal)),
        "stage_summary_rows": int(len(summary)),
        "probation_window_rows": int(len(windows)),
        "trajectory_rows": int(len(trajectory)),
        "expected_trajectory": list(expected_trajectory()),
        "proposal_uses_task_label": False,
        "geometry_routing_uses_task_label": False,
        "commit_uses_task_label": False,
    }
    (OUT / "invariants.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def write_checkpoint_manifest() -> dict[str, object]:
    checkpoint_dir = OUT / "checkpoints"
    files = [
        {"name": path.name, "bytes": path.stat().st_size}
        for path in sorted(checkpoint_dir.glob("*.pt"))
    ] if checkpoint_dir.is_dir() else []
    manifest = {
        "format": "minicells.sequential-probationary-genesis-checkpoint-manifest.v1",
        "experiment": "024",
        "file_count": len(files),
        "expected_file_count": EXPECTED_CHECKPOINTS,
        "files": files,
        "published_model_checkpoints": False,
        "purpose": "Kaggle-local recovery for one parent plus five sequential stage checkpoints per replicate",
    }
    (OUT / "checkpoint-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if len(files) != EXPECTED_CHECKPOINTS:
        raise RuntimeError(f"Experiment 024 checkpoint set incomplete: {len(files)}/{EXPECTED_CHECKPOINTS}")
    return manifest


def plot_trajectory(trajectory: pd.DataFrame) -> None:
    order = ["START", *STAGES]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for replicate, part in trajectory.groupby("replicate"):
        part = part.copy()
        part["x"] = part["point"].map({name: index for index, name in enumerate(order)})
        ax.plot(part["x"], part["active_k"], marker="o", label=f"r{int(replicate)}")
    ax.plot(range(len(order)), expected_trajectory(), linestyle="--", linewidth=1.2, label="preregistered")
    ax.set_xticks(range(len(order)), [name.split("_")[0] for name in order])
    ax.set_yticks(range(1, MAX_TRAITS + 1))
    ax.set_ylabel("Committed trait count K")
    ax.set_title("Sequential probationary trait genesis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "trait-count-trajectory.png", dpi=180)
    plt.close(fig)


def plot_utility(windows: pd.DataFrame) -> None:
    means = windows.groupby(["stage", "window"], as_index=False)["geometry_net_utility"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    for stage in STAGES:
        part = means.loc[means["stage"] == stage]
        ax.plot(part["window"], part["geometry_net_utility"], marker="o", label=stage)
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Probation window")
    ax.set_ylabel("Geometry candidate normalized net utility")
    ax.set_title("Does each proposed newborn earn its structural cost?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "stage-probation-utility.png", dpi=180)
    plt.close(fig)


def plot_identity(summary: pd.DataFrame) -> None:
    rows = summary.sort_values(["stage", "replicate"])
    positions = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(positions, rows["identity_margin"].fillna(0).to_numpy(float))
    ax.set_xticks(
        positions,
        [f"r{int(row.replicate)}\n{str(row.stage).split('_')[0]}" for row in rows.itertuples()],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("Committed functional identity margin")
    ax.set_title("Functional identity across one continuous organism")
    fig.tight_layout()
    fig.savefig(OUT / "stage-identity.png", dpi=180)
    plt.close(fig)


def decide(data: dict[str, object], gpu_count: int, checkpoint_manifest: dict[str, object]) -> dict[str, object]:
    summary: pd.DataFrame = data["stage_summary"]
    per_replicate = []
    for replicate in range(N_REPLICATES):
        rows = summary.loc[summary["replicate"] == replicate].sort_values(
            "stage", key=lambda values: values.map({stage: i for i, stage in enumerate(STAGES)})
        )
        result = classify_replicate(rows.to_dict(orient="records"))
        result["replicate"] = replicate
        per_replicate.append(result)
    status = aggregate_status(per_replicate)
    results = {
        "story_null_reject_replicates": sum(int(row["story_null_reject"]) for row in per_replicate),
        "arithmetic_birth_replicates": sum(int(row["arithmetic_birth"]) for row in per_replicate),
        "duplicate_reject_replicates": sum(int(row["duplicate_reject"]) for row in per_replicate),
        "weak_transform_reject_replicates": sum(int(row["weak_transform_reject"]) for row in per_replicate),
        "transform_birth_replicates": sum(int(row["transform_birth"]) for row in per_replicate),
        "final_k3_replicates": sum(int(row["final_k"] == 3) for row in per_replicate),
        "per_replicate": per_replicate,
    }
    pd.DataFrame(per_replicate).to_csv(OUT / "replicate-decision.csv", index=False)
    decision = {
        "format": "minicells.sequential-probationary-genesis.v1",
        "experiment": "MINI Cells Experiment 024 — Sequential Probationary Genesis",
        "question": "Can one continuous TextNCA organism repeatedly reject or commit temporary task-label-free trait births and thereby grow from one to two to three persistent functional traits?",
        "design": {
            "replicates": N_REPLICATES,
            "gpu_count": gpu_count,
            "stages": [
                {
                    "stage": stage,
                    "expected_outcome": stage_spec(stage).expected_outcome,
                    "expected_start_k": stage_spec(stage).expected_start_k,
                    "expected_end_k": stage_spec(stage).expected_end_k,
                    "counts": stage_spec(stage).counts,
                }
                for stage in STAGES
            ],
            "expected_trajectory": list(expected_trajectory()),
            "structural_cost_fraction": STRUCTURAL_COST_FRACTION,
            "geometry_advantage_min": GEOMETRY_ADVANTAGE_MIN,
            "routing_purity_min": ROUTING_PURITY_MIN,
            "positive_replicates_min": POSITIVE_REPLICATES_MIN,
            "proposal_always_opened": True,
            "proposal_uses_task_label": False,
            "geometry_routing_uses_task_label": False,
            "commit_uses_task_label": False,
            "sensor": "fixed pretrained parent phenotype gradient",
            "sequential_commit_semantics": "accepted stage continues from geometry-shadow model+Adam state; rejected stage continues from incumbent model+Adam state",
        },
        "results": results,
        "checkpoint_manifest": {
            "file_count": checkpoint_manifest["file_count"],
            "expected_file_count": checkpoint_manifest["expected_file_count"],
            "published_model_checkpoints": False,
        },
        "interpretation": {
            "success": "Strong positive requires Story null rejection in 3/3; a 1->2 Story/Arithmetic birth with functional identity and routing purity >=0.75 in >=2/3; duplicate-Arithmetic and 10% weak-Transform rejection with two-trait retention in 3/3; and a later 2->3 Story/Arithmetic/Transform birth with three-trait functional identity and routing purity >=0.75 in >=2/3, ending at exactly K=3.",
            "scope": "024 validates repeated birth valuation in one persistent organism. It still uses a fixed pretrained gradient sensor and does not establish local sensing, rewiring, pruning, merging, or inference-time recruitment.",
        },
        "status": status,
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return decision


def write_task_spec() -> None:
    payload = {
        "format": "minicells.sequential-probationary-genesis-task.v1",
        "stages": {
            stage: {
                "counts": stage_spec(stage).counts,
                "expected_outcome": stage_spec(stage).expected_outcome,
            }
            for stage in STAGES
        },
        "duplicate_negative_control": "ARITH_A and ARITH_B are independent schedules over the exact same arithmetic token distribution",
        "weak_negative_control": "D_WEAK_TRANSFORM contains only 10% transform samples while preserving Story and Arithmetic exposure",
        "task_labels_used_by_proposal": False,
        "task_labels_used_by_geometry_routing": False,
        "task_labels_used_by_commit": False,
        "posthoc_labels_used_for_scientific_validation": True,
        "capacity_control_uses_stream_identity_only_for_local matched split": True,
    }
    (OUT / "task-spec.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    cache = prepare_corpus()
    gpu_count = run_workers(cache)
    data = collect()
    validate_invariants(data)
    checkpoint_manifest = write_checkpoint_manifest()
    write_task_spec()
    plot_trajectory(data["trajectory"])
    plot_utility(data["windows"])
    plot_identity(data["stage_summary"])
    decision = decide(data, gpu_count, checkpoint_manifest)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())