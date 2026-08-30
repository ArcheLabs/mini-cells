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

import run_online_nonparametric_trait_genesis_worker as worker  # noqa: E402
from minicells.language_conflict_differentiation import prepare_arithmetic_cache  # noqa: E402
from minicells.language_data import load_tokenizer, prepare_tinystories_corpus  # noqa: E402
from minicells.language_online_trait_genesis import (  # noqa: E402
    MAX_TRAITS,
    MODE_STABILITY_MIN,
    PERSISTENCE_EVALS,
    ROUTING_PURITY_MIN,
    SENSOR_BUFFER,
    SENSOR_INTERVAL,
    STRUCTURAL_PENALTY,
    developmental_curriculum,
    prepare_transform_cache,
    stage_end_steps,
)


OUT = ROOT / "results" / "online-nonparametric-trait-genesis-v1"
WORKER = ROOT / "scripts" / "run_online_nonparametric_trait_genesis_worker.py"
N_REPLICATES = worker.N_REPLICATES
EXPECTED_CHECKPOINTS = N_REPLICATES * (1 + len(worker.STAGES))


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
    source_022 = ROOT / "artifacts" / "experiments" / "022-emergent-trait-bifurcation" / "decision.json"
    if not source_022.is_file():
        raise FileNotFoundError("Experiment 022 decision.json must be merged before Experiment 023")
    shutil.copy2(source_022, OUT / "source-022-decision.json")
    return cache


def _worker_complete(replicate: int) -> bool:
    meta = OUT / f"r{replicate}-worker.json"
    summary = OUT / f"r{replicate}-stage-summary.csv"
    structure = OUT / f"r{replicate}-structure.csv"
    if not meta.is_file() or not summary.is_file() or not structure.is_file():
        return False
    payload = json.loads(meta.read_text(encoding="utf-8"))
    return (
        payload.get("format") == "minicells.online-nonparametric-trait-genesis-worker.v1"
        and int(payload.get("replicate", -1)) == replicate
    )


def run_workers(cache: Path) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 023 requires a Kaggle GPU accelerator")
    gpu_count = min(2, available)
    missing = [replicate for replicate in range(N_REPLICATES) if not _worker_complete(replicate)]
    if not missing:
        print("reusing complete Experiment 023 workers")
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
                command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True
            )
            active.append((replicate, local_gpu, process, log, handle))
            print(f"started Experiment 023 r{replicate} on physical GPU {local_gpu}")
        failures = []
        for replicate, gpu, process, log, handle in active:
            code = process.wait()
            handle.close()
            print(f"--- Experiment 023 r{replicate} / GPU {gpu} ---")
            print(log.read_text(encoding="utf-8").rstrip())
            if code != 0:
                failures.append(f"r{replicate} exited {code}; see {log}")
        if failures:
            raise RuntimeError("; ".join(failures))
    return gpu_count


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def collect() -> dict[str, object]:
    workers = []
    stage_summary = []
    structure = []
    routing = []
    genesis = []
    evaluation = []
    pretrain = []
    for replicate in range(N_REPLICATES):
        workers.append(json.loads((OUT / f"r{replicate}-worker.json").read_text(encoding="utf-8")))
        stage_summary.append(pd.read_csv(OUT / f"r{replicate}-stage-summary.csv"))
        structure.append(pd.read_csv(OUT / f"r{replicate}-structure.csv"))
        routing.append(pd.read_csv(OUT / f"r{replicate}-routing.csv"))
        optional_genesis = _read_optional_csv(OUT / f"r{replicate}-genesis.csv")
        if not optional_genesis.empty:
            genesis.append(optional_genesis)
        evaluation.append(pd.read_csv(OUT / f"r{replicate}-evaluation.csv"))
        pretrain.append(pd.read_csv(OUT / f"r{replicate}-pretrain.csv"))
    result = {
        "workers": workers,
        "stage_summary": pd.concat(stage_summary, ignore_index=True),
        "structure": pd.concat(structure, ignore_index=True),
        "routing": pd.concat(routing, ignore_index=True),
        "genesis": pd.concat(genesis, ignore_index=True) if genesis else pd.DataFrame(
            columns=["replicate", "step", "stage", "from_k", "to_k", "selected_k", "parent_branch", "structural_penalty", "stability"]
        ),
        "evaluation": pd.concat(evaluation, ignore_index=True),
        "pretrain": pd.concat(pretrain, ignore_index=True),
    }
    for key, filename in (
        ("stage_summary", "stage-summary.csv"),
        ("structure", "structure.csv"),
        ("routing", "routing.csv"),
        ("genesis", "genesis.csv"),
        ("evaluation", "evaluation.csv"),
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
        "format": "minicells.online-nonparametric-trait-genesis-checkpoint-manifest.v1",
        "experiment": "023",
        "file_count": len(files),
        "expected_file_count": EXPECTED_CHECKPOINTS,
        "files": files,
        "published_model_checkpoints": False,
        "purpose": "Kaggle-local recovery for parent and developmental stage checkpoints",
    }
    (OUT / "checkpoint-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if len(files) != EXPECTED_CHECKPOINTS:
        raise RuntimeError(f"Experiment 023 checkpoint set incomplete: {len(files)}/{EXPECTED_CHECKPOINTS}")
    return manifest


def write_invariants(data: dict[str, object]) -> dict[str, object]:
    curriculum = developmental_curriculum(0)
    counts = {}
    for row in curriculum:
        key = f"{row['stage']}:{row['stream_key']}"
        counts[key] = counts.get(key, 0) + 1
    payload = {
        "format": "minicells.online-nonparametric-trait-genesis-invariants.v1",
        "curriculum_steps": len(curriculum),
        "stage_end_steps": stage_end_steps(),
        "stream_counts": counts,
        "trigger_uses_task_label": False,
        "routing_uses_task_label": False,
        "max_traits": MAX_TRAITS,
        "structural_penalty": STRUCTURAL_PENALTY,
        "sensor_buffer": SENSOR_BUFFER,
        "sensor_interval": SENSOR_INTERVAL,
        "persistence_evaluations": PERSISTENCE_EVALS,
        "mode_stability_min": MODE_STABILITY_MIN,
        "finite_structure_metrics": bool(
            np.isfinite(
                data["structure"].select_dtypes(include=[np.number]).to_numpy(dtype=float)
            ).all()
        ),
    }
    (OUT / "invariants.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def plot_k_trajectory(structure: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for replicate, part in structure.groupby("replicate"):
        ax.plot(part["step"], part["active_k_after"], marker="o", label=f"active r{int(replicate)}")
        ax.plot(part["step"], part["selected_k"], linestyle="--", alpha=0.6, label=f"selected r{int(replicate)}")
    for step in stage_end_steps().values():
        ax.axvline(step, linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Online developmental step")
    ax.set_ylabel("Trait count K")
    ax.set_yticks(range(1, MAX_TRAITS + 1))
    ax.set_title("Experiment 023 online structural model selection")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "trait-count-trajectory.png", dpi=180)
    plt.close(fig)


def plot_objectives(structure: pd.DataFrame) -> None:
    means = structure.groupby("step", as_index=False)[[f"k{k}_objective" for k in range(1, MAX_TRAITS + 1)]].mean()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k in range(1, MAX_TRAITS + 1):
        column = f"k{k}_objective"
        if column in means:
            ax.plot(means["step"], means[column], label=f"J{k}")
    ax.set_xlabel("Online developmental step")
    ax.set_ylabel("Penalized structural objective")
    ax.set_title("Nonparametric developmental objective")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "structural-objectives.png", dpi=180)
    plt.close(fig)


def plot_stage_identity(summary: pd.DataFrame) -> None:
    selected = summary.loc[summary["stage"].isin(["B_EMERGING_MATH", "C_DUPLICATE_CONTROL", "D_THIRD_MODE"])].copy()
    values = pd.to_numeric(selected["normalized_identity_margin"], errors="coerce").fillna(0.0).to_numpy(float)
    positions = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(positions, values)
    ax.set_xticks(
        positions,
        [f"r{int(row.replicate)}\n{row.stage.split('_')[0]}" for row in selected.itertuples()],
    )
    ax.set_ylabel("Normalized functional identity margin")
    ax.set_title("Online trait identity after developmental stages")
    fig.tight_layout()
    fig.savefig(OUT / "stage-identity.png", dpi=180)
    plt.close(fig)


def decide(data: dict[str, object], gpu_count: int, checkpoint_manifest: dict[str, object]) -> dict[str, object]:
    summary: pd.DataFrame = data["stage_summary"]
    genesis: pd.DataFrame = data["genesis"]

    def row_for(replicate: int, stage: str):
        part = summary.loc[(summary["replicate"] == replicate) & (summary["stage"] == stage)]
        if len(part) != 1:
            raise RuntimeError(f"missing unique stage summary for r{replicate} {stage}")
        return part.iloc[0]

    null_pass = 0
    two_mode_pass = 0
    duplicate_pass = 0
    third_mode_pass = 0
    overgrowth_replicates = 0
    per_replicate = []
    for replicate in range(N_REPLICATES):
        a = row_for(replicate, "A_STORY_ONLY")
        b = row_for(replicate, "B_EMERGING_MATH")
        c = row_for(replicate, "C_DUPLICATE_CONTROL")
        d = row_for(replicate, "D_THIRD_MODE")
        rep_genesis = genesis.loc[genesis["replicate"] == replicate] if not genesis.empty else genesis
        a_ok = int(a["active_k"] == 1 and a["genesis_events_in_stage"] == 0)
        b_ok = int(
            b["active_k"] == 2
            and b["genesis_events_in_stage"] == 1
            and float(b.get("identity_pass", 0) or 0) == 1
            and float(b.get("routing_purity_pass", 0) or 0) == 1
        )
        c_ok = int(c["active_k"] == 2 and c["genesis_events_in_stage"] == 0)
        d_ok = int(
            d["active_k"] == 3
            and d["genesis_events_in_stage"] == 1
            and float(d.get("identity_pass", 0) or 0) == 1
            and float(d.get("routing_purity_pass", 0) or 0) == 1
        )
        overgrowth = int((not rep_genesis.empty) and (pd.to_numeric(rep_genesis["to_k"], errors="coerce") >= 4).any())
        null_pass += a_ok
        two_mode_pass += b_ok
        duplicate_pass += c_ok
        third_mode_pass += d_ok
        overgrowth_replicates += overgrowth
        per_replicate.append(
            {
                "replicate": replicate,
                "story_only_no_false_genesis": a_ok,
                "two_mode_genesis": b_ok,
                "duplicate_control_no_extra_genesis": c_ok,
                "three_mode_genesis": d_ok,
                "overgrowth": overgrowth,
                "final_active_k": int(d["active_k"]),
            }
        )

    if null_pass < N_REPLICATES:
        status = "FALSE_POSITIVE_GENESIS_ON_UNIMODAL_STREAM"
    elif duplicate_pass < N_REPLICATES or overgrowth_replicates > 0:
        status = "DUPLICATE_MODE_OVERSEGMENTATION"
    elif two_mode_pass >= 2 and third_mode_pass >= 2:
        status = "ONLINE_NONPARAMETRIC_TRAIT_GENESIS_SIGNAL"
    elif two_mode_pass >= 2:
        status = "TWO_MODE_GENESIS_WITHOUT_THIRD_MODE"
    elif third_mode_pass >= 2:
        status = "THIRD_MODE_WITHOUT_STABLE_TWO_MODE"
    else:
        status = "NO_ONLINE_TRAIT_GENESIS"

    decision = {
        "format": "minicells.online-nonparametric-trait-genesis.v1",
        "experiment": "MINI Cells Experiment 023 — Online Nonparametric Trait Genesis",
        "question": "Can one TextNCA organism decide online when and how many persistent phenotype traits to create from an unlabeled gradient field under an explicit structural complexity cost?",
        "design": {
            "replicates": N_REPLICATES,
            "gpu_count": gpu_count,
            "max_traits": MAX_TRAITS,
            "structural_objective": "J_K = R_K / R_1 + lambda * (K - 1)",
            "structural_penalty": STRUCTURAL_PENALTY,
            "sensor_buffer": SENSOR_BUFFER,
            "sensor_interval": SENSOR_INTERVAL,
            "persistence_evaluations": PERSISTENCE_EVALS,
            "mode_stability_min": MODE_STABILITY_MIN,
            "routing_purity_min": ROUTING_PURITY_MIN,
            "trigger_uses_task_label": False,
            "routing_uses_task_label": False,
            "sensor": "fixed parent phenotype gradient shadow",
            "curriculum": [
                "A: STORY only",
                "B: ARITHMETIC enters at 10%, 30%, 50%",
                "C: same arithmetic distribution split into A/B negative-control labels",
                "D: STORY + ARITHMETIC + TRANSFORM",
            ],
        },
        "results": {
            "story_only_no_false_genesis_replicates": null_pass,
            "two_mode_genesis_replicates": two_mode_pass,
            "duplicate_control_no_extra_genesis_replicates": duplicate_pass,
            "three_mode_genesis_replicates": third_mode_pass,
            "overgrowth_replicates": overgrowth_replicates,
            "per_replicate": per_replicate,
        },
        "checkpoint_manifest": {
            "file_count": checkpoint_manifest["file_count"],
            "expected_file_count": checkpoint_manifest["expected_file_count"],
            "published_model_checkpoints": False,
        },
        "interpretation": {
            "success": "A strong positive requires zero false genesis on STORY-only in all replicates, stable 1->2 genesis with two-domain functional identity in >=2/3, no 2->3 growth under the duplicate-arithmetic control in all replicates, and genuine 2->3 genesis with three-domain functional identity in >=2/3.",
            "scope": "023 tests online unknown-K structural genesis with a fixed shadow gradient sensor. It does not yet establish a purely local NCA sensing rule, rewiring, pruning, merging, or inference-time recruitment.",
        },
        "status": status,
    }
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(per_replicate).to_csv(OUT / "replicate-decision.csv", index=False)
    return decision


def write_task_spec() -> None:
    curriculum = developmental_curriculum(0)
    payload = {
        "format": "minicells.online-nonparametric-trait-genesis-task.v1",
        "story_domain": "TinyStories token stream",
        "arithmetic_domain": "same deterministic arithmetic corpus as Experiments 021/022",
        "transform_domain": "deterministic six-digit reverse transformation rendered as language",
        "duplicate_negative_control": "ARITH_A and ARITH_B are independent schedules over the exact same arithmetic token distribution",
        "curriculum_steps": len(curriculum),
        "stage_end_steps": stage_end_steps(),
        "task_boundary_exposed_to_model": False,
        "task_labels_used_by_trigger": False,
        "task_labels_used_by_routing": False,
        "posthoc_labels_used_for_scientific_validation": True,
    }
    (OUT / "task-spec.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    cache = prepare_corpus()
    gpu_count = run_workers(cache)
    data = collect()
    checkpoint_manifest = write_checkpoint_manifest()
    write_invariants(data)
    write_task_spec()
    plot_k_trajectory(data["structure"])
    plot_objectives(data["structure"])
    plot_stage_identity(data["stage_summary"])
    decision = decide(data, gpu_count, checkpoint_manifest)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
