from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
import torch

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))

from minicells.clm_v2_closed_loop_validation import (  # noqa: E402
    FINAL_HANDOFF_RATIO_MAX,
    HANDOFF_PROGRESS_RATIO_MAX,
    HANDOFF_SAFETY_RATIO_MAX,
    LOCAL_COSINE_MIN,
    LOCAL_RELATIVE_MSE_MAX,
    NORMALIZED_ADVANTAGE_MIN,
    QUALITY_SAFE_K_MAX,
    RECEPTOR_RATIO_MAX,
    SAMPLE_VARIATION_MIN,
    make_closed_loop_decision,
)
from minicells.language_scaling import prepare_scaling_corpus  # noqa: E402

OUT = ROOT / "results" / "clm-v2-validation-001b-closed-loop-handoff"
SOURCE_005 = ROOT / "artifacts" / "experiments" / "005-consumer-language-bridge"
SOURCE_006 = ROOT / "artifacts" / "experiments" / "006-consumer-language-scaling"
WORKER = ROOT / "scripts" / "run_clm_v2_validation_001b_worker.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Run CLM v2 Validation 001b.")
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def complete(replicate: int) -> bool:
    path = OUT / f"r{replicate}-worker.json"
    return path.is_file() and json.loads(path.read_text()).get("complete") is True


def command(replicate: int, cache: Path) -> list[str]:
    return [
        sys.executable, str(WORKER), "--replicate", str(replicate),
        "--cache-dir", str(cache), "--output-dir", str(OUT),
        "--checkpoint", str(SOURCE_006 / "minicells-v2-10m.pt"),
        "--model-config", str(SOURCE_006 / "model-configs.json"),
    ]


def run_workers(cache: Path) -> int:
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("CUDA is required")
    used = min(2, gpu_count)
    queue = [index for index in range(3) if not complete(index)]
    while queue:
        active = []
        for gpu in range(min(used, len(queue))):
            replicate = queue.pop(0)
            log = OUT / f"r{replicate}.log"
            handle = log.open("w")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                command(replicate, cache), cwd=ROOT, env=env,
                stdout=handle, stderr=subprocess.STDOUT, text=True,
            )
            active.append((replicate, process, handle, log))
        failures = []
        for replicate, process, handle, log in active:
            code = process.wait()
            handle.close()
            print(log.read_text())
            if code:
                failures.append(f"r{replicate} exited {code}")
        if failures:
            raise RuntimeError("; ".join(failures))
    return used


def _save_empty_plot(name: str, message: str) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.text(0.5, 0.5, message, ha="center", va="center")
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


def plots(progression: pd.DataFrame, arms: pd.DataFrame) -> None:
    handoff = progression[progression["phase"].str.startswith("alpha-")]
    if handoff.empty:
        _save_empty_plot("scaffold-handoff.png", "Closed-loop handoff not reached")
    else:
        fig, axis = plt.subplots(figsize=(7, 4.5))
        for replicate, group in handoff.groupby("replicate"):
            group = group.sort_values("alpha", ascending=False)
            axis.plot(group["alpha"], group["quality_ratio"], marker="o", label=f"r{replicate}")
        axis.axhline(HANDOFF_SAFETY_RATIO_MAX, linestyle="--", label="safety gate")
        axis.set(xlabel="Scaffold alpha", ylabel="PPL / teacher PPL", title="Closed-loop scaffold handoff")
        if axis.lines:
            axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT / "scaffold-handoff.png", dpi=160)
        plt.close(fig)

    imitation = progression[progression["phase"].str.startswith("imitation-")]
    if imitation.empty:
        _save_empty_plot("local-imitation.png", "No local imitation results")
    else:
        fig, axis = plt.subplots(figsize=(7, 4.5))
        for replicate, group in imitation.groupby("replicate"):
            axis.plot(
                range(1, len(group) + 1), group["validation_local_relative_mse"],
                marker="o", label=f"r{replicate}",
            )
        axis.axhline(LOCAL_RELATIVE_MSE_MAX, linestyle="--", label="MSE gate")
        axis.set(xlabel="500K-token imitation block", ylabel="Relative MSE", title="Local scaffold imitation")
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT / "local-imitation.png", dpi=160)
        plt.close(fig)

    if handoff.empty:
        _save_empty_plot("handoff-recovery.png", "Closed-loop handoff not reached")
    else:
        fig, axis = plt.subplots(figsize=(7, 4.5))
        for replicate, group in handoff.groupby("replicate"):
            axis.plot(group["alpha"], group["handoff_progress_ratio"], marker="o", label=f"r{replicate}")
        axis.axhline(HANDOFF_PROGRESS_RATIO_MAX, linestyle="--", label="progress gate")
        axis.set(xlabel="Scaffold alpha", ylabel="PPL after / before stage training", title="Closed-loop recovery")
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT / "handoff-recovery.png", dpi=160)
        plt.close(fig)

    sparse = progression[
        progression["top_k"].isin([6, 5, 4, 3]) & (progression["alpha"] == 0)
    ]
    if sparse.empty:
        _save_empty_plot("quality-vs-k.png", "Scaffold-free K progression not reached")
    else:
        fig, axis = plt.subplots(figsize=(7, 4.5))
        for replicate, group in sparse.groupby("replicate"):
            axis.plot(group["top_k"], group["quality_ratio"], marker="o", label=f"r{replicate}")
        axis.axhline(FINAL_HANDOFF_RATIO_MAX, linestyle="--", label="quality gate")
        axis.set(xlabel="Conditional K", ylabel="Quality ratio", title="Quality-gated K")
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT / "quality-vs-k.png", dpi=160)
        plt.close(fig)

    if not arms.empty:
        fig, axis = plt.subplots(figsize=(7, 4.5))
        pivot = arms.pivot(index="replicate", columns="arm", values="nll")
        available = [name for name in ("dynamic", "static", "shuffled") if name in pivot.columns]
        pivot[available].plot.bar(ax=axis)
        axis.set(ylabel="NLL", title="Dynamic vs matched controls")
        fig.tight_layout()
        fig.savefig(OUT / "routing-controls.png", dpi=160)
        plt.close(fig)

        dynamic = arms[arms["arm"] == "dynamic"]
        fig, axis = plt.subplots(figsize=(7, 4.5))
        for name in ("sample_variation", "position_variation", "temporal_variation"):
            axis.plot(dynamic["replicate"], dynamic[name], marker="o", label=name)
        axis.legend()
        axis.set(title="Routing variation", xlabel="Replicate")
        fig.tight_layout()
        fig.savefig(OUT / "routing-variation.png", dpi=160)
        plt.close(fig)

        usage = pd.DataFrame([json.loads(value) for value in dynamic["program_usage"]])
        fig, axis = plt.subplots(figsize=(7, 4.5))
        axis.bar(range(12), usage.mean(0))
        axis.set(title="Program usage")
        fig.tight_layout()
        fig.savefig(OUT / "program-usage.png", dpi=160)
        plt.close(fig)

        coactivation = torch.tensor(
            [json.loads(value) for value in dynamic["program_coactivation"]]
        ).mean(0)
        fig, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(coactivation, vmin=0, vmax=1)
        fig.colorbar(image, ax=axis)
        axis.set(title="Program coactivation")
        fig.tight_layout()
        fig.savefig(OUT / "program-coactivation.png", dpi=160)
        plt.close(fig)
    else:
        for name in (
            "routing-controls.png", "routing-variation.png",
            "program-usage.png", "program-coactivation.png",
        ):
            _save_empty_plot(name, "Final routing controls not reached")

    fig, axis = plt.subplots(figsize=(7, 4.5))
    labels = ["Genome", "K6", "K5", "K4", "K3"]
    values = [1.75, 1.0, 0.875, 0.75, 0.625]
    axis.bar(labels, values)
    axis.axhline(1, linestyle="--")
    axis.set(ylabel="FFN hidden-capacity / dense", title="Genome capacity vs active compute")
    fig.tight_layout()
    fig.savefig(OUT / "capacity-vs-compute.png", dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.fresh and OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    train, validation, tokenizer, manifest = prepare_scaling_corpus(
        ROOT, source_005_dir=SOURCE_005
    )
    del train, validation
    shutil.copy2(tokenizer, OUT / "tokenizer.json")
    (OUT / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    gpu_count = run_workers(tokenizer.parent)
    progression = pd.concat([
        pd.read_csv(OUT / f"r{replicate}-progression.csv") for replicate in range(3)
    ], ignore_index=True)
    progression.to_csv(OUT / "progression.csv", index=False)

    arm_frames = []
    for replicate in range(3):
        path = OUT / f"r{replicate}-arms.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            if not frame.empty:
                arm_frames.append(frame)
    arms = pd.concat(arm_frames, ignore_index=True) if arm_frames else pd.DataFrame(columns=[
        "replicate", "arm", "top_k", "nll", "ppl", "sample_variation",
        "position_variation", "temporal_variation", "program_usage",
        "program_coactivation", "receptor_ratio", "active_ffn_ratio",
        "effective_ffn_ratio", "tokens_per_second", "shuffle_permutations",
    ])
    arms.to_csv(OUT / "arms.csv", index=False)

    diagnostics = pd.concat([
        pd.read_csv(OUT / f"r{replicate}-router-diagnostics.csv") for replicate in range(3)
    ], ignore_index=True)
    diagnostics.to_csv(OUT / "router-diagnostics.csv", index=False)

    workers = [
        json.loads((OUT / f"r{replicate}-worker.json").read_text()) for replicate in range(3)
    ]
    stage0 = [
        json.loads((OUT / f"r{replicate}-stage0.json").read_text()) for replicate in range(3)
    ]
    decision = make_closed_loop_decision(
        workers, arms.to_dict("records"), teacher_nll=float(stage0[0]["teacher_nll"])
    )
    decision["provenance"] = {
        "source": "Experiment 006 minicells-v2-10m.pt",
        "clm_v1_commit": "63af81d47755b4975d4e300560c1c72f779e0d11",
        "validation_001b": "CLM_NO_QUALITY_SAFE_SPARSITY",
        "clm_v2_validation_001": (
            "CLMV2_LOCAL_IMITATION_FAILURE; local relative MSE reached about 0.052-0.055 "
            "and cosine about 0.970-0.972, but direct alpha=0 rollout remained about 1.12x PPL"
        ),
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )

    task = {
        "format": "minicells.clm-v2-validation-001b-task.v1",
        "experiment": "CLM v2 Validation 001b — Closed-Loop Scaffold Handoff",
        "programs": 12,
        "shared_hidden": 128,
        "expert_hidden": 64,
        "initial_k": 6,
        "genome_capacity_ratio": 1.75,
        "cell_activation": 1.0,
        "local_gate": {
            "relative_mse_max": LOCAL_RELATIVE_MSE_MAX,
            "cosine_min": LOCAL_COSINE_MIN,
            "direct_alpha_zero_ppl": "telemetry_only",
        },
        "alphas": [1, 0.75, 0.5, 0.25, 0],
        "handoff_safety_ratio_max": HANDOFF_SAFETY_RATIO_MAX,
        "handoff_progress_ratio_max": HANDOFF_PROGRESS_RATIO_MAX,
        "final_handoff_ratio_max": FINAL_HANDOFF_RATIO_MAX,
        "k_progression": [6, 5, 4, 3],
        "quality_safe_k_max": QUALITY_SAFE_K_MAX,
        "sample_variation_min": SAMPLE_VARIATION_MIN,
        "advantage_min": NORMALIZED_ADVANTAGE_MIN,
        "receptor_ratio_max": RECEPTOR_RATIO_MAX,
        "imitation_tokens_initial": 500_000,
        "imitation_tokens_max": 1_000_000,
        "handoff_tokens_per_alpha": 250_000,
        "consolidation_tokens": 500_000,
        "k_reduction_tokens_per_stage": 375_000,
        "replicates": 3,
        "seeds": {
            "model_router": "103001+r",
            "schedule": "104001+r*100+stage",
            "shuffle": "112001+r*100+permutation",
        },
    }
    (OUT / "task-spec.json").write_text(
        json.dumps(task, indent=2, sort_keys=True) + "\n"
    )
    runtime = {
        "format": "minicells.clm-v2-validation-001b-runtime.v1",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus_used": gpu_count,
    }
    (OUT / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    plots(progression, arms)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
