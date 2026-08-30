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
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))

from minicells.language_clm_validation import make_validation_001b_decision  # noqa: E402
from minicells.language_scaling import prepare_scaling_corpus  # noqa: E402


OUT = ROOT / "results" / "clm-validation-001b-stable-program-conditionality"
SOURCE_005 = ROOT / "artifacts" / "experiments" / "005-consumer-language-bridge"
SOURCE_006 = ROOT / "artifacts" / "experiments" / "006-consumer-language-scaling"
CHECKPOINT = SOURCE_006 / "minicells-v2-10m.pt"
MODEL_CONFIG = SOURCE_006 / "model-configs.json"
WORKER = ROOT / "scripts" / "run_clm_validation_001b_worker.py"
DIAGNOSIS = (
    ROOT / "artifacts" / "experiments" / "clm-validation-001-program-conditionality"
    / "DIAGNOSIS.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLM Validation 001b.")
    parser.add_argument("--fresh", action="store_true", help="Delete existing 001b outputs first.")
    return parser.parse_args()


def worker_command(replicate: int, cache_dir: Path) -> list[str]:
    return [
        sys.executable, str(WORKER), "--replicate", str(replicate),
        "--cache-dir", str(cache_dir), "--output-dir", str(OUT),
        "--checkpoint", str(CHECKPOINT), "--model-config", str(MODEL_CONFIG),
    ]


def replicate_complete(replicate: int) -> bool:
    path = OUT / f"r{replicate}-worker.json"
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("complete") is True


def run_replicates(cache_dir: Path) -> int:
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("CLM Validation 001b requires CUDA")
    used = min(2, gpu_count)
    queue = [replicate for replicate in range(3) if not replicate_complete(replicate)]
    while queue:
        active = []
        for gpu in range(min(used, len(queue))):
            replicate = queue.pop(0)
            log_path = OUT / f"r{replicate}.log"
            handle = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                worker_command(replicate, cache_dir), cwd=ROOT, env=env,
                stdout=handle, stderr=subprocess.STDOUT, text=True,
            )
            active.append((replicate, gpu, process, handle, log_path))
            print(f"started replicate {replicate} on physical GPU {gpu}")
        failures = []
        for replicate, gpu, process, handle, log_path in active:
            code = process.wait()
            handle.close()
            print(f"--- replicate {replicate} / GPU {gpu} ---")
            print(log_path.read_text(encoding="utf-8").rstrip())
            if code:
                failures.append(f"replicate {replicate} exited {code}")
        if failures:
            raise RuntimeError("; ".join(failures))
    return used


def save_plots(progression: pd.DataFrame, arms: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for replicate, group in progression.groupby("replicate"):
        ordered = group.sort_values("top_k", ascending=False)
        axis.plot(ordered["program_ratio"], ordered["quality_ratio"], marker="o",
                  label=f"r{replicate}")
    axis.axhline(1.03, color="red", linestyle="--", label="quality gate")
    axis.set(xlabel="Active program ratio K/8", ylabel="Dynamic PPL / same-student dense PPL",
             title="Quality-gated discrete continuation")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "quality-vs-k.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    arms.pivot(index="replicate", columns="arm", values="validation_nll")[
        ["dynamic", "static", "shuffled"]
    ].plot.bar(ax=axis)
    axis.set(ylabel="Validation NLL", title="Controls at replicate-specific K*")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "routing-controls-nll.png", dpi=160)
    plt.close(fig)

    dynamic = arms[arms["arm"] == "dynamic"]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(dynamic["replicate"] - 0.15, dynamic["structural_variation"], width=0.3,
             label="sample")
    axis.bar(dynamic["replicate"] + 0.15, dynamic["temporal_variation"], width=0.3,
             label="temporal")
    axis.axhline(0.05, color="red", linestyle="--")
    axis.set(xlabel="Replicate", ylabel="Routing variation", title="Dynamic routing variation")
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "routing-variation.png", dpi=160)
    plt.close(fig)

    usages = pd.DataFrame([json.loads(value) for value in dynamic["program_usage"]])
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(range(8), usages.mean(0), yerr=usages.std(0))
    axis.set(xlabel="Program", ylabel="Usage", title="Program usage at K*")
    axis.set_xticks(range(8))
    fig.tight_layout()
    fig.savefig(OUT / "program-usage.png", dpi=160)
    plt.close(fig)

    coactivation = torch.tensor(
        [json.loads(value) for value in dynamic["program_coactivation"]]
    ).mean(0)
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(coactivation, vmin=0, vmax=1, cmap="viridis")
    axis.set(title="Program coactivation at K*", xlabel="Program", ylabel="Program")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(OUT / "program-coactivation.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    for replicate, group in progression.groupby("replicate"):
        axis.plot(group["effective_compute_ratio"], group["quality_ratio"], marker="o",
                  label=f"r{replicate}")
    axis.set(xlabel="Effective compute ratio", ylabel="Quality ratio",
             title="Compute vs quality")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "compute-vs-quality.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    for arm, group in arms.groupby("arm"):
        axis.scatter(group["effective_compute_ratio"], group["tokens_per_second"], label=arm)
    axis.set(xlabel="Effective compute ratio", ylabel="Tokens / second", title="Throughput")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "throughput.png", dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.fresh and OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    train, validation, tokenizer_path, manifest = prepare_scaling_corpus(
        ROOT, source_005_dir=SOURCE_005,
    )
    del train, validation
    shutil.copy2(tokenizer_path, OUT / "tokenizer.json")
    (OUT / "corpus-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gpu_count = run_replicates(tokenizer_path.parent)
    progression = pd.concat([
        pd.read_csv(OUT / f"r{r}-progression.csv") for r in range(3)
    ], ignore_index=True)
    arms = pd.concat([pd.read_csv(OUT / f"r{r}-arms.csv") for r in range(3)], ignore_index=True)
    diagnostics = pd.concat([
        pd.read_csv(OUT / f"r{r}-router-diagnostics.csv") for r in range(3)
    ], ignore_index=True)
    progression.to_csv(OUT / "progression.csv", index=False)
    arms.to_csv(OUT / "arms.csv", index=False)
    diagnostics.to_csv(OUT / "router-diagnostics.csv", index=False)
    workers = [json.loads((OUT / f"r{r}-worker.json").read_text()) for r in range(3)]
    decision = make_validation_001b_decision(
        arms.to_dict("records"), progression.to_dict("records"),
        router_warmup_ok={int(row["replicate"]): bool(row["router_warmup_ok"]) for row in workers},
    )
    decision["provenance"] = {
        "source_model": "artifacts/experiments/006-consumer-language-scaling/minicells-v2-10m.pt",
        "clm_implementation_commit": "63af81d47755b4975d4e300560c1c72f779e0d11",
        "validation_001_status": "FAIL / CLM_PROGRAM_SPARSITY_QUALITY_FAILURE",
        "validation_001_diagnosis": "CONTINUATION_DID_NOT_LEAVE_DENSE_BASIN",
    }
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    task_spec = {
        "format": "minicells.clm-validation-001b-task.v1",
        "programs": 8, "cell_activation": 1.0,
        "progression": [8, 7, 6, 5, 4], "budgets": [250000, 250000, 375000, 500000, 500000],
        "quality_ratio_max": 1.03, "required_safe_k_max": 6,
        "sample_variation_min": 0.05, "static_advantage_min": 0.002,
        "shuffle_advantage_min": 0.002, "receptor_ratio_max": 0.05,
        "replicates": 3, "shuffle_permutations": 3,
        "seeds": {"receptor": "72001+r", "schedule": "82001+r*100+stage",
                  "shuffle": "92001+r*100+permutation"},
    }
    (OUT / "task-spec.json").write_text(
        json.dumps(task_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    runtime = {
        "format": "minicells.clm-validation-001b-runtime.v1",
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu_count_used": gpu_count,
        "gpus": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
    }
    (OUT / "runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(DIAGNOSIS, OUT / "VALIDATION_001_DIAGNOSIS.md")
    save_plots(progression, arms)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
