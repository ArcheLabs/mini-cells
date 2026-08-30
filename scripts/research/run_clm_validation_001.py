from __future__ import annotations

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

from minicells.language_clm_validation import make_validation_decision  # noqa: E402
from minicells.language_scaling import prepare_scaling_corpus  # noqa: E402


OUT = ROOT / "results" / "clm-validation-001-program-conditionality"
SOURCE_005 = ROOT / "artifacts" / "experiments" / "005-consumer-language-bridge"
SOURCE_006 = ROOT / "artifacts" / "experiments" / "006-consumer-language-scaling"
CHECKPOINT = SOURCE_006 / "minicells-v2-10m.pt"
MODEL_CONFIG = SOURCE_006 / "model-configs.json"
WORKER = ROOT / "scripts" / "run_clm_validation_001_worker.py"


def command(replicate: int, cache_dir: Path) -> list[str]:
    return [sys.executable, str(WORKER), "--replicate", str(replicate),
            "--cache-dir", str(cache_dir), "--output-dir", str(OUT),
            "--checkpoint", str(CHECKPOINT), "--model-config", str(MODEL_CONFIG)]


def run_replicates(cache_dir: Path) -> int:
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("CLM Validation 001 requires at least one CUDA GPU")
    queue = [0, 1, 2]
    used = min(2, gpu_count)
    while queue:
        active: list[tuple[int, int, subprocess.Popen[str], object, Path]] = []
        for gpu in range(min(used, len(queue))):
            replicate = queue.pop(0)
            log_path = OUT / f"r{replicate}.log"
            handle = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                command(replicate, cache_dir), cwd=ROOT, env=env,
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


def save_plots(rows: pd.DataFrame) -> None:
    colors = {"dense": "#4c78a8", "dynamic": "#59a14f",
              "static": "#f28e2b", "shuffled": "#e15759"}
    fig, axis = plt.subplots(figsize=(7, 5))
    for arm, group in rows.groupby("arm"):
        axis.scatter(group["effective_compute_ratio"], group["validation_ppl"],
                     label=arm, color=colors[arm])
    axis.set(xlabel="Effective compute ratio (receptor included)", ylabel="Validation PPL",
             title="CLM Validation 001 — quality vs compute")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "quality-vs-compute.png", dpi=160)
    plt.close(fig)

    k4 = rows[rows["top_k"] == 4]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    pivot = k4.pivot(index="replicate", columns="arm", values="validation_nll")
    control_arms = ("dynamic", "static", "shuffled")
    pivot[list(control_arms)].plot.bar(
        ax=axis, color=[colors[arm] for arm in control_arms]
    )
    axis.set(ylabel="Validation NLL", title="Matched-compute routing controls (4/8)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "routing-controls-nll.png", dpi=160)
    plt.close(fig)

    dynamic = rows[rows["arm"] == "dynamic"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for replicate, group in dynamic.groupby("replicate"):
        axes[0].plot(group["top_k"], group["structural_variation"], marker="o",
                     label=f"r{replicate}")
        axes[1].plot(group["top_k"], group["temporal_variation"], marker="o",
                     label=f"r{replicate}")
    axes[0].set(
        title="Across-input mask variation",
        xlabel="Active programs",
        ylabel="Hamming distance",
    )
    axes[1].set(
        title="Across-step routing change",
        xlabel="Active programs",
        ylabel="Hamming distance",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "routing-variation.png", dpi=160)
    plt.close(fig)

    k4_dynamic = dynamic[dynamic["top_k"] == 4]
    usages = pd.DataFrame([json.loads(value) for value in k4_dynamic["program_usage"]])
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(range(8), usages.mean(0), yerr=usages.std(0), color="#59a14f")
    axis.set(xlabel="Program id", ylabel="Activation frequency",
             title="Dynamic program usage (4/8)")
    axis.set_xticks(range(8))
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "program-usage.png", dpi=160)
    plt.close(fig)

    matrices = torch.tensor(
        [json.loads(value) for value in k4_dynamic["program_coactivation"]]
    ).mean(0)
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrices, vmin=0, vmax=1, cmap="viridis")
    axis.set(xlabel="Program", ylabel="Program", title="Dynamic program coactivation (4/8)")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(OUT / "program-coactivation.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    for arm, group in rows.groupby("arm"):
        axis.scatter(group["effective_compute_ratio"], group["tokens_per_second"],
                     label=arm, color=colors[arm])
    axis.set(xlabel="Effective compute ratio", ylabel="Tokens / second",
             title="Sparse-dispatch throughput")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "throughput-vs-compute.png", dpi=160)
    plt.close(fig)


def main() -> int:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    train, validation, tokenizer_path, manifest = prepare_scaling_corpus(
        ROOT, source_005_dir=SOURCE_005,
    )
    del train, validation
    shutil.copy2(tokenizer_path, OUT / "tokenizer.json")
    (OUT / "corpus-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gpu_count = run_replicates(tokenizer_path.parent)
    arms = pd.concat(
        [pd.read_csv(OUT / f"r{replicate}-arms.csv") for replicate in range(3)],
        ignore_index=True,
    )
    phases = pd.concat(
        [pd.read_csv(OUT / f"r{replicate}-phases.csv").assign(replicate=replicate)
         for replicate in range(3)], ignore_index=True,
    )
    arms.to_csv(OUT / "arms.csv", index=False)
    phases.to_csv(OUT / "phases.csv", index=False)
    parity = [json.loads((OUT / f"r{r}-worker.json").read_text())["dense_conversion"]
              for r in range(3)]
    if any(item["status"] != "CLM_DENSE_EQUIVALENCE" for item in parity):
        raise RuntimeError("real-checkpoint dense conversion failed; sparse results are invalid")
    decision = make_validation_decision(arms.to_dict("records"))
    decision["stage_0"] = parity
    decision["source"] = {
        "commit": "63af81d47755b4975d4e300560c1c72f779e0d11",
        "checkpoint": "artifacts/experiments/006-consumer-language-scaling/minicells-v2-10m.pt",
        "training_tokens": 10_000_000,
    }
    decision["runtime"] = {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu_count_used": gpu_count,
        "gpus": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
    }
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    task_spec = {
        "format": "minicells.clm-validation-001-task.v1",
        "question": (
            "Does state-dependent sparse program routing preserve LM quality better "
            "than matched static pruning?"
        ),
        "programs": 8, "replicates": 3, "cell_activation": 1.0,
        "arms": ["dense", "dynamic", "static", "shuffled"],
        "top_k": [6, 4], "continuation_tokens_per_replicate": 1_500_000,
        "forbidden": ["cell sparsity", "phenotype differentiation", "topology changes",
                      "capability labels", "fork/death", "semantic experts"],
    }
    (OUT / "task-spec.json").write_text(
        json.dumps(task_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    save_plots(arms)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
