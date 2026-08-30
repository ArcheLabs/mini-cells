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

from minicells.clm_upcycling_validation import make_upcycling_decision  # noqa: E402
from minicells.language_scaling import prepare_scaling_corpus  # noqa: E402

OUT = ROOT / "results" / "clm-upcycling-study-001-inherit-then-differentiate"
SOURCE_005 = ROOT / "artifacts" / "experiments" / "005-consumer-language-bridge"
SOURCE_006 = ROOT / "artifacts" / "experiments" / "006-consumer-language-scaling"
WORKER = ROOT / "scripts" / "run_clm_upcycling_study_001_worker.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLM Upcycling Study 001.")
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def complete(replicate: int) -> bool:
    path = OUT / f"r{replicate}-worker.json"
    return path.is_file() and json.loads(path.read_text()).get("complete") is True


def command(replicate: int, cache: Path) -> list[str]:
    return [
        sys.executable,
        str(WORKER),
        "--replicate", str(replicate),
        "--cache-dir", str(cache),
        "--output-dir", str(OUT),
        "--checkpoint", str(SOURCE_006 / "minicells-v2-10m.pt"),
        "--model-config", str(SOURCE_006 / "model-configs.json"),
    ]


def run_workers(cache: Path) -> int:
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("CUDA is required")
    used = min(2, gpu_count)
    queue = [replicate for replicate in range(3) if not complete(replicate)]
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


def _save_or_placeholder(fig_name: str, draw) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    drawn = draw(axis)
    if not drawn:
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
        axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT / fig_name, dpi=160)
    plt.close(fig)


def plots(progression: pd.DataFrame, controls: pd.DataFrame, workers: list[dict[str, object]]) -> None:
    def quality(axis):
        if progression.empty:
            return False
        for (method, replicate), group in progression.groupby(["method", "replicate"]):
            group = group.sort_values("tokens")
            axis.plot(group["tokens"], group["validation_ppl"], marker="o",
                      label=f"{method}-r{replicate}")
        axis.set(xlabel="Continuation tokens", ylabel="Validation PPL",
                 title="Matched-budget continuation")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.25)
        return True
    _save_or_placeholder("continuation-quality.png", quality)

    def final_quality(axis):
        rows = []
        for worker in workers:
            r = int(worker["replicate"])
            rows.extend([
                (r, "source", float(worker["source_ppl"])),
                (r, "dense", float(worker["dense_ppl"])),
                (r, "copy_random", float(worker["random_final_ppl"])),
                (r, "copy_geometry", float(worker["geometry_final_ppl"])),
            ])
        if not rows:
            return False
        frame = pd.DataFrame(rows, columns=["replicate", "method", "ppl"])
        frame.pivot(index="replicate", columns="method", values="ppl").plot.bar(ax=axis)
        axis.set(ylabel="PPL", title="Final quality at equal continuation budget")
        return True
    _save_or_placeholder("final-quality.png", final_quality)

    def divergence(axis):
        sparse = progression[progression["method"].isin(["copy_random", "copy_geometry"])]
        if sparse.empty:
            return False
        for (method, replicate), group in sparse.groupby(["method", "replicate"]):
            group = group.sort_values("tokens")
            axis.plot(group["tokens"], group["expert_pairwise_relative_l2"], marker="o",
                      label=f"{method}-r{replicate}")
        axis.set(xlabel="Continuation tokens", ylabel="Mean pairwise relative L2",
                 title="Expert differentiation")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.25)
        return True
    _save_or_placeholder("expert-divergence.png", divergence)

    def entropy(axis):
        sparse = progression[progression["method"].isin(["copy_random", "copy_geometry"])]
        if sparse.empty:
            return False
        for (method, replicate), group in sparse.groupby(["method", "replicate"]):
            group = group.sort_values("tokens")
            axis.plot(group["tokens"], group["usage_entropy"], marker="o",
                      label=f"{method}-r{replicate}")
        axis.axhline(0.8, linestyle="--")
        axis.set(xlabel="Continuation tokens", ylabel="Normalized usage entropy",
                 title="Router utilization")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.25)
        return True
    _save_or_placeholder("usage-entropy.png", entropy)

    def routing_controls(axis):
        if controls.empty:
            return False
        table = controls.pivot_table(
            index=["replicate", "method"], columns="arm", values="nll"
        )
        table[["dynamic", "static", "shuffled"]].plot.bar(ax=axis)
        axis.set(ylabel="NLL", title="Dynamic routing vs matched controls")
        return True
    _save_or_placeholder("routing-controls.png", routing_controls)

    def variation(axis):
        if controls.empty:
            return False
        dynamic = controls[controls["arm"] == "dynamic"]
        if dynamic.empty:
            return False
        positions = range(len(dynamic))
        axis.bar([x - 0.2 for x in positions], dynamic["sample_variation"], width=0.4,
                 label="sample")
        axis.bar([x + 0.2 for x in positions], dynamic["temporal_variation"], width=0.4,
                 label="temporal")
        axis.set_xticks(list(positions), [
            f"r{int(row.replicate)}-{row.method}" for row in dynamic.itertuples()
        ], rotation=30, ha="right")
        axis.axhline(0.05, linestyle="--")
        axis.set(ylabel="Mask variation", title="Routing variation")
        axis.legend()
        return True
    _save_or_placeholder("routing-variation.png", variation)

    def capacity(axis):
        labels = ["Dense FFN", "Upcycled total experts", "Upcycled active expert"]
        values = [1.0, 4.0, 1.0]
        axis.bar(labels, values)
        axis.set(ylabel="FFN parameter capacity / dense", title="Inheritance without active-compute inflation")
        axis.tick_params(axis="x", rotation=15)
        return True
    _save_or_placeholder("capacity-vs-active.png", capacity)


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

    progression = pd.concat(
        [pd.read_csv(OUT / f"r{r}-progression.csv") for r in range(3)],
        ignore_index=True,
    )
    controls = pd.concat(
        [pd.read_csv(OUT / f"r{r}-controls.csv") for r in range(3)],
        ignore_index=True,
    )
    workers = [json.loads((OUT / f"r{r}-worker.json").read_text()) for r in range(3)]
    progression.to_csv(OUT / "progression.csv", index=False)
    controls.to_csv(OUT / "controls.csv", index=False)
    decision = make_upcycling_decision(workers, controls.to_dict("records"))
    decision["provenance"] = {
        "source": "Experiment 006 minicells-v2-10m.pt",
        "clm_v1": "CLM_NO_QUALITY_SAFE_SPARSITY at K=7 in Validation 001b",
        "clm_v2_random_newborn": (
            "CLMV2_CLOSED_LOOP_HANDOFF_FAILURE; closed-loop homotopy reduced the direct "
            "zero-scaffold gap but scaffold-free K6 plateaued near 1.098x teacher PPL"
        ),
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    task = {
        "format": "minicells.clm-upcycling-study-001-task.v1",
        "experiment": "CLM Upcycling Study 001 — Inherit Then Differentiate",
        "scientific_question": (
            "Can exact dense-FFN inheritance create a function-preserving routed cellular model, "
            "and does local-state geometry improve subsequent expert differentiation?"
        ),
        "formal_arms": ["dense_continued", "copy_random", "copy_geometry"],
        "historical_control": "random-newborn CLM v2 Validation 001b",
        "num_experts": 4,
        "top_k": 1,
        "total_ffn_capacity_ratio": 4.0,
        "active_ffn_capacity_ratio": 1.0,
        "training_tokens_per_arm": 1_000_000,
        "training_blocks": 4,
        "block_tokens": 250_000,
        "same_data_schedule_across_arms": True,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "distillation_weight": 0.5,
        "balance_weight": 0.01,
        "geometry": {
            "source": "frozen teacher norm_ffn local perceptions",
            "method": "cosine k-means",
            "clusters": 4,
            "max_samples_per_stage": 8192,
            "labels_used": False,
        },
        "thresholds": decision["thresholds"],
        "replicates": 3,
        "cell_activation": 1.0,
        "forbidden": [
            "capability labels", "semantic experts", "phenotype", "cell sparsity",
            "topology growth", "cell lifecycle",
        ],
    }
    (OUT / "task-spec.json").write_text(json.dumps(task, indent=2, sort_keys=True) + "\n")
    runtime = {
        "format": "minicells.clm-upcycling-study-001-runtime.v1",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus_used": gpu_count,
    }
    (OUT / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    plots(progression, controls, workers)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
