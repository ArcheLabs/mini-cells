from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import functional as F

matplotlib.use("Agg")

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))

from minicells.language_2d import build_minicells_2d  # noqa: E402
from minicells.language_data import (  # noqa: E402
    batch_from_starts,
    encode_story_stream,
    fixed_validation_starts,
    iter_tinystories,
    load_tokenizer,
)
from minicells.language_scaling import build_minicells_v2  # noqa: E402
from minicells.language_settling import relaxation_forward  # noqa: E402

OUT = ROOT / "results" / "language-settling-dynamics-v1"
SOURCE_005 = ROOT / "artifacts" / "experiments" / "005-consumer-language-bridge"
SOURCE_011 = ROOT / "artifacts" / "experiments" / "011-stabilizing-cost"
WORKER = ROOT / "scripts" / "run_language_settling_variant.py"
NEW_MODELS = ("minicells-v2-settling", "minicells-2d-k4-settling")
SWEEP_MODELS = (
    "minicells-v2-stable-011",
    "minicells-v2-settling-012",
    "minicells-2d-k4-stable-011",
    "minicells-2d-k4-settling-012",
)
DISPLAY = {
    "minicells-v2-stable-011": "1D stabilizing (011)",
    "minicells-v2-settling-012": "1D settling (012)",
    "minicells-2d-k4-stable-011": "2D stabilizing (011)",
    "minicells-2d-k4-settling-012": "2D settling (012)",
}
DEPTHS = (2, 4, 6, 8, 12, 16)
TRAIN_STREAM_TOKENS = 3_000_000
VALIDATION_STREAM_TOKENS = 200_000
SWEEP_VALIDATION_BATCHES = 16


def tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def prepare_corpus() -> tuple[Path, dict[str, object]]:
    source_tokenizer = SOURCE_005 / "tokenizer.json"
    source_manifest_path = SOURCE_011 / "corpus-manifest.json"
    if not source_tokenizer.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError("Experiments 005 and 011 artifacts must be merged before 012")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    tokenizer_sha = hashlib.sha256(source_tokenizer.read_bytes()).hexdigest()
    if tokenizer_sha != source_manifest.get("tokenizer_sha256"):
        raise RuntimeError("Experiment 011 tokenizer hash mismatch")

    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer_path = cache / "tokenizer.json"
    train_path = cache / "train-tokens.pt"
    validation_path = cache / "validation-tokens.pt"
    manifest_path = cache / "corpus-manifest.json"
    expected = {
        "format": "minicells.language-settling-corpus.v1",
        "source_011_tokenizer_sha256": tokenizer_sha,
        "train_stream_tokens": TRAIN_STREAM_TOKENS,
        "validation_stream_tokens": VALIDATION_STREAM_TOKENS,
    }

    if tokenizer_path.exists() and train_path.exists() and validation_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in expected.items()):
            train = torch.load(train_path, map_location="cpu")
            validation = torch.load(validation_path, map_location="cpu")
            if (
                tensor_sha256(train) == manifest.get("train_token_sha256")
                and tensor_sha256(validation) == manifest.get("validation_token_sha256")
            ):
                return cache, manifest

    shutil.copy2(source_tokenizer, tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    train, train_stories = encode_story_stream(
        tokenizer,
        iter_tinystories("train"),
        target_tokens=TRAIN_STREAM_TOKENS,
    )
    validation, validation_stories = encode_story_stream(
        tokenizer,
        iter_tinystories("validation"),
        target_tokens=VALIDATION_STREAM_TOKENS,
    )
    train_sha = tensor_sha256(train)
    validation_sha = tensor_sha256(validation)
    if train_sha != source_manifest.get("train_token_sha256"):
        raise RuntimeError("Experiment 012 training stream does not reproduce Experiment 011")
    if validation_sha != source_manifest.get("validation_token_sha256"):
        raise RuntimeError("Experiment 012 validation stream does not reproduce Experiment 011")

    torch.save(train, train_path)
    torch.save(validation, validation_path)
    manifest = {
        **expected,
        "dataset": source_manifest.get("dataset"),
        "streaming": True,
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "train_stories_consumed": train_stories,
        "validation_stories_consumed": validation_stories,
        "train_token_sha256": train_sha,
        "validation_token_sha256": validation_sha,
        "tokenizer_sha256": tokenizer_sha,
        "reproduces_011_corpus": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache, manifest


def worker_command(model: str, cache_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(WORKER),
        "--model",
        model,
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(OUT),
    ]


def run_training(cache_dir: Path) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 012 requires CUDA")
    gpu_count = min(2, available)
    active: list[tuple[str, int, subprocess.Popen[str], Path, object]] = []
    for index, model in enumerate(NEW_MODELS):
        gpu_index = index % gpu_count
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        log_path = OUT / f"{model}.log"
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            worker_command(model, cache_dir),
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        active.append((model, gpu_index, process, log_path, handle))
        print(f"started {model:28s} on physical GPU {gpu_index}")

    failures: list[str] = []
    for model, gpu_index, process, log_path, handle in active:
        code = process.wait()
        handle.close()
        print(f"--- {model} / GPU {gpu_index} ---")
        print(log_path.read_text(encoding="utf-8").rstrip())
        if code != 0:
            failures.append(f"{model} exited {code}; see {log_path}")
    if failures:
        raise RuntimeError("; ".join(failures))
    return gpu_count


def load_sweep_models(vocab_size: int) -> dict[str, torch.nn.Module]:
    paths = {
        "minicells-v2-stable-011": SOURCE_011 / "minicells-v2-stable-2m.pt",
        "minicells-v2-settling-012": OUT / "minicells-v2-settling-2m.pt",
        "minicells-2d-k4-stable-011": SOURCE_011 / "minicells-2d-k4-stable-2m.pt",
        "minicells-2d-k4-settling-012": OUT / "minicells-2d-k4-settling-2m.pt",
    }
    models: dict[str, torch.nn.Module] = {
        "minicells-v2-stable-011": build_minicells_v2(vocab_size),
        "minicells-v2-settling-012": build_minicells_v2(vocab_size),
        "minicells-2d-k4-stable-011": build_minicells_2d(vocab_size, tissue_height=4),
        "minicells-2d-k4-settling-012": build_minicells_2d(vocab_size, tissue_height=4),
    }
    for name, model in models.items():
        path = paths[name]
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    return models


@torch.no_grad()
def evaluate_depth(
    model: torch.nn.Module,
    validation: torch.Tensor,
    starts: tuple[tuple[int, ...], ...],
    *,
    depth: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    residuals = [0.0, 0.0, 0.0]
    batches = 0
    for batch_starts in starts:
        inputs, targets = batch_from_starts(validation, batch_starts, 128, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            relaxed = relaxation_forward(model, inputs, stage_depths=(depth, depth, depth))
            loss = F.cross_entropy(
                relaxed.output.logits.reshape(-1, relaxed.output.logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
        total_loss += float(loss.item())
        total_tokens += int(targets.numel())
        for index, value in enumerate(relaxed.stage_last_residuals):
            residuals[index] += float(value.detach().cpu())
        batches += 1
    nll = total_loss / total_tokens
    mean_stage = [value / batches for value in residuals]
    return {
        "validation_nll": nll,
        "validation_ppl": math.exp(min(nll, 20.0)),
        "stage1_relative_residual": mean_stage[0],
        "stage2_relative_residual": mean_stage[1],
        "stage3_relative_residual": mean_stage[2],
        "mean_relative_residual": sum(mean_stage) / 3.0,
        "validation_tokens": total_tokens,
    }


def run_relaxation_sweep(cache_dir: Path) -> pd.DataFrame:
    tokenizer = load_tokenizer(cache_dir / "tokenizer.json")
    validation = torch.load(cache_dir / "validation-tokens.pt", map_location="cpu")
    starts = fixed_validation_starts(
        int(validation.numel()),
        batches=SWEEP_VALIDATION_BATCHES,
        batch_size=8,
        sequence_length=128,
        seed=52_012,
    )
    models = load_sweep_models(tokenizer.get_vocab_size())
    device = torch.device("cuda:0")
    rows: list[dict[str, object]] = []
    for name in SWEEP_MODELS:
        model = models[name].to(device)
        for depth in DEPTHS:
            result = evaluate_depth(model, validation, starts, depth=depth, device=device)
            row = {
                "model": name,
                "display_name": DISPLAY[name],
                "depth_per_stage": depth,
                "total_recurrent_iterations": depth * 3,
                **result,
            }
            rows.append(row)
            print(
                f"{name:30s} depth={depth:2d} ppl={result['validation_ppl']:.3f} "
                f"residual={result['mean_relative_residual']:.6f}"
            )
        model.to("cpu")
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "relaxation-sweep.csv", index=False)
    return frame


def summarize(sweep: pd.DataFrame) -> pd.DataFrame:
    source_summary = pd.read_csv(SOURCE_011 / "model-summary.csv").set_index("model")
    rows: list[dict[str, object]] = []
    source_map = {
        "minicells-v2-stable-011": "minicells-v2-stable",
        "minicells-2d-k4-stable-011": "minicells-2d-k4-stable",
    }
    for model, group in sweep.groupby("model", sort=False):
        indexed = group.set_index("depth_per_stage")
        late = indexed.loc[[8, 12, 16], "validation_ppl"]
        residual4 = float(indexed.loc[4, "mean_relative_residual"])
        residual16 = float(indexed.loc[16, "mean_relative_residual"])
        row: dict[str, object] = {
            "model": model,
            "display_name": DISPLAY[model],
            "ppl_depth2": float(indexed.loc[2, "validation_ppl"]),
            "ppl_depth4": float(indexed.loc[4, "validation_ppl"]),
            "ppl_depth8": float(indexed.loc[8, "validation_ppl"]),
            "ppl_depth12": float(indexed.loc[12, "validation_ppl"]),
            "ppl_depth16": float(indexed.loc[16, "validation_ppl"]),
            "late_ppl_plateau_ratio": float(late.max() / late.min()),
            "late_ppl_drift_percent": float((late.max() / late.min() - 1.0) * 100.0),
            "residual_depth4": residual4,
            "residual_depth16": residual16,
            "residual_ratio_16_to_4": residual16 / residual4 if residual4 > 0 else math.nan,
            "best_ppl": float(group["validation_ppl"].min()),
            "best_depth": int(group.loc[group["validation_ppl"].idxmin(), "depth_per_stage"]),
        }
        if model in source_map:
            source = source_summary.loc[source_map[model]]
            row["standard_011_ppl_2m"] = float(source["final_ppl_2m"])
            row["training_seconds_per_million"] = float(source["seconds_per_million_tokens"])
        else:
            worker_name = model.replace("-012", "")
            worker = json.loads((OUT / f"{worker_name}-worker.json").read_text(encoding="utf-8"))
            row["standard_011_ppl_2m"] = math.nan
            row["training_seconds_per_million"] = float(worker["seconds_per_million_tokens"])
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "model-summary.csv", index=False)
    return summary


def _settling_pass(summary: pd.DataFrame, settling_name: str, baseline_name: str) -> dict[str, object]:
    by_model = summary.set_index("model")
    candidate = by_model.loc[settling_name]
    baseline = by_model.loc[baseline_name]
    baseline_standard = float(baseline["standard_011_ppl_2m"])
    quality_ratio = float(candidate["ppl_depth4"] / baseline_standard)
    plateau = float(candidate["late_ppl_plateau_ratio"])
    contraction = float(candidate["residual_ratio_16_to_4"])
    baseline_plateau = float(baseline["late_ppl_plateau_ratio"])
    baseline_contraction = float(baseline["residual_ratio_16_to_4"])
    passed = quality_ratio <= 1.05 and plateau <= 1.02 and contraction <= 0.75
    return {
        "pass": passed,
        "depth4_ppl_ratio_to_011_standard": quality_ratio,
        "late_ppl_plateau_ratio_8_to_16": plateau,
        "late_ppl_drift_percent": float(candidate["late_ppl_drift_percent"]),
        "residual_ratio_16_to_4": contraction,
        "baseline_011_free_run_plateau_ratio": baseline_plateau,
        "baseline_011_free_run_residual_ratio_16_to_4": baseline_contraction,
        "improves_plateau_vs_011": plateau < baseline_plateau,
        "improves_residual_contraction_vs_011": contraction < baseline_contraction,
    }


def make_decision(summary: pd.DataFrame, gpu_count: int) -> dict[str, object]:
    one_d = _settling_pass(
        summary,
        "minicells-v2-settling-012",
        "minicells-v2-stable-011",
    )
    two_d = _settling_pass(
        summary,
        "minicells-2d-k4-settling-012",
        "minicells-2d-k4-stable-011",
    )
    if one_d["pass"] and two_d["pass"]:
        status = "PASS_SETTLING_BOTH"
        diagnosis = "BOTH_1D_AND_2D_LEARN_BOUNDED_FREE_RUNNING_SETTLING_DYNAMICS"
    elif one_d["pass"] or two_d["pass"]:
        status = "PASS_SETTLING_PARTIAL"
        diagnosis = "SETTLING_SIGNAL_IS_ARCHITECTURE_DEPENDENT"
    else:
        status = "NO_SETTLING_YET"
        diagnosis = "CURRENT_FIXED_POINT_OBJECTIVE_DOES_NOT_YET_CREATE_A_STABLE_FREE_RUNNING_PLATEAU"
    return {
        "format": "minicells.language-settling-dynamics.v1",
        "experiment": "MINI Cells Experiment 012 — Shared-Rule Settling Dynamics",
        "status": status,
        "diagnosis": diagnosis,
        "question": "Can a language NCA learn a shared update rule whose state and predictions settle under free-running iterations beyond the trained depth?",
        "training": {
            "tokens_per_new_model": 2_000_000,
            "new_models": list(NEW_MODELS),
            "random_main_depth_per_stage": [2, 4],
            "absolute_step_embedding_used": False,
            "fixed_point_probe": "one additional shared-rule update after each sampled stage depth",
            "state_stability_weight": 0.10,
            "logit_consistency_weight": 0.05,
            "note": "The three probe updates participate in training loss; Experiment 012 tests settling, not lower training cost.",
        },
        "evaluation": {
            "free_run_depths_per_stage": list(DEPTHS),
            "maximum_total_recurrent_iterations": 48,
            "shared_rule": "identical update rule at every iteration; no absolute step identity",
            "pass_criteria": {
                "depth4_ppl_vs_011_standard_max": 1.05,
                "late_ppl_plateau_ratio_8_to_16_max": 1.02,
                "residual_ratio_16_to_4_max": 0.75,
            },
            "gpu_count_used_for_new_training": gpu_count,
        },
        "one_d": one_d,
        "two_d": two_d,
        "interpretation": {
            "pass": "A pass is evidence for bounded settling under an autonomous shared cellular rule, not proof of a unique mathematical fixed point or global convergence.",
            "baseline": "Experiment 011 stable checkpoints are free-run with the same step-independent rule to show whether Experiment 012 improves beyond random-depth training alone.",
            "next_level": "Per-cell event-driven sleeping should only be attempted after a reproducible settling plateau exists.",
        },
    }


def save_plots(sweep: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    for model, group in sweep.groupby("model", sort=False):
        ordered = group.sort_values("depth_per_stage")
        axis.plot(ordered["depth_per_stage"], ordered["validation_ppl"], marker="o", label=DISPLAY[model])
    axis.set_xlabel("Shared-rule iterations per stage")
    axis.set_ylabel("Validation perplexity")
    axis.set_title("Experiment 012 — PPL under free-running relaxation")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ppl-vs-relaxation-depth.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for model, group in sweep.groupby("model", sort=False):
        ordered = group.sort_values("depth_per_stage")
        axis.plot(ordered["depth_per_stage"], ordered["mean_relative_residual"], marker="o", label=DISPLAY[model])
    axis.set_yscale("log")
    axis.set_xlabel("Shared-rule iterations per stage")
    axis.set_ylabel("Mean relative state residual")
    axis.set_title("Experiment 012 — State residual under relaxation")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "residual-vs-relaxation-depth.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(summary["display_name"], summary["late_ppl_drift_percent"])
    axis.axhline(2.0, linestyle="--", linewidth=1, label="2% pass boundary")
    axis.set_ylabel("PPL range from depth 8–16 (%)")
    axis.set_title("Experiment 012 — Late-depth prediction drift")
    axis.tick_params(axis="x", rotation=20)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "late-ppl-drift.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(summary["display_name"], summary["residual_ratio_16_to_4"])
    axis.axhline(0.75, linestyle="--", linewidth=1, label="0.75 pass boundary")
    axis.set_ylabel("Residual(depth 16) / residual(depth 4)")
    axis.set_title("Experiment 012 — Residual contraction")
    axis.tick_params(axis="x", rotation=20)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "residual-contraction.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for model, group in sweep.groupby("model", sort=False):
        ordered = group.sort_values("total_recurrent_iterations")
        axis.plot(
            ordered["total_recurrent_iterations"],
            ordered["validation_ppl"],
            marker="o",
            label=DISPLAY[model],
        )
    axis.set_xlabel("Total recurrent rule applications")
    axis.set_ylabel("Validation perplexity")
    axis.set_title("Experiment 012 — Quality vs relaxation compute")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "quality-vs-relaxation-compute.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache_dir, manifest = prepare_corpus()
    gpu_count = run_training(cache_dir)
    sweep = run_relaxation_sweep(cache_dir)
    summary = summarize(sweep)
    decision = make_decision(summary, gpu_count)
    save_plots(sweep, summary)

    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUT / "corpus-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    task = {
        "format": "minicells.language-settling-task.v1",
        "experiment": "012",
        "new_training_models": list(NEW_MODELS),
        "baseline_source": "artifacts/experiments/011-stabilizing-cost",
        "free_run_depths": list(DEPTHS),
        "validation_batches_per_depth": SWEEP_VALIDATION_BATCHES,
        "primary_metrics": [
            "late_ppl_plateau_ratio",
            "relative_state_residual",
            "residual_ratio_16_to_4",
            "depth4_language_quality",
        ],
    }
    (OUT / "task-spec.json").write_text(
        json.dumps(task, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== decision ===")
    print(json.dumps(decision, indent=2))
    print("=== summary ===")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
