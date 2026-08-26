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

matplotlib.use("Agg")

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))

from minicells.language_data import encode_story_stream, iter_tinystories, load_tokenizer  # noqa: E402
from minicells.language_depth_ablation import (  # noqa: E402
    VARIANTS,
    factorial_contrast,
    geometric_ratio_from_log_contrast,
)


OUT = ROOT / "results" / "language-depth-ablation-v1"
SOURCE_005 = ROOT / "artifacts" / "experiments" / "005-consumer-language-bridge"
SOURCE_009 = ROOT / "artifacts" / "experiments" / "009-2d-latent-tissue"
SOURCE_011 = ROOT / "artifacts" / "experiments" / "011-stabilizing-cost"
WORKER = ROOT / "scripts" / "run_language_depth_ablation_variant.py"
TOPOLOGIES = ("1d", "2d")
CODES = tuple(variant.code for variant in VARIANTS)
CHECKPOINTS = (250_000, 500_000, 1_000_000, 2_000_000)
TRAIN_STREAM_TOKENS = 3_000_000
VALIDATION_STREAM_TOKENS = 200_000
FACTOR_TERMS = (
    ("random_depth",),
    ("low_step_init",),
    ("stability_loss",),
    ("random_depth", "low_step_init"),
    ("random_depth", "stability_loss"),
    ("low_step_init", "stability_loss"),
    ("random_depth", "low_step_init", "stability_loss"),
)
FACTOR_LABEL = {
    ("random_depth",): "Random depth",
    ("low_step_init",): "Step init 0.25",
    ("stability_loss",): "Residual loss",
    ("random_depth", "low_step_init"): "Random × step init",
    ("random_depth", "stability_loss"): "Random × residual",
    ("low_step_init", "stability_loss"): "Step init × residual",
    ("random_depth", "low_step_init", "stability_loss"): "3-way interaction",
}
PURE_CONTRASTS = {
    "random_only_scale1_reg0": ("B", "A"),
    "random_only_scale025_reg0": ("C", "E"),
    "random_only_scale1_reg01": ("H", "F"),
    "random_only_scale025_reg01": ("D", "G"),
    "step_init_after_random_reg0": ("C", "B"),
    "step_init_only_fixed_reg0": ("E", "A"),
    "residual_after_random_scale025": ("D", "C"),
    "residual_only_fixed_scale1": ("F", "A"),
}


def tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def prepare_corpus() -> tuple[Path, dict[str, object]]:
    source_tokenizer = SOURCE_005 / "tokenizer.json"
    source_manifest_path = SOURCE_005 / "corpus-manifest.json"
    source_009_manifest_path = SOURCE_009 / "corpus-manifest.json"
    if not source_tokenizer.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError("Experiment 005 tokenizer/corpus manifest must exist before 013")
    if not source_009_manifest_path.is_file():
        raise FileNotFoundError("Experiment 009 corpus manifest must exist before 013")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_009_manifest = json.loads(source_009_manifest_path.read_text(encoding="utf-8"))
    source_tokenizer_sha = hashlib.sha256(source_tokenizer.read_bytes()).hexdigest()
    if source_tokenizer_sha != source_manifest.get("tokenizer_sha256"):
        raise RuntimeError("Experiment 005 tokenizer hash mismatch")

    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer_path = cache / "tokenizer.json"
    train_path = cache / "train-tokens.pt"
    validation_path = cache / "validation-tokens.pt"
    manifest_path = cache / "corpus-manifest.json"
    expected = {
        "format": "minicells.language-depth-ablation-corpus.v1",
        "source_005_tokenizer_sha256": source_tokenizer_sha,
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
                and hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == source_tokenizer_sha
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
    if train_sha != source_009_manifest.get("train_token_sha256"):
        raise RuntimeError("Experiment 013 train stream does not reproduce Experiment 009")
    if validation_sha != source_009_manifest.get("validation_token_sha256"):
        raise RuntimeError("Experiment 013 validation stream does not reproduce Experiment 009")

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
        "tokenizer_sha256": source_tokenizer_sha,
        "reproduces_009_corpus": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache, manifest


def worker_command(topology: str, code: str, cache_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(WORKER),
        "--topology",
        topology,
        "--variant",
        code,
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(OUT),
    ]


def run_models(cache_dir: Path) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 013 requires CUDA")
    gpu_count = min(2, available)

    jobs: list[tuple[str, str]] = []
    for index, code in enumerate(CODES):
        pair = [("1d", code), ("2d", code)]
        # Balance the two topologies across physical GPUs over the eight cells.
        if index % 2:
            pair.reverse()
        jobs.extend(pair)

    while jobs:
        batch = jobs[:gpu_count]
        jobs = jobs[gpu_count:]
        active: list[tuple[str, int, subprocess.Popen[str], Path, object]] = []
        for gpu_index, (topology, code) in enumerate(batch):
            run_name = f"{topology}-{code}"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            log_path = OUT / f"{run_name}.log"
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                worker_command(topology, code, cache_dir),
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append((run_name, gpu_index, process, log_path, handle))
            print(f"started {run_name:6s} on physical GPU {gpu_index}")
        failures: list[str] = []
        for run_name, gpu_index, process, log_path, handle in active:
            code = process.wait()
            handle.close()
            print(f"--- {run_name} / GPU {gpu_index} ---")
            print(log_path.read_text(encoding="utf-8").rstrip())
            if code != 0:
                failures.append(f"{run_name} exited {code}; see {log_path}")
        if failures:
            raise RuntimeError("; ".join(failures))
    return gpu_count


def summarize() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint_frames = []
    depth_frames = []
    worker_rows: list[dict[str, object]] = []
    for topology in TOPOLOGIES:
        for variant in VARIANTS:
            run_name = f"{topology}-{variant.code}"
            checkpoint_frames.append(pd.read_csv(OUT / f"{run_name}-checkpoints.csv"))
            depth_frames.append(pd.read_csv(OUT / f"{run_name}-depth-eval.csv"))
            worker = json.loads((OUT / f"{run_name}-worker.json").read_text(encoding="utf-8"))
            final = checkpoint_frames[-1].sort_values("consumed_tokens").iloc[-1]
            worker_rows.append(
                {
                    "run": run_name,
                    "topology": topology,
                    "variant": variant.code,
                    "random_depth": variant.random_depth,
                    "low_step_init": variant.low_step_init,
                    "step_embedding_init_scale": variant.step_embedding_init_scale,
                    "stability_loss": variant.uses_stability_loss,
                    "stability_weight": variant.stability_weight,
                    "parameters": int(worker["parameters"]),
                    "final_ppl_2m": float(final["validation_ppl"]),
                    "final_nll_2m": float(final["validation_nll"]),
                    "training_elapsed_seconds": float(worker["training_elapsed_seconds"]),
                    "training_tokens_per_second": float(worker["training_tokens_per_second"]),
                    "seconds_per_million_tokens": float(worker["seconds_per_million_tokens"]),
                    "peak_vram_gib": float(worker["peak_vram_bytes"] / (1024**3)),
                    "avg_recurrent_iterations": float(worker["avg_recurrent_iterations"]),
                    "initial_step_embedding_rms": float(worker["initial_step_embedding_rms"]),
                    "final_step_embedding_rms": float(worker["final_step_embedding_rms"]),
                    "step_embedding_rms_growth_ratio": float(worker["step_embedding_rms_growth_ratio"]),
                    "depth_robustness_ratio_2_to_4": float(worker["depth_robustness_ratio_2_to_4"]),
                    "ppl_depth2": float(worker["ppl_depth2"]),
                    "ppl_depth3": float(worker["ppl_depth3"]),
                    "ppl_depth4": float(worker["ppl_depth4"]),
                }
            )
    checkpoints = pd.concat(checkpoint_frames, ignore_index=True)
    depths = pd.concat(depth_frames, ignore_index=True)
    summary = pd.DataFrame(worker_rows)
    checkpoints.to_csv(OUT / "checkpoints.csv", index=False)
    depths.to_csv(OUT / "depth-eval.csv", index=False)
    summary.to_csv(OUT / "model-summary.csv", index=False)
    return checkpoints, depths, summary


def factor_effects(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = (
        "final_ppl_2m",
        "seconds_per_million_tokens",
        "depth_robustness_ratio_2_to_4",
        "final_step_embedding_rms",
    )
    for topology in TOPOLOGIES:
        group = summary.loc[summary["topology"] == topology].set_index("variant")
        for metric in metrics:
            log_values = {code: math.log(float(group.loc[code, metric])) for code in CODES}
            for factors in FACTOR_TERMS:
                contrast = factorial_contrast(log_values, factors)
                ratio = geometric_ratio_from_log_contrast(contrast)
                rows.append(
                    {
                        "topology": topology,
                        "metric": metric,
                        "term": "*".join(factors),
                        "label": FACTOR_LABEL[factors],
                        "order": len(factors),
                        "log_contrast": contrast,
                        "ratio_high_to_low": ratio,
                        "percent_change_high_vs_low": (ratio - 1.0) * 100.0,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "factorial-effects.csv", index=False)
    return frame


def pure_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = ("final_ppl_2m", "seconds_per_million_tokens", "depth_robustness_ratio_2_to_4")
    for topology in TOPOLOGIES:
        group = summary.loc[summary["topology"] == topology].set_index("variant")
        for label, (high, low) in PURE_CONTRASTS.items():
            for metric in metrics:
                high_value = float(group.loc[high, metric])
                low_value = float(group.loc[low, metric])
                rows.append(
                    {
                        "topology": topology,
                        "contrast": label,
                        "high_variant": high,
                        "low_variant": low,
                        "metric": metric,
                        "high_value": high_value,
                        "low_value": low_value,
                        "ratio_high_to_low": high_value / low_value,
                        "percent_change": (high_value / low_value - 1.0) * 100.0,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "pure-contrasts.csv", index=False)
    return frame


def replication_check(summary: pd.DataFrame) -> pd.DataFrame:
    if not (SOURCE_011 / "model-summary.csv").is_file():
        raise FileNotFoundError("Experiment 011 results must be merged before 013")
    source = pd.read_csv(SOURCE_011 / "model-summary.csv").set_index("model")
    current = summary.set_index(["topology", "variant"])
    mapping = {
        ("1d", "A"): "minicells-v2-fixed",
        ("1d", "D"): "minicells-v2-stable",
        ("2d", "A"): "minicells-2d-k4-fixed",
        ("2d", "D"): "minicells-2d-k4-stable",
    }
    rows = []
    for key, source_name in mapping.items():
        row = current.loc[key]
        source_row = source.loc[source_name]
        rows.append(
            {
                "topology": key[0],
                "variant": key[1],
                "source_011_model": source_name,
                "ppl_013": float(row["final_ppl_2m"]),
                "ppl_011": float(source_row["final_ppl_2m"]),
                "ppl_ratio_013_to_011": float(row["final_ppl_2m"] / source_row["final_ppl_2m"]),
                "cost_013": float(row["seconds_per_million_tokens"]),
                "cost_011": float(source_row["seconds_per_million_tokens"]),
                "cost_ratio_013_to_011": float(row["seconds_per_million_tokens"] / source_row["seconds_per_million_tokens"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "replication-check.csv", index=False)
    return frame


def _ratio(contrasts: pd.DataFrame, topology: str, name: str, metric: str) -> float:
    row = contrasts.loc[
        (contrasts["topology"] == topology)
        & (contrasts["contrast"] == name)
        & (contrasts["metric"] == metric)
    ].iloc[0]
    return float(row["ratio_high_to_low"])


def _main_effect(effects: pd.DataFrame, topology: str, factor: str, metric: str) -> float:
    row = effects.loc[
        (effects["topology"] == topology)
        & (effects["term"] == factor)
        & (effects["metric"] == metric)
    ].iloc[0]
    return float(row["ratio_high_to_low"])


def make_decision(
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    contrasts: pd.DataFrame,
    replication: pd.DataFrame,
    gpu_count: int,
) -> dict[str, object]:
    topology_results: dict[str, object] = {}
    random_pure_both = True
    for topology in TOPOLOGIES:
        pure_random = _ratio(contrasts, topology, "random_only_scale1_reg0", "final_ppl_2m")
        random_main = _main_effect(effects, topology, "random_depth", "final_ppl_2m")
        scale_main = _main_effect(effects, topology, "low_step_init", "final_ppl_2m")
        residual_main = _main_effect(effects, topology, "stability_loss", "final_ppl_2m")
        random_robustness = _main_effect(effects, topology, "random_depth", "depth_robustness_ratio_2_to_4")
        random_cost = _main_effect(effects, topology, "random_depth", "seconds_per_million_tokens")
        random_pure_both = random_pure_both and pure_random < 1.0
        topology_results[topology] = {
            "pure_random_depth_ppl_ratio_B_to_A": pure_random,
            "factorial_main_effect_ppl_ratios": {
                "random_depth": random_main,
                "low_step_init": scale_main,
                "stability_loss": residual_main,
            },
            "random_depth_main_effect_training_cost_ratio": random_cost,
            "random_depth_main_effect_depth_robustness_ratio": random_robustness,
            "step_init_after_random_no_reg_ppl_ratio_C_to_B": _ratio(
                contrasts, topology, "step_init_after_random_reg0", "final_ppl_2m"
            ),
            "residual_after_random_low_step_ppl_ratio_D_to_C": _ratio(
                contrasts, topology, "residual_after_random_scale025", "final_ppl_2m"
            ),
            "stability_loss_only_fixed_scale1_ppl_ratio_F_to_A": _ratio(
                contrasts, topology, "residual_only_fixed_scale1", "final_ppl_2m"
            ),
        }

    if random_pure_both:
        diagnosis = "RANDOM_DEPTH_HAS_A_MATCHED_SINGLE_SEED_QUALITY_SIGNAL_IN_BOTH_TOPOLOGIES"
    else:
        diagnosis = "RANDOM_DEPTH_EFFECT_IS_MIXED_ACROSS_TOPOLOGIES"

    replication_max_ppl_deviation = float((replication["ppl_ratio_013_to_011"] - 1.0).abs().max())
    return {
        "format": "minicells.language-depth-ablation.v1",
        "experiment": "MINI Cells Experiment 013 — Random-Depth Ablation",
        "status": "ABLATION_COMPLETE",
        "diagnosis": diagnosis,
        "question": "Which Experiment 011 ingredient explains the quality/cost gain: random recurrent depth, low step-embedding initialization, residual stability loss, or their interactions?",
        "design": {
            "type": "complete 2x2x2 matched-seed factorial",
            "topologies": list(TOPOLOGIES),
            "variants_per_topology": len(VARIANTS),
            "models_total": len(VARIANTS) * len(TOPOLOGIES),
            "tokens_per_model": 2_000_000,
            "factors": {
                "random_depth": {"low": [4, 4, 4], "high": "per-stage Uniform{2,3,4}"},
                "step_embedding_init_scale": {"low": 1.0, "high": 0.25, "note": "initialization-only scaling; embeddings remain trainable"},
                "stability_weight": {"low": 0.0, "high": 0.10},
            },
            "same_1d_initialization_across_cells": True,
            "same_2d_initialization_across_cells": True,
            "same_training_schedule_across_cells": True,
            "same_random_depth_schedule_where_enabled": True,
        },
        "evaluation": {
            "standard_quality": "fixed (4,4,4) validation PPL at 2M tokens",
            "temporal_robustness": "PPL at (2,2,2), (3,3,3), and (4,4,4); lower max/min ratio is better",
            "training_cost": "synchronized forward/backward/optimizer wall clock, validation excluded",
            "step_clock_diagnostic": "initial and final trainable step-embedding RMS",
            "gpu_count_used_for_parallel_execution": gpu_count,
        },
        "results": topology_results,
        "replication": {
            "compares_A_and_D_to_experiment_011": True,
            "max_absolute_ppl_ratio_deviation_from_011": replication_max_ppl_deviation,
        },
        "scope": {
            "single_seed": True,
            "interpretation": "Matched initialization isolates within-seed effects, but a multi-seed confirmation is required before treating any small effect as architecture-general.",
            "important": "This experiment tests the actual Experiment 011 recipe. The 0.25 step-embedding factor is initialization scaling, not a permanent forward-time clock attenuation.",
        },
    }


def _save(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_plots(summary: pd.DataFrame, depths: pd.DataFrame, effects: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    ordered = list(CODES)
    x = list(range(len(ordered)))
    width = 0.36

    fig, axis = plt.subplots(figsize=(10, 5.5))
    for offset, topology in ((-width / 2, "1d"), (width / 2, "2d")):
        group = summary.loc[summary["topology"] == topology].set_index("variant").loc[ordered]
        axis.bar([value + offset for value in x], group["final_ppl_2m"], width=width, label=topology.upper())
    axis.set_xticks(x, ordered)
    axis.set_ylabel("Validation PPL @ 2M (fixed 4/4/4 evaluation)")
    axis.set_title("Experiment 013 — full random-depth ablation")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save(fig, "factorial-ppl.png")

    fig, axis = plt.subplots(figsize=(10, 5.5))
    for offset, topology in ((-width / 2, "1d"), (width / 2, "2d")):
        group = summary.loc[summary["topology"] == topology].set_index("variant").loc[ordered]
        axis.bar([value + offset for value in x], group["seconds_per_million_tokens"], width=width, label=topology.upper())
    axis.set_xticks(x, ordered)
    axis.set_ylabel("Measured training seconds / 1M tokens")
    axis.set_title("Training cost by factorial cell")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save(fig, "factorial-training-cost.png")

    main = effects.loc[(effects["metric"] == "final_ppl_2m") & (effects["order"] == 1)].copy()
    labels = ["Random depth", "Step init 0.25", "Residual loss"]
    terms = ["random_depth", "low_step_init", "stability_loss"]
    x_main = list(range(3))
    fig, axis = plt.subplots(figsize=(9, 5.2))
    for offset, topology in ((-width / 2, "1d"), (width / 2, "2d")):
        values = []
        for term in terms:
            row = main.loc[(main["topology"] == topology) & (main["term"] == term)].iloc[0]
            values.append(float(row["percent_change_high_vs_low"]))
        axis.bar([value + offset for value in x_main], values, width=width, label=topology.upper())
    axis.axhline(0.0, linewidth=1)
    axis.set_xticks(x_main, labels)
    axis.set_ylabel("Factorial main effect on PPL (%)\nnegative = improvement")
    axis.set_title("Which Experiment 011 ingredient changes quality?")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save(fig, "factor-main-effects-ppl.png")

    random_rows = contrasts.loc[
        (contrasts["metric"] == "final_ppl_2m")
        & contrasts["contrast"].str.startswith("random_only_")
    ].copy()
    contexts = [
        "random_only_scale1_reg0",
        "random_only_scale025_reg0",
        "random_only_scale1_reg01",
        "random_only_scale025_reg01",
    ]
    context_labels = ["scale1 / no reg", "scale.25 / no reg", "scale1 / reg", "scale.25 / reg"]
    fig, axis = plt.subplots(figsize=(10, 5.2))
    x_ctx = list(range(4))
    for offset, topology in ((-width / 2, "1d"), (width / 2, "2d")):
        values = []
        for context in contexts:
            row = random_rows.loc[
                (random_rows["topology"] == topology) & (random_rows["contrast"] == context)
            ].iloc[0]
            values.append(float(row["percent_change"]))
        axis.bar([value + offset for value in x_ctx], values, width=width, label=topology.upper())
    axis.axhline(0.0, linewidth=1)
    axis.set_xticks(x_ctx, context_labels, rotation=15)
    axis.set_ylabel("Random-depth matched PPL change (%)\nnegative = improvement")
    axis.set_title("Random depth isolated in all four matched contexts")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save(fig, "random-depth-isolation.png")

    for topology in TOPOLOGIES:
        fig, axis = plt.subplots(figsize=(10, 6))
        group = depths.loc[depths["topology"] == topology]
        for variant in ordered:
            line = group.loc[group["variant"] == variant].sort_values("depth_per_stage")
            axis.plot(line["depth_per_stage"], line["validation_ppl"], marker="o", label=variant)
        axis.set_xlabel("Recurrent depth per stage")
        axis.set_ylabel("Validation PPL")
        axis.set_title(f"{topology.upper()} temporal robustness after training")
        axis.set_xticks([2, 3, 4])
        axis.legend(ncol=4, fontsize=8)
        axis.grid(alpha=0.25)
        _save(fig, f"{topology}-depth-robustness.png")

    fig, axis = plt.subplots(figsize=(10, 5.2))
    for offset, topology in ((-width / 2, "1d"), (width / 2, "2d")):
        group = summary.loc[summary["topology"] == topology].set_index("variant").loc[ordered]
        axis.bar([value + offset for value in x], group["step_embedding_rms_growth_ratio"], width=width, label=topology.upper())
    axis.axhline(1.0, linewidth=1)
    axis.set_xticks(x, ordered)
    axis.set_ylabel("Final / initial step-embedding RMS")
    axis.set_title("Did the trainable step clock regrow after initialization scaling?")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save(fig, "step-embedding-growth.png")


def write_task_spec() -> None:
    payload = {
        "format": "minicells.language-depth-ablation-task.v1",
        "experiment": "013",
        "name": "Random-Depth Ablation",
        "variants": [
            {
                "code": variant.code,
                "random_depth": variant.random_depth,
                "step_embedding_init_scale": variant.step_embedding_init_scale,
                "stability_weight": variant.stability_weight,
            }
            for variant in VARIANTS
        ],
        "topologies": list(TOPOLOGIES),
        "tokens_per_model": 2_000_000,
        "checkpoints": list(CHECKPOINTS),
        "primary_metrics": [
            "fixed-depth validation PPL at 2M",
            "measured seconds per 1M tokens",
            "depth 2/3/4 PPL robustness",
            "step-embedding RMS growth",
        ],
    }
    (OUT / "task-spec.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache, manifest = prepare_corpus()
    shutil.copy2(cache / "corpus-manifest.json", OUT / "corpus-manifest.json")
    write_task_spec()
    gpu_count = run_models(cache)
    checkpoints, depths, summary = summarize()
    effects = factor_effects(summary)
    contrasts = pure_contrasts(summary)
    replication = replication_check(summary)
    decision = make_decision(summary, effects, contrasts, replication, gpu_count)
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_plots(summary, depths, effects, contrasts)
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"results: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
