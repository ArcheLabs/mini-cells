from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .language_data import encode_story_stream, iter_tinystories, load_tokenizer
from .language_models import TextNCALM, build_parameter_matched_transformer, count_parameters
from .language_training import estimate_learning_slope


SCALING_CHECKPOINTS = (500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000)
TRAIN_STREAM_TOKENS = 15_000_000
VALIDATION_STREAM_TOKENS = 200_000
MODEL_SEED = 55005
TRANSFORMER_SEED = MODEL_SEED + 1
SCHEDULE_SEED = 5005
WARMUP_STEPS = 100


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def prepare_scaling_corpus(
    root: Path,
    *,
    source_005_dir: Path,
    train_stream_tokens: int = TRAIN_STREAM_TOKENS,
    validation_stream_tokens: int = VALIDATION_STREAM_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor, Path, dict[str, object]]:
    source_tokenizer = source_005_dir / "tokenizer.json"
    source_manifest_path = source_005_dir / "corpus-manifest.json"
    if not source_tokenizer.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError("Experiment 005 tokenizer/corpus manifest must be merged before 006")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_tokenizer_sha = hashlib.sha256(source_tokenizer.read_bytes()).hexdigest()
    if source_tokenizer_sha != source_manifest.get("tokenizer_sha256"):
        raise RuntimeError("Experiment 005 tokenizer hash does not match its corpus manifest")

    cache = root / "results" / "consumer-language-scaling-v1" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer_path = cache / "tokenizer.json"
    train_path = cache / "train-tokens.pt"
    validation_path = cache / "validation-tokens.pt"
    manifest_path = cache / "corpus-manifest.json"

    expected = {
        "format": "minicells.tinystories-scaling-corpus.v1",
        "source_005_tokenizer_sha256": source_tokenizer_sha,
        "train_stream_tokens": train_stream_tokens,
        "validation_stream_tokens": validation_stream_tokens,
    }
    if tokenizer_path.exists() and train_path.exists() and validation_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in expected.items()):
            train = torch.load(train_path, map_location="cpu")
            validation = torch.load(validation_path, map_location="cpu")
            if (
                _tensor_sha256(train) == manifest.get("train_token_sha256")
                and _tensor_sha256(validation) == manifest.get("validation_token_sha256")
                and hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == source_tokenizer_sha
            ):
                return train, validation, tokenizer_path, manifest
        for path in (tokenizer_path, train_path, validation_path, manifest_path):
            path.unlink(missing_ok=True)

    shutil.copy2(source_tokenizer, tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    train, train_stories = encode_story_stream(
        tokenizer,
        iter_tinystories("train"),
        target_tokens=train_stream_tokens,
    )
    validation, validation_stories = encode_story_stream(
        tokenizer,
        iter_tinystories("validation"),
        target_tokens=validation_stream_tokens,
    )

    source_train_tokens = int(source_manifest["train_stream_tokens"])
    source_validation_tokens = int(source_manifest["validation_stream_tokens"])
    prefix_train_sha = _tensor_sha256(train[:source_train_tokens])
    prefix_validation_sha = _tensor_sha256(validation[:source_validation_tokens])
    if prefix_train_sha != source_manifest.get("train_token_sha256"):
        raise RuntimeError("006 training corpus prefix does not reproduce Experiment 005")
    if prefix_validation_sha != source_manifest.get("validation_token_sha256"):
        raise RuntimeError("006 validation corpus prefix does not reproduce Experiment 005")

    torch.save(train, train_path)
    torch.save(validation, validation_path)
    manifest = {
        **expected,
        "dataset": source_manifest.get("dataset"),
        "streaming": True,
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "source_005_train_prefix_tokens": source_train_tokens,
        "source_005_validation_prefix_tokens": source_validation_tokens,
        "source_005_train_prefix_sha256": prefix_train_sha,
        "source_005_validation_prefix_sha256": prefix_validation_sha,
        "train_stories_consumed": train_stories,
        "validation_stories_consumed": validation_stories,
        "train_token_sha256": _tensor_sha256(train),
        "validation_token_sha256": _tensor_sha256(validation),
        "tokenizer_sha256": source_tokenizer_sha,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return train, validation, tokenizer_path, manifest


def build_minicells_v2(vocab_size: int) -> TextNCALM:
    return TextNCALM(
        vocab_size=vocab_size,
        rms_norm=False,
        carry_bias=2.0,
        tie_embeddings=True,
        stage_supervision=False,
    )


def build_scaling_models(vocab_size: int):
    torch.manual_seed(MODEL_SEED)
    minicells = build_minicells_v2(vocab_size)
    target_parameters = count_parameters(minicells)
    torch.manual_seed(TRANSFORMER_SEED)
    transformer, transformer_config = build_parameter_matched_transformer(
        vocab_size,
        target_parameters,
    )
    return minicells, transformer, transformer_config


def summarize_scaling(checkpoints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_models = {"minicells-v2", "transformer-s"}
    if set(checkpoints["model"].unique()) != required_models:
        raise ValueError(f"expected exactly {sorted(required_models)}")
    expected_tokens = set(SCALING_CHECKPOINTS)
    for model, group in checkpoints.groupby("model"):
        if set(group["consumed_tokens"].astype(int)) != expected_tokens:
            raise ValueError(f"{model} is missing one or more scaling checkpoints")

    pivot = checkpoints.pivot(index="consumed_tokens", columns="model", values="validation_ppl").sort_index()
    ratios = pd.DataFrame(
        {
            "consumed_tokens": pivot.index.astype(int),
            "minicells_ppl": pivot["minicells-v2"].to_numpy(),
            "transformer_ppl": pivot["transformer-s"].to_numpy(),
            "ppl_ratio": (pivot["minicells-v2"] / pivot["transformer-s"]).to_numpy(),
        }
    )

    rows: list[dict[str, object]] = []
    for model, group in checkpoints.groupby("model"):
        ordered = group.sort_values("consumed_tokens")
        last = ordered.iloc[-1]
        rows.append(
            {
                "model": model,
                "parameters": int(ordered.iloc[0]["parameters"]),
                "ppl_500k": float(ordered.loc[ordered["consumed_tokens"] == 500_000, "validation_ppl"].iloc[0]),
                "ppl_1m": float(ordered.loc[ordered["consumed_tokens"] == 1_000_000, "validation_ppl"].iloc[0]),
                "ppl_2m": float(ordered.loc[ordered["consumed_tokens"] == 2_000_000, "validation_ppl"].iloc[0]),
                "ppl_5m": float(ordered.loc[ordered["consumed_tokens"] == 5_000_000, "validation_ppl"].iloc[0]),
                "ppl_10m": float(last["validation_ppl"]),
                "nll_10m": float(last["validation_nll"]),
                "learning_slope_alpha": estimate_learning_slope(ordered),
                "elapsed_seconds": float(last["elapsed_seconds"]),
                "tokens_per_second": float(last["tokens_per_second"]),
                "peak_vram_bytes": int(last["peak_vram_bytes"]),
            }
        )
    return pd.DataFrame(rows), ratios


def make_scaling_decision(
    summary: pd.DataFrame,
    ratios: pd.DataFrame,
    *,
    source_005b_ppl: float,
) -> dict[str, object]:
    by_model = summary.set_index("model")
    candidate = by_model.loc["minicells-v2"]
    transformer = by_model.loc["transformer-s"]
    ratios = ratios.sort_values("consumed_tokens")
    ratio_500k = float(ratios.iloc[0]["ppl_ratio"])
    ratio_10m = float(ratios.iloc[-1]["ppl_ratio"])
    ratio_change = ratio_10m - ratio_500k
    alpha_candidate = float(candidate["learning_slope_alpha"])
    alpha_transformer = float(transformer["learning_slope_alpha"])
    slope_ratio = alpha_candidate / alpha_transformer if alpha_transformer > 0 else 0.0

    if ratio_10m <= 1.15 and slope_ratio >= 0.90 and ratio_change <= 0.05:
        status = "GREEN"
        diagnosis = "MINICELLS_SCALING_COMPETITIVE_TO_10M"
    elif ratio_10m <= 1.30 and slope_ratio >= 0.80 and ratio_change <= 0.10:
        status = "YELLOW"
        diagnosis = "MINICELLS_SCALING_GAP_PERSISTS_BUT_REMAINS_VIABLE"
    else:
        status = "RED"
        diagnosis = "MINICELLS_SCALING_DISADVANTAGE_EMERGES"

    return {
        "format": "minicells.consumer-language-scaling.v1",
        "experiment": "MINI Cells Experiment 006 — 10M Language Scaling",
        "status": status,
        "diagnosis": diagnosis,
        "budget": {
            "tokens_per_model": 10_000_000,
            "checkpoints": list(SCALING_CHECKPOINTS),
            "train_stream_tokens": TRAIN_STREAM_TOKENS,
            "validation_stream_tokens": VALIDATION_STREAM_TOKENS,
        },
        "candidate": {
            "model": "minicells-v2",
            "source_005b_best_ppl_500k": source_005b_ppl,
            "ppl_500k": float(candidate["ppl_500k"]),
            "ppl_10m": float(candidate["ppl_10m"]),
            "learning_slope_alpha": alpha_candidate,
        },
        "transformer": {
            "ppl_500k": float(transformer["ppl_500k"]),
            "ppl_10m": float(transformer["ppl_10m"]),
            "learning_slope_alpha": alpha_transformer,
        },
        "comparison": {
            "ppl_ratio_500k": ratio_500k,
            "ppl_ratio_10m": ratio_10m,
            "ratio_change_500k_to_10m": ratio_change,
            "slope_ratio_to_transformer": slope_ratio,
            "ratio_trajectory": [float(value) for value in ratios["ppl_ratio"]],
        },
        "interpretation": {
            "green": "10M PPL ratio <=1.15, slope ratio >=0.90, and the relative gap does not materially worsen.",
            "yellow": "10M PPL ratio <=1.30 with a still-competitive slope and bounded gap growth.",
            "red": "The relative gap or learning slope indicates a scaling disadvantage before 10M tokens.",
        },
    }


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_scaling_plots(checkpoints: pd.DataFrame, ratios: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    for model, group in checkpoints.groupby("model"):
        ordered = group.sort_values("consumed_tokens")
        axis.plot(ordered["consumed_tokens"], ordered["validation_ppl"], marker="o", label=model)
    axis.set_xscale("log")
    axis.set_xlabel("Consumed training tokens")
    axis.set_ylabel("Validation perplexity")
    axis.set_title("Experiment 006 — 10M perplexity scaling")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(fig, output_dir / "ppl-scaling.png")

    fig, axis = plt.subplots(figsize=(8, 5))
    for model, group in checkpoints.groupby("model"):
        ordered = group.sort_values("consumed_tokens")
        axis.plot(ordered["consumed_tokens"], ordered["validation_nll"], marker="o", label=model)
    axis.set_xscale("log")
    axis.set_xlabel("Consumed training tokens")
    axis.set_ylabel("Validation NLL")
    axis.set_title("Experiment 006 — validation NLL scaling")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(fig, output_dir / "nll-scaling.png")

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(ratios["consumed_tokens"], ratios["ppl_ratio"], marker="o")
    axis.axhline(1.0, linewidth=1, linestyle="--")
    axis.axhline(1.15, linewidth=1, linestyle=":", label="competitive threshold 1.15×")
    axis.axhline(1.30, linewidth=1, linestyle=":", label="viability ceiling 1.30×")
    axis.set_xscale("log")
    axis.set_xlabel("Consumed training tokens")
    axis.set_ylabel("MiniCells / Transformer PPL")
    axis.set_title("Experiment 006 — relative scaling gap")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(fig, output_dir / "relative-gap.png")


def write_generation_progression(generations: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Experiment 006 Generation Progression",
        "",
        "Fixed prompts use the same sampling policy and deterministic generation seeds as Experiment 005.",
        "Samples are qualitative; the scaling decision uses validation perplexity/NLL.",
        "",
    ]
    frame = pd.DataFrame(generations).sort_values(["consumed_tokens", "model", "prompt"])
    for consumed, token_group in frame.groupby("consumed_tokens"):
        lines.extend([f"## {int(consumed):,} consumed tokens", ""])
        for model, model_group in token_group.groupby("model"):
            lines.extend([f"### {model}", ""])
            for row in model_group.itertuples(index=False):
                lines.extend([f"**Prompt:** `{row.prompt}`", "", str(row.text).replace("\n", " "), ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
