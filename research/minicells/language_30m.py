from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .language_data import iter_tinystories, load_tokenizer
from .language_models import TextNCALM, TransformerLM, count_parameters
from .language_training import estimate_learning_slope


EXPERIMENT_ID = "007"
FORMAT = "minicells.language-30m.v1"
MODEL_NAME = "minicells-30m-v0"
TRANSFORMER_NAME = "transformer-30m"

TARGET_TOKENS = 100_000_000
CHECKPOINT_TOKENS = (10_000_000, 25_000_000, 50_000_000, 75_000_000, 100_000_000)
RESUME_INTERVAL_TOKENS = 5_000_000
TRAIN_STREAM_TOKENS = 120_000_000
VALIDATION_STREAM_TOKENS = 1_000_000

CONTEXT_LENGTH = 128
TRAIN_SEQUENCE_LENGTH = 125
BATCH_SIZE = 8
TOKENS_PER_STEP = BATCH_SIZE * TRAIN_SEQUENCE_LENGTH
MODEL_SEED = 57007
TRANSFORMER_SEED = MODEL_SEED + 1
SCHEDULE_SEED = 7007
VALIDATION_SEED = 7107
WARMUP_STEPS = 1_000
BASE_LR = 3e-4
WEIGHT_DECAY = 0.1

MINICELLS_DIM = 720
MINICELLS_HEADS = 8
MINICELLS_FFN = 2_880
MINICELLS_WINDOWS = (8, 32, 128)
MINICELLS_ITERATIONS = (4, 4, 4)
MINICELLS_CARRY_BIAS = 2.0

TRANSFORMER_DIM = 512
TRANSFORMER_HEADS = 8
TRANSFORMER_FFN = 2_048
TRANSFORMER_LAYERS = 9


@dataclass(frozen=True)
class MemmapCorpus:
    train_path: Path
    validation_path: Path
    tokenizer_path: Path
    manifest: dict[str, object]


def build_minicells_30m(vocab_size: int) -> TextNCALM:
    return TextNCALM(
        vocab_size=vocab_size,
        max_context=CONTEXT_LENGTH,
        dim=MINICELLS_DIM,
        heads=MINICELLS_HEADS,
        ffn_dim=MINICELLS_FFN,
        windows=MINICELLS_WINDOWS,
        iterations=MINICELLS_ITERATIONS,
        rms_norm=False,
        carry_bias=MINICELLS_CARRY_BIAS,
        tie_embeddings=True,
        stage_supervision=False,
    )


def build_transformer_30m(vocab_size: int) -> tuple[TransformerLM, dict[str, int | float]]:
    model = TransformerLM(
        vocab_size=vocab_size,
        max_context=CONTEXT_LENGTH,
        dim=TRANSFORMER_DIM,
        heads=TRANSFORMER_HEADS,
        ffn_dim=TRANSFORMER_FFN,
        layers=TRANSFORMER_LAYERS,
        tie_embeddings=True,
    )
    target = count_parameters(build_minicells_30m(vocab_size))
    parameters = count_parameters(model)
    error = abs(parameters - target) / target
    return model, {
        "dim": TRANSFORMER_DIM,
        "heads": TRANSFORMER_HEADS,
        "ffn_dim": TRANSFORMER_FFN,
        "layers": TRANSFORMER_LAYERS,
        "parameters": parameters,
        "target_parameters": target,
        "relative_parameter_error": error,
    }


def model_parameter_summary(vocab_size: int) -> dict[str, object]:
    minicells = build_minicells_30m(vocab_size)
    transformer, match = build_transformer_30m(vocab_size)
    result = {
        MODEL_NAME: count_parameters(minicells),
        TRANSFORMER_NAME: count_parameters(transformer),
        "relative_parameter_error": float(match["relative_parameter_error"]),
    }
    del minicells, transformer
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_int64_sha256(values: np.memmap, count: int) -> str:
    if count < 0 or count > len(values):
        raise ValueError("invalid prefix length")
    digest = hashlib.sha256()
    chunk = 1_000_000
    for start in range(0, count, chunk):
        stop = min(count, start + chunk)
        converted = np.asarray(values[start:stop], dtype=np.int64)
        digest.update(converted.tobytes())
    return digest.hexdigest()


def _write_story_stream(
    tokenizer: object,
    *,
    split: str,
    output_path: Path,
    target_tokens: int,
) -> int:
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise RuntimeError("tokenizer does not contain <eos>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(target_tokens,))
    cursor = 0
    stories = 0
    try:
        for text in iter_tinystories(split):
            ids = tokenizer.encode(text).ids
            if not ids:
                continue
            ids.append(eos_id)
            take = min(len(ids), target_tokens - cursor)
            values[cursor : cursor + take] = np.asarray(ids[:take], dtype=np.uint16)
            cursor += take
            stories += 1
            if cursor >= target_tokens:
                break
        if cursor != target_tokens:
            raise RuntimeError(
                f"TinyStories {split} stream ended at {cursor:,} tokens; required {target_tokens:,}"
            )
        values.flush()
    finally:
        del values
    return stories


def open_memmap(path: Path) -> np.memmap:
    size = path.stat().st_size
    if size % np.dtype(np.uint16).itemsize:
        raise RuntimeError(f"invalid uint16 token file size: {path}")
    return np.memmap(path, dtype=np.uint16, mode="r")


def prepare_30m_corpus(
    root: Path,
    *,
    source_006_dir: Path,
    train_stream_tokens: int = TRAIN_STREAM_TOKENS,
    validation_stream_tokens: int = VALIDATION_STREAM_TOKENS,
) -> MemmapCorpus:
    source_tokenizer = source_006_dir / "tokenizer.json"
    source_manifest_path = source_006_dir / "corpus-manifest.json"
    if not source_tokenizer.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError("Experiment 006 tokenizer/corpus manifest must exist before 007")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_tokenizer_sha = hashlib.sha256(source_tokenizer.read_bytes()).hexdigest()
    if source_tokenizer_sha != source_manifest.get("tokenizer_sha256"):
        raise RuntimeError("Experiment 006 tokenizer hash does not match its manifest")

    cache = root / "results" / "consumer-language-30m-v1" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer_path = cache / "tokenizer.json"
    train_path = cache / "train.u16"
    validation_path = cache / "validation.u16"
    manifest_path = cache / "corpus-manifest.json"

    expected = {
        "format": "minicells.tinystories-30m-corpus.v1",
        "dtype": "uint16",
        "source_006_tokenizer_sha256": source_tokenizer_sha,
        "train_stream_tokens": train_stream_tokens,
        "validation_stream_tokens": validation_stream_tokens,
    }
    if tokenizer_path.exists() and train_path.exists() and validation_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        valid_shape = (
            train_path.stat().st_size == train_stream_tokens * 2
            and validation_path.stat().st_size == validation_stream_tokens * 2
        )
        if (
            valid_shape
            and all(manifest.get(key) == value for key, value in expected.items())
            and _sha256_file(train_path) == manifest.get("train_raw_sha256")
            and _sha256_file(validation_path) == manifest.get("validation_raw_sha256")
            and hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == source_tokenizer_sha
        ):
            return MemmapCorpus(train_path, validation_path, tokenizer_path, manifest)
        for path in (tokenizer_path, train_path, validation_path, manifest_path):
            path.unlink(missing_ok=True)

    shutil.copy2(source_tokenizer, tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max:
        raise RuntimeError("uint16 corpus storage requires vocabulary <= 65535")

    train_stories = _write_story_stream(
        tokenizer,
        split="train",
        output_path=train_path,
        target_tokens=train_stream_tokens,
    )
    validation_stories = _write_story_stream(
        tokenizer,
        split="validation",
        output_path=validation_path,
        target_tokens=validation_stream_tokens,
    )

    train = open_memmap(train_path)
    validation = open_memmap(validation_path)
    source_train_tokens = int(source_manifest["train_stream_tokens"])
    source_validation_tokens = int(source_manifest["validation_stream_tokens"])
    train_prefix_sha = _legacy_int64_sha256(train, source_train_tokens)
    validation_prefix_sha = _legacy_int64_sha256(validation, source_validation_tokens)
    if train_prefix_sha != source_manifest.get("train_token_sha256"):
        raise RuntimeError("007 training corpus prefix does not reproduce Experiment 006")
    if validation_prefix_sha != source_manifest.get("validation_token_sha256"):
        raise RuntimeError("007 validation corpus prefix does not reproduce Experiment 006")

    manifest = {
        **expected,
        "dataset": source_manifest.get("dataset"),
        "streaming": True,
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "source_006_train_prefix_tokens": source_train_tokens,
        "source_006_validation_prefix_tokens": source_validation_tokens,
        "source_006_train_prefix_sha256": train_prefix_sha,
        "source_006_validation_prefix_sha256": validation_prefix_sha,
        "train_stories_consumed": train_stories,
        "validation_stories_consumed": validation_stories,
        "train_raw_sha256": _sha256_file(train_path),
        "validation_raw_sha256": _sha256_file(validation_path),
        "tokenizer_sha256": source_tokenizer_sha,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del train, validation
    return MemmapCorpus(train_path, validation_path, tokenizer_path, manifest)


def memmap_batch(
    token_stream: np.memmap,
    starts: tuple[int, ...],
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = np.stack(
        [np.asarray(token_stream[start : start + sequence_length + 1], dtype=np.int64) for start in starts],
        axis=0,
    )
    packed = torch.from_numpy(rows).to(device, non_blocking=True)
    return packed[:, :-1], packed[:, 1:]


def summarize_30m(checkpoints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {MODEL_NAME, TRANSFORMER_NAME}
    if set(checkpoints["model"].unique()) != required:
        raise ValueError(f"expected exactly {sorted(required)}")
    expected_tokens = set(CHECKPOINT_TOKENS)
    for model, group in checkpoints.groupby("model"):
        if set(group["consumed_tokens"].astype(int)) != expected_tokens:
            raise ValueError(f"{model} is missing one or more 30M checkpoints")

    pivot = checkpoints.pivot(index="consumed_tokens", columns="model", values="validation_ppl").sort_index()
    ratios = pd.DataFrame(
        {
            "consumed_tokens": pivot.index.astype(int),
            "minicells_ppl": pivot[MODEL_NAME].to_numpy(),
            "transformer_ppl": pivot[TRANSFORMER_NAME].to_numpy(),
            "ppl_ratio": (pivot[MODEL_NAME] / pivot[TRANSFORMER_NAME]).to_numpy(),
        }
    )
    rows: list[dict[str, object]] = []
    for model, group in checkpoints.groupby("model"):
        ordered = group.sort_values("consumed_tokens")
        last = ordered.iloc[-1]
        row: dict[str, object] = {
            "model": model,
            "parameters": int(ordered.iloc[0]["parameters"]),
            "nll_100m": float(last["validation_nll"]),
            "learning_slope_alpha": estimate_learning_slope(ordered),
            "elapsed_seconds": float(last["elapsed_seconds"]),
            "tokens_per_second": float(last["tokens_per_second"]),
            "peak_vram_bytes": int(last["peak_vram_bytes"]),
        }
        for tokens in CHECKPOINT_TOKENS:
            label = f"ppl_{tokens // 1_000_000}m"
            row[label] = float(
                ordered.loc[ordered["consumed_tokens"] == tokens, "validation_ppl"].iloc[0]
            )
        rows.append(row)
    return pd.DataFrame(rows), ratios


def make_30m_decision(
    summary: pd.DataFrame,
    ratios: pd.DataFrame,
    *,
    source_006_ratio_10m: float,
) -> dict[str, object]:
    by_model = summary.set_index("model")
    candidate = by_model.loc[MODEL_NAME]
    transformer = by_model.loc[TRANSFORMER_NAME]
    ordered = ratios.sort_values("consumed_tokens")
    ratio_10m = float(ordered.iloc[0]["ppl_ratio"])
    ratio_100m = float(ordered.iloc[-1]["ppl_ratio"])
    ratio_change = ratio_100m - ratio_10m
    alpha_candidate = float(candidate["learning_slope_alpha"])
    alpha_transformer = float(transformer["learning_slope_alpha"])
    slope_ratio = alpha_candidate / alpha_transformer if alpha_transformer > 0 else 0.0

    if ratio_100m <= 1.15 and slope_ratio >= 0.90 and ratio_change <= 0.05:
        status = "GREEN"
        diagnosis = "MINICELLS_30M_PARAMETER_SCALING_COMPETITIVE"
    elif ratio_100m <= 1.30 and slope_ratio >= 0.80 and ratio_change <= 0.10:
        status = "YELLOW"
        diagnosis = "MINICELLS_30M_REMAINS_VIABLE_WITH_PARAMETER_SCALE"
    else:
        status = "RED"
        diagnosis = "MINICELLS_30M_PARAMETER_SCALING_DISADVANTAGE"

    return {
        "format": FORMAT,
        "experiment": "MINI Cells Experiment 007 — MiniCells-30M v0",
        "status": status,
        "diagnosis": diagnosis,
        "budget": {
            "tokens_per_model": TARGET_TOKENS,
            "checkpoints": list(CHECKPOINT_TOKENS),
            "train_stream_tokens": TRAIN_STREAM_TOKENS,
            "validation_stream_tokens": VALIDATION_STREAM_TOKENS,
        },
        "candidate": {
            "model": MODEL_NAME,
            "parameters": int(candidate["parameters"]),
            "ppl_100m": float(candidate["ppl_100m"]),
            "nll_100m": float(candidate["nll_100m"]),
            "learning_slope_alpha": alpha_candidate,
        },
        "transformer": {
            "model": TRANSFORMER_NAME,
            "parameters": int(transformer["parameters"]),
            "ppl_100m": float(transformer["ppl_100m"]),
            "nll_100m": float(transformer["nll_100m"]),
            "learning_slope_alpha": alpha_transformer,
        },
        "comparison": {
            "ppl_ratio_10m": ratio_10m,
            "ppl_ratio_100m": ratio_100m,
            "ratio_change_10m_to_100m": ratio_change,
            "ratio_trajectory": [float(value) for value in ordered["ppl_ratio"]],
            "slope_ratio_to_transformer": slope_ratio,
            "source_006_1m_parameter_ratio_at_10m": source_006_ratio_10m,
        },
        "interpretation": {
            "green": "At ~30M parameters, MiniCells remains within 1.15x Transformer PPL at 100M tokens with a competitive learning slope and bounded gap growth.",
            "yellow": "The 30M model remains viable but shows a measurable parameter-scaling gap that warrants optimization.",
            "red": "A material parameter-scaling disadvantage emerges by 30M parameters.",
        },
    }


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_30m_plots(checkpoints: pd.DataFrame, ratios: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    for model, group in checkpoints.groupby("model"):
        ordered = group.sort_values("consumed_tokens")
        axis.plot(ordered["consumed_tokens"], ordered["validation_ppl"], marker="o", label=model)
    axis.set_xscale("log")
    axis.set_xlabel("Consumed training tokens")
    axis.set_ylabel("Validation perplexity")
    axis.set_title("Experiment 007 — ~30M parameter language scaling")
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
    axis.set_title("Experiment 007 — validation NLL")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(fig, output_dir / "nll-scaling.png")

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(ratios["consumed_tokens"], ratios["ppl_ratio"], marker="o")
    axis.axhline(1.0, linewidth=1, linestyle="--")
    axis.axhline(1.15, linewidth=1, linestyle=":", label="competitive threshold 1.15x")
    axis.axhline(1.30, linewidth=1, linestyle=":", label="viability ceiling 1.30x")
    axis.set_xscale("log")
    axis.set_xlabel("Consumed training tokens")
    axis.set_ylabel("MiniCells-30M / Transformer-30M PPL")
    axis.set_title("Experiment 007 — relative gap")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(fig, output_dir / "relative-gap.png")


def write_generation_progression(generations: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Experiment 007 Generation Progression",
        "",
        "Fixed prompts use deterministic sampling. Quantitative decisions use validation NLL/PPL.",
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
