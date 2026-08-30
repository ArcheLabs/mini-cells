from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))

from minicells.language_data import (  # noqa: E402
    fixed_validation_starts,
    load_tokenizer,
    make_training_schedule,
    prepare_tinystories_corpus,
)
from minicells.language_models import (  # noqa: E402
    build_minitextnca_plus,
    build_parameter_matched_transformer,
    build_textnca_control,
    count_parameters,
)
from minicells.language_report import (  # noqa: E402
    save_consumer_summary,
    save_learning_slopes,
    save_ppl_scaling,
    save_relative_gap,
    save_throughput,
    save_training_curves,
    write_generation_progression,
    write_model_configs,
)
from minicells.language_training import (  # noqa: E402
    estimate_learning_slope,
    train_language_model,
)

OUT = ROOT / "results" / "consumer-language-bridge-v1"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUDGET_TOKENS = 500_000
VOCAB_SIZE = 2048
TRAIN_STREAM_TOKENS = 800_000
VALIDATION_STREAM_TOKENS = 100_000
MODEL_SEED = 55005

print(
    {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(DEVICE),
        "budget_tokens_per_model": BUDGET_TOKENS,
    }
)
if DEVICE.type != "cuda":
    print("WARNING: Experiment 005 is designed for a Kaggle T4; CUDA is not available.")

corpus = prepare_tinystories_corpus(
    ROOT,
    vocab_size=VOCAB_SIZE,
    train_stream_tokens=TRAIN_STREAM_TOKENS,
    validation_stream_tokens=VALIDATION_STREAM_TOKENS,
)
tokenizer = load_tokenizer(corpus.tokenizer_path)
actual_vocab = tokenizer.get_vocab_size()
shutil.copy2(corpus.tokenizer_path, OUT / "tokenizer.json")
(OUT / "corpus-manifest.json").write_text(
    json.dumps(corpus.manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

schedule = make_training_schedule(
    int(corpus.train.numel()),
    budget_tokens=BUDGET_TOKENS,
    batch_size=8,
    sequence_length=125,
)
validation_starts = fixed_validation_starts(
    int(corpus.validation.numel()),
    batches=24,
    batch_size=8,
    sequence_length=128,
)
assert schedule.consumed_tokens == BUDGET_TOKENS

# Use the same seed for the two NCA variants so matching-shaped parameters begin
# from comparable random draws. The S+ differences are architectural/training choices.
torch.manual_seed(MODEL_SEED)
textnca = build_textnca_control(actual_vocab)
torch.manual_seed(MODEL_SEED)
minitextnca = build_minitextnca_plus(actual_vocab)
textnca_parameters = count_parameters(textnca)
minitextnca_parameters = count_parameters(minitextnca)

torch.manual_seed(MODEL_SEED + 1)
transformer, transformer_match = build_parameter_matched_transformer(
    actual_vocab,
    minitextnca_parameters,
)
transformer_parameters = count_parameters(transformer)
if float(transformer_match["relative_parameter_error"]) > 0.05:
    raise RuntimeError(
        "Unable to parameter-match Transformer within 5%: "
        f"{transformer_match['relative_parameter_error']:.2%}"
    )

model_configs = {
    "format": "minicells.consumer-language-models.v1",
    "shared": {
        "dataset": "roneneldan/TinyStories",
        "vocab_size": actual_vocab,
        "context_length": 128,
        "training_budget_tokens": BUDGET_TOKENS,
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "warmup_steps": 50,
        "gradient_clip": 1.0,
        "token_embedding_lm_head_tied": True,
    },
    "textnca-s": {
        "parameters": textnca_parameters,
        "dim": 128,
        "heads": 4,
        "ffn_dim": 512,
        "windows": [8, 32, 128],
        "iterations": [4, 4, 4],
        "normalization": "LayerNorm",
        "gru_carry_bias": 0.0,
        "auxiliary_stage_losses": None,
    },
    "minitextnca-s-plus": {
        "parameters": minitextnca_parameters,
        "dim": 128,
        "heads": 4,
        "ffn_dim": 512,
        "windows": [8, 32, 128],
        "iterations": [4, 4, 4],
        "normalization": "RMSNorm",
        "gru_carry_bias": 2.0,
        "auxiliary_stage_losses": [0.1, 0.2],
    },
    "transformer-s": {
        **transformer_match,
        "dim": 128,
        "heads": 4,
        "normalization": "RMSNorm",
    },
}
write_model_configs(model_configs, OUT / "model-configs.json")
print("model parameters", {
    "textnca-s": textnca_parameters,
    "minitextnca-s-plus": minitextnca_parameters,
    "transformer-s": transformer_parameters,
    "transformer_match_error": transformer_match["relative_parameter_error"],
})

runs = []
runs.append(
    train_language_model(
        name="textnca-s",
        model=textnca,
        train_stream=corpus.train,
        validation_stream=corpus.validation,
        schedule=schedule,
        validation_starts=validation_starts,
        tokenizer=tokenizer,
        output_dir=OUT,
        device=DEVICE,
        auxiliary_weights=None,
        seed=MODEL_SEED,
    )
)
runs.append(
    train_language_model(
        name="minitextnca-s-plus",
        model=minitextnca,
        train_stream=corpus.train,
        validation_stream=corpus.validation,
        schedule=schedule,
        validation_starts=validation_starts,
        tokenizer=tokenizer,
        output_dir=OUT,
        device=DEVICE,
        auxiliary_weights=(0.1, 0.2),
        seed=MODEL_SEED,
    )
)
runs.append(
    train_language_model(
        name="transformer-s",
        model=transformer,
        train_stream=corpus.train,
        validation_stream=corpus.validation,
        schedule=schedule,
        validation_starts=validation_starts,
        tokenizer=tokenizer,
        output_dir=OUT,
        device=DEVICE,
        auxiliary_weights=None,
        seed=MODEL_SEED + 1,
    )
)

combined = pd.concat([run.metrics for run in runs], ignore_index=True)
combined.to_csv(OUT / "checkpoints.csv", index=False)
all_generations = [item for run in runs for item in run.generations]
(OUT / "generation-samples.json").write_text(
    json.dumps(all_generations, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
write_generation_progression(all_generations, OUT / "generation-progression.md")

parameter_map = {
    "textnca-s": textnca_parameters,
    "minitextnca-s-plus": minitextnca_parameters,
    "transformer-s": transformer_parameters,
}
summary_rows = []
for run in runs:
    frame = run.metrics.sort_values("consumed_tokens")
    by_tokens = frame.set_index("consumed_tokens")
    summary_rows.append(
        {
            "model": run.name,
            "parameters": parameter_map[run.name],
            "ppl_125k": float(by_tokens.loc[125_000, "validation_ppl"]),
            "ppl_250k": float(by_tokens.loc[250_000, "validation_ppl"]),
            "ppl_500k": float(by_tokens.loc[500_000, "validation_ppl"]),
            "nll_500k": float(by_tokens.loc[500_000, "validation_nll"]),
            "learning_slope_alpha": estimate_learning_slope(frame),
            "elapsed_seconds": run.elapsed_seconds,
            "tokens_per_second": BUDGET_TOKENS / run.elapsed_seconds,
            "peak_vram_bytes": run.peak_vram_bytes,
        }
    )
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "model-summary.csv", index=False)

pivot = combined.pivot(index="consumed_tokens", columns="model", values="validation_ppl").sort_index()
ratios = pd.DataFrame(
    {
        "consumed_tokens": pivot.index,
        "textnca_to_transformer": pivot["textnca-s"] / pivot["transformer-s"],
        "minitextnca_plus_to_transformer": pivot["minitextnca-s-plus"] / pivot["transformer-s"],
    }
)
ratios.to_csv(OUT / "relative-gap.csv", index=False)

candidate = summary.set_index("model").loc["minitextnca-s-plus"]
baseline = summary.set_index("model").loc["transformer-s"]
control = summary.set_index("model").loc["textnca-s"]
ratio_125 = float(ratios.loc[ratios["consumed_tokens"] == 125_000, "minitextnca_plus_to_transformer"].iloc[0])
ratio_250 = float(ratios.loc[ratios["consumed_tokens"] == 250_000, "minitextnca_plus_to_transformer"].iloc[0])
ratio_500 = float(ratios.loc[ratios["consumed_tokens"] == 500_000, "minitextnca_plus_to_transformer"].iloc[0])
ratio_improvement = ratio_125 - ratio_500
slope_ratio = (
    float(candidate["learning_slope_alpha"]) / float(baseline["learning_slope_alpha"])
    if float(baseline["learning_slope_alpha"]) > 0
    else 0.0
)

if ratio_500 <= 1.25:
    status = "GREEN"
    diagnosis = "MINITEXTNCA_COMPETITIVE_AT_500K_CONTINUE_SCALING"
elif ratio_500 <= 1.60 and (slope_ratio >= 0.80 or ratio_improvement >= 0.05):
    status = "YELLOW"
    diagnosis = "MINITEXTNCA_SCALING_SIGNAL_WORTH_CONTINUING"
else:
    status = "RED"
    diagnosis = "MINITEXTNCA_EARLY_SCALING_GAP_TOO_LARGE"

decision = {
    "format": "minicells.consumer-language-bridge.v1",
    "experiment": "MINI Cells Experiment 005 — 500K Consumer Language Model Bridge",
    "status": status,
    "diagnosis": diagnosis,
    "budget": {
        "tokens_per_model": BUDGET_TOKENS,
        "checkpoints": [125_000, 250_000, 500_000],
        "same_training_schedule": True,
    },
    "parameter_matching": {
        "minitextnca_plus_parameters": minitextnca_parameters,
        "transformer_parameters": transformer_parameters,
        "relative_error": transformer_match["relative_parameter_error"],
        "within_5_percent": float(transformer_match["relative_parameter_error"]) <= 0.05,
    },
    "candidate": {
        "model": "minitextnca-s-plus",
        "ppl_125k": float(candidate["ppl_125k"]),
        "ppl_250k": float(candidate["ppl_250k"]),
        "ppl_500k": float(candidate["ppl_500k"]),
        "ppl_ratio_trajectory": [ratio_125, ratio_250, ratio_500],
        "ppl_ratio_500k": ratio_500,
        "ratio_improvement_125k_to_500k": ratio_improvement,
        "learning_slope_alpha": float(candidate["learning_slope_alpha"]),
        "slope_ratio_to_transformer": slope_ratio,
    },
    "transformer": {
        "ppl_500k": float(baseline["ppl_500k"]),
        "learning_slope_alpha": float(baseline["learning_slope_alpha"]),
    },
    "structural_control": {
        "model": "textnca-s",
        "ppl_500k": float(control["ppl_500k"]),
        "plus_improvement_fraction": (
            float(control["ppl_500k"]) - float(candidate["ppl_500k"])
        ) / float(control["ppl_500k"]),
    },
    "interpretation": {
        "green": "Candidate PPL is <=1.25x Transformer at 500K; continue to >=2M tokens.",
        "yellow": "Gap is 1.25-1.60x but early scaling slope/gap trend remains competitive; extend cautiously.",
        "red": "Gap is too large and is not closing fast enough; redesign before spending a larger token budget.",
    },
    "runtime": {
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    },
}
(OUT / "decision.json").write_text(
    json.dumps(decision, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

task_spec = {
    "format": "minicells.consumer-language-task.v1",
    "dataset": "roneneldan/TinyStories",
    "tokenizer": "ByteLevel BPE",
    "vocab_size": actual_vocab,
    "max_context": 128,
    "training_sequence_length": 125,
    "batch_size": 8,
    "tokens_per_step": 1000,
    "training_steps": 500,
    "consumed_tokens": BUDGET_TOKENS,
    "checkpoints": [125_000, 250_000, 500_000],
    "validation_batches": 24,
    "generation": {"temperature": 0.8, "top_k": 40, "new_tokens": 32},
}
(OUT / "task-spec.json").write_text(json.dumps(task_spec, indent=2) + "\n", encoding="utf-8")

save_training_curves(combined, OUT / "training-curves.png")
save_ppl_scaling(combined, OUT / "ppl-scaling.png")
save_relative_gap(combined, "transformer-s", OUT / "relative-gap.png")
save_learning_slopes(summary, OUT / "learning-slope.png")
save_throughput(summary, OUT / "throughput.png")
save_consumer_summary(summary, decision, OUT / "consumer-readiness-summary.png")

print("=== decision ===")
print(json.dumps(decision, indent=2))
print("=== files ===")
for path in sorted(OUT.iterdir()):
    if path.is_file():
        print(path.name, path.stat().st_size)
