from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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

OUT = ROOT / "results" / "language-2d-latent-tissue-v1"
SOURCE_005 = ROOT / "artifacts" / "experiments" / "005-consumer-language-bridge"
WORKER = ROOT / "scripts" / "run_language_2d_variant.py"
DEFAULT_MODELS = ("minicells-v2", "minicells-2d-k4")
CHECKPOINTS = (250_000, 500_000, 1_000_000, 2_000_000)
TRAIN_STREAM_TOKENS = 3_000_000
VALIDATION_STREAM_TOKENS = 200_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MINI Cells Experiment 009.")
    parser.add_argument(
        "--include-k2",
        action="store_true",
        help="Also run the K=2 tissue variant after the primary 1D vs K=4 comparison.",
    )
    return parser.parse_args()


def tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def prepare_corpus() -> tuple[Path, dict[str, object]]:
    source_tokenizer = SOURCE_005 / "tokenizer.json"
    source_manifest_path = SOURCE_005 / "corpus-manifest.json"
    if not source_tokenizer.is_file() or not source_manifest_path.is_file():
        raise FileNotFoundError("Experiment 005 tokenizer/corpus manifest must exist before 009")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_tokenizer_sha = hashlib.sha256(source_tokenizer.read_bytes()).hexdigest()
    if source_tokenizer_sha != source_manifest.get("tokenizer_sha256"):
        raise RuntimeError("Experiment 005 tokenizer hash does not match its corpus manifest")

    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer_path = cache / "tokenizer.json"
    train_path = cache / "train-tokens.pt"
    validation_path = cache / "validation-tokens.pt"
    manifest_path = cache / "corpus-manifest.json"
    expected = {
        "format": "minicells.language-2d-corpus.v1",
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
        for path in (tokenizer_path, train_path, validation_path, manifest_path):
            path.unlink(missing_ok=True)

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

    source_train_tokens = int(source_manifest["train_stream_tokens"])
    source_validation_tokens = int(source_manifest["validation_stream_tokens"])
    if tensor_sha256(train[:source_train_tokens]) != source_manifest.get("train_token_sha256"):
        raise RuntimeError("009 training corpus prefix does not reproduce Experiment 005")
    if tensor_sha256(validation[:source_validation_tokens]) != source_manifest.get("validation_token_sha256"):
        raise RuntimeError("009 validation corpus prefix does not reproduce Experiment 005")

    torch.save(train, train_path)
    torch.save(validation, validation_path)
    manifest = {
        **expected,
        "dataset": source_manifest.get("dataset"),
        "streaming": True,
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "train_stories_consumed": train_stories,
        "validation_stories_consumed": validation_stories,
        "train_token_sha256": tensor_sha256(train),
        "validation_token_sha256": tensor_sha256(validation),
        "tokenizer_sha256": source_tokenizer_sha,
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


def run_batch(models: tuple[str, ...], cache_dir: Path) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 009 requires at least one CUDA GPU")
    if available == 1 or len(models) == 1:
        for model in models:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            log_path = OUT / f"{model}.log"
            with log_path.open("w", encoding="utf-8") as handle:
                result = subprocess.run(
                    worker_command(model, cache_dir),
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            print(f"--- {model} / GPU 0 ---")
            print(log_path.read_text(encoding="utf-8").rstrip())
            if result.returncode != 0:
                raise RuntimeError(f"{model} worker failed with exit code {result.returncode}")
        return 1

    active: list[tuple[str, int, subprocess.Popen[str], Path]] = []
    for gpu_index, model in enumerate(models[:2]):
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
        active.append((model, gpu_index, process, log_path))
        handle.close()
        print(f"started {model:18s} on physical GPU {gpu_index}")

    failures: list[str] = []
    for model, gpu_index, process, log_path in active:
        code = process.wait()
        print(f"--- {model} / GPU {gpu_index} ---")
        print(log_path.read_text(encoding="utf-8").rstrip())
        if code != 0:
            failures.append(f"{model} exited {code}; see {log_path}")
    if failures:
        raise RuntimeError("; ".join(failures))
    return min(2, available)


def summarize(models: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [pd.read_csv(OUT / f"{model}-checkpoints.csv") for model in models]
    checkpoints = pd.concat(frames, ignore_index=True)
    checkpoints.to_csv(OUT / "checkpoints.csv", index=False)
    rows: list[dict[str, object]] = []
    for model, group in checkpoints.groupby("model"):
        ordered = group.sort_values("consumed_tokens")
        final = ordered.iloc[-1]
        rows.append(
            {
                "model": model,
                "parameters": int(final["parameters"]),
                "ppl_250k": float(ordered.loc[ordered["consumed_tokens"] == 250_000, "validation_ppl"].iloc[0]),
                "ppl_500k": float(ordered.loc[ordered["consumed_tokens"] == 500_000, "validation_ppl"].iloc[0]),
                "ppl_1m": float(ordered.loc[ordered["consumed_tokens"] == 1_000_000, "validation_ppl"].iloc[0]),
                "ppl_2m": float(final["validation_ppl"]),
                "tokens_per_second": float(final["tokens_per_second"]),
                "peak_vram_bytes": int(final["peak_vram_bytes"]),
                "elapsed_seconds": float(final["elapsed_seconds"]),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "model-summary.csv", index=False)
    return checkpoints, summary


def make_decision(summary: pd.DataFrame) -> dict[str, object]:
    by_model = summary.set_index("model")
    baseline = by_model.loc["minicells-v2"]
    candidate = by_model.loc["minicells-2d-k4"]
    ppl_ratio = float(candidate["ppl_2m"] / baseline["ppl_2m"])
    parameter_ratio = float(candidate["parameters"] / baseline["parameters"])
    throughput_ratio = float(candidate["tokens_per_second"] / baseline["tokens_per_second"])
    vram_ratio = float(candidate["peak_vram_bytes"] / max(1, baseline["peak_vram_bytes"]))

    diagnostics = json.loads((OUT / "minicells-2d-k4-tissue-diagnostics.json").read_text(encoding="utf-8"))
    latent_cosines = [abs(float(value)) for value in diagnostics["row_cosine_to_token"][1:]]
    differentiated = bool(latent_cosines) and max(latent_cosines) < 0.98

    if ppl_ratio <= 1.0:
        status = "PASS_PERFORMANCE"
        diagnosis = "2D_LATENT_TISSUE_MATCHES_OR_BEATS_1D_AT_2M"
    elif ppl_ratio <= 1.10 and differentiated:
        status = "PASS_DYNAMICS"
        diagnosis = "2D_TISSUE_SHOWS_DISTINCT_REPRESENTATIONS_WITH_BOUNDED_PPL_COST"
    else:
        status = "NO_SIGNAL"
        diagnosis = "2D_TISSUE_DOES_NOT_YET_JUSTIFY_ITS_COMPUTE_COST"

    return {
        "format": "minicells.language-2d-latent-tissue.v1",
        "experiment": "MINI Cells Experiment 009 — 2D Latent Tissue",
        "status": status,
        "diagnosis": diagnosis,
        "primary_question": "Does a causal latent-tissue axis provide useful computation beyond the 1D MiniCells-v2 topology?",
        "budget": {
            "tokens_per_model": 2_000_000,
            "checkpoints": list(CHECKPOINTS),
            "context_length": 128,
        },
        "comparison": {
            "ppl_ratio_2d_to_1d_at_2m": ppl_ratio,
            "parameter_ratio_2d_to_1d": parameter_ratio,
            "throughput_ratio_2d_to_1d": throughput_ratio,
            "peak_vram_ratio_2d_to_1d": vram_ratio,
        },
        "tissue": {
            "row_cosine_to_token": diagnostics["row_cosine_to_token"],
            "differentiated_below_abs_cosine_0_98": differentiated,
            "latent_row_ablation_ppl": diagnostics["latent_row_ablation_ppl"],
            "row_update_rms_flat_by_stage": diagnostics["row_update_rms_flat_by_stage"],
        },
        "interpretation": {
            "pass_performance": "K=4 reaches equal or lower validation perplexity than the unchanged 1D baseline at the same token budget.",
            "pass_dynamics": "K=4 stays within 10% PPL while its latent rows remain measurably distinct from the token row.",
            "no_signal": "This first factorized 2D topology does not yet show enough language or tissue-differentiation signal to justify its added compute.",
        },
    }


def save_plots(checkpoints: pd.DataFrame, summary: pd.DataFrame, decision: dict[str, object]) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    for model, group in checkpoints.groupby("model"):
        ordered = group.sort_values("consumed_tokens")
        axis.plot(ordered["consumed_tokens"], ordered["validation_ppl"], marker="o", label=model)
    axis.set_xscale("log")
    axis.set_xlabel("Consumed training tokens")
    axis.set_ylabel("Validation perplexity")
    axis.set_title("Experiment 009 — 1D vs 2D latent tissue")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ppl-comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(summary["model"], summary["tokens_per_second"])
    axis.set_ylabel("Training tokens / second")
    axis.set_title("Experiment 009 — training throughput")
    axis.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(OUT / "throughput.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    cosines = decision["tissue"]["row_cosine_to_token"]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar([f"row {index}" for index in range(len(cosines))], cosines)
    axis.set_ylabel("Mean cosine to token row")
    axis.set_title("Experiment 009 — final tissue differentiation")
    fig.tight_layout()
    fig.savefig(OUT / "tissue-cosine.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_task_files(models: tuple[str, ...], corpus_manifest: dict[str, object]) -> None:
    task = {
        "format": "minicells.language-2d-task.v1",
        "goal": "Isolate the effect of adding a causal latent-tissue axis while preserving the MiniCells-v2 language recipe.",
        "models": list(models),
        "primary_models": list(DEFAULT_MODELS),
        "budget_tokens_per_model": 2_000_000,
        "checkpoints": list(CHECKPOINTS),
        "same_tokenizer_as_005": True,
        "same_language_recipe_as_006": {
            "dim": 128,
            "heads": 4,
            "ffn_dim": 512,
            "windows": [8, 32, 128],
            "iterations": [4, 4, 4],
            "normalization": "LayerNorm",
            "gru_carry_bias": 2.0,
        },
        "2d_change_only": {
            "tissue_height": 4,
            "horizontal_update": "existing causal local attention, shared independently over tissue rows",
            "vertical_update": "depthwise 3-cell same-position convolution",
            "decoder": "token row 0 only",
        },
    }
    (OUT / "task-spec.json").write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "corpus-manifest.json").write_text(
        json.dumps(corpus_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    cache_dir, corpus_manifest = prepare_corpus()
    models = DEFAULT_MODELS + (("minicells-2d-k2",) if args.include_k2 else ())

    print(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "models": models,
            "budget_tokens_per_model": 2_000_000,
        }
    )

    gpu_count_used = run_batch(DEFAULT_MODELS, cache_dir)
    if args.include_k2:
        run_batch(("minicells-2d-k2",), cache_dir)

    checkpoints, summary = summarize(models)
    decision = make_decision(summary)
    decision["runtime"] = {
        "gpu_count_used_for_primary_comparison": gpu_count_used,
        "physical_gpus": [torch.cuda.get_device_name(i) for i in range(min(gpu_count_used, torch.cuda.device_count()))],
    }
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_plots(checkpoints, summary, decision)
    write_task_files(models, corpus_manifest)

    print("=== decision ===")
    print(json.dumps(decision, indent=2))
    print("=== model summary ===")
    print(summary.to_string(index=False))
    print("=== files ===")
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            print(path.name, path.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
