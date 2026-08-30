from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "src").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))

from minicells.language_ablation import (  # noqa: E402
    FACTORIAL_SPECS,
    factorial_effects,
    save_effects,
    save_factorial_learning_curves,
    save_factorial_ppl,
    save_replication_comparison,
    validate_factorial_specs,
)
from minicells.language_data import prepare_tinystories_corpus  # noqa: E402

OUT = ROOT / "results" / "consumer-language-ablation-v1"
SOURCE_005 = ROOT / "artifacts" / "experiments" / "005-consumer-language-bridge"
SOURCE_DECISION = SOURCE_005 / "decision.json"
SOURCE_MANIFEST = SOURCE_005 / "corpus-manifest.json"
WORKER = ROOT / "scripts" / "run_consumer_language_ablation_variant.py"
BUDGET_TOKENS = 500_000
VOCAB_SIZE = 2048
TRAIN_STREAM_TOKENS = 800_000
VALIDATION_STREAM_TOKENS = 100_000
REPLICATION_TOLERANCE = 0.05


def require_source_005() -> tuple[dict[str, object], dict[str, object]]:
    if not SOURCE_DECISION.is_file() or not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(
            "Experiment 005 artifacts must be merged before 005B. "
            f"Missing {SOURCE_DECISION} or {SOURCE_MANIFEST}."
        )
    decision = json.loads(SOURCE_DECISION.read_text(encoding="utf-8"))
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if decision.get("format") != "minicells.consumer-language-bridge.v1":
        raise RuntimeError(f"unexpected Experiment 005 format: {decision.get('format')!r}")
    return decision, manifest


def verify_same_corpus(current: dict[str, object], baseline: dict[str, object]) -> None:
    keys = (
        "dataset",
        "vocab_size_actual",
        "train_stream_tokens",
        "validation_stream_tokens",
        "train_token_sha256",
        "validation_token_sha256",
        "tokenizer_sha256",
    )
    mismatches = {
        key: {"005": baseline.get(key), "005B": current.get(key)}
        for key in keys
        if current.get(key) != baseline.get(key)
    }
    if mismatches:
        raise RuntimeError(
            "Experiment 005B must reuse the exact Experiment 005 corpus/tokenizer identity. "
            f"Mismatches: {json.dumps(mismatches, sort_keys=True)}"
        )


def worker_command(spec, cache_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(WORKER),
        "--name",
        spec.name,
        "--rms-norm",
        "1" if spec.rms_norm else "0",
        "--carry-bias",
        "1" if spec.carry_bias else "0",
        "--auxiliary-loss",
        "1" if spec.auxiliary_loss else "0",
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(OUT),
    ]


def run_wave(specs, gpu_indices: list[int], cache_dir: Path) -> None:
    active: list[tuple[object, int, subprocess.Popen[str], Path]] = []
    for spec, gpu_index in zip(specs, gpu_indices):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        log_path = OUT / f"{spec.name}.log"
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            worker_command(spec, cache_dir),
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        active.append((spec, gpu_index, process, log_path))
        handle.close()
        print(f"started {spec.name:12s} on physical GPU {gpu_index}")

    failures: list[str] = []
    for spec, gpu_index, process, log_path in active:
        code = process.wait()
        log = log_path.read_text(encoding="utf-8")
        print(f"--- {spec.name} / GPU {gpu_index} ---")
        print(log.rstrip())
        if code != 0:
            failures.append(f"{spec.name} exited {code}; see {log_path}")
    if failures:
        raise RuntimeError("; ".join(failures))


def write_generation_progression(generations: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Experiment 005B Generation Progression",
        "",
        "Qualitative fixed-prompt samples from the eight factorial cells.",
        "Quantitative attribution uses validation NLL/perplexity, not sample preference.",
        "",
    ]
    frame = pd.DataFrame(generations).sort_values(["consumed_tokens", "model", "prompt"])
    for consumed, token_group in frame.groupby("consumed_tokens"):
        lines.extend([f"## {int(consumed):,} consumed tokens", ""])
        for model, model_group in token_group.groupby("model"):
            lines.extend([f"### {model}", ""])
            for row in model_group.itertuples(index=False):
                lines.extend(
                    [
                        f"**Prompt:** `{row.prompt}`",
                        "",
                        str(row.text).replace("\n", " "),
                        "",
                    ]
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    validate_factorial_specs()
    source_decision, source_manifest = require_source_005()

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    print(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "budget_tokens_per_cell": BUDGET_TOKENS,
            "factorial_cells": len(FACTORIAL_SPECS),
        }
    )
    if torch.cuda.device_count() < 1:
        raise RuntimeError("Experiment 005B requires at least one CUDA GPU")

    corpus = prepare_tinystories_corpus(
        ROOT,
        vocab_size=VOCAB_SIZE,
        train_stream_tokens=TRAIN_STREAM_TOKENS,
        validation_stream_tokens=VALIDATION_STREAM_TOKENS,
    )
    verify_same_corpus(corpus.manifest, source_manifest)
    cache_dir = corpus.tokenizer_path.parent
    shutil.copy2(corpus.tokenizer_path, OUT / "tokenizer.json")
    (OUT / "corpus-manifest.json").write_text(
        json.dumps(corpus.manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Release parent copies before GPU workers start; workers read the exact
    # cached token tensors independently.
    del corpus

    gpu_count = min(2, torch.cuda.device_count())
    physical_gpus = list(range(gpu_count))
    for start in range(0, len(FACTORIAL_SPECS), gpu_count):
        wave = FACTORIAL_SPECS[start : start + gpu_count]
        run_wave(wave, physical_gpus[: len(wave)], cache_dir)

    summaries: list[dict[str, object]] = []
    checkpoint_frames: list[pd.DataFrame] = []
    generations: list[dict[str, object]] = []
    for spec in FACTORIAL_SPECS:
        summary_path = OUT / f"{spec.name}-worker.json"
        checkpoint_path = OUT / f"{spec.name}-checkpoints.csv"
        generations_path = OUT / f"{spec.name}-generations.json"
        if not summary_path.is_file() or not checkpoint_path.is_file() or not generations_path.is_file():
            raise FileNotFoundError(f"incomplete worker artifacts for {spec.name}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        factors = summary["factors"]
        summaries.append(
            {
                "name": spec.name,
                "rms_norm": bool(factors["rms_norm"]),
                "carry_bias": bool(factors["carry_bias"]),
                "auxiliary_loss": bool(factors["auxiliary_loss"]),
                "parameters": int(summary["parameters"]),
                "ppl_125k": float(summary["ppl_125k"]),
                "ppl_250k": float(summary["ppl_250k"]),
                "validation_nll": float(summary["validation_nll"]),
                "validation_ppl": float(summary["validation_ppl"]),
                "learning_slope_alpha": float(summary["learning_slope_alpha"]),
                "elapsed_seconds": float(summary["elapsed_seconds"]),
                "tokens_per_second": float(summary["tokens_per_second"]),
                "peak_vram_bytes": int(summary["peak_vram_bytes"]),
            }
        )
        checkpoint_frames.append(pd.read_csv(checkpoint_path))
        generations.extend(json.loads(generations_path.read_text(encoding="utf-8")))

    factorial = pd.DataFrame(summaries)
    if not factorial["validation_nll"].map(math.isfinite).all():
        raise RuntimeError("non-finite factorial validation NLL")
    factorial.to_csv(OUT / "factorial-results.csv", index=False)
    checkpoints = pd.concat(checkpoint_frames, ignore_index=True)
    checkpoints.to_csv(OUT / "checkpoints.csv", index=False)
    effects = factorial_effects(factorial)
    effects.to_csv(OUT / "factorial-effects.csv", index=False)

    (OUT / "generation-samples.json").write_text(
        json.dumps(generations, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_generation_progression(generations, OUT / "generation-progression.md")

    source_textnca = float(source_decision["structural_control"]["ppl_500k"])
    source_plus = float(source_decision["candidate"]["ppl_500k"])
    by_name = factorial.set_index("name")
    replicated_textnca = float(by_name.loc["ln-c0-a0", "validation_ppl"])
    replicated_plus = float(by_name.loc["rms-c2-aux", "validation_ppl"])
    replication = pd.DataFrame(
        [
            {
                "condition": "textnca-baseline",
                "ppl_005": source_textnca,
                "ppl_005b": replicated_textnca,
                "relative_error": abs(replicated_textnca - source_textnca) / source_textnca,
            },
            {
                "condition": "minitextnca-s-plus",
                "ppl_005": source_plus,
                "ppl_005b": replicated_plus,
                "relative_error": abs(replicated_plus - source_plus) / source_plus,
            },
        ]
    )
    replication.to_csv(OUT / "replication.csv", index=False)
    replication_pass = bool((replication["relative_error"] <= REPLICATION_TOLERANCE).all())

    best_row = factorial.sort_values(["validation_ppl", "learning_slope_alpha"], ascending=[True, False]).iloc[0]
    best_name = str(best_row["name"])
    best_checkpoint = OUT / f"{best_name}-500k.pt"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(best_checkpoint)
    shutil.copy2(best_checkpoint, OUT / "best-500k.pt")

    # Keep only the selected checkpoint in curated results. The other seven are
    # regenerable ablation intermediates and would unnecessarily bloat Git history.
    for spec in FACTORIAL_SPECS:
        (OUT / f"{spec.name}-500k.pt").unlink(missing_ok=True)

    main_effects = effects[effects["order"] == 1].copy()
    dominant_main = main_effects.sort_values("abs_effect_nll", ascending=False).iloc[0]
    dominant_overall = effects.sort_values("abs_effect_nll", ascending=False).iloc[0]
    parameter_spread = (factorial["parameters"].max() - factorial["parameters"].min()) / factorial[
        "parameters"
    ].max()

    model_configs = {
        "format": "minicells.consumer-language-ablation-models.v1",
        "shared": {
            "dataset": "roneneldan/TinyStories",
            "budget_tokens": BUDGET_TOKENS,
            "context_length": 128,
            "vocab_size": int(source_manifest["vocab_size_actual"]),
            "dim": 128,
            "heads": 4,
            "ffn_dim": 512,
            "windows": [8, 32, 128],
            "iterations": [4, 4, 4],
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "model_seed": 55005,
            "schedule_seed": 5005,
        },
        "factors": {
            "R": {"off": "LayerNorm", "on": "RMSNorm"},
            "C": {"off": 0.0, "on": 2.0},
            "A": {"off": None, "on": [0.1, 0.2]},
        },
        "cells": [
            {
                "name": spec.name,
                "rms_norm": spec.rms_norm,
                "carry_bias": spec.carry_bias,
                "auxiliary_loss": spec.auxiliary_loss,
            }
            for spec in FACTORIAL_SPECS
        ],
    }
    (OUT / "model-configs.json").write_text(
        json.dumps(model_configs, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    task_spec = {
        "format": "minicells.consumer-language-ablation-task.v1",
        "design": "2^3 full factorial",
        "factors": ["RMSNorm", "carry bias +2", "auxiliary stage losses 0.1/0.2"],
        "cells": 8,
        "budget_tokens_per_cell": BUDGET_TOKENS,
        "checkpoints": [125_000, 250_000, 500_000],
        "response_for_factorial_effects": "validation NLL at 500K",
        "negative_effect_is_better": True,
        "replication_tolerance": REPLICATION_TOLERANCE,
        "same_seed_and_training_schedule": True,
        "max_parallel_gpus": 2,
    }
    (OUT / "task-spec.json").write_text(
        json.dumps(task_spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    status = "PASS" if replication_pass else "NEEDS_ITERATION"
    diagnosis = (
        "RECURRENT_OPTIMIZATION_FACTORS_LOCALIZED"
        if replication_pass
        else "EXPERIMENT_005_REPLICATION_DRIFT"
    )
    decision = {
        "format": "minicells.consumer-language-ablation.v1",
        "experiment": "MINI Cells Experiment 005B — Recurrent Optimization Factorial Ablation",
        "status": status,
        "diagnosis": diagnosis,
        "design": {
            "cells": 8,
            "budget_tokens_per_cell": BUDGET_TOKENS,
            "same_corpus_as_005": True,
            "same_model_seed": True,
            "same_training_schedule": True,
            "parameter_spread_fraction": float(parameter_spread),
        },
        "replication": {
            "tolerance": REPLICATION_TOLERANCE,
            "pass": replication_pass,
            "textnca_relative_error": float(replication.iloc[0]["relative_error"]),
            "minitextnca_plus_relative_error": float(replication.iloc[1]["relative_error"]),
        },
        "best": {
            "name": best_name,
            "validation_ppl": float(best_row["validation_ppl"]),
            "validation_nll": float(best_row["validation_nll"]),
            "learning_slope_alpha": float(best_row["learning_slope_alpha"]),
            "rms_norm": bool(best_row["rms_norm"]),
            "carry_bias": bool(best_row["carry_bias"]),
            "auxiliary_loss": bool(best_row["auxiliary_loss"]),
        },
        "dominant_main_effect": {
            "term": str(dominant_main["term"]),
            "label": str(dominant_main["label"]),
            "effect_nll": float(dominant_main["effect_nll"]),
            "ppl_multiplier": float(dominant_main["ppl_multiplier"]),
        },
        "dominant_overall_effect": {
            "term": str(dominant_overall["term"]),
            "label": str(dominant_overall["label"]),
            "effect_nll": float(dominant_overall["effect_nll"]),
            "ppl_multiplier": float(dominant_overall["ppl_multiplier"]),
        },
        "recommended_006_variant": best_name if replication_pass else None,
        "runtime": {
            "gpu_count_used": gpu_count,
            "physical_gpus": [torch.cuda.get_device_name(index) for index in range(gpu_count)],
        },
        "caveat": (
            "Single-seed factorial effects localize deterministic treatment effects for this 500K run; "
            "they are not statistical confidence intervals."
        ),
    }
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    save_factorial_ppl(factorial, OUT / "factorial-ppl.png")
    save_factorial_learning_curves(checkpoints, OUT / "factorial-learning-curves.png")
    save_effects(
        effects,
        OUT / "main-effects.png",
        order=1,
        title="Experiment 005B — main effects on validation NLL",
    )
    save_effects(
        effects,
        OUT / "interaction-effects.png",
        order=2,
        title="Experiment 005B — pairwise interaction effects",
    )
    save_effects(
        effects,
        OUT / "triple-interaction.png",
        order=3,
        title="Experiment 005B — three-way interaction",
    )
    save_replication_comparison(
        {"textnca_ppl": source_textnca, "minitextnca_plus_ppl": source_plus},
        factorial,
        OUT / "replication.png",
    )

    print("=== decision ===")
    print(json.dumps(decision, indent=2))
    print("=== factorial results ===")
    print(factorial.sort_values("validation_ppl").to_string(index=False))
    print("=== factorial effects ===")
    print(effects.sort_values("abs_effect_nll", ascending=False).to_string(index=False))
    print("=== files ===")
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            print(path.name, path.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
