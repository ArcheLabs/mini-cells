from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

import native_clm_runtime as base
from native_clm_replication import (
    MODEL_IDS,
    PARAMETER_TOLERANCE,
    REPLICATION_MODELS,
    REPLICATION_SEEDS,
    build_replication_model,
    estimate_replication_flops,
    replication_config,
)


def read_rows(path: Path) -> list[dict[str, object]]:
    numeric = {
        "step", "consumed_tokens", "train_loss", "learning_rate", "grad_norm",
        "parameters", "elapsed_seconds", "tokens_per_second", "peak_vram_bytes",
        "validation_nll", "validation_ppl", "validation_tokens",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, object]] = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def log_token_auc(rows: list[dict[str, object]]) -> dict[str, float]:
    x = np.log(np.array([float(row["consumed_tokens"]) for row in rows]))
    y = np.array([float(row["validation_nll"]) for row in rows])
    auc = float(np.trapezoid(y, x))
    return {"auc": auc, "mean_nll": auc / float(x[-1] - x[0])}


def train_one_replication(
    name: str,
    *,
    corpus: base.Corpus,
    profile: base.Profile,
    output_dir: Path,
    seed: int,
    device: torch.device,
    resume: bool,
) -> dict[str, object]:
    if seed not in REPLICATION_SEEDS:
        raise RuntimeError(f"seed {seed} outside frozen replication seeds")
    params = base.count_parameters(build_replication_model(name))
    t1_params = base.count_parameters(build_replication_model(base.T1_NAME))
    if abs(params / t1_params - 1.0) >= PARAMETER_TOLERANCE:
        raise RuntimeError(f"parameter mismatch: {name}={params}, T1={t1_params}")

    # Preserve established initialization semantics: T1 keeps the original
    # baseline model_seed=seed+1; prior M4/X3/X4 runners use model_seed=seed.
    if name == base.T1_NAME:
        summary = base.train_one(
            name, corpus=corpus, profile=profile, output_dir=output_dir,
            seed=seed, device=device, resume=resume,
        )
    else:
        old = (base.C0_NAME, base.build_model, base.model_config, base.estimate_flops)
        base.C0_NAME = name
        base.build_model = lambda model_name, vocab_size=base.VOCAB_SIZE: build_replication_model(model_name, vocab_size)
        base.model_config = lambda model_name, vocab_size=base.VOCAB_SIZE: replication_config(model_name, vocab_size)
        base.estimate_flops = lambda model_name, sequence_length, tokens: estimate_replication_flops(
            model_name, sequence_length=sequence_length, tokens=tokens
        )
        try:
            summary = base.train_one(
                name, corpus=corpus, profile=profile, output_dir=output_dir,
                seed=seed, device=device, resume=resume,
            )
        finally:
            base.C0_NAME, base.build_model, base.model_config, base.estimate_flops = old

    rows = read_rows(output_dir / name / "checkpoints.csv")
    summary["checkpoints"] = rows
    summary["validation_log_token_nll_auc"] = log_token_auc(rows)
    summary["replication_seed"] = seed
    summary["replication_id"] = MODEL_IDS[name]
    summary["parameter_ratio_to_t1"] = params / t1_params
    (output_dir / name / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def seed_leaderboard(output_dir: Path, seed: int):
    rows: list[dict[str, object]] = []
    curves: dict[str, list[dict[str, object]]] = {}
    for model in REPLICATION_MODELS:
        model_id = MODEL_IDS[model]
        summary = json.loads((output_dir / model / "summary.json").read_text(encoding="utf-8"))
        checkpoints = read_rows(output_dir / model / "checkpoints.csv")
        curves[model_id] = checkpoints
        final = summary["final"]
        auc = summary["validation_log_token_nll_auc"]
        rows.append({
            "seed": seed,
            "id": model_id,
            "model": model,
            "parameters": int(summary["parameters"]),
            "parameter_ratio_to_t1": float(summary["parameter_ratio_to_t1"]),
            "validation_ppl_10m": float(final["validation_ppl"]),
            "validation_nll_10m": float(final["validation_nll"]),
            "log_token_nll_auc": float(auc["auc"]),
            "log_token_nll_auc_mean": float(auc["mean_nll"]),
            "train_flops_estimate": float(summary["flops"]["train_flops_estimate"]),
            "inference_flops_per_token_estimate": float(summary["flops"]["inference_forward_flops_per_token"]),
            "tokens_per_second": float(final["tokens_per_second"]),
            "peak_vram_bytes": int(final["peak_vram_bytes"]),
        })
    t1 = next(row for row in rows if row["id"] == "T1")
    for row in rows:
        row["ppl_delta_vs_same_seed_t1"] = float(row["validation_ppl_10m"]) - float(t1["validation_ppl_10m"])
        row["ppl_ratio_vs_same_seed_t1"] = float(row["validation_ppl_10m"]) / float(t1["validation_ppl_10m"])
        row["train_flops_ratio_vs_same_seed_t1"] = float(row["train_flops_estimate"]) / float(t1["train_flops_estimate"])
    return rows, curves


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(per_seed: dict[int, list[dict[str, object]]]) -> list[dict[str, object]]:
    aggregate = []
    for model in REPLICATION_MODELS:
        model_id = MODEL_IDS[model]
        model_rows = [next(row for row in per_seed[seed] if row["id"] == model_id) for seed in REPLICATION_SEEDS]
        ppl = [float(row["validation_ppl_10m"]) for row in model_rows]
        auc = [float(row["log_token_nll_auc_mean"]) for row in model_rows]
        paired = [float(row["ppl_delta_vs_same_seed_t1"]) for row in model_rows]
        throughput = [float(row["tokens_per_second"]) for row in model_rows]
        vram = [int(row["peak_vram_bytes"]) for row in model_rows]
        aggregate.append({
            "id": model_id,
            "model": model,
            "seeds": list(REPLICATION_SEEDS),
            "parameters": int(model_rows[0]["parameters"]),
            "validation_ppl_mean": sum(ppl) / len(ppl),
            "validation_ppl_min": min(ppl),
            "validation_ppl_max": max(ppl),
            "log_token_nll_auc_mean_across_seeds": sum(auc) / len(auc),
            "paired_ppl_delta_vs_t1_mean": sum(paired) / len(paired),
            "paired_ppl_delta_vs_t1_min": min(paired),
            "paired_ppl_delta_vs_t1_max": max(paired),
            "train_flops_estimate": float(model_rows[0]["train_flops_estimate"]),
            "inference_flops_per_token_estimate": float(model_rows[0]["inference_flops_per_token_estimate"]),
            "tokens_per_second_mean": sum(throughput) / len(throughput),
            "peak_vram_bytes_max": max(vram),
        })
    return aggregate
