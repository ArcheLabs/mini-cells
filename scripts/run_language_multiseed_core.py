from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "scripts"))

from minicells.language_multiseed_core import (  # noqa: E402
    CORE_COST_RATIO_MAX,
    CORE_PPL_RATIO_MAX,
    CORE_VARIANT_CODES,
    MIN_JOINT_PASS_REPLICATES,
    N_REPLICATES,
    PER_SEED_COST_RATIO_MAX,
    PER_SEED_PPL_RATIO_MAX,
    core_recipe_confirmation,
    core_recipe_ratio,
    factor_ratio,
    ratio_summary,
    seed_bundle,
)
from run_language_depth_ablation import prepare_corpus as prepare_013_corpus  # noqa: E402


OUT = ROOT / "results" / "language-multiseed-core-v1"
SOURCE_013 = ROOT / "artifacts" / "experiments" / "013-random-depth-ablation"
WORKER = ROOT / "scripts" / "run_language_multiseed_core_variant.py"
TOPOLOGIES = ("1d", "2d")
CHECKPOINTS = (250_000, 500_000, 1_000_000, 2_000_000)
METRICS = (
    "final_ppl_2m",
    "seconds_per_million_tokens",
    "depth_robustness_ratio_2_to_4",
)


def prepare_corpus() -> tuple[Path, dict[str, object]]:
    """Reuse the exact Experiment 013 token streams and verify provenance."""

    cache, generated = prepare_013_corpus()
    source_path = SOURCE_013 / "corpus-manifest.json"
    if not source_path.is_file():
        raise FileNotFoundError("Experiment 013 results must be merged before 014")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    for key in ("tokenizer_sha256", "train_token_sha256", "validation_token_sha256"):
        if generated.get(key) != source.get(key):
            raise RuntimeError(f"Experiment 014 corpus differs from Experiment 013: {key}")
    manifest = {
        **generated,
        "format": "minicells.language-multiseed-core-corpus.v1",
        "source_experiment": "013-random-depth-ablation",
        "source_013_train_token_sha256": source.get("train_token_sha256"),
        "source_013_validation_token_sha256": source.get("validation_token_sha256"),
        "reproduces_013_corpus": True,
    }
    return cache, manifest


def worker_command(topology: str, replicate: int, code: str, cache_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(WORKER),
        "--topology",
        topology,
        "--variant",
        code,
        "--replicate",
        str(replicate),
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(OUT),
    ]


def _run_batch(jobs: list[tuple[str, int, str, int]], cache_dir: Path) -> None:
    active: list[tuple[str, int, subprocess.Popen[str], Path, object]] = []
    for topology, replicate, code, gpu_index in jobs:
        run_name = f"{topology}-r{replicate}-{code}"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        log_path = OUT / f"{run_name}.log"
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            worker_command(topology, replicate, code, cache_dir),
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        active.append((run_name, gpu_index, process, log_path, handle))
        print(f"started {run_name:9s} on physical GPU {gpu_index}")

    failures: list[str] = []
    for run_name, gpu_index, process, log_path, handle in active:
        exit_code = process.wait()
        handle.close()
        print(f"--- {run_name} / GPU {gpu_index} ---")
        print(log_path.read_text(encoding="utf-8").rstrip())
        if exit_code != 0:
            failures.append(f"{run_name} exited {exit_code}; see {log_path}")
    if failures:
        raise RuntimeError("; ".join(failures))


def run_models(cache_dir: Path) -> int:
    """Run matched timing cells without cross-GPU timing confounding."""

    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 014 requires CUDA")
    gpu_count = min(2, available)

    for replicate in range(N_REPLICATES):
        if gpu_count == 1:
            for topology in TOPOLOGIES:
                for code in CORE_VARIANT_CODES:
                    _run_batch([(topology, replicate, code, 0)], cache_dir)
            continue

        assignment = {"1d": replicate % 2, "2d": 1 - (replicate % 2)}
        for code in CORE_VARIANT_CODES:
            _run_batch(
                [
                    ("1d", replicate, code, assignment["1d"]),
                    ("2d", replicate, code, assignment["2d"]),
                ],
                cache_dir,
            )
    return gpu_count


def summarize() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint_frames: list[pd.DataFrame] = []
    depth_frames: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for topology in TOPOLOGIES:
        for replicate in range(N_REPLICATES):
            for code in CORE_VARIANT_CODES:
                run_name = f"{topology}-r{replicate}-{code}"
                checkpoints = pd.read_csv(OUT / f"{run_name}-checkpoints.csv")
                depths = pd.read_csv(OUT / f"{run_name}-depth-eval.csv")
                worker = json.loads((OUT / f"{run_name}-worker.json").read_text(encoding="utf-8"))
                final = checkpoints.sort_values("consumed_tokens").iloc[-1]
                checkpoint_frames.append(checkpoints)
                depth_frames.append(depths)
                rows.append(
                    {
                        "run": run_name,
                        "replicate": replicate,
                        "topology": topology,
                        "variant": code,
                        "random_depth": bool(worker["random_depth"]),
                        "stability_loss": float(worker["stability_weight"]) > 0.0,
                        "stability_weight": float(worker["stability_weight"]),
                        "parameters": int(worker["parameters"]),
                        "model_seed": int(worker["seed"]),
                        "schedule_seed": int(worker["seed_bundle"]["schedule_seed"]),
                        "depth_seed": int(worker["seed_bundle"]["depth_seed"]),
                        "final_ppl_2m": float(final["validation_ppl"]),
                        "final_nll_2m": float(final["validation_nll"]),
                        "training_elapsed_seconds": float(worker["training_elapsed_seconds"]),
                        "training_tokens_per_second": float(worker["training_tokens_per_second"]),
                        "seconds_per_million_tokens": float(worker["seconds_per_million_tokens"]),
                        "peak_vram_gib": float(worker["peak_vram_bytes"] / (1024**3)),
                        "avg_recurrent_iterations": float(worker["avg_recurrent_iterations"]),
                        "depth_robustness_ratio_2_to_4": float(worker["depth_robustness_ratio_2_to_4"]),
                        "ppl_depth2": float(worker["ppl_depth2"]),
                        "ppl_depth3": float(worker["ppl_depth3"]),
                        "ppl_depth4": float(worker["ppl_depth4"]),
                    }
                )
    all_checkpoints = pd.concat(checkpoint_frames, ignore_index=True)
    all_depths = pd.concat(depth_frames, ignore_index=True)
    summary = pd.DataFrame(rows)
    all_checkpoints.to_csv(OUT / "checkpoints.csv", index=False)
    all_depths.to_csv(OUT / "depth-eval.csv", index=False)
    summary.to_csv(OUT / "model-summary.csv", index=False)
    return all_checkpoints, all_depths, summary


def paired_ratios(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for topology in TOPOLOGIES:
        for replicate in range(N_REPLICATES):
            group = summary.loc[
                (summary["topology"] == topology) & (summary["replicate"] == replicate)
            ].set_index("variant")
            if set(group.index) != set(CORE_VARIANT_CODES):
                raise RuntimeError(f"incomplete matched cell for {topology} replicate {replicate}")
            for metric in METRICS:
                values = {code: float(group.loc[code, metric]) for code in CORE_VARIANT_CODES}
                rows.extend(
                    [
                        {
                            "topology": topology,
                            "replicate": replicate,
                            "metric": metric,
                            "effect": "core_recipe_H_over_A",
                            "ratio": core_recipe_ratio(values),
                        },
                        {
                            "topology": topology,
                            "replicate": replicate,
                            "metric": metric,
                            "effect": "random_depth_main",
                            "ratio": factor_ratio(values, "random_depth"),
                        },
                        {
                            "topology": topology,
                            "replicate": replicate,
                            "metric": metric,
                            "effect": "stability_loss_main",
                            "ratio": factor_ratio(values, "stability_loss"),
                        },
                    ]
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "paired-ratios.csv", index=False)
    return frame


def aggregate_effects(ratios: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for topology in TOPOLOGIES:
        for effect in ratios["effect"].drop_duplicates():
            for metric in METRICS:
                values = ratios.loc[
                    (ratios["topology"] == topology)
                    & (ratios["effect"] == effect)
                    & (ratios["metric"] == metric),
                    "ratio",
                ].tolist()
                stats = ratio_summary(values)
                rows.append(
                    {
                        "topology": topology,
                        "effect": effect,
                        "metric": metric,
                        **stats,
                        "percent_change": (float(stats["geometric_mean_ratio"]) - 1.0) * 100.0,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "effect-summary.csv", index=False)
    return frame


def make_decision(ratios: pd.DataFrame, effects: pd.DataFrame, gpu_count: int) -> dict[str, object]:
    results: dict[str, object] = {}
    confirmed_topologies = 0
    for topology in TOPOLOGIES:
        subset = ratios.loc[
            (ratios["topology"] == topology) & (ratios["effect"] == "core_recipe_H_over_A")
        ]
        ppl = subset.loc[subset["metric"] == "final_ppl_2m"].sort_values("replicate")["ratio"].tolist()
        cost = subset.loc[subset["metric"] == "seconds_per_million_tokens"].sort_values("replicate")["ratio"].tolist()
        confirmation = core_recipe_confirmation(ppl, cost)
        confirmed_topologies += int(bool(confirmation["confirmed"]))
        random_effects = effects.loc[
            (effects["topology"] == topology) & (effects["effect"] == "random_depth_main")
        ].set_index("metric")
        stability_effects = effects.loc[
            (effects["topology"] == topology) & (effects["effect"] == "stability_loss_main")
        ].set_index("metric")
        results[topology] = {
            "core_recipe_H_over_A": confirmation,
            "random_depth_main": {
                metric: float(random_effects.loc[metric, "geometric_mean_ratio"])
                for metric in METRICS
            },
            "stability_loss_main": {
                metric: float(stability_effects.loc[metric, "geometric_mean_ratio"])
                for metric in METRICS
            },
        }

    if confirmed_topologies == len(TOPOLOGIES):
        status = "CORE_RECIPE_CONFIRMED_BOTH_TOPOLOGIES"
    elif confirmed_topologies:
        status = "CORE_RECIPE_MIXED_ACROSS_TOPOLOGIES"
    else:
        status = "CORE_RECIPE_NOT_CONFIRMED"

    return {
        "format": "minicells.language-multiseed-core.v1",
        "experiment": "MINI Cells Experiment 014 — Multi-Seed Core Recipe Confirmation",
        "status": status,
        "question": (
            "Does the Experiment 013 scale-1.0 core recipe (random recurrent depth + residual "
            "stability loss) reproducibly reduce training cost without worsening 2M-token PPL?"
        ),
        "design": {
            "type": "5-replicate matched-seed 2x2 factorial",
            "topologies": list(TOPOLOGIES),
            "variants": list(CORE_VARIANT_CODES),
            "replicates": N_REPLICATES,
            "models_total": len(TOPOLOGIES) * len(CORE_VARIANT_CODES) * N_REPLICATES,
            "tokens_per_model": 2_000_000,
            "step_embedding_init_scale": 1.0,
            "matched_within_replicate": [
                "model initialization within topology",
                "training schedule",
                "random-depth schedule",
                "validation sample schedule",
                "physical GPU for all A/B/F/H timing cells within topology",
            ],
            "gpu_assignment": "topology stays on one GPU within a replicate; assignments swap across replicates",
        },
        "pre_registered_confirmation": {
            "aggregate_H_over_A_ppl_ratio_max": CORE_PPL_RATIO_MAX,
            "aggregate_H_over_A_cost_ratio_max": CORE_COST_RATIO_MAX,
            "per_seed_joint_ppl_ratio_max": PER_SEED_PPL_RATIO_MAX,
            "per_seed_joint_cost_ratio_max": PER_SEED_COST_RATIO_MAX,
            "minimum_joint_pass_replicates": MIN_JOINT_PASS_REPLICATES,
        },
        "evaluation": {
            "standard_quality": "fixed (4,4,4) validation PPL at 2M tokens",
            "training_cost": "synchronized forward/backward/optimizer wall clock, validation excluded",
            "temporal_robustness": "PPL at (2,2,2), (3,3,3), and (4,4,4)",
            "aggregate_estimator": "paired geometric mean ratio across five replicates",
            "uncertainty": "deterministic exact 5**5 percentile bootstrap on paired ratios",
            "gpu_count_used_for_parallel_execution": gpu_count,
        },
        "scope": {
            "source_experiment": "013-random-depth-ablation",
            "replicate_zero": "preserves Experiment 013 model/schedule/depth/evaluation seeds",
            "interpretation": (
                "Confirmation establishes a reproducible training recipe for the current 1D and K=4 "
                "2D topologies; it does not establish that the current 2D topology is compute-efficient."
            ),
        },
        "results": results,
    }


def write_task_spec() -> None:
    payload = {
        "format": "minicells.language-multiseed-core-task.v1",
        "experiment": "014",
        "name": "Multi-Seed Core Recipe Confirmation",
        "topologies": list(TOPOLOGIES),
        "variants": {
            "A": {"random_depth": False, "stability_weight": 0.0},
            "B": {"random_depth": True, "stability_weight": 0.0},
            "F": {"random_depth": False, "stability_weight": 0.1},
            "H": {"random_depth": True, "stability_weight": 0.1},
        },
        "step_embedding_init_scale": 1.0,
        "replicates": [seed_bundle(index).as_dict() for index in range(N_REPLICATES)],
        "tokens_per_model": 2_000_000,
        "checkpoints": list(CHECKPOINTS),
        "timing_pairing": "same physical GPU for A/B/F/H within topology and replicate",
        "primary_metrics": [
            "fixed-depth validation PPL at 2M",
            "measured seconds per 1M tokens",
            "H/A paired ratio",
            "random-depth 2x2 main effect",
            "stability-loss 2x2 main effect",
        ],
        "confirmation_thresholds": {
            "aggregate_H_over_A_ppl_ratio_max": CORE_PPL_RATIO_MAX,
            "aggregate_H_over_A_cost_ratio_max": CORE_COST_RATIO_MAX,
            "per_seed_joint_ppl_ratio_max": PER_SEED_PPL_RATIO_MAX,
            "per_seed_joint_cost_ratio_max": PER_SEED_COST_RATIO_MAX,
            "minimum_joint_pass_replicates": MIN_JOINT_PASS_REPLICATES,
        },
    }
    (OUT / "task-spec.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_core_recipe_ratios(ratios: pd.DataFrame) -> None:
    core = ratios.loc[ratios["effect"] == "core_recipe_H_over_A"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    specs = (
        ("final_ppl_2m", CORE_PPL_RATIO_MAX, "H/A validation PPL ratio"),
        ("seconds_per_million_tokens", CORE_COST_RATIO_MAX, "H/A training-cost ratio"),
    )
    for ax, (metric, threshold, title) in zip(axes, specs, strict=True):
        for topology in TOPOLOGIES:
            group = core.loc[(core["topology"] == topology) & (core["metric"] == metric)].sort_values("replicate")
            ax.plot(group["replicate"], group["ratio"], marker="o", label=topology)
        ax.axhline(threshold, linestyle="--", linewidth=1)
        ax.set_xticks(range(N_REPLICATES))
        ax.set_xlabel("replicate")
        ax.set_ylabel("ratio; lower is better")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "core-recipe-seed-ratios.png", dpi=160)
    plt.close(fig)


def plot_tradeoff(ratios: pd.DataFrame) -> None:
    core = ratios.loc[ratios["effect"] == "core_recipe_H_over_A"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for topology in TOPOLOGIES:
        group = core.loc[core["topology"] == topology]
        ppl = group.loc[group["metric"] == "final_ppl_2m"].set_index("replicate")["ratio"]
        cost = group.loc[group["metric"] == "seconds_per_million_tokens"].set_index("replicate")["ratio"]
        aligned = pd.concat({"ppl": ppl, "cost": cost}, axis=1).sort_index()
        ax.scatter(aligned["cost"], aligned["ppl"], label=topology)
        for replicate, row in aligned.iterrows():
            ax.annotate(f"r{replicate}", (row["cost"], row["ppl"]), xytext=(4, 4), textcoords="offset points")
    ax.axvline(CORE_COST_RATIO_MAX, linestyle="--", linewidth=1)
    ax.axhline(CORE_PPL_RATIO_MAX, linestyle="--", linewidth=1)
    ax.set_xlabel("H/A training-cost ratio")
    ax.set_ylabel("H/A PPL ratio")
    ax.set_title("Core recipe: paired quality/cost tradeoff")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "core-recipe-ppl-cost.png", dpi=160)
    plt.close(fig)


def plot_factor_effects(effects: pd.DataFrame) -> None:
    subset = effects.loc[
        effects["metric"].isin(["final_ppl_2m", "seconds_per_million_tokens"])
        & effects["effect"].isin(["random_depth_main", "stability_loss_main"])
    ].copy()
    labels: list[str] = []
    values: list[float] = []
    for topology in TOPOLOGIES:
        for effect in ("random_depth_main", "stability_loss_main"):
            for metric in ("final_ppl_2m", "seconds_per_million_tokens"):
                row = subset.loc[
                    (subset["topology"] == topology)
                    & (subset["effect"] == effect)
                    & (subset["metric"] == metric)
                ].iloc[0]
                labels.append(f"{topology}\n{effect.replace('_main', '')}\n{'PPL' if metric == 'final_ppl_2m' else 'cost'}")
                values.append(float(row["geometric_mean_ratio"]))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(values)), values)
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_ylabel("geometric mean high/low ratio")
    ax.set_title("Experiment 014 matched-seed main effects")
    fig.tight_layout()
    fig.savefig(OUT / "factor-main-effects.png", dpi=160)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache_dir, corpus_manifest = prepare_corpus()
    write_task_spec()
    gpu_count = run_models(cache_dir)
    _, _, summary = summarize()
    ratios = paired_ratios(summary)
    effects = aggregate_effects(ratios)
    decision = make_decision(ratios, effects, gpu_count)
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "corpus-manifest.json").write_text(
        json.dumps(corpus_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_core_recipe_ratios(ratios)
    plot_tradeoff(ratios)
    plot_factor_effects(effects)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
