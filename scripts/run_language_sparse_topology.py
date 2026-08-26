from __future__ import annotations

import hashlib
import json
import math
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

from minicells.language_skill_data import (  # noqa: E402
    ALL_TASKS,
    BASE_TASKS,
    COMPOSITION_MAP,
    MODEL_LENGTH,
    VOCAB_SIZE,
    generate_skill_corpus,
)
from minicells.language_sparse_topology import (  # noqa: E402
    ACTIVE_LATENT,
    BALANCE_WEIGHT,
    STABILITY_WEIGHT,
    TISSUE_HEIGHT,
    VARIANT_CODES,
)


OUT = ROOT / "results" / "language-sparse-topology-v1"
SOURCE_014 = ROOT / "artifacts" / "experiments" / "014-multiseed-core-recipe"
WORKER = ROOT / "scripts" / "run_language_sparse_topology_variant.py"
N_REPLICATES = 3
TRAIN_EXAMPLES = 50_000
VALIDATION_EXAMPLES = 1_000
TRAIN_CORPUS_SEED = 15_015
VALIDATION_CORPUS_SEED = 25_015
BUDGET_TOKENS = 960_000
QUALITY_NLL_RATIO_MAX = 1.10
MIN_SIGNAL_REPLICATES = 2
MAX_LOGICAL_ACTIVE_FRACTION = 0.40


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def prepare_corpus() -> tuple[Path, dict[str, object]]:
    decision_path = SOURCE_014 / "decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError("Experiment 014 artifacts must be merged before 015")
    source_014 = json.loads(decision_path.read_text(encoding="utf-8"))
    if source_014.get("status") != "CORE_RECIPE_CONFIRMED_BOTH_TOPOLOGIES":
        raise RuntimeError("Experiment 015 requires the confirmed Experiment 014 H recipe")
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    train_path = cache / "train-corpus.pt"
    validation_path = cache / "validation-corpus.pt"
    manifest_path = cache / "corpus-manifest.json"
    expected = {
        "format": "minicells.language-sparse-topology-corpus.v1",
        "train_examples": TRAIN_EXAMPLES,
        "validation_examples": VALIDATION_EXAMPLES,
        "train_seed": TRAIN_CORPUS_SEED,
        "validation_seed": VALIDATION_CORPUS_SEED,
        "tasks": list(ALL_TASKS),
        "vocab_size": VOCAB_SIZE,
        "model_length": MODEL_LENGTH,
    }
    if train_path.is_file() and validation_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in expected.items()):
            train = torch.load(train_path, map_location="cpu")
            validation = torch.load(validation_path, map_location="cpu")
            if (
                tensor_sha256(train["sequences"]) == manifest.get("train_sequence_sha256")
                and tensor_sha256(validation["sequences"]) == manifest.get("validation_sequence_sha256")
            ):
                return cache, manifest
    train = generate_skill_corpus(TRAIN_EXAMPLES, seed=TRAIN_CORPUS_SEED)
    validation = generate_skill_corpus(VALIDATION_EXAMPLES, seed=VALIDATION_CORPUS_SEED)
    train_payload = {
        "sequences": train.sequences,
        "task_ids": train.task_ids,
        "task_names": train.task_names,
        "loss_mask": train.loss_mask,
    }
    validation_payload = {
        "sequences": validation.sequences,
        "task_ids": validation.task_ids,
        "task_names": validation.task_names,
        "loss_mask": validation.loss_mask,
    }
    torch.save(train_payload, train_path)
    torch.save(validation_payload, validation_path)
    manifest = {
        **expected,
        "base_tasks": list(BASE_TASKS),
        "composition_map": {key: list(value) for key, value in COMPOSITION_MAP.items()},
        "train_sequence_sha256": tensor_sha256(train.sequences),
        "validation_sequence_sha256": tensor_sha256(validation.sequences),
        "train_task_sha256": tensor_sha256(train.task_ids),
        "validation_task_sha256": tensor_sha256(validation.task_ids),
        "source_014_status": source_014["status"],
        "source_014_format": source_014["format"],
        "objective": "autoregressive next-token prediction with loss restricted to transformed output + EOS",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cache, manifest


def worker_command(replicate: int, variant: str, cache_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(WORKER),
        "--replicate",
        str(replicate),
        "--variant",
        variant,
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(OUT),
    ]


def run_models(cache_dir: Path) -> int:
    available = torch.cuda.device_count()
    if available < 1:
        raise RuntimeError("Experiment 015 requires CUDA")
    gpu_count = min(2, available)
    replicate_groups = [
        tuple(range(start, min(start + gpu_count, N_REPLICATES)))
        for start in range(0, N_REPLICATES, gpu_count)
    ]
    for group in replicate_groups:
        gpu_for = {replicate: index for index, replicate in enumerate(group)}
        # A/B/C for a replicate stay on one physical GPU for paired timing.
        for variant in VARIANT_CODES:
            active = []
            for replicate in group:
                gpu_index = gpu_for[replicate]
                run_name = f"r{replicate}-{variant}"
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
                log_path = OUT / f"{run_name}.log"
                handle = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    worker_command(replicate, variant, cache_dir),
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active.append((run_name, gpu_index, process, log_path, handle))
                print(f"started {run_name} on physical GPU {gpu_index}")
            failures = []
            for run_name, gpu_index, process, log_path, handle in active:
                exit_code = process.wait()
                handle.close()
                print(f"--- {run_name} / GPU {gpu_index} ---")
                print(log_path.read_text(encoding="utf-8").rstrip())
                if exit_code != 0:
                    failures.append(f"{run_name} exited {exit_code}; see {log_path}")
            if failures:
                raise RuntimeError("; ".join(failures))
    return gpu_count


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(float(value)) for value in values) / len(values))


def collect_results() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    workers = []
    checkpoints = []
    task_metrics = []
    regions = []
    edges = []
    ablations = []
    reuse = []
    for replicate in range(N_REPLICATES):
        for variant in VARIANT_CODES:
            run_name = f"r{replicate}-{variant}"
            worker = json.loads((OUT / f"{run_name}-worker.json").read_text(encoding="utf-8"))
            workers.append(
                {
                    "run": run_name,
                    "replicate": replicate,
                    "variant": variant,
                    "parameters": worker["parameters"],
                    "seconds_per_million_tokens": worker["seconds_per_million_tokens"],
                    "peak_vram_gib": worker["peak_vram_bytes"] / (1024**3),
                    "avg_recurrent_iterations": worker["avg_recurrent_iterations"],
                    "logical_active_fraction": worker["topology"]["logical_active_fraction"],
                    "mean_nll": worker["topology"]["mean_nll"],
                    "mean_token_accuracy": worker["topology"]["mean_token_accuracy"],
                    "mean_exact_match": worker["topology"]["mean_exact_match"],
                    "region_mi": worker["topology"]["task_region_mi"]["observed"],
                    "region_mi_null_p99": worker["topology"]["task_region_mi"]["null_p99"],
                    "region_mi_p": worker["topology"]["task_region_mi"]["empirical_p"],
                    "edge_mi": worker["topology"]["task_edge_mi"]["observed"],
                    "edge_mi_null_p99": worker["topology"]["task_edge_mi"]["null_p99"],
                    "edge_mi_p": worker["topology"]["task_edge_mi"]["empirical_p"],
                    "composition_reuse_margin_mean": worker["topology"]["composition_reuse_margin_mean"],
                    "composition_reuse_positive": worker["topology"]["composition_reuse_positive"],
                    "functional_specific_rows": worker["ablation"]["functional_specific_rows"],
                }
            )
            checkpoints.append(pd.read_csv(OUT / f"{run_name}-checkpoints.csv"))
            task_metrics.append(pd.read_csv(OUT / f"{run_name}-task-metrics.csv"))
            regions.append(pd.read_csv(OUT / f"{run_name}-task-region.csv"))
            edge_path = OUT / f"{run_name}-task-edge.csv"
            if edge_path.stat().st_size > 1:
                edge_frame = pd.read_csv(edge_path)
                if not edge_frame.empty:
                    edges.append(edge_frame)
            ablations.append(pd.read_csv(OUT / f"{run_name}-ablation.csv"))
            reuse.append(pd.read_csv(OUT / f"{run_name}-composition-reuse.csv"))
    summary = pd.DataFrame(workers)
    if summary.groupby("replicate")["parameters"].nunique().max() != 1:
        raise RuntimeError("A/B/C parameter counts differ within a matched replicate")
    checkpoint_frame = pd.concat(checkpoints, ignore_index=True)
    task_frame = pd.concat(task_metrics, ignore_index=True)
    region_frame = pd.concat(regions, ignore_index=True)
    edge_frame = (
        pd.concat(edges, ignore_index=True)
        if edges
        else pd.DataFrame(columns=["run", "task", "source", "receiver", "mean_edge_usage"])
    )
    ablation_frame = pd.concat(ablations, ignore_index=True)
    reuse_frame = pd.concat(reuse, ignore_index=True)
    summary.to_csv(OUT / "model-summary.csv", index=False)
    checkpoint_frame.to_csv(OUT / "checkpoints.csv", index=False)
    task_frame.to_csv(OUT / "task-metrics.csv", index=False)
    region_frame.to_csv(OUT / "task-region.csv", index=False)
    edge_frame.to_csv(OUT / "task-edge.csv", index=False)
    ablation_frame.to_csv(OUT / "ablation.csv", index=False)
    reuse_frame.to_csv(OUT / "composition-reuse.csv", index=False)
    return summary, checkpoint_frame, task_frame, region_frame, edge_frame, ablation_frame, reuse_frame


def make_paired_ratios(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for replicate in range(N_REPLICATES):
        group = summary.loc[summary["replicate"] == replicate].set_index("variant")
        for variant in ("B", "C"):
            for metric in ("mean_nll", "seconds_per_million_tokens"):
                rows.append(
                    {
                        "replicate": replicate,
                        "variant": variant,
                        "baseline": "A",
                        "metric": metric,
                        "ratio": float(group.loc[variant, metric] / group.loc["A", metric]),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "paired-ratios.csv", index=False)
    return frame


def make_decision(summary: pd.DataFrame, ratios: pd.DataFrame, gpu_count: int) -> dict[str, object]:
    c = summary.loc[summary["variant"] == "C"].sort_values("replicate")
    c_nll_ratios = ratios.loc[
        (ratios["variant"] == "C") & (ratios["metric"] == "mean_nll"), "ratio"
    ].tolist()
    quality_ratio = geometric_mean(c_nll_ratios)
    region_passes = int((c["region_mi"] > c["region_mi_null_p99"]).sum())
    edge_passes = int((c["edge_mi"] > c["edge_mi_null_p99"]).sum())
    ablation_passes = int((c["functional_specific_rows"] >= 1).sum())
    composition_passes = int((c["composition_reuse_margin_mean"] > 0.0).sum())
    sparse_passes = int((c["logical_active_fraction"] <= MAX_LOGICAL_ACTIVE_FRACTION).sum())
    pass_flags = {
        "quality": quality_ratio <= QUALITY_NLL_RATIO_MAX,
        "task_region_specialization": region_passes >= MIN_SIGNAL_REPLICATES,
        "task_edge_routing": edge_passes >= MIN_SIGNAL_REPLICATES,
        "functional_ablation": ablation_passes >= MIN_SIGNAL_REPLICATES,
        "composition_reuse": composition_passes >= MIN_SIGNAL_REPLICATES,
        "logical_sparsity": sparse_passes == N_REPLICATES,
    }
    if all(pass_flags.values()):
        status = "EMERGENT_SPARSE_TOPOLOGY_SIGNAL"
    elif sum(bool(value) for value in pass_flags.values()) >= 3:
        status = "PARTIAL_TOPOLOGY_SIGNAL"
    else:
        status = "NO_EMERGENT_TOPOLOGY_SIGNAL"
    controls = {}
    for variant in VARIANT_CODES:
        group = summary.loc[summary["variant"] == variant]
        controls[variant] = {
            "mean_nll_geomean": geometric_mean(group["mean_nll"].tolist()),
            "mean_token_accuracy": float(group["mean_token_accuracy"].mean()),
            "logical_active_fraction": float(group["logical_active_fraction"].mean()),
            "region_mi_significant_replicates": int((group["region_mi"] > group["region_mi_null_p99"]).sum()),
            "edge_mi_significant_replicates": int((group["edge_mi"] > group["edge_mi_null_p99"]).sum()),
            "mean_functional_specific_rows": float(group["functional_specific_rows"].mean()),
            "mean_composition_reuse_margin": float(group["composition_reuse_margin_mean"].mean()),
        }
    return {
        "format": "minicells.language-sparse-topology.v1",
        "experiment": "MINI Cells Experiment 015 — Emergent Sparse Tissue Topology",
        "status": status,
        "question": (
            "Can a homogeneous cellular language tissue develop task-dependent sparse regions and "
            "dynamic region-to-region routing without predefined experts?"
        ),
        "source_recipe": {
            "experiment": "014-multiseed-core-recipe",
            "random_depth": True,
            "stability_weight": STABILITY_WEIGHT,
            "step_embedding_init_scale": 1.0,
        },
        "design": {
            "variants": {
                "A": "dense local tissue",
                "B": "top-2 latent sparse local tissue",
                "C": "top-2 latent sparse tissue + state-derived top-1 long-range source per receiver",
            },
            "replicates": N_REPLICATES,
            "models_total": N_REPLICATES * len(VARIANT_CODES),
            "tokens_per_model": BUDGET_TOKENS,
            "tissue_height": TISSUE_HEIGHT,
            "active_latent": ACTIVE_LATENT,
            "trainable_row_embedding": False,
            "fixed_coordinate_buffer": True,
            "equal_parameterization_across_variants": True,
            "dynamic_edges_same_token_position_only": True,
            "dynamic_edge_router_parameters": 0,
            "gpu_count": gpu_count,
        },
        "corpus": {
            "kind": "deterministic synthetic multi-skill autoregressive corpus",
            "base_tasks": list(BASE_TASKS),
            "composition_map": {key: list(value) for key, value in COMPOSITION_MAP.items()},
            "loss_scope": "transformed output tokens + EOS",
        },
        "pre_registered_signal": {
            "dynamic_C_nll_ratio_to_dense_A_max": QUALITY_NLL_RATIO_MAX,
            "minimum_signal_replicates": MIN_SIGNAL_REPLICATES,
            "task_region_mi": "observed > label-shuffled p99",
            "task_edge_mi": "observed > label-shuffled p99",
            "functional_ablation": "at least one row with delta NLL >=0.02 and specificity >=1.5",
            "composition_reuse": "mean true-component reuse margin versus best wrong pair > 0",
            "logical_active_fraction_max": MAX_LOGICAL_ACTIVE_FRACTION,
        },
        "dynamic_C": {
            "nll_ratio_to_dense_A_geometric_mean": quality_ratio,
            "region_mi_signal_replicates": region_passes,
            "edge_mi_signal_replicates": edge_passes,
            "functional_ablation_replicates": ablation_passes,
            "composition_reuse_replicates": composition_passes,
            "logical_sparsity_replicates": sparse_passes,
            "pass_flags": pass_flags,
        },
        "controls": controls,
        "interpretation": {
            "pass": (
                "Evidence that sparse expert-like functional topology can emerge from shared cellular "
                "rules. It is not yet evidence for tissue transplantation, birth/death, or unbounded growth."
            ),
            "partial": "Some specialization/routing signal exists but one or more required properties are missing.",
            "fail": "Do not add growth or transplantation until specialization/routing is made reproducible.",
        },
    }


def write_task_spec() -> None:
    payload = {
        "format": "minicells.language-sparse-topology-task.v1",
        "experiment": "015",
        "name": "Emergent Sparse Tissue Topology",
        "variants": list(VARIANT_CODES),
        "replicates": N_REPLICATES,
        "tokens_per_model": BUDGET_TOKENS,
        "tissue_height": TISSUE_HEIGHT,
        "active_latent": ACTIVE_LATENT,
        "stability_weight": STABILITY_WEIGHT,
        "balance_weight": BALANCE_WEIGHT,
        "tasks": list(ALL_TASKS),
        "composition_map": {key: list(value) for key, value in COMPOSITION_MAP.items()},
        "primary_endpoints": [
            "masked output NLL",
            "task-region mutual information vs shuffled null",
            "task-edge mutual information vs shuffled null",
            "targeted row ablation specificity",
            "composition reuse margin",
            "logical active-cell fraction",
        ],
        "confirmation_thresholds": {
            "dynamic_C_nll_ratio_to_dense_A_max": QUALITY_NLL_RATIO_MAX,
            "minimum_signal_replicates": MIN_SIGNAL_REPLICATES,
            "logical_active_fraction_max": MAX_LOGICAL_ACTIVE_FRACTION,
        },
    }
    (OUT / "task-spec.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def save_plots(
    summary: pd.DataFrame,
    regions: pd.DataFrame,
    ablations: pd.DataFrame,
    reuse: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for variant, group in summary.groupby("variant"):
        ax.scatter(group["logical_active_fraction"], group["mean_nll"], label=variant)
    ax.set_xlabel("Logical active-cell fraction")
    ax.set_ylabel("Validation output NLL")
    ax.set_title("Experiment 015 — quality vs cellular sparsity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "quality-vs-sparsity.png", dpi=160)
    plt.close(fig)

    representative = regions.loc[regions["run"] == "r0-C"]
    pivot = representative.pivot(index="task", columns="row", values="mean_activity").reindex(ALL_TASKS)
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto")
    ax.set_xticks(range(TISSUE_HEIGHT), [str(index) for index in range(TISSUE_HEIGHT)])
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("Tissue row")
    ax.set_title("Dynamic C / replicate 0 — task-conditioned region activity")
    fig.colorbar(image, ax=ax, label="mean hard activity")
    fig.tight_layout()
    fig.savefig(OUT / "task-region-heatmap.png", dpi=160)
    plt.close(fig)

    representative_ablation = ablations.loc[ablations["run"] == "r0-C"]
    pivot = representative_ablation.pivot(index="task", columns="row", values="delta_nll").reindex(ALL_TASKS)
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto")
    ax.set_xticks(range(TISSUE_HEIGHT - 1), [str(index) for index in range(1, TISSUE_HEIGHT)])
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("Ablated latent row")
    ax.set_title("Dynamic C / replicate 0 — functional ablation ΔNLL")
    fig.colorbar(image, ax=ax, label="ablated NLL - baseline NLL")
    fig.tight_layout()
    fig.savefig(OUT / "ablation-heatmap.png", dpi=160)
    plt.close(fig)

    dynamic_reuse = reuse.loc[reuse["run"].str.endswith("-C")]
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    for run, group in dynamic_reuse.groupby("run"):
        ax.plot(group["composite"], group["reuse_margin_vs_best_wrong"], marker="o", label=run)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_ylabel("true-pair reuse - best wrong-pair reuse")
    ax.set_title("Dynamic C — compositional region reuse")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "composition-reuse.png", dpi=160)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache_dir, manifest = prepare_corpus()
    write_task_spec()
    gpu_count = run_models(cache_dir)
    summary, _, _, regions, _, ablations, reuse = collect_results()
    ratios = make_paired_ratios(summary)
    decision = make_decision(summary, ratios, gpu_count)
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUT / "corpus-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    save_plots(summary, regions, ablations, reuse)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
