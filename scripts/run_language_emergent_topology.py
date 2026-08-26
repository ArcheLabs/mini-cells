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
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))

from minicells.language_emergent_topology import (  # noqa: E402
    ACTIVITY_BUDGET,
    ACTIVITY_RATE,
    LOCAL_COUPLING,
    LONG_RANGE_MAX_COUPLING,
    PLASTICITY_RATE,
    STABILITY_WEIGHT,
    TISSUE_HEIGHT,
)
from minicells.language_skill_data import (  # noqa: E402
    ALL_TASKS,
    BASE_TASKS,
    COMPOSITION_MAP,
    MODEL_LENGTH,
    VOCAB_SIZE,
    generate_skill_corpus,
)

OUT = ROOT / "results" / "local-substrate-emergent-topology-v1"
SOURCE_015 = ROOT / "artifacts" / "experiments" / "015-emergent-sparse-topology"
WORKER = ROOT / "scripts" / "run_language_emergent_topology_variant.py"
VARIANTS = ("L", "E")
N_REPLICATES = 3
TRAIN_EXAMPLES = 50_000
VALIDATION_EXAMPLES = 1_000
TRAIN_CORPUS_SEED = 15_015
VALIDATION_CORPUS_SEED = 25_015
BUDGET_TOKENS = 960_000

QUALITY_NLL_RATIO_MAX = 1.05
PER_SEED_NLL_RATIO_MAX = 1.10
MAX_ROBUSTNESS_RATIO_TO_L = 1.15
MIN_SIGNAL_REPLICATES = 2
MAX_EFFECTIVE_ACTIVE_FRACTION = 0.75
MIN_FINAL_PLASTIC_STRENGTH = 0.005
MIN_FINAL_PLASTIC_TV = 0.05


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(float(value)) for value in values) / len(values))


def prepare_corpus() -> tuple[Path, dict[str, object]]:
    decision_path = SOURCE_015 / "decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError("Experiment 015 artifacts must be merged before 015c")
    source_015 = json.loads(decision_path.read_text(encoding="utf-8"))
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    train_path = cache / "train-corpus.pt"
    validation_path = cache / "validation-corpus.pt"
    manifest_path = cache / "corpus-manifest.json"
    expected = {
        "format": "minicells.local-substrate-emergent-topology-corpus.v1",
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
            if tensor_sha256(train["sequences"]) == manifest.get("train_sequence_sha256") and tensor_sha256(validation["sequences"]) == manifest.get("validation_sequence_sha256"):
                return cache, manifest

    train = generate_skill_corpus(TRAIN_EXAMPLES, seed=TRAIN_CORPUS_SEED)
    validation = generate_skill_corpus(VALIDATION_EXAMPLES, seed=VALIDATION_CORPUS_SEED)
    for corpus, path in ((train, train_path), (validation, validation_path)):
        torch.save(
            {
                "sequences": corpus.sequences,
                "task_ids": corpus.task_ids,
                "task_names": corpus.task_names,
                "loss_mask": corpus.loss_mask,
            },
            path,
        )
    manifest = {
        **expected,
        "base_tasks": list(BASE_TASKS),
        "composition_map": {key: list(value) for key, value in COMPOSITION_MAP.items()},
        "train_sequence_sha256": tensor_sha256(train.sequences),
        "validation_sequence_sha256": tensor_sha256(validation.sequences),
        "train_task_sha256": tensor_sha256(train.task_ids),
        "validation_task_sha256": tensor_sha256(validation.task_ids),
        "source_015_status": source_015.get("status"),
        "source_015_format": source_015.get("format"),
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
        raise RuntimeError("Experiment 015c requires CUDA")
    gpu_count = min(2, available)
    groups = [tuple(range(start, min(start + gpu_count, N_REPLICATES))) for start in range(0, N_REPLICATES, gpu_count)]
    for group in groups:
        gpu_for = {replicate: index for index, replicate in enumerate(group)}
        # L and E for one replicate remain on the same physical GPU.
        for variant in VARIANTS:
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


def _read_if_present(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.is_file() else None


def collect_results() -> dict[str, pd.DataFrame]:
    workers = []
    collections: dict[str, list[pd.DataFrame]] = {
        "checkpoints": [],
        "task_metrics": [],
        "depth_eval": [],
        "activity": [],
        "plastic_topology": [],
        "final_activity": [],
        "final_plastic_topology": [],
        "dynamics": [],
        "interventions": [],
        "ablation": [],
    }
    suffixes = {
        "checkpoints": "checkpoints",
        "task_metrics": "task-metrics",
        "depth_eval": "depth-eval",
        "activity": "activity",
        "plastic_topology": "plastic-topology",
        "final_activity": "final-activity",
        "final_plastic_topology": "final-plastic-topology",
        "dynamics": "dynamics",
        "interventions": "interventions",
        "ablation": "ablation",
    }
    for replicate in range(N_REPLICATES):
        for variant in VARIANTS:
            run_name = f"r{replicate}-{variant}"
            worker = json.loads((OUT / f"{run_name}-worker.json").read_text(encoding="utf-8"))
            topology = worker.get("topology") or {}
            workers.append(
                {
                    "run": run_name,
                    "replicate": replicate,
                    "variant": variant,
                    "parameters": worker["parameters"],
                    "seconds_per_million_tokens": worker["seconds_per_million_tokens"],
                    "peak_vram_gib": worker["peak_vram_bytes"] / (1024**3),
                    "avg_recurrent_iterations": worker["avg_recurrent_iterations"],
                    "mean_nll": worker["mean_nll"],
                    "mean_token_accuracy": worker["mean_token_accuracy"],
                    "mean_exact_match": worker["mean_exact_match"],
                    "effective_active_fraction": topology.get("effective_active_fraction"),
                    "final_activity_entropy": topology.get("final_activity_entropy"),
                    "final_plastic_entropy": topology.get("final_plastic_entropy"),
                    "final_plastic_tv": topology.get("final_plastic_tv"),
                    "final_mean_plastic_strength": topology.get("final_mean_plastic_strength"),
                    "mean_plastic_to_reaction": topology.get("mean_plastic_to_reaction"),
                    "plastic_distribution_change_l1": topology.get("plastic_distribution_change_l1"),
                    "activity_mi": (topology.get("task_activity_mi") or {}).get("observed"),
                    "activity_mi_null_p99": (topology.get("task_activity_mi") or {}).get("null_p99"),
                    "topology_mi": (topology.get("task_topology_mi") or {}).get("observed"),
                    "topology_mi_null_p99": (topology.get("task_topology_mi") or {}).get("null_p99"),
                    "functional_specific_rows": (worker.get("ablation") or {}).get("functional_specific_rows"),
                }
            )
            for key, suffix in suffixes.items():
                frame = _read_if_present(OUT / f"{run_name}-{suffix}.csv")
                if frame is not None:
                    collections[key].append(frame)

    result = {"model_summary": pd.DataFrame(workers)}
    result["model_summary"].to_csv(OUT / "model-summary.csv", index=False)
    for key, frames in collections.items():
        result[key] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        result[key].to_csv(OUT / f"{key.replace('_', '-')}.csv", index=False)
    return result


def make_paired_ratios(summary: pd.DataFrame, depth_eval: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for replicate in range(N_REPLICATES):
        group = summary.loc[summary["replicate"] == replicate].set_index("variant")
        for metric in ("mean_nll", "seconds_per_million_tokens", "parameters"):
            rows.append({"replicate": replicate, "metric": metric, "E_over_L": float(group.loc["E", metric] / group.loc["L", metric])})
        depth_group = depth_eval.loc[depth_eval["replicate"] == replicate]
        local = depth_group.loc[depth_group["variant"] == "L"].set_index("depth")
        emergent = depth_group.loc[depth_group["variant"] == "E"].set_index("depth")
        local_robust = float(local.loc[2, "mean_nll"] / local.loc[4, "mean_nll"])
        emergent_robust = float(emergent.loc[2, "mean_nll"] / emergent.loc[4, "mean_nll"])
        rows.append({"replicate": replicate, "metric": "depth_robustness_2_over_4", "E_over_L": emergent_robust / local_robust})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "paired-ratios.csv", index=False)
    return frame


def _intervention_passes(interventions: pd.DataFrame, name: str) -> int:
    selected = interventions.loc[interventions["intervention"] == name]
    return int((selected["bootstrap_95_lower"] > 0.0).sum())


def make_decision(summary: pd.DataFrame, ratios: pd.DataFrame, interventions: pd.DataFrame, gpu_count: int) -> dict[str, object]:
    emergent = summary.loc[summary["variant"] == "E"].sort_values("replicate")
    nll_ratios = ratios.loc[ratios["metric"] == "mean_nll", "E_over_L"].tolist()
    robustness_ratios = ratios.loc[ratios["metric"] == "depth_robustness_2_over_4", "E_over_L"].tolist()
    quality_geomean = geometric_mean(nll_ratios)
    quality_pass = quality_geomean <= QUALITY_NLL_RATIO_MAX and max(nll_ratios) <= PER_SEED_NLL_RATIO_MAX
    stability_pass = geometric_mean(robustness_ratios) <= MAX_ROBUSTNESS_RATIO_TO_L
    activity_signal = int((emergent["activity_mi"] > emergent["activity_mi_null_p99"]).sum())
    topology_signal = int((emergent["topology_mi"] > emergent["topology_mi_null_p99"]).sum())
    topology_emergence = int(((emergent["final_mean_plastic_strength"] >= MIN_FINAL_PLASTIC_STRENGTH) & (emergent["final_plastic_tv"] >= MIN_FINAL_PLASTIC_TV)).sum())
    sparse_activity = int((emergent["effective_active_fraction"] <= MAX_EFFECTIVE_ACTIVE_FRACTION).sum())
    plastic_off = _intervention_passes(interventions, "plastic_diffusion_off")
    shuffled = _intervention_passes(interventions, "topology_shuffled")
    plasticity_off = _intervention_passes(interventions, "plasticity_off")
    local_off = _intervention_passes(interventions, "local_diffusion_off")
    causal_topology = plastic_off >= MIN_SIGNAL_REPLICATES and shuffled >= MIN_SIGNAL_REPLICATES
    causal_plasticity = plasticity_off >= MIN_SIGNAL_REPLICATES
    emerged = topology_emergence >= MIN_SIGNAL_REPLICATES and topology_signal >= MIN_SIGNAL_REPLICATES

    pass_flags = {
        "quality": quality_pass,
        "depth_stability": stability_pass,
        "activity_competition": sparse_activity >= MIN_SIGNAL_REPLICATES,
        "task_activity_specialization": activity_signal >= MIN_SIGNAL_REPLICATES,
        "plastic_topology_emergence": topology_emergence >= MIN_SIGNAL_REPLICATES,
        "task_plastic_topology_specialization": topology_signal >= MIN_SIGNAL_REPLICATES,
        "causal_plastic_topology": causal_topology,
        "causal_plasticity": causal_plasticity,
    }

    if all(pass_flags.values()):
        status = "STABLE_CAUSAL_EMERGENT_TOPOLOGY"
    elif all(value for key, value in pass_flags.items() if key != "activity_competition"):
        status = "STABLE_EMERGENT_TOPOLOGY_DENSE_ACTIVITY"
    elif emerged and causal_topology and not quality_pass:
        status = "EMERGENT_TOPOLOGY_DESTABILIZING"
    elif emerged and not causal_topology:
        status = "EMERGENT_TOPOLOGY_EPIPHENOMENAL"
    elif topology_emergence >= MIN_SIGNAL_REPLICATES:
        status = "TOPOLOGY_EMERGED_WITHOUT_TASK_STRUCTURE"
    else:
        status = "NO_EMERGENT_TOPOLOGY"

    return {
        "format": "minicells.local-substrate-emergent-topology.v1",
        "experiment": "MINI Cells Experiment 015c — Local Substrate + Emergent Plastic Topology",
        "status": status,
        "question": "Can a fixed local NCA substrate plus replicator activity and an initially structureless long-range plastic graph form stable, causally useful functional topology without a router or expert-specific parameters?",
        "source_experiment": {
            "experiment": "015b-plastic-reaction-diffusion",
            "finding": "015b made diffusion causally important but mixed fixed geometry with plastic topology, remained dense, and over-coupled computation.",
        },
        "design": {
            "variants": {"L": "replicator activity + fixed nearest-neighbor local substrate only", "E": "same L dynamics plus emergent long-range plastic topology"},
            "replicates": N_REPLICATES,
            "models_total": N_REPLICATES * len(VARIANTS),
            "tokens_per_model": BUDGET_TOKENS,
            "tissue_height": TISSUE_HEIGHT,
            "gpu_count": gpu_count,
            "identical_parameterization": True,
            "router_parameters": 0,
            "expert_specific_parameters": 0,
            "activity_budget": ACTIVITY_BUDGET,
            "activity_dynamics": "multiplicative replicator competition driven by standardized local reaction magnitude",
            "activity_rate": ACTIVITY_RATE,
            "local_substrate": "fixed nearest-neighbor graph diffusion",
            "local_coupling": LOCAL_COUPLING,
            "plastic_candidate_graph": "long-range only, uniform distribution at initialization",
            "initial_plastic_effect": 0.0,
            "plastic_edge_strength": "LONG_RANGE_MAX_COUPLING * (1 - normalized_entropy(distribution))",
            "long_range_max_coupling": LONG_RANGE_MAX_COUPLING,
            "plasticity_rate": PLASTICITY_RATE,
            "plasticity": "multiplicative Hebbian competition over long-range candidate sources with row homeostasis",
        },
        "pre_registered_signal": {
            "E_over_L_nll_geomean_max": QUALITY_NLL_RATIO_MAX,
            "per_seed_E_over_L_nll_max": PER_SEED_NLL_RATIO_MAX,
            "depth_robustness_E_over_L_max": MAX_ROBUSTNESS_RATIO_TO_L,
            "minimum_signal_replicates": MIN_SIGNAL_REPLICATES,
            "effective_active_fraction_max": MAX_EFFECTIVE_ACTIVE_FRACTION,
            "final_plastic_strength_min": MIN_FINAL_PLASTIC_STRENGTH,
            "final_plastic_tv_min": MIN_FINAL_PLASTIC_TV,
            "task_activity_mi": "observed > label-shuffled p99",
            "task_plastic_topology_mi": "observed > label-shuffled p99",
            "causal_topology": "normal beats plastic-diffusion-off and topology-shuffled with paired bootstrap 95% lower > 0",
            "causal_plasticity": "normal beats plasticity-off with paired bootstrap 95% lower > 0",
        },
        "results": {
            "E_over_L_nll_geometric_mean": quality_geomean,
            "E_over_L_nll_ratios": nll_ratios,
            "E_over_L_depth_robustness_geometric_mean": geometric_mean(robustness_ratios),
            "activity_competition_replicates": sparse_activity,
            "task_activity_signal_replicates": activity_signal,
            "plastic_topology_emergence_replicates": topology_emergence,
            "task_plastic_topology_signal_replicates": topology_signal,
            "plastic_diffusion_off_causal_replicates": plastic_off,
            "topology_shuffled_causal_replicates": shuffled,
            "plasticity_off_causal_replicates": plasticity_off,
            "local_diffusion_off_causal_replicates": local_off,
            "mean_effective_active_fraction": float(emergent["effective_active_fraction"].mean()),
            "mean_final_plastic_strength": float(emergent["final_mean_plastic_strength"].mean()),
            "mean_final_plastic_tv": float(emergent["final_plastic_tv"].mean()),
            "mean_plastic_to_reaction": float(emergent["mean_plastic_to_reaction"].mean()),
            "pass_flags": pass_flags,
        },
        "observability": {
            "final_node_topology": "final-node-topology.png",
            "task_topology_atlas": "task-node-topology-atlas.png",
            "final_plastic_topology_heatmap": "final-plastic-topology-heatmap.png",
            "activity_competition": "activity-competition.png",
            "plastic_topology_growth": "plastic-topology-growth.png",
            "reaction_diffusion_dynamics": "reaction-diffusion-dynamics.png",
            "graph_semantics": "dashed neighbor edges are fixed local substrate; directed solid arrows are only emergent long-range plastic edges; node size is mean activity",
        },
        "scope": {
            "claim": "transient emergent long-range functional topology over a fixed local substrate",
            "not_claimed": [
                "persistent long-term cellular memory",
                "skill transplantation",
                "birth/death",
                "unbounded growth",
                "clean zero-shot capability composition",
                "one globally time-homogeneous rule across all three existing stages",
            ],
        },
    }


def _circle_positions(n: int) -> dict[int, tuple[float, float]]:
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
    return {i: (float(np.cos(angle)), float(np.sin(angle))) for i, angle in enumerate(angles)}


def _draw_topology(ax, activity: np.ndarray, weights: np.ndarray, *, title: str) -> None:
    positions = _circle_positions(TISSUE_HEIGHT)
    for row in range(TISSUE_HEIGHT - 1):
        x1, y1 = positions[row]
        x2, y2 = positions[row + 1]
        ax.plot([x1, x2], [y1, y2], linestyle="--", linewidth=1.0, alpha=0.45)
    max_activity = max(1e-6, float(activity.max()))
    for row in range(TISSUE_HEIGHT):
        x, y = positions[row]
        size = 450.0 + 1400.0 * float(activity[row] / max_activity)
        ax.scatter([x], [y], s=size, zorder=3)
        label = f"R{row}" + ("\nI/O" if row == 0 else "")
        ax.text(x, y, label, ha="center", va="center", fontsize=8, zorder=4)
    max_weight = max(1e-8, float(weights.max()))
    for receiver in range(TISSUE_HEIGHT):
        candidates = [(source, float(weights[receiver, source])) for source in range(TISSUE_HEIGHT) if abs(receiver - source) > 1]
        for source, weight in sorted(candidates, key=lambda item: item[1], reverse=True)[:2]:
            if weight <= 1e-5:
                continue
            x1, y1 = positions[source]
            x2, y2 = positions[receiver]
            arrow = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.7 + 3.0 * weight / max_weight,
                alpha=0.7,
                connectionstyle="arc3,rad=0.12",
                shrinkA=18,
                shrinkB=18,
            )
            ax.add_patch(arrow)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")


def make_figures(results: dict[str, pd.DataFrame]) -> None:
    final_activity = results["final_activity"]
    final_topology = results["final_plastic_topology"]
    activity = results["activity"]
    topology = results["plastic_topology"]
    dynamics = results["dynamics"]

    mean_a = final_activity.groupby("row")["mean_activity"].mean().reindex(range(TISSUE_HEIGHT)).to_numpy()
    mean_w = final_topology.groupby(["receiver", "source"])["mean_weight"].mean().unstack(fill_value=0.0).reindex(index=range(TISSUE_HEIGHT), columns=range(TISSUE_HEIGHT), fill_value=0.0).to_numpy()
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_topology(ax, mean_a, mean_w, title="Experiment 015c — Final substrate + emergent topology")
    fig.tight_layout()
    fig.savefig(OUT / "final-node-topology.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for ax, task in zip(axes.flat, ALL_TASKS):
        task_a = activity.loc[activity["task"] == task].groupby("row")["mean_activity"].mean().reindex(range(TISSUE_HEIGHT)).to_numpy()
        task_w = topology.loc[topology["task"] == task].groupby(["receiver", "source"])["mean_weight"].mean().unstack(fill_value=0.0).reindex(index=range(TISSUE_HEIGHT), columns=range(TISSUE_HEIGHT), fill_value=0.0).to_numpy()
        _draw_topology(ax, task_a, task_w, title=task)
    fig.tight_layout()
    fig.savefig(OUT / "task-node-topology-atlas.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(mean_w)
    ax.set_xlabel("source row")
    ax.set_ylabel("receiver row")
    ax.set_title("Emergent long-range topology only")
    fig.colorbar(image, ax=ax, label="mean plastic weight")
    fig.tight_layout()
    fig.savefig(OUT / "final-plastic-topology-heatmap.png", dpi=180)
    plt.close(fig)

    dyn = dynamics.groupby("evolution_step", as_index=False).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in range(TISSUE_HEIGHT):
        ax.plot(dyn["evolution_step"], dyn[f"activity_r{row}"], label=f"R{row}")
    ax.set_xlabel("evolution step")
    ax.set_ylabel("activity resource")
    ax.set_title("Replicator activity competition")
    ax.legend(ncol=4, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "activity-competition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dyn["evolution_step"], dyn["mean_plastic_strength"], label="mean plastic strength")
    ax.plot(dyn["evolution_step"], dyn["plastic_tv"], label="TV from uniform")
    ax.plot(dyn["evolution_step"], 1.0 - dyn["plastic_entropy"], label="1 - topology entropy")
    ax.set_xlabel("evolution step")
    ax.set_title("Emergence of long-range topology")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "plastic-topology-growth.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dyn["evolution_step"], dyn["reaction_rms"], label="reaction")
    ax.plot(dyn["evolution_step"], dyn["local_diffusion_rms"], label="fixed local diffusion")
    ax.plot(dyn["evolution_step"], dyn["plastic_diffusion_rms"], label="emergent plastic diffusion")
    ax.set_xlabel("evolution step")
    ax.set_ylabel("RMS")
    ax.set_title("Reaction / local substrate / plastic topology dynamics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "reaction-diffusion-dynamics.png", dpi=180)
    plt.close(fig)


def write_specs(manifest: dict[str, object]) -> None:
    (OUT / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    task_spec = {
        "format": "minicells.local-substrate-emergent-topology-task.v1",
        "tasks": list(ALL_TASKS),
        "base_tasks": list(BASE_TASKS),
        "composition_map": {key: list(value) for key, value in COMPOSITION_MAP.items()},
        "variants": {"L": "local substrate only", "E": "same substrate plus emergent plastic long-range topology"},
        "mechanism": {
            "activity": "replicator competition with fixed total budget",
            "local": "fixed nearest-neighbor diffusion",
            "plastic": "initially uniform long-range candidate distribution; actual coupling is zero at maximum entropy and grows only as the distribution differentiates",
        },
    }
    (OUT / "task-spec.json").write_text(json.dumps(task_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache, manifest = prepare_corpus()
    write_specs(manifest)
    gpu_count = run_models(cache)
    results = collect_results()
    ratios = make_paired_ratios(results["model_summary"], results["depth_eval"])
    decision = make_decision(results["model_summary"], ratios, results["interventions"], gpu_count)
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_figures(results)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
