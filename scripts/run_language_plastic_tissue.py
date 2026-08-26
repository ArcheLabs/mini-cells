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

from minicells.language_plastic_tissue import (  # noqa: E402
    ACTIVITY_BUDGET,
    ACTIVITY_MOMENTUM,
    NONLOCAL_PRIOR,
    PLASTICITY_RATE,
    STABILITY_WEIGHT,
    SYNAPTIC_BUDGET,
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


OUT = ROOT / "results" / "language-plastic-reaction-diffusion-v1"
SOURCE_015 = ROOT / "artifacts" / "experiments" / "015-emergent-sparse-topology"
WORKER = ROOT / "scripts" / "run_language_plastic_tissue_variant.py"
VARIANTS = ("B", "D")
N_REPLICATES = 3
TRAIN_EXAMPLES = 50_000
VALIDATION_EXAMPLES = 1_000
TRAIN_CORPUS_SEED = 15_015
VALIDATION_CORPUS_SEED = 25_015
BUDGET_TOKENS = 960_000

QUALITY_NLL_RATIO_MAX = 1.05
PER_SEED_NLL_RATIO_MAX = 1.10
MAX_ROBUSTNESS_RATIO_TO_B = 1.15
MIN_SIGNAL_REPLICATES = 2


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(float(value)) for value in values) / len(values))


def prepare_corpus() -> tuple[Path, dict[str, object]]:
    decision_path = SOURCE_015 / "decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError("Experiment 015 artifacts must be merged before 015b")
    source_015 = json.loads(decision_path.read_text(encoding="utf-8"))
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    train_path = cache / "train-corpus.pt"
    validation_path = cache / "validation-corpus.pt"
    manifest_path = cache / "corpus-manifest.json"
    expected = {
        "format": "minicells.language-plastic-reaction-diffusion-corpus.v1",
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
        "source_015_status": source_015.get("status"),
        "source_015_format": source_015.get("format"),
        "source_015_commit_scope": "same synthetic corpus protocol and seed family",
        "objective": "autoregressive next-token prediction with loss restricted to transformed output + EOS",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
        raise RuntimeError("Experiment 015b requires CUDA")
    gpu_count = min(2, available)
    groups = [
        tuple(range(start, min(start + gpu_count, N_REPLICATES)))
        for start in range(0, N_REPLICATES, gpu_count)
    ]
    for group in groups:
        gpu_for = {replicate: index for index, replicate in enumerate(group)}
        # B and D for one replicate remain on the same physical GPU.
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
    if not path.is_file():
        return None
    return pd.read_csv(path)


def collect_results() -> dict[str, pd.DataFrame]:
    workers = []
    collections: dict[str, list[pd.DataFrame]] = {
        "checkpoints": [],
        "task_metrics": [],
        "depth_eval": [],
        "activity": [],
        "connectome": [],
        "final_activity": [],
        "final_connectome": [],
        "dynamics": [],
        "interventions": [],
        "ablation": [],
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
                    "final_connectome_entropy": topology.get("final_connectome_entropy"),
                    "final_nonlocal_mass": topology.get("final_nonlocal_mass"),
                    "mean_diffusion_to_reaction": topology.get("mean_diffusion_to_reaction"),
                    "connectome_change_l1": topology.get("connectome_change_l1"),
                    "activity_mi": (topology.get("task_activity_mi") or {}).get("observed"),
                    "activity_mi_null_p99": (topology.get("task_activity_mi") or {}).get("null_p99"),
                    "connectome_mi": (topology.get("task_connectome_mi") or {}).get("observed"),
                    "connectome_mi_null_p99": (topology.get("task_connectome_mi") or {}).get("null_p99"),
                    "functional_specific_rows": (worker.get("ablation") or {}).get("functional_specific_rows"),
                }
            )
            for key, suffix in (
                ("checkpoints", "checkpoints"),
                ("task_metrics", "task-metrics"),
                ("depth_eval", "depth-eval"),
                ("activity", "activity"),
                ("connectome", "connectome"),
                ("final_activity", "final-activity"),
                ("final_connectome", "final-connectome"),
                ("dynamics", "dynamics"),
                ("interventions", "interventions"),
                ("ablation", "ablation"),
            ):
                frame = _read_if_present(OUT / f"{run_name}-{suffix}.csv")
                if frame is not None:
                    collections[key].append(frame)

    result = {"model_summary": pd.DataFrame(workers)}
    for key, frames in collections.items():
        result[key] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result["model_summary"].to_csv(OUT / "model-summary.csv", index=False)
    for key in collections:
        result[key].to_csv(OUT / f"{key.replace('_', '-')}.csv", index=False)
    return result


def make_paired_ratios(summary: pd.DataFrame, depth_eval: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for replicate in range(N_REPLICATES):
        group = summary.loc[summary["replicate"] == replicate].set_index("variant")
        for metric in ("mean_nll", "seconds_per_million_tokens", "parameters"):
            rows.append(
                {
                    "replicate": replicate,
                    "metric": metric,
                    "D_over_B": float(group.loc["D", metric] / group.loc["B", metric]),
                }
            )
        depth_group = depth_eval.loc[depth_eval["replicate"] == replicate]
        b = depth_group.loc[depth_group["variant"] == "B"].set_index("depth")
        d = depth_group.loc[depth_group["variant"] == "D"].set_index("depth")
        b_robust = float(b.loc[2, "mean_nll"] / b.loc[4, "mean_nll"])
        d_robust = float(d.loc[2, "mean_nll"] / d.loc[4, "mean_nll"])
        rows.append(
            {
                "replicate": replicate,
                "metric": "depth_robustness_2_over_4",
                "D_over_B": d_robust / b_robust,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "paired-ratios.csv", index=False)
    return frame


def _intervention_passes(interventions: pd.DataFrame, name: str) -> int:
    selected = interventions.loc[interventions["intervention"] == name]
    return int((selected["bootstrap_95_lower"] > 0.0).sum())


def make_decision(
    summary: pd.DataFrame,
    ratios: pd.DataFrame,
    interventions: pd.DataFrame,
    gpu_count: int,
) -> dict[str, object]:
    d = summary.loc[summary["variant"] == "D"].sort_values("replicate")
    nll_ratios = ratios.loc[ratios["metric"] == "mean_nll", "D_over_B"].tolist()
    robustness_ratios = ratios.loc[
        ratios["metric"] == "depth_robustness_2_over_4", "D_over_B"
    ].tolist()
    quality_geomean = geometric_mean(nll_ratios)
    quality_pass = quality_geomean <= QUALITY_NLL_RATIO_MAX and max(nll_ratios) <= PER_SEED_NLL_RATIO_MAX
    stability_pass = geometric_mean(robustness_ratios) <= MAX_ROBUSTNESS_RATIO_TO_B
    activity_signal = int((d["activity_mi"] > d["activity_mi_null_p99"]).sum())
    connectome_signal = int((d["connectome_mi"] > d["connectome_mi_null_p99"]).sum())
    diffusion_off_passes = _intervention_passes(interventions, "diffusion_off")
    shuffled_passes = _intervention_passes(interventions, "connectome_shuffled")
    plasticity_off_passes = _intervention_passes(interventions, "plasticity_off")
    causal_routing_pass = (
        diffusion_off_passes >= MIN_SIGNAL_REPLICATES
        and shuffled_passes >= MIN_SIGNAL_REPLICATES
    )
    plasticity_pass = plasticity_off_passes >= MIN_SIGNAL_REPLICATES
    topology_signal = (
        activity_signal >= MIN_SIGNAL_REPLICATES
        and connectome_signal >= MIN_SIGNAL_REPLICATES
    )
    pass_flags = {
        "quality": quality_pass,
        "depth_stability": stability_pass,
        "task_activity_specialization": activity_signal >= MIN_SIGNAL_REPLICATES,
        "task_connectome_specialization": connectome_signal >= MIN_SIGNAL_REPLICATES,
        "causal_diffusion_routing": causal_routing_pass,
        "causal_plasticity": plasticity_pass,
    }
    if all(pass_flags.values()):
        status = "STABLE_CAUSAL_PLASTIC_TOPOLOGY"
    elif topology_signal and not quality_pass:
        status = "PLASTIC_TOPOLOGY_DESTABILIZING"
    elif topology_signal and not (causal_routing_pass and plasticity_pass):
        status = "PLASTIC_TOPOLOGY_EPIPHENOMENAL"
    elif topology_signal:
        status = "PARTIAL_PLASTIC_TOPOLOGY_SIGNAL"
    else:
        status = "NO_PLASTIC_TOPOLOGY_SIGNAL"

    return {
        "format": "minicells.language-plastic-reaction-diffusion.v1",
        "experiment": "MINI Cells Experiment 015b — Plastic Reaction-Diffusion Tissue",
        "status": status,
        "question": (
            "Can one shared cellular rule plus resource-constrained activity and a plastic "
            "reaction-diffusion connectome produce stable, causally useful tissue topology "
            "without a router, top-k gate, expert parameters, or load-balancing loss?"
        ),
        "source_experiment": {
            "experiment": "015-emergent-sparse-topology",
            "reason": "015 showed specialization and routing signals but destructive dynamic messages.",
        },
        "design": {
            "variants": {
                "B": "Experiment-015 sparse-local control",
                "D": "plastic reaction-diffusion tissue",
            },
            "replicates": N_REPLICATES,
            "models_total": N_REPLICATES * len(VARIANTS),
            "tokens_per_model": BUDGET_TOKENS,
            "tissue_height": TISSUE_HEIGHT,
            "gpu_count": gpu_count,
            "shared_initialization": "D copies all shape-compatible B tensors before training",
            "router_parameters": 0,
            "top_k": False,
            "balance_loss": False,
            "trainable_connectome": False,
            "activity_budget": ACTIVITY_BUDGET,
            "activity_momentum": ACTIVITY_MOMENTUM,
            "synaptic_budget": SYNAPTIC_BUDGET,
            "plasticity_rate": PLASTICITY_RATE,
            "nonlocal_prior": NONLOCAL_PRIOR,
            "reaction_diffusion": "h <- h + saturating_activity * (local_reaction + W(h_neighbor-h))",
            "plasticity": "co-active reaction similarity -> multiplicative Hebbian growth -> row homeostasis",
        },
        "pre_registered_signal": {
            "D_over_B_nll_geomean_max": QUALITY_NLL_RATIO_MAX,
            "per_seed_D_over_B_nll_max": PER_SEED_NLL_RATIO_MAX,
            "depth_robustness_D_over_B_max": MAX_ROBUSTNESS_RATIO_TO_B,
            "minimum_signal_replicates": MIN_SIGNAL_REPLICATES,
            "task_activity_mi": "observed > label-shuffled p99",
            "task_connectome_mi": "observed > label-shuffled p99",
            "causal_routing": "normal beats diffusion-off and connectome-shuffled with paired bootstrap 95% lower > 0",
            "causal_plasticity": "normal beats plasticity-off with paired bootstrap 95% lower > 0",
        },
        "results": {
            "D_over_B_nll_geometric_mean": quality_geomean,
            "D_over_B_nll_ratios": nll_ratios,
            "D_over_B_depth_robustness_geometric_mean": geometric_mean(robustness_ratios),
            "task_activity_signal_replicates": activity_signal,
            "task_connectome_signal_replicates": connectome_signal,
            "diffusion_off_causal_replicates": diffusion_off_passes,
            "connectome_shuffled_causal_replicates": shuffled_passes,
            "plasticity_off_causal_replicates": plasticity_off_passes,
            "mean_effective_active_fraction": float(d["effective_active_fraction"].mean()),
            "mean_final_nonlocal_mass": float(d["final_nonlocal_mass"].mean()),
            "mean_connectome_change_l1": float(d["connectome_change_l1"].mean()),
            "mean_diffusion_to_reaction": float(d["mean_diffusion_to_reaction"].mean()),
            "pass_flags": pass_flags,
        },
        "observability": {
            "final_node_topology": "final-node-topology.png",
            "task_topology_atlas": "task-node-topology-atlas.png",
            "final_connectome_heatmap": "final-connectome-heatmap.png",
            "activity_dynamics": "activity-dynamics.png",
            "connectome_dynamics": "connectome-dynamics.png",
            "reaction_diffusion_dynamics": "reaction-diffusion-dynamics.png",
            "machine_readable": [
                "activity.csv",
                "connectome.csv",
                "final-activity.csv",
                "final-connectome.csv",
                "dynamics.csv",
                "interventions.csv",
                "ablation.csv",
            ],
        },
        "scope": {
            "claim": "transient plastic functional topology only",
            "not_claimed": [
                "persistent long-term cellular memory",
                "skill transplantation",
                "birth/death",
                "unbounded growth",
                "clean zero-shot capability composition",
            ],
        },
    }


def _aggregate_graph(
    activity: pd.DataFrame,
    connectome: pd.DataFrame,
    *,
    task: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    a = activity.copy()
    c = connectome.copy()
    if task is not None:
        a = a.loc[a["task"] == task]
        c = c.loc[c["task"] == task]
    mean_activity = a.groupby("row")["mean_activity"].mean().reindex(range(TISSUE_HEIGHT)).to_numpy()
    pivot = (
        c.groupby(["receiver", "source"])["mean_weight"]
        .mean()
        .unstack("source")
        .reindex(index=range(TISSUE_HEIGHT), columns=range(TISSUE_HEIGHT), fill_value=0.0)
    )
    return mean_activity, pivot.to_numpy()


def _draw_topology(ax, activity: np.ndarray, connectome: np.ndarray, title: str) -> None:
    theta = np.linspace(0, 2 * np.pi, TISSUE_HEIGHT, endpoint=False) + np.pi / 2
    xy = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    max_activity = max(1e-8, float(activity.max()))
    node_sizes = 500.0 + 1500.0 * activity / max_activity
    max_weight = max(1e-8, float(connectome.max()))

    # For readability the graph shows the two strongest incoming sources per receiver.
    for receiver in range(TISSUE_HEIGHT):
        candidates = [
            source
            for source in np.argsort(connectome[receiver])[::-1]
            if source != receiver and connectome[receiver, source] > 0
        ][:2]
        for source in candidates:
            weight = float(connectome[receiver, source])
            start = xy[source]
            end = xy[receiver]
            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.5 + 4.0 * weight / max_weight,
                alpha=0.25 + 0.65 * weight / max_weight,
                connectionstyle="arc3,rad=0.12",
            )
            ax.add_patch(arrow)

    ax.scatter(xy[:, 0], xy[:, 1], s=node_sizes)
    for row, (x, y) in enumerate(xy):
        label = "R0 / I-O" if row == 0 else f"R{row}"
        ax.text(x, y, label, ha="center", va="center", fontsize=8)
    ax.set_title(title)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")


def save_plots(frames: dict[str, pd.DataFrame]) -> None:
    activity = frames["activity"].loc[frames["activity"]["run"].str.endswith("-D")]
    connectome = frames["connectome"].loc[frames["connectome"]["run"].str.endswith("-D")]
    dynamics = frames["dynamics"].loc[frames["dynamics"]["run"].str.endswith("-D")]

    mean_activity, mean_connectome = _aggregate_graph(activity, connectome)
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    _draw_topology(
        ax,
        mean_activity,
        mean_connectome,
        "Experiment 015b — final plastic tissue topology\n(3 seeds × all validation tasks; top-2 incoming edges shown)",
    )
    fig.tight_layout()
    fig.savefig(OUT / "final-node-topology.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, task in zip(axes.flat, ALL_TASKS):
        task_activity, task_connectome = _aggregate_graph(activity, connectome, task=task)
        _draw_topology(ax, task_activity, task_connectome, task)
    fig.suptitle("Experiment 015b — task-conditioned topology atlas")
    fig.tight_layout()
    fig.savefig(OUT / "task-node-topology-atlas.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    image = ax.imshow(mean_connectome, aspect="auto")
    ax.set_xlabel("Source row")
    ax.set_ylabel("Receiver row")
    ax.set_xticks(range(TISSUE_HEIGHT))
    ax.set_yticks(range(TISSUE_HEIGHT))
    ax.set_title("Experiment 015b — final mean connectome")
    fig.colorbar(image, ax=ax, label="mean synaptic weight")
    fig.tight_layout()
    fig.savefig(OUT / "final-connectome-heatmap.png", dpi=170)
    plt.close(fig)

    mean_dynamics = dynamics.groupby("evolution_step").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for row in range(TISSUE_HEIGHT):
        ax.plot(
            mean_dynamics.index,
            mean_dynamics[f"activity_r{row}"],
            marker="o",
            label=f"R{row}",
        )
    ax.set_xlabel("Evolution step")
    ax.set_ylabel("Mean metabolic activity")
    ax.set_title("Experiment 015b — activity allocation dynamics")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "activity-dynamics.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(mean_dynamics.index, mean_dynamics["connectome_entropy"], marker="o", label="connectome entropy")
    ax.plot(mean_dynamics.index, mean_dynamics["nonlocal_mass"], marker="o", label="non-local mass")
    ax.set_xlabel("Evolution step")
    ax.set_ylabel("Normalized topology statistic")
    ax.set_title("Experiment 015b — connectome self-organization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "connectome-dynamics.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(mean_dynamics.index, mean_dynamics["reaction_rms"], marker="o", label="reaction RMS")
    ax.plot(mean_dynamics.index, mean_dynamics["diffusion_rms"], marker="o", label="diffusion RMS")
    ax.plot(
        mean_dynamics.index,
        mean_dynamics["diffusion_to_reaction"],
        marker="o",
        label="diffusion / reaction",
    )
    ax.set_xlabel("Evolution step")
    ax.set_ylabel("Magnitude")
    ax.set_title("Experiment 015b — reaction / diffusion dynamics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "reaction-diffusion-dynamics.png", dpi=170)
    plt.close(fig)


def write_task_spec() -> None:
    payload = {
        "format": "minicells.language-plastic-reaction-diffusion-task.v1",
        "experiment": "015b",
        "name": "Plastic Reaction-Diffusion Tissue",
        "variants": list(VARIANTS),
        "replicates": N_REPLICATES,
        "tokens_per_model": BUDGET_TOKENS,
        "tissue_height": TISSUE_HEIGHT,
        "activity_budget": ACTIVITY_BUDGET,
        "activity_momentum": ACTIVITY_MOMENTUM,
        "synaptic_budget": SYNAPTIC_BUDGET,
        "plasticity_rate": PLASTICITY_RATE,
        "nonlocal_prior": NONLOCAL_PRIOR,
        "stability_weight": STABILITY_WEIGHT,
        "mechanism": [
            "one shared cellular reaction rule",
            "continuous finite metabolic activity budget",
            "graph-Laplacian-like diffusion",
            "co-activity + reaction-correlation Hebbian plasticity",
            "per-cell synaptic homeostasis",
        ],
        "removed_from_015": [
            "trainable activity gate",
            "hard top-k",
            "dynamic top-1 router",
            "load-balancing loss",
            "trainable edge parameters",
        ],
        "primary_endpoints": [
            "D/B validation NLL",
            "depth robustness",
            "task-activity MI vs shuffled null",
            "task-connectome MI vs shuffled null",
            "paired diffusion-off causal intervention",
            "paired connectome-shuffled causal intervention",
            "paired plasticity-off causal intervention",
        ],
    }
    (OUT / "task-spec.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache_dir, manifest = prepare_corpus()
    write_task_spec()
    gpu_count = run_models(cache_dir)
    frames = collect_results()
    ratios = make_paired_ratios(frames["model_summary"], frames["depth_eval"])
    decision = make_decision(
        frames["model_summary"],
        ratios,
        frames["interventions"],
        gpu_count,
    )
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUT / "corpus-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_plots(frames)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
