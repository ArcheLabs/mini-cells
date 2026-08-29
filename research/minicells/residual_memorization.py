"""Core Validation 001b: generalization vs residual memorization.

This module is intentionally diagnostic-only. It consumes checkpoints produced by
Core Validation 001 and never changes the parent training algorithm. The main
intervention ranks every non-DC conjugate Fourier pair by late embedding energy
and then cumulatively removes or retains the first k pairs. If late old and
held-out accuracy degrade together, the surviving computation is consistent
with shared generalization. A persistent old-only advantage is the operational
signature of residual memorization.

The assay is calibrated in two ways:
1. The early checkpoint must show a large seen-vs-unseen gap, proving the same
   model family can expose membership-specific memorization.
2. A cumulative-replay oracle must itself show coupled seen-vs-heldout decay;
   otherwise the intervention is considered invalid rather than negative
   evidence for the hypothesis.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .knowledge_subsumption import (
    CellularModularNet,
    Curriculum,
    KnowledgeSubsumptionConfig,
    evaluate_indices,
    fourier_filter_embedding,
    fourier_pair_energy,
    make_curriculum,
)


@dataclass(frozen=True)
class ResidualMemorizationConfig:
    minimum_early_gap: float = 0.50
    minimum_correlation: float = 0.95
    maximum_mean_absolute_gap: float = 0.05
    maximum_positive_gap: float = 0.10
    maximum_dc_only_accuracy: float = 0.15
    oracle_minimum_correlation: float = 0.95
    oracle_maximum_mean_absolute_gap: float = 0.05
    oracle_maximum_positive_gap: float = 0.10
    oracle_maximum_dc_only_accuracy: float = 0.15

    @classmethod
    def from_protocol(cls, path: str | Path) -> "ResidualMemorizationConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        gates = payload["gates"]
        coupling = gates["late_membership_coupling"]
        endpoint = gates["sweep_endpoint"]
        oracle = gates["oracle_validity"]
        return cls(
            minimum_early_gap=float(
                gates["assay_sensitivity"][
                    "minimum_early_seen_minus_unseen_gap_at_k0"
                ]
            ),
            minimum_correlation=float(
                coupling["minimum_exclusion_accuracy_correlation"]
            ),
            maximum_mean_absolute_gap=float(
                coupling["maximum_mean_absolute_old_heldout_gap"]
            ),
            maximum_positive_gap=float(
                coupling["maximum_positive_old_heldout_gap"]
            ),
            maximum_dc_only_accuracy=max(
                float(endpoint["maximum_dc_only_old_accuracy"]),
                float(endpoint["maximum_dc_only_heldout_accuracy"]),
            ),
            oracle_minimum_correlation=float(
                oracle["minimum_exclusion_accuracy_correlation"]
            ),
            oracle_maximum_mean_absolute_gap=float(
                oracle["maximum_mean_absolute_seen_heldout_gap"]
            ),
            oracle_maximum_positive_gap=float(
                oracle["maximum_positive_seen_heldout_gap"]
            ),
            oracle_maximum_dc_only_accuracy=max(
                float(oracle["maximum_dc_only_seen_accuracy"]),
                float(oracle["maximum_dc_only_heldout_accuracy"]),
            ),
        )


def rank_all_frequency_pairs(embedding: torch.Tensor) -> tuple[int, ...]:
    """Return every non-DC conjugate pair ranked by embedding energy."""
    energy = fourier_pair_energy(embedding)
    order = torch.argsort(energy, descending=True) + 1
    return tuple(int(value) for value in order.tolist())


@torch.no_grad()
def cumulative_fourier_sweep(
    model: CellularModularNet,
    curriculum: Curriculum,
    ranking: tuple[int, ...],
    partitions: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Measure cumulative exclusion and restriction for k=0..all pairs."""
    model.eval()
    rows: list[dict[str, Any]] = []
    original = model.embedding.weight.detach()
    for k in range(len(ranking) + 1):
        selected = tuple(ranking[:k])
        excluded = fourier_filter_embedding(original, selected, keep_keys=False)
        restricted = fourier_filter_embedding(original, selected, keep_keys=True)
        row: dict[str, Any] = {
            "k": k,
            "selected_frequency_pairs": list(selected),
            "excluded": {},
            "restricted": {},
        }
        for name, indices in partitions.items():
            row["excluded"][name] = evaluate_indices(
                model,
                curriculum,
                indices,
                device=device,
                embedding_override=excluded,
            )
            row["restricted"][name] = evaluate_indices(
                model,
                curriculum,
                indices,
                device=device,
                embedding_override=restricted,
            )
        rows.append(row)
    return rows


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("correlation inputs must have equal non-zero length")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered))
    left_norm = math.sqrt(sum(value * value for value in left_centered))
    right_norm = math.sqrt(sum(value * value for value in right_centered))
    denominator = left_norm * right_norm
    if denominator <= 1e-12:
        return 1.0 if all(abs(a - b) <= 1e-12 for a, b in zip(left, right)) else 0.0
    return numerator / denominator


def coupling_metrics(
    sweep: list[dict[str, Any]],
    left_partition: str,
    right_partition: str,
) -> dict[str, float]:
    """Summarize whether two random partitions degrade together under exclusion."""
    left = [
        float(row["excluded"][left_partition]["accuracy"])
        for row in sweep
    ]
    right = [
        float(row["excluded"][right_partition]["accuracy"])
        for row in sweep
    ]
    gaps = [a - b for a, b in zip(left, right)]
    absolute = [abs(value) for value in gaps]
    return {
        "exclusion_accuracy_correlation": _pearson(left, right),
        "mean_absolute_gap": sum(absolute) / len(absolute),
        "maximum_positive_gap": max(0.0, max(gaps)),
        "maximum_absolute_gap": max(absolute),
        "endpoint_left_accuracy": left[-1],
        "endpoint_right_accuracy": right[-1],
        "left_auc_mean": sum(left) / len(left),
        "right_auc_mean": sum(right) / len(right),
    }


def _load_model(
    config: KnowledgeSubsumptionConfig,
    checkpoint: str | Path,
    *,
    device: torch.device,
) -> CellularModularNet:
    model = CellularModularNet(config).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def analyze_checkpoint_pair(
    training_config: KnowledgeSubsumptionConfig,
    residual_config: ResidualMemorizationConfig,
    source_run: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Analyze one Core-001 early/late checkpoint pair without retraining it."""
    seed = int(source_run["seed"])
    task = str(source_run["task"])
    checkpoints = source_run.get("checkpoints", {})
    early_path = checkpoints.get("early")
    late_path = checkpoints.get("late")
    if not early_path or not late_path:
        raise RuntimeError(
            f"Core Validation 001 checkpoints are required for task={task} seed={seed}"
        )
    curriculum = make_curriculum(training_config, seed=seed, task=task)
    early_model = _load_model(training_config, early_path, device=device)
    late_model = _load_model(training_config, late_path, device=device)
    ranking = rank_all_frequency_pairs(late_model.embedding.weight)

    early_sweep = cumulative_fourier_sweep(
        early_model,
        curriculum,
        ranking,
        {
            "seen": curriculum.phases[0],
            "unseen": curriculum.future_after_first,
        },
        device=device,
    )
    late_sweep = cumulative_fourier_sweep(
        late_model,
        curriculum,
        ranking,
        {
            "old": curriculum.late_old,
            "current": curriculum.late_current,
            "heldout": curriculum.heldout,
        },
        device=device,
    )
    late_coupling = coupling_metrics(late_sweep, "old", "heldout")
    early_gap = (
        float(early_sweep[0]["excluded"]["seen"]["accuracy"])
        - float(early_sweep[0]["excluded"]["unseen"]["accuracy"])
    )
    parent_gates = source_run["gates"]
    parent_preconditions = bool(
        parent_gates["early_memorization"]
        and parent_gates["late_generalization"]
        and parent_gates["generalizing_circuit"]
    )
    gates = {
        "parent_preconditions": parent_preconditions,
        "assay_sensitivity": early_gap >= residual_config.minimum_early_gap,
        "synchronized_decay": (
            late_coupling["exclusion_accuracy_correlation"]
            >= residual_config.minimum_correlation
        ),
        "no_material_membership_advantage": (
            late_coupling["mean_absolute_gap"]
            <= residual_config.maximum_mean_absolute_gap
            and late_coupling["maximum_positive_gap"]
            <= residual_config.maximum_positive_gap
        ),
        "dc_endpoint_destroyed": (
            late_coupling["endpoint_left_accuracy"]
            <= residual_config.maximum_dc_only_accuracy
            and late_coupling["endpoint_right_accuracy"]
            <= residual_config.maximum_dc_only_accuracy
        ),
    }
    gates["pass"] = all(gates.values())
    return {
        "task": task,
        "seed": seed,
        "source_control_valid": source_run.get("control_valid"),
        "source_parent_gates": parent_gates,
        "frequency_ranking": list(ranking),
        "early_seen_minus_unseen_gap_at_k0": early_gap,
        "late_coupling": late_coupling,
        "gates": gates,
        "early_sweep": early_sweep,
        "late_sweep": late_sweep,
    }


def train_oracle_model(
    config: KnowledgeSubsumptionConfig,
    *,
    seed: int,
    device: torch.device,
) -> tuple[CellularModularNet, Curriculum]:
    """Reproduce the parent cumulative-replay oracle, returning the trained model."""
    torch.manual_seed(seed + 10000)
    random.seed(seed + 10000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 10000)
    curriculum = make_curriculum(config, seed=seed, task="modular_addition")
    model = CellularModularNet(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    seen = curriculum.late_seen
    pairs = curriculum.pairs[seen]
    labels = curriculum.labels[seen]
    generator = torch.Generator().manual_seed(seed + 10001)
    for _ in range(sum(config.phase_steps)):
        model.train()
        if len(seen) <= config.batch_size:
            selection = torch.arange(len(seen))
        else:
            selection = torch.randint(
                len(seen),
                (config.batch_size,),
                generator=generator,
            )
        batch_pairs = pairs[selection].to(device)
        batch_labels = labels[selection].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch_pairs), batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
    model.eval()
    return model, curriculum


def analyze_oracle(
    training_config: KnowledgeSubsumptionConfig,
    residual_config: ResidualMemorizationConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model, curriculum = train_oracle_model(training_config, seed=seed, device=device)
    ranking = rank_all_frequency_pairs(model.embedding.weight)
    sweep = cumulative_fourier_sweep(
        model,
        curriculum,
        ranking,
        {"seen": curriculum.late_seen, "heldout": curriculum.heldout},
        device=device,
    )
    coupling = coupling_metrics(sweep, "seen", "heldout")
    gates = {
        "synchronized_decay": (
            coupling["exclusion_accuracy_correlation"]
            >= residual_config.oracle_minimum_correlation
        ),
        "no_material_membership_advantage": (
            coupling["mean_absolute_gap"]
            <= residual_config.oracle_maximum_mean_absolute_gap
            and coupling["maximum_positive_gap"]
            <= residual_config.oracle_maximum_positive_gap
        ),
        "dc_endpoint_destroyed": (
            coupling["endpoint_left_accuracy"]
            <= residual_config.oracle_maximum_dc_only_accuracy
            and coupling["endpoint_right_accuracy"]
            <= residual_config.oracle_maximum_dc_only_accuracy
        ),
    }
    gates["valid"] = all(gates.values())
    return {
        "seed": seed,
        "frequency_ranking": list(ranking),
        "coupling": coupling,
        "gates": gates,
        "sweep": sweep,
    }


def summarize_experiment(
    runs: list[dict[str, Any]],
    oracle: dict[str, Any],
    *,
    positive_status: str,
    negative_status: str,
) -> dict[str, Any]:
    primary = [run for run in runs if run["task"] == "modular_addition"]
    controls = [run for run in runs if run["task"] == "balanced_random_labels"]
    primary_passes = sum(bool(run["gates"]["pass"]) for run in primary)
    control_false_positives = sum(bool(run["gates"]["pass"]) for run in controls)
    valid_controls = sum(bool(run.get("source_control_valid")) for run in controls)
    all_primary = bool(primary) and primary_passes == len(primary)
    all_controls_valid = bool(controls) and valid_controls == len(controls)
    zero_control = bool(controls) and control_false_positives == 0
    oracle_valid = bool(oracle.get("gates", {}).get("valid"))
    supported = all_primary and all_controls_valid and zero_control and oracle_valid
    return {
        "status": positive_status if supported else negative_status,
        "supported": supported,
        "primary_runs": len(primary),
        "primary_passes": primary_passes,
        "control_runs": len(controls),
        "valid_control_runs": valid_controls,
        "control_false_positives": control_false_positives,
        "oracle_valid": oracle_valid,
        "requirements": {
            "all_primary_seeds_pass": all_primary,
            "all_controls_memorization_valid": all_controls_valid,
            "zero_control_false_positives": zero_control,
            "oracle_valid": oracle_valid,
        },
    }
