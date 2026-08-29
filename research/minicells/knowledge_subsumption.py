"""Core Validation 001: knowledge subsumption and computational reorganization.

A factored one-hidden-layer modular-addition network is trained on a sequential
curriculum with no replay. Hidden neurons are grouped into fixed causal cells
only so interventions can be measured; there is no routing, growth, mitosis, or
tissue mechanism in the formal experiment.

The primary mechanistic readout follows the modular-addition grokking literature:
late Fourier structure is identified from the learned embedding, then the same
frequency set is used to construct restricted (generalizing frequencies only)
and excluded (generalizing frequencies removed) models at early and late
checkpoints. This directly tests circuit formation and cleanup rather than
assuming that a historical fact permanently belongs to a particular cell.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

TaskName = Literal["modular_addition", "balanced_random_labels"]


@dataclass(frozen=True)
class KnowledgeSubsumptionConfig:
    modulus: int = 31
    curriculum_fractions: tuple[float, ...] = (0.10, 0.20, 0.20, 0.20)
    phase_steps: tuple[int, ...] = (3000, 5000, 8000, 12000)
    eval_interval_steps: int = 250
    embedding_dim: int = 64
    num_cells: int = 16
    neurons_per_cell: int = 8
    learning_rate: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 1.0
    gradient_clip_norm: float = 1.0
    batch_size: int = 256
    probe_examples_per_partition: int = 64
    path_cells: int = 3
    key_frequency_pairs: int = 3
    early_minimum_seen_accuracy: float = 0.98
    early_maximum_unseen_accuracy: float = 0.20
    late_minimum_old_accuracy: float = 0.90
    late_minimum_current_accuracy: float = 0.90
    late_minimum_heldout_accuracy: float = 0.90
    restricted_minimum_old_accuracy: float = 0.80
    restricted_minimum_heldout_accuracy: float = 0.80
    early_excluded_minimum_seen_accuracy: float = 0.80
    late_excluded_maximum_old_accuracy: float = 0.30
    late_excluded_maximum_heldout_accuracy: float = 0.20

    def __post_init__(self) -> None:
        if self.modulus < 5 or self.modulus % 2 == 0:
            raise ValueError("modulus must be an odd integer >= 5")
        if len(self.curriculum_fractions) < 2:
            raise ValueError("at least two curriculum phases are required")
        if len(self.curriculum_fractions) != len(self.phase_steps):
            raise ValueError("curriculum_fractions and phase_steps must have equal length")
        if any(value <= 0 for value in self.curriculum_fractions):
            raise ValueError("curriculum fractions must be positive")
        if sum(self.curriculum_fractions) >= 1.0:
            raise ValueError("curriculum must leave a held-out partition")
        if any(value <= 0 for value in self.phase_steps):
            raise ValueError("phase steps must be positive")
        if self.num_cells < 1 or self.neurons_per_cell < 1:
            raise ValueError("cell dimensions must be positive")
        if not 1 <= self.path_cells <= self.num_cells:
            raise ValueError("path_cells must be in [1, num_cells]")
        if not 1 <= self.key_frequency_pairs <= (self.modulus - 1) // 2:
            raise ValueError("invalid key_frequency_pairs")

    @property
    def hidden_dim(self) -> int:
        return self.num_cells * self.neurons_per_cell

    @classmethod
    def from_protocol(cls, path: str | Path) -> "KnowledgeSubsumptionConfig":
        payload = json.loads(Path(path).read_text())
        task = payload["task"]
        model = payload["model"]
        optimizer = payload["optimizer"]
        probe = payload["causal_probe"]
        gates = payload["gates"]
        return cls(
            modulus=int(task["modulus"]),
            curriculum_fractions=tuple(float(v) for v in task["curriculum_fractions"]),
            phase_steps=tuple(int(v) for v in task["phase_steps"]),
            eval_interval_steps=int(task["evaluation_interval_steps"]),
            embedding_dim=int(model["embedding_dim"]),
            num_cells=int(model["num_cells"]),
            neurons_per_cell=int(model["neurons_per_cell"]),
            learning_rate=float(optimizer["learning_rate"]),
            betas=tuple(float(v) for v in optimizer["betas"]),
            weight_decay=float(optimizer["weight_decay"]),
            gradient_clip_norm=float(optimizer["gradient_clip_norm"]),
            batch_size=int(optimizer["batch_size"]),
            probe_examples_per_partition=int(probe["examples_per_partition"]),
            path_cells=int(probe["path_cells"]),
            key_frequency_pairs=int(probe["key_frequency_pairs"]),
            early_minimum_seen_accuracy=float(gates["early_memorization"]["minimum_seen_accuracy"]),
            early_maximum_unseen_accuracy=float(
                gates["early_memorization"]["maximum_unseen_accuracy"]
            ),
            late_minimum_old_accuracy=float(gates["late_generalization"]["minimum_old_accuracy"]),
            late_minimum_current_accuracy=float(
                gates["late_generalization"]["minimum_current_accuracy"]
            ),
            late_minimum_heldout_accuracy=float(
                gates["late_generalization"]["minimum_heldout_accuracy"]
            ),
            restricted_minimum_old_accuracy=float(
                gates["generalizing_circuit"]["minimum_restricted_old_accuracy"]
            ),
            restricted_minimum_heldout_accuracy=float(
                gates["generalizing_circuit"]["minimum_restricted_heldout_accuracy"]
            ),
            early_excluded_minimum_seen_accuracy=float(
                gates["memorization_cleanup"]["minimum_early_excluded_seen_accuracy"]
            ),
            late_excluded_maximum_old_accuracy=float(
                gates["memorization_cleanup"]["maximum_late_excluded_old_accuracy"]
            ),
            late_excluded_maximum_heldout_accuracy=float(
                gates["memorization_cleanup"]["maximum_late_excluded_heldout_accuracy"]
            ),
        )


@dataclass(frozen=True)
class Curriculum:
    pairs: torch.Tensor
    labels: torch.Tensor
    phases: tuple[torch.Tensor, ...]
    heldout: torch.Tensor

    @property
    def future_after_first(self) -> torch.Tensor:
        return torch.cat((*self.phases[1:], self.heldout))

    @property
    def late_old(self) -> torch.Tensor:
        return torch.cat(self.phases[:-1])

    @property
    def late_current(self) -> torch.Tensor:
        return self.phases[-1]

    @property
    def late_seen(self) -> torch.Tensor:
        return torch.cat(self.phases)


class CellularModularNet(nn.Module):
    """Factored modular-addition MLP with exact hidden-cell ablation."""

    def __init__(self, config: KnowledgeSubsumptionConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.modulus, config.embedding_dim)
        self.input_projection = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.embedding_dim, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(config.modulus))

    def forward(
        self,
        pairs: torch.Tensor,
        *,
        ablate_cells: tuple[int, ...] | list[int] = (),
        embedding_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self.embedding.weight if embedding_override is None else embedding_override
        left = F.embedding(pairs[:, 0], embedding)
        right = F.embedding(pairs[:, 1], embedding)
        hidden = F.relu(self.input_projection(left + right))
        if ablate_cells:
            mask = torch.ones(self.config.hidden_dim, device=hidden.device, dtype=hidden.dtype)
            width = self.config.neurons_per_cell
            for cell in ablate_cells:
                start = int(cell) * width
                stop = start + width
                mask[start:stop] = 0
            hidden = hidden * mask
        latent = self.output_projection(hidden)
        return latent @ embedding.transpose(0, 1) + self.output_bias


def make_curriculum(
    config: KnowledgeSubsumptionConfig,
    *,
    seed: int,
    task: TaskName,
) -> Curriculum:
    modulus = config.modulus
    values = torch.arange(modulus, dtype=torch.long)
    pairs = torch.cartesian_prod(values, values)
    true_labels = (pairs[:, 0] + pairs[:, 1]) % modulus
    label_generator = torch.Generator().manual_seed(seed + 19)
    if task == "modular_addition":
        labels = true_labels
    elif task == "balanced_random_labels":
        labels = true_labels[torch.randperm(len(true_labels), generator=label_generator)]
    else:
        raise ValueError(f"unknown task: {task}")

    split_generator = torch.Generator().manual_seed(seed + 97)
    order = torch.randperm(len(pairs), generator=split_generator)
    phase_sizes = [max(1, int(round(len(pairs) * value))) for value in config.curriculum_fractions]
    if sum(phase_sizes) >= len(pairs):
        raise ValueError("curriculum leaves no held-out examples after integer rounding")
    phases: list[torch.Tensor] = []
    offset = 0
    for size in phase_sizes:
        phases.append(order[offset : offset + size])
        offset += size
    return Curriculum(pairs, labels, tuple(phases), order[offset:])


def _partition_tensors(curriculum: Curriculum, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return curriculum.pairs[indices], curriculum.labels[indices]


@torch.no_grad()
def evaluate_indices(
    model: CellularModularNet,
    curriculum: Curriculum,
    indices: torch.Tensor,
    *,
    device: torch.device,
    ablate_cells: tuple[int, ...] | list[int] = (),
    embedding_override: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    pairs, labels = _partition_tensors(curriculum, indices)
    pairs = pairs.to(device)
    labels = labels.to(device)
    if embedding_override is not None:
        embedding_override = embedding_override.to(device)
    logits = model(
        pairs,
        ablate_cells=ablate_cells,
        embedding_override=embedding_override,
    )
    loss = F.cross_entropy(logits, labels)
    accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    return {"nll": float(loss.item()), "accuracy": float(accuracy.item())}


def _sample_indices(indices: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    if len(indices) <= count:
        return indices.clone()
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(len(indices), generator=generator)[:count]
    return indices[selected]


@torch.no_grad()
def responsibility_matrix(
    model: CellularModularNet,
    curriculum: Curriculum,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Positive per-example NLL increase caused by ablating each hidden cell."""

    model.eval()
    pairs, labels = _partition_tensors(curriculum, indices)
    pairs = pairs.to(device)
    labels = labels.to(device)
    baseline = F.cross_entropy(model(pairs), labels, reduction="none")
    rows: list[torch.Tensor] = []
    for cell_index in range(model.config.num_cells):
        ablated = model(pairs, ablate_cells=(cell_index,))
        losses = F.cross_entropy(ablated, labels, reduction="none")
        rows.append((losses - baseline).clamp_min(0).detach().cpu())
    return torch.stack(rows, dim=0)


def path_fingerprints(responsibility: torch.Tensor, path_cells: int) -> list[tuple[int, ...]]:
    if responsibility.ndim != 2:
        raise ValueError("responsibility must have shape [cells, examples]")
    k = min(path_cells, responsibility.shape[0])
    indices = torch.topk(responsibility, k=k, dim=0).indices.transpose(0, 1)
    return [tuple(sorted(int(value) for value in row.tolist())) for row in indices]


def mean_pairwise_jaccard(paths: list[tuple[int, ...]]) -> float:
    if len(paths) < 2:
        return 1.0
    total = 0.0
    count = 0
    for left_index, left_path in enumerate(paths):
        left = set(left_path)
        for right_path in paths[left_index + 1 :]:
            right = set(right_path)
            total += len(left & right) / max(1, len(left | right))
            count += 1
    return total / max(1, count)


def fourier_pair_energy(embedding: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.fft(embedding.detach().float(), dim=0)
    energy = spectrum.abs().square().sum(dim=1)
    half = (embedding.shape[0] - 1) // 2
    return torch.stack([energy[index] + energy[-index] for index in range(1, half + 1)])


def select_key_frequency_pairs(embedding: torch.Tensor, count: int) -> tuple[int, ...]:
    pair_energy = fourier_pair_energy(embedding)
    k = min(count, len(pair_energy))
    selected = torch.topk(pair_energy, k=k).indices + 1
    return tuple(sorted(int(value) for value in selected.tolist()))


def fourier_filter_embedding(
    embedding: torch.Tensor,
    key_pairs: tuple[int, ...],
    *,
    keep_keys: bool,
) -> torch.Tensor:
    """Keep/remove selected conjugate Fourier pairs while always retaining DC."""

    spectrum = torch.fft.fft(embedding.detach().float(), dim=0)
    mask = torch.zeros(embedding.shape[0], device=spectrum.device, dtype=spectrum.real.dtype)
    if keep_keys:
        mask[0] = 1
        for frequency in key_pairs:
            mask[frequency] = 1
            mask[-frequency] = 1
    else:
        mask.fill_(1)
        for frequency in key_pairs:
            mask[frequency] = 0
            mask[-frequency] = 0
    filtered = torch.fft.ifft(spectrum * mask[:, None], dim=0).real
    return filtered.to(dtype=embedding.dtype)


def fourier_concentration(embedding: torch.Tensor, key_pairs: tuple[int, ...]) -> float:
    pair_energy = fourier_pair_energy(embedding)
    total = float(pair_energy.sum().item())
    selected = sum(float(pair_energy[index - 1].item()) for index in key_pairs)
    return selected / max(total, 1e-12)


def _fourier_eval_bundle(
    model: CellularModularNet,
    curriculum: Curriculum,
    key_pairs: tuple[int, ...],
    *,
    device: torch.device,
    partitions: dict[str, torch.Tensor],
) -> dict[str, dict[str, dict[str, float]]]:
    restricted = fourier_filter_embedding(model.embedding.weight, key_pairs, keep_keys=True)
    excluded = fourier_filter_embedding(model.embedding.weight, key_pairs, keep_keys=False)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for name, indices in partitions.items():
        result[name] = {
            "full": evaluate_indices(model, curriculum, indices, device=device),
            "restricted": evaluate_indices(
                model,
                curriculum,
                indices,
                device=device,
                embedding_override=restricted,
            ),
            "excluded": evaluate_indices(
                model,
                curriculum,
                indices,
                device=device,
                embedding_override=excluded,
            ),
        }
    return result


def mechanistic_diagnostics(
    early_model: CellularModularNet,
    late_model: CellularModularNet,
    curriculum: Curriculum,
    config: KnowledgeSubsumptionConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    key_pairs = select_key_frequency_pairs(late_model.embedding.weight, config.key_frequency_pairs)
    early_fourier = _fourier_eval_bundle(
        early_model,
        curriculum,
        key_pairs,
        device=device,
        partitions={
            "seen": curriculum.phases[0],
            "unseen": curriculum.future_after_first,
        },
    )
    late_fourier = _fourier_eval_bundle(
        late_model,
        curriculum,
        key_pairs,
        device=device,
        partitions={
            "old": curriculum.late_old,
            "current": curriculum.late_current,
            "heldout": curriculum.heldout,
        },
    )

    early_probe = _sample_indices(
        curriculum.phases[0], config.probe_examples_per_partition, seed + 301
    )
    late_probe = torch.cat(
        (
            _sample_indices(curriculum.late_old, config.probe_examples_per_partition, seed + 302),
            _sample_indices(
                curriculum.late_current, config.probe_examples_per_partition, seed + 303
            ),
            _sample_indices(curriculum.heldout, config.probe_examples_per_partition, seed + 304),
        )
    )
    early_paths = path_fingerprints(
        responsibility_matrix(early_model, curriculum, early_probe, device=device),
        config.path_cells,
    )
    late_paths = path_fingerprints(
        responsibility_matrix(late_model, curriculum, late_probe, device=device),
        config.path_cells,
    )
    return {
        "key_frequency_pairs": list(key_pairs),
        "early_fourier_concentration": fourier_concentration(
            early_model.embedding.weight, key_pairs
        ),
        "late_fourier_concentration": fourier_concentration(
            late_model.embedding.weight, key_pairs
        ),
        "fourier_concentration_gain": (
            fourier_concentration(late_model.embedding.weight, key_pairs)
            - fourier_concentration(early_model.embedding.weight, key_pairs)
        ),
        "early": early_fourier,
        "late": late_fourier,
        "early_path_reuse": mean_pairwise_jaccard(early_paths),
        "late_path_reuse": mean_pairwise_jaccard(late_paths),
        "path_reuse_gain": mean_pairwise_jaccard(late_paths) - mean_pairwise_jaccard(early_paths),
    }


def _run_gate(
    early: dict[str, float],
    late: dict[str, dict[str, float]],
    mechanistic: dict[str, Any],
    config: KnowledgeSubsumptionConfig,
) -> dict[str, bool]:
    early_fourier = mechanistic["early"]
    late_fourier = mechanistic["late"]
    gates = {
        "early_memorization": (
            early["seen_accuracy"] >= config.early_minimum_seen_accuracy
            and early["unseen_accuracy"] <= config.early_maximum_unseen_accuracy
        ),
        "late_generalization": (
            late["old"]["accuracy"] >= config.late_minimum_old_accuracy
            and late["current"]["accuracy"] >= config.late_minimum_current_accuracy
            and late["heldout"]["accuracy"] >= config.late_minimum_heldout_accuracy
        ),
        "generalizing_circuit": (
            late_fourier["old"]["restricted"]["accuracy"]
            >= config.restricted_minimum_old_accuracy
            and late_fourier["heldout"]["restricted"]["accuracy"]
            >= config.restricted_minimum_heldout_accuracy
        ),
        "memorization_cleanup": (
            early_fourier["seen"]["excluded"]["accuracy"]
            >= config.early_excluded_minimum_seen_accuracy
            and late_fourier["old"]["excluded"]["accuracy"]
            <= config.late_excluded_maximum_old_accuracy
            and late_fourier["heldout"]["excluded"]["accuracy"]
            <= config.late_excluded_maximum_heldout_accuracy
        ),
    }
    gates["pass"] = all(gates.values())
    return gates


def _clone_model(model: CellularModularNet, device: torch.device) -> CellularModularNet:
    return copy.deepcopy(model).to(device)


def train_sequential_run(
    config: KnowledgeSubsumptionConfig,
    *,
    seed: int,
    task: TaskName,
    device: torch.device,
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train only on the current phase; earlier examples are never replayed."""

    torch.manual_seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    curriculum = make_curriculum(config, seed=seed, task=task)
    model = CellularModularNet(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(seed + 401)
    phase_history: list[dict[str, Any]] = []
    early_model: CellularModularNet | None = None
    early_metrics: dict[str, float] | None = None
    phase_one_model: CellularModularNet | None = None
    phase_one_metrics: dict[str, float] | None = None
    global_step = 0

    for phase_index, (phase_indices, maximum_steps) in enumerate(
        zip(curriculum.phases, config.phase_steps)
    ):
        phase_pairs, phase_labels = _partition_tensors(curriculum, phase_indices)
        phase_size = len(phase_indices)
        for phase_step in range(1, maximum_steps + 1):
            model.train()
            if phase_size <= config.batch_size:
                selection = torch.arange(phase_size)
            else:
                selection = torch.randint(
                    phase_size,
                    (config.batch_size,),
                    generator=generator,
                )
            pairs = phase_pairs[selection].to(device)
            labels = phase_labels[selection].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(pairs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            global_step += 1

            should_eval = phase_step % config.eval_interval_steps == 0 or phase_step == maximum_steps
            if not should_eval:
                continue
            current = evaluate_indices(model, curriculum, phase_indices, device=device)
            if phase_index == 0:
                future = evaluate_indices(
                    model, curriculum, curriculum.future_after_first, device=device
                )
                phase_one_model = _clone_model(model, device)
                phase_one_metrics = {
                    "seen_accuracy": current["accuracy"],
                    "seen_nll": current["nll"],
                    "unseen_accuracy": future["accuracy"],
                    "unseen_nll": future["nll"],
                    "captured_global_step": float(global_step),
                }
                if (
                    current["accuracy"] >= config.early_minimum_seen_accuracy
                    and future["accuracy"] <= config.early_maximum_unseen_accuracy
                ):
                    early_model = phase_one_model
                    early_metrics = dict(phase_one_metrics)
                    break
        phase_history.append(
            {
                "phase": phase_index + 1,
                "global_step": global_step,
                "current": evaluate_indices(model, curriculum, phase_indices, device=device),
            }
        )

    if early_model is None:
        if phase_one_model is None or phase_one_metrics is None:
            raise RuntimeError("phase-one checkpoint was not captured")
        early_model = phase_one_model
        early_metrics = phase_one_metrics
    assert early_metrics is not None

    late = {
        "old": evaluate_indices(model, curriculum, curriculum.late_old, device=device),
        "current": evaluate_indices(model, curriculum, curriculum.late_current, device=device),
        "heldout": evaluate_indices(model, curriculum, curriculum.heldout, device=device),
    }
    mechanistic = mechanistic_diagnostics(
        early_model,
        model,
        curriculum,
        config,
        seed=seed,
        device=device,
    )
    gates = _run_gate(early_metrics, late, mechanistic, config)

    checkpoint_paths: dict[str, str] = {}
    if save_dir is not None:
        destination = Path(save_dir)
        destination.mkdir(parents=True, exist_ok=True)
        early_path = destination / f"{task}-seed{seed}-early.pt"
        late_path = destination / f"{task}-seed{seed}-late.pt"
        torch.save(early_model.state_dict(), early_path)
        torch.save(model.state_dict(), late_path)
        checkpoint_paths = {"early": str(early_path), "late": str(late_path)}

    return {
        "task": task,
        "seed": seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "total_steps": global_step,
        "phase_history": phase_history,
        "early": early_metrics,
        "late": late,
        "mechanistic": mechanistic,
        "gates": gates,
        "checkpoints": checkpoint_paths,
    }


def train_oracle_reference(
    config: KnowledgeSubsumptionConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Descriptive reference trained on the cumulative union of formal training pairs."""

    torch.manual_seed(seed + 10000)
    curriculum = make_curriculum(config, seed=seed, task="modular_addition")
    model = CellularModularNet(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    seen = curriculum.late_seen
    pairs, labels = _partition_tensors(curriculum, seen)
    generator = torch.Generator().manual_seed(seed + 10001)
    for _ in range(sum(config.phase_steps)):
        model.train()
        if len(seen) <= config.batch_size:
            selection = torch.arange(len(seen))
        else:
            selection = torch.randint(len(seen), (config.batch_size,), generator=generator)
        batch_pairs = pairs[selection].to(device)
        batch_labels = labels[selection].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch_pairs), batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()

    key_pairs = select_key_frequency_pairs(model.embedding.weight, config.key_frequency_pairs)
    restricted = fourier_filter_embedding(model.embedding.weight, key_pairs, keep_keys=True)
    excluded = fourier_filter_embedding(model.embedding.weight, key_pairs, keep_keys=False)
    return {
        "task": "oracle_modular_addition",
        "seed": seed,
        "training_mode": "cumulative_replay_reference",
        "key_frequency_pairs": list(key_pairs),
        "fourier_concentration": fourier_concentration(model.embedding.weight, key_pairs),
        "seen": {
            "full": evaluate_indices(model, curriculum, seen, device=device),
            "restricted": evaluate_indices(
                model, curriculum, seen, device=device, embedding_override=restricted
            ),
            "excluded": evaluate_indices(
                model, curriculum, seen, device=device, embedding_override=excluded
            ),
        },
        "heldout": {
            "full": evaluate_indices(model, curriculum, curriculum.heldout, device=device),
            "restricted": evaluate_indices(
                model,
                curriculum,
                curriculum.heldout,
                device=device,
                embedding_override=restricted,
            ),
            "excluded": evaluate_indices(
                model,
                curriculum,
                curriculum.heldout,
                device=device,
                embedding_override=excluded,
            ),
        },
    }


def summarize_experiment(
    runs: list[dict[str, Any]],
    *,
    positive_status: str = "KNOWLEDGE_SUBSUMPTION_SUPPORTED",
    negative_status: str = "KNOWLEDGE_SUBSUMPTION_NOT_SUPPORTED",
) -> dict[str, Any]:
    primary = [run for run in runs if run.get("task") == "modular_addition"]
    control = [run for run in runs if run.get("task") == "balanced_random_labels"]
    primary_passes = sum(bool(run["gates"]["pass"]) for run in primary)
    control_false_positives = sum(bool(run["gates"]["pass"]) for run in control)
    all_primary = bool(primary) and primary_passes == len(primary)
    zero_control = bool(control) and control_false_positives == 0
    supported = all_primary and zero_control
    return {
        "status": positive_status if supported else negative_status,
        "supported": supported,
        "primary_runs": len(primary),
        "primary_passes": primary_passes,
        "control_runs": len(control),
        "control_false_positives": control_false_positives,
        "requirements": {
            "all_primary_seeds_pass": all_primary,
            "zero_control_false_positives": zero_control,
        },
    }


def chance_accuracy(modulus: int) -> float:
    return 1.0 / modulus


def effective_heldout_fraction(config: KnowledgeSubsumptionConfig) -> float:
    return 1.0 - math.fsum(config.curriculum_fractions)
