from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .language_conflict_differentiation import FORK_EPSILON, ForkableTextNCA


FAMILIES = ("STORY", "ARITHMETIC", "TRANSFORM")
STREAM_KEYS = ("STORY", "ARITH_A", "ARITH_B", "TRANSFORM")
MAX_TRAITS = 4
STRUCTURAL_PENALTY = 0.20
MIN_CLUSTER_FRACTION = 0.12
MIN_SILHOUETTE_Q10 = 0.15
INVALID_MODEL_OBJECTIVE = 1_000_000.0
SENSOR_BUFFER = 96
SENSOR_INTERVAL = 32
PERSISTENCE_EVALS = 3
MODE_STABILITY_MIN = 0.65
IDENTITY_NORMALIZED_MARGIN_MIN = 0.01
ROUTING_PURITY_MIN = 0.75
KMEANS_STEPS = 32


@dataclass(frozen=True)
class ModeFit:
    k: int
    centroids: torch.Tensor
    assignment: torch.Tensor
    residual: float
    normalized_residual: float
    min_cluster_fraction: float
    silhouette_q10: float
    objective: float


@dataclass(frozen=True)
class ModelSelection:
    selected_k: int
    fits: tuple[ModeFit, ...]

    def fit(self, k: int) -> ModeFit:
        for item in self.fits:
            if item.k == k:
                return item
        raise KeyError(k)


@dataclass
class GrowthEvidence:
    candidate_k: int = 0
    stable_evaluations: int = 0
    previous_centroids: torch.Tensor | None = None
    last_stability: float = 0.0

    def reset(self) -> None:
        self.candidate_k = 0
        self.stable_evaluations = 0
        self.previous_centroids = None
        self.last_stability = 0.0


@dataclass(frozen=True)
class MultiIdentitySummary:
    assignment: tuple[int, ...]
    normalized_margins: tuple[float, ...]
    normalized_identity_margin: float
    passes: bool


class OnlineTraitTextNCA(ForkableTextNCA):
    """Experiment-023 TextNCA with a bounded pool of latent phenotype slots.

    Only the first active traits are physically active. The shared TextNCA
    genome and the frozen parent phenotype act as the organism and its fixed
    shadow developmental sensor respectively. Replacing the shadow gradient
    oracle is deliberately deferred to the next experiment.
    """

    def __init__(self, vocab_size: int, *, max_traits: int = MAX_TRAITS) -> None:
        super().__init__(vocab_size)
        self.max_traits = int(max_traits)
        self.online_traits = nn.Parameter(torch.zeros(self.max_traits, self.dim))
        with torch.no_grad():
            self.online_traits.copy_(self.parent_trait.detach()[None, :].expand_as(self.online_traits))

    @torch.no_grad()
    def initialize_online_population(self) -> None:
        self.online_traits.copy_(self.parent_trait.detach()[None, :].expand_as(self.online_traits))

    def freeze_for_online_development(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.parent_trait.requires_grad_(False)
        self.child_traits.requires_grad_(False)
        self.online_traits.requires_grad_(True)

    def forward_trait(self, input_ids: torch.Tensor, branch: int) -> torch.Tensor:
        if not 0 <= branch < self.max_traits:
            raise ValueError("branch outside latent phenotype pool")
        return self.logits_with_trait(input_ids, self.online_traits[branch])

    @torch.no_grad()
    def spawn_first_bifurcation(self, ordered_centroids: torch.Tensor) -> None:
        if ordered_centroids.shape[0] != 2:
            raise ValueError("first bifurcation requires two centroids")
        axis = F.normalize(ordered_centroids[0] - ordered_centroids[1], dim=0, eps=1e-8)
        parent = self.online_traits[0].clone()
        self.online_traits[0].copy_(parent + FORK_EPSILON * axis)
        self.online_traits[1].copy_(parent - FORK_EPSILON * axis)

    @torch.no_grad()
    def spawn_additional_trait(
        self,
        *,
        new_branch: int,
        parent_branch: int,
        parent_centroid: torch.Tensor,
        new_centroid: torch.Tensor,
    ) -> None:
        if not 1 <= new_branch < self.max_traits:
            raise ValueError("new branch outside latent phenotype pool")
        direction = F.normalize(new_centroid - parent_centroid, dim=0, eps=1e-8)
        self.online_traits[new_branch].copy_(
            self.online_traits[parent_branch] + FORK_EPSILON * direction
        )


def _farthest_first(unit: torch.Tensor, k: int) -> list[int]:
    mean = unit.mean(dim=0)
    first = int((unit - mean).square().sum(dim=1).argmax().item())
    selected = [first]
    while len(selected) < k:
        distance = torch.stack(
            [(unit - unit[index]).square().sum(dim=1) for index in selected], dim=1
        ).min(dim=1).values
        for index in selected:
            distance[index] = -1.0
        selected.append(int(distance.argmax().item()))
    return selected


def fit_k_modes(
    gradients: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    if gradients.ndim != 2 or len(gradients) < k:
        raise ValueError("gradients must be [samples, phenotype_dim] with samples >= k")
    if not torch.isfinite(gradients).all():
        raise ValueError("mode gradients must be finite")
    unit = F.normalize(gradients.float(), dim=1, eps=1e-8)
    if k == 1:
        centroids = unit.mean(dim=0, keepdim=True)
        assignment = torch.zeros(len(unit), dtype=torch.long, device=unit.device)
    else:
        centroids = unit[_farthest_first(unit, k)].clone()
        assignment = torch.zeros(len(unit), dtype=torch.long, device=unit.device)
        previous = None
        for _ in range(KMEANS_STEPS):
            distance = (unit[:, None, :] - centroids[None, :, :]).square().sum(dim=-1)
            updated = distance.argmin(dim=1)
            if any(int((updated == cluster).sum()) == 0 for cluster in range(k)):
                break
            new_centroids = torch.stack(
                [unit[updated == cluster].mean(dim=0) for cluster in range(k)], dim=0
            )
            assignment = updated
            centroids = new_centroids
            if previous is not None and torch.equal(updated, previous):
                break
            previous = updated
    residual = float((unit - centroids[assignment]).square().sum())
    fractions = [float((assignment == cluster).float().mean()) for cluster in range(k)]
    return centroids.detach(), assignment.detach(), residual, min(fractions)


def lower_tail_silhouette(
    gradients: torch.Tensor,
    assignment: torch.Tensor,
    *,
    quantile: float = 0.10,
) -> float:
    """Return a lower-tail silhouette score for a proposed discrete mode split.

    K-means will reduce SSE even when it merely quantizes one smooth unimodal
    cloud.  Such artificial splits contain many boundary samples with poor
    cluster membership.  A genuine mode split should keep even the lower tail
    of samples more cohesive with its own cluster than with the nearest other
    cluster.  The sensor buffer is only 96 rows, so the O(N^2) distance matrix
    is negligible relative to the TextNCA gradient probes.
    """
    if gradients.ndim != 2 or len(gradients) != len(assignment):
        raise ValueError("gradients and assignments must align")
    clusters = sorted(set(int(value) for value in assignment.detach().cpu().tolist()))
    if len(clusters) <= 1:
        return 1.0
    unit = F.normalize(gradients.float(), dim=1, eps=1e-8)
    pairwise = torch.cdist(unit, unit, p=2)
    values = []
    for index in range(len(unit)):
        own = int(assignment[index].item())
        own_mask = assignment == own
        own_mask = own_mask.clone()
        own_mask[index] = False
        if int(own_mask.sum()) == 0:
            values.append(torch.tensor(0.0, device=unit.device))
            continue
        cohesion = pairwise[index][own_mask].mean()
        alternatives = []
        for cluster in clusters:
            if cluster == own:
                continue
            mask = assignment == cluster
            if int(mask.sum()) > 0:
                alternatives.append(pairwise[index][mask].mean())
        if not alternatives:
            values.append(torch.tensor(0.0, device=unit.device))
            continue
        separation = torch.stack(alternatives).min()
        denominator = torch.maximum(cohesion, separation).clamp_min(1e-8)
        values.append((separation - cohesion) / denominator)
    scores = torch.stack(values)
    return float(torch.quantile(scores, quantile))


def select_model_order(
    gradients: torch.Tensor,
    *,
    max_k: int = MAX_TRAITS,
    structural_penalty: float = STRUCTURAL_PENALTY,
) -> ModelSelection:
    _, _, residual_k1, _ = fit_k_modes(gradients, 1)
    fits: list[ModeFit] = []
    for k in range(1, min(max_k, len(gradients)) + 1):
        centroids, assignment, residual, min_fraction = fit_k_modes(gradients, k)
        if residual_k1 <= 1e-12:
            normalized = 1.0
        else:
            normalized = residual / residual_k1
        silhouette_q10 = lower_tail_silhouette(gradients, assignment) if k > 1 else 1.0
        objective = normalized + structural_penalty * (k - 1)
        if k > 1 and (
            min_fraction < MIN_CLUSTER_FRACTION
            or silhouette_q10 < MIN_SILHOUETTE_Q10
        ):
            objective = INVALID_MODEL_OBJECTIVE
        fits.append(
            ModeFit(
                k=k,
                centroids=centroids,
                assignment=assignment,
                residual=residual,
                normalized_residual=float(normalized),
                min_cluster_fraction=min_fraction,
                silhouette_q10=float(silhouette_q10),
                objective=float(objective),
            )
        )
    selected = min(fits, key=lambda item: (item.objective, item.k))
    return ModelSelection(selected_k=selected.k, fits=tuple(fits))


def _best_permutation(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[int, ...]:
    if reference.shape != candidate.shape:
        raise ValueError("centroid sets must have equal shapes")
    k = reference.shape[0]
    best = None
    best_cost = float("inf")
    for permutation in itertools.permutations(range(k)):
        ordered = candidate[list(permutation)]
        cost = float((reference - ordered).square().sum())
        if cost < best_cost:
            best_cost = cost
            best = permutation
    assert best is not None
    return tuple(int(value) for value in best)


def mode_set_stability(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    permutation = _best_permutation(reference, candidate)
    ordered = candidate[list(permutation)]
    left = F.normalize(reference.float(), dim=1, eps=1e-8)
    right = F.normalize(ordered.float(), dim=1, eps=1e-8)
    return float((left * right).sum(dim=1).mean())


def align_same_k(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    permutation = _best_permutation(reference, candidate)
    return candidate[list(permutation)].detach()


def align_growth_centroids(
    old_centroids: torch.Tensor,
    new_centroids: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    """Align K+1 centroids to existing branch order and identify the newborn."""
    old_k = old_centroids.shape[0]
    if new_centroids.shape[0] != old_k + 1:
        raise ValueError("growth alignment requires exactly one new centroid")
    best_indices = None
    best_cost = float("inf")
    for indices in itertools.permutations(range(old_k + 1), old_k):
        ordered_existing = new_centroids[list(indices)]
        cost = float((old_centroids - ordered_existing).square().sum())
        if cost < best_cost:
            best_cost = cost
            best_indices = tuple(int(value) for value in indices)
    assert best_indices is not None
    unmatched = next(index for index in range(old_k + 1) if index not in best_indices)
    ordered = torch.cat(
        [new_centroids[list(best_indices)], new_centroids[unmatched : unmatched + 1]], dim=0
    ).detach()
    newborn = old_k
    distances = (old_centroids - ordered[newborn][None, :]).square().sum(dim=1)
    parent = int(distances.argmin().item())
    return ordered, newborn, parent


def update_growth_evidence(
    evidence: GrowthEvidence,
    *,
    active_k: int,
    selection: ModelSelection,
) -> tuple[GrowthEvidence, bool]:
    if selection.selected_k <= active_k or active_k >= MAX_TRAITS:
        evidence.reset()
        return evidence, False
    candidate_k = active_k + 1
    candidate_centroids = selection.fit(candidate_k).centroids
    if evidence.candidate_k != candidate_k or evidence.previous_centroids is None:
        evidence.candidate_k = candidate_k
        evidence.stable_evaluations = 1
        evidence.previous_centroids = candidate_centroids.detach().cpu()
        evidence.last_stability = 1.0
        return evidence, False
    stability = mode_set_stability(evidence.previous_centroids, candidate_centroids.cpu())
    evidence.last_stability = stability
    if stability >= MODE_STABILITY_MIN:
        evidence.stable_evaluations += 1
    else:
        evidence.stable_evaluations = 1
    evidence.previous_centroids = candidate_centroids.detach().cpu()
    return evidence, evidence.stable_evaluations >= PERSISTENCE_EVALS


def route_to_centroid(gradient: torch.Tensor, centroids: torch.Tensor) -> tuple[int, float]:
    unit = F.normalize(gradient.detach().float(), dim=0, eps=1e-8)
    target = centroids.to(unit.device)
    distances = (target - unit[None, :]).square().sum(dim=1)
    branch = int(distances.argmin().item())
    sorted_distance = torch.sort(distances).values
    margin = (
        float(sorted_distance[1] - sorted_distance[0]) if len(sorted_distance) > 1 else 0.0
    )
    return branch, margin


def cluster_purity(assignment: torch.Tensor, labels: list[str]) -> float:
    if len(assignment) != len(labels) or not labels:
        raise ValueError("assignments and labels must align")
    total = 0
    labels_array = np.asarray(labels)
    assignment_array = assignment.detach().cpu().numpy()
    for cluster in sorted(set(int(value) for value in assignment.tolist())):
        mask = assignment_array == cluster
        values, counts = np.unique(labels_array[mask], return_counts=True)
        if len(values):
            total += int(counts.max())
    return float(total / len(labels))


def summarize_multi_identity(
    losses: dict[str, tuple[float, ...]],
    baselines: dict[str, float],
    domains: tuple[str, ...],
) -> MultiIdentitySummary:
    if not domains:
        raise ValueError("at least one domain is required")
    branch_count = len(next(iter(losses.values())))
    if branch_count < len(domains):
        return MultiIdentitySummary((), (), 0.0, False)
    best_assignment = None
    best_total = float("inf")
    for assignment in itertools.permutations(range(branch_count), len(domains)):
        total = sum(losses[domain][branch] for domain, branch in zip(domains, assignment))
        if total < best_total:
            best_total = total
            best_assignment = assignment
    assert best_assignment is not None
    margins = []
    for domain, branch in zip(domains, best_assignment):
        matched = losses[domain][branch]
        alternatives = [value for index, value in enumerate(losses[domain]) if index != branch]
        margin = min(alternatives) - matched if alternatives else 0.0
        margins.append(float(margin / max(abs(baselines[domain]), 1e-8)))
    passes = bool(all(value >= IDENTITY_NORMALIZED_MARGIN_MIN for value in margins))
    return MultiIdentitySummary(
        assignment=tuple(int(value) for value in best_assignment),
        normalized_margins=tuple(margins),
        normalized_identity_margin=float(np.mean(margins)),
        passes=passes,
    )


def _shuffle_counts(counts: dict[str, int], seed: int) -> list[str]:
    values = [name for name, count in counts.items() for _ in range(count)]
    rng = random.Random(seed)
    rng.shuffle(values)
    return values


def developmental_curriculum(replicate: int) -> list[dict[str, object]]:
    """Return an exact-count online stream with no task-boundary signal to the model."""
    specs = (
        ("A_STORY_ONLY", "story-only", {"STORY": 192}),
        ("B_EMERGING_MATH", "math-10", {"STORY": 115, "ARITH_A": 13}),
        ("B_EMERGING_MATH", "math-30", {"STORY": 90, "ARITH_A": 38}),
        ("B_EMERGING_MATH", "math-50", {"STORY": 64, "ARITH_A": 64}),
        (
            "C_DUPLICATE_CONTROL",
            "duplicate-arithmetic",
            {"STORY": 64, "ARITH_A": 64, "ARITH_B": 64},
        ),
        (
            "D_THIRD_MODE",
            "three-way",
            {"STORY": 128, "ARITH_A": 128, "TRANSFORM": 128},
        ),
    )
    rows: list[dict[str, object]] = []
    global_step = 0
    for index, (stage, subphase, counts) in enumerate(specs):
        schedule = _shuffle_counts(counts, seed=323_000 + 10_000 * replicate + index)
        for stream_key in schedule:
            global_step += 1
            family = "ARITHMETIC" if stream_key.startswith("ARITH") else stream_key
            rows.append(
                {
                    "step": global_step,
                    "stage": stage,
                    "subphase": subphase,
                    "stream_key": stream_key,
                    "family": family,
                }
            )
    return rows


def stage_end_steps() -> dict[str, int]:
    rows = developmental_curriculum(0)
    result: dict[str, int] = {}
    for row in rows:
        result[str(row["stage"])] = int(row["step"])
    return result


def _transform_text(rng: random.Random) -> str:
    values = [rng.randrange(0, 10) for _ in range(6)]
    source = " ".join(str(value) for value in values)
    target = " ".join(str(value) for value in reversed(values))
    return f"Transform sequence {source}. Reverse answer {target}."


def _transform_stream(
    tokenizer: object, *, target_tokens: int, seed: int
) -> tuple[torch.Tensor, int]:
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise RuntimeError("tokenizer does not contain EOS")
    rng = random.Random(seed)
    values: list[int] = []
    examples = 0
    while len(values) < target_tokens:
        encoded = tokenizer.encode(_transform_text(rng)).ids
        if encoded:
            values.extend(encoded)
            values.append(int(eos_id))
            examples += 1
    return torch.tensor(values[:target_tokens], dtype=torch.long), examples


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def prepare_transform_cache(cache_dir: Path, tokenizer: object) -> dict[str, object]:
    root = cache_dir / "online-trait-genesis-transform"
    root.mkdir(parents=True, exist_ok=True)
    train_path = root / "train-tokens.pt"
    validation_path = root / "validation-tokens.pt"
    manifest_path = root / "manifest.json"
    expected = {"train_tokens": 300_000, "validation_tokens": 60_000, "seed": 23023}
    if train_path.is_file() and validation_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train = torch.load(train_path, map_location="cpu")
        validation = torch.load(validation_path, map_location="cpu")
        if (
            all(manifest.get(key) == value for key, value in expected.items())
            and _tensor_sha256(train) == manifest.get("train_sha256")
            and _tensor_sha256(validation) == manifest.get("validation_sha256")
        ):
            return {
                "train": train,
                "validation": validation,
                "manifest": manifest,
                "path": manifest_path,
            }
    train, train_examples = _transform_stream(
        tokenizer, target_tokens=expected["train_tokens"], seed=expected["seed"]
    )
    validation, validation_examples = _transform_stream(
        tokenizer, target_tokens=expected["validation_tokens"], seed=expected["seed"] + 1
    )
    torch.save(train, train_path)
    torch.save(validation, validation_path)
    manifest = {
        "format": "minicells.online-trait-genesis-transform-corpus.v1",
        **expected,
        "train_examples": train_examples,
        "validation_examples": validation_examples,
        "train_sha256": _tensor_sha256(train),
        "validation_sha256": _tensor_sha256(validation),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "train": train,
        "validation": validation,
        "manifest": manifest,
        "path": manifest_path,
    }
