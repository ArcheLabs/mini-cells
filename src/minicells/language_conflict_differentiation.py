from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .language_models import TextNCALM


DOMAINS = ("STORY", "ARITHMETIC")
ARMS = ("unified", "capacity-fork", "differentiation-fork")
SEQUENCE_LENGTH = 64
BATCH_SIZE = 8
PRETRAIN_STEPS = 300
POSTFORK_STEPS = 400
PRETRAIN_LR = 3e-4
PHENOTYPE_LR = 5e-3
FORK_EPSILON = 2e-2
CALIBRATION_WINDOWS = 3
CALIBRATION_BATCHES_PER_DOMAIN = 8
INTERFERENCE_STEP = 2e-2
CONFLICT_CANCELLATION_MIN = 0.15
CONFLICT_PC1_RATIO_MIN = 0.20
CONFLICT_SPLIT_BALANCE_MIN = 0.25
INTERFERENCE_MIN = 1e-3
IDENTITY_NORMALIZED_MARGIN_MIN = 1e-2
ROUTING_PURITY_MIN = 0.75
DIFFERENTIATION_REPLICATES_MIN = 2


@dataclass(frozen=True)
class ConflictGeometry:
    axis: torch.Tensor
    mean_unit_gradient: torch.Tensor
    directional_cancellation: float
    pc1_variance_ratio: float
    split_balance: float


@dataclass(frozen=True)
class IdentitySummary:
    assignment: tuple[int, int]
    story_margin: float
    arithmetic_margin: float
    normalized_story_margin: float
    normalized_arithmetic_margin: float
    normalized_identity_margin: float
    opposite_preference: bool

    @property
    def passes(self) -> bool:
        return bool(
            self.opposite_preference
            and self.normalized_story_margin >= IDENTITY_NORMALIZED_MARGIN_MIN
            and self.normalized_arithmetic_margin >= IDENTITY_NORMALIZED_MARGIN_MIN
        )


class ForkableTextNCA(nn.Module):
    """A stable TextNCA genome with a forkable population-level phenotype.

    The first two NCA stages are a shared cellular stem.  A phenotype vector is
    broadcast into the final 1-D token-cell population before the third shared
    NCA stage.  Forking creates two phenotypic copies while keeping every genome
    parameter shared.  Experiment 021 therefore tests differential experience,
    not duplicated expert weights.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.base = TextNCALM(
            vocab_size=vocab_size,
            max_context=128,
            dim=128,
            heads=4,
            ffn_dim=512,
            windows=(8, 32, 128),
            iterations=(4, 4, 4),
            rms_norm=False,
            carry_bias=2.0,
            tie_embeddings=True,
            stage_supervision=False,
        )
        dim = self.base.token_embedding.embedding_dim
        self.parent_trait = nn.Parameter(torch.zeros(dim))
        self.child_traits = nn.Parameter(torch.zeros(2, dim))
        nn.init.normal_(self.parent_trait, mean=0.0, std=0.02)
        with torch.no_grad():
            self.child_traits.copy_(self.parent_trait[None, :].expand_as(self.child_traits))

    @property
    def dim(self) -> int:
        return int(self.parent_trait.numel())

    def shared_state(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.base.max_context:
            raise ValueError(f"input_ids must be [batch, <= {self.base.max_context}]")
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)
        state = self.base.token_embedding(input_ids) + self.base.position_embedding(positions)[None, :, :]
        for stage in self.base.stages[:2]:
            state = stage(state)
        return state

    def logits_from_shared(self, shared: torch.Tensor, trait: torch.Tensor) -> torch.Tensor:
        if trait.ndim != 1 or trait.shape[0] != self.dim:
            raise ValueError("trait must be one phenotype vector")
        state = self.base.stages[2](shared + trait.view(1, 1, -1))
        return self.base.lm_head(self.base.final_norm(state))

    def logits_with_trait(self, input_ids: torch.Tensor, trait: torch.Tensor) -> torch.Tensor:
        return self.logits_from_shared(self.shared_state(input_ids), trait)

    def forward_parent(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.logits_with_trait(input_ids, self.parent_trait)

    def forward_child(self, input_ids: torch.Tensor, branch: int) -> torch.Tensor:
        if branch not in (0, 1):
            raise ValueError("branch must be 0 or 1")
        return self.logits_with_trait(input_ids, self.child_traits[branch])

    @torch.no_grad()
    def initialize_children(self, axis: torch.Tensor, *, symmetry_break: bool) -> None:
        direction = axis.to(device=self.parent_trait.device, dtype=self.parent_trait.dtype)
        direction = direction / direction.norm().clamp_min(1e-8)
        self.child_traits[0].copy_(self.parent_trait)
        self.child_traits[1].copy_(self.parent_trait)
        if symmetry_break:
            self.child_traits[0].add_(direction, alpha=FORK_EPSILON)
            self.child_traits[1].add_(direction, alpha=-FORK_EPSILON)

    def freeze_genome_for_fork(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.parent_trait.requires_grad_(False)
        self.child_traits.requires_grad_(True)

    def pretrain_parameters(self) -> list[nn.Parameter]:
        return [*self.base.parameters(), self.parent_trait]



def language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def trait_gradient(
    model: ForkableTextNCA,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    trait: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    probe = (model.parent_trait.detach() if trait is None else trait.detach()).clone().requires_grad_(True)
    logits = model.logits_with_trait(inputs, probe)
    loss = language_model_loss(logits.float(), targets)
    gradient = torch.autograd.grad(loss, probe, retain_graph=False, create_graph=False)[0]
    return gradient.detach(), float(loss.detach())


def learn_conflict_geometry(gradients: torch.Tensor) -> ConflictGeometry:
    if gradients.ndim != 2 or gradients.shape[0] < 4:
        raise ValueError("gradients must be [microbatches, phenotype_dim]")
    if not torch.isfinite(gradients).all():
        raise ValueError("conflict gradients must be finite")
    unit = F.normalize(gradients.float(), dim=1, eps=1e-8)
    mean = unit.mean(dim=0)
    centered = unit - mean
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projections = centered @ axis
    positive = float((projections > 0).float().mean())
    negative = float((projections < 0).float().mean())
    variance = singular.square()
    pc1_ratio = float(variance[0] / variance.sum().clamp_min(1e-12))
    cancellation = float(1.0 - mean.square().sum().clamp(max=1.0))
    return ConflictGeometry(
        axis=axis.detach(),
        mean_unit_gradient=mean.detach(),
        directional_cancellation=cancellation,
        pc1_variance_ratio=pc1_ratio,
        split_balance=min(positive, negative),
    )


def conflict_gate(geometry: ConflictGeometry) -> bool:
    return bool(
        geometry.directional_cancellation >= CONFLICT_CANCELLATION_MIN
        and geometry.pc1_variance_ratio >= CONFLICT_PC1_RATIO_MIN
        and geometry.split_balance >= CONFLICT_SPLIT_BALANCE_MIN
    )


def route_gradient(gradient: torch.Tensor, geometry: ConflictGeometry) -> tuple[int, float]:
    unit = F.normalize(gradient.detach().float(), dim=0, eps=1e-8)
    score = float(torch.dot(unit - geometry.mean_unit_gradient.to(unit.device), geometry.axis.to(unit.device)))
    return (0 if score >= 0.0 else 1), score


def mean_gradient_cosine(gradients: torch.Tensor, labels: list[str]) -> float:
    if len(labels) != len(gradients):
        raise ValueError("labels and gradients must align")
    means = []
    for domain in DOMAINS:
        positions = [index for index, label in enumerate(labels) if label == domain]
        if not positions:
            raise ValueError(f"missing calibration domain: {domain}")
        means.append(F.normalize(gradients[positions].mean(dim=0).float(), dim=0, eps=1e-8))
    return float(torch.dot(means[0], means[1]))


def routing_purity(scores: list[float], labels: list[str]) -> float:
    if len(scores) != len(labels) or not scores:
        raise ValueError("routing scores and labels must align")
    predicted = np.asarray([0 if score >= 0.0 else 1 for score in scores], dtype=np.int64)
    truth = np.asarray([0 if label == DOMAINS[0] else 1 for label in labels], dtype=np.int64)
    direct = float((predicted == truth).mean())
    swapped = float(((1 - predicted) == truth).mean())
    return max(direct, swapped)


def summarize_identity(losses: dict[str, tuple[float, float]], parent_losses: dict[str, float]) -> IdentitySummary:
    story = losses[DOMAINS[0]]
    arithmetic = losses[DOMAINS[1]]
    direct_total = story[0] + arithmetic[1]
    swapped_total = story[1] + arithmetic[0]
    assignment = (0, 1) if direct_total <= swapped_total else (1, 0)
    story_branch, arithmetic_branch = assignment
    story_other = 1 - story_branch
    arithmetic_other = 1 - arithmetic_branch
    story_margin = float(story[story_other] - story[story_branch])
    arithmetic_margin = float(arithmetic[arithmetic_other] - arithmetic[arithmetic_branch])
    story_norm = story_margin / max(abs(parent_losses[DOMAINS[0]]), 1e-8)
    arithmetic_norm = arithmetic_margin / max(abs(parent_losses[DOMAINS[1]]), 1e-8)
    return IdentitySummary(
        assignment=assignment,
        story_margin=story_margin,
        arithmetic_margin=arithmetic_margin,
        normalized_story_margin=story_norm,
        normalized_arithmetic_margin=arithmetic_norm,
        normalized_identity_margin=0.5 * (story_norm + arithmetic_norm),
        opposite_preference=bool(story_margin > 0.0 and arithmetic_margin > 0.0),
    )


def counterfactual_interference(
    model: ForkableTextNCA,
    story_batch: tuple[torch.Tensor, torch.Tensor],
    arithmetic_batch: tuple[torch.Tensor, torch.Tensor],
    story_gradient: torch.Tensor,
    arithmetic_gradient: torch.Tensor,
) -> tuple[float, float]:
    parent = model.parent_trait.detach()
    story_inputs, story_targets = story_batch
    math_inputs, math_targets = arithmetic_batch
    with torch.no_grad():
        story_base = float(language_model_loss(model.logits_with_trait(story_inputs, parent).float(), story_targets))
        math_base = float(language_model_loss(model.logits_with_trait(math_inputs, parent).float(), math_targets))
        story_step = parent - INTERFERENCE_STEP * F.normalize(story_gradient, dim=0, eps=1e-8)
        math_step = parent - INTERFERENCE_STEP * F.normalize(arithmetic_gradient, dim=0, eps=1e-8)
        math_after_story = float(language_model_loss(model.logits_with_trait(math_inputs, story_step).float(), math_targets))
        story_after_math = float(language_model_loss(model.logits_with_trait(story_inputs, math_step).float(), story_targets))
    return math_after_story - math_base, story_after_math - story_base


def conflict_window_pass(geometry: ConflictGeometry, interference_story_to_math: float, interference_math_to_story: float) -> bool:
    return bool(
        conflict_gate(geometry)
        and interference_story_to_math >= INTERFERENCE_MIN
        and interference_math_to_story >= INTERFERENCE_MIN
    )


def _arithmetic_text(rng: random.Random) -> str:
    kind = rng.randrange(4)
    if kind == 0:
        a, b = rng.randrange(0, 100), rng.randrange(0, 100)
        return f"Calculate {a} + {b}. Answer {a + b}."
    if kind == 1:
        a, b = rng.randrange(0, 100), rng.randrange(0, 100)
        hi, lo = max(a, b), min(a, b)
        return f"Calculate {hi} - {lo}. Answer {hi - lo}."
    if kind == 2:
        a, b = rng.randrange(0, 21), rng.randrange(0, 21)
        return f"Calculate {a} * {b}. Answer {a * b}."
    x, a = rng.randrange(0, 100), rng.randrange(0, 100)
    total = x + a
    return f"Solve x + {a} = {total}. Answer x = {x}."


def arithmetic_stream(tokenizer: object, *, target_tokens: int, seed: int) -> tuple[torch.Tensor, int]:
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise RuntimeError("tokenizer does not contain EOS")
    rng = random.Random(seed)
    values: list[int] = []
    examples = 0
    while len(values) < target_tokens:
        encoded = tokenizer.encode(_arithmetic_text(rng)).ids
        if encoded:
            values.extend(encoded)
            values.append(int(eos_id))
            examples += 1
    return torch.tensor(values[:target_tokens], dtype=torch.long), examples


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def prepare_arithmetic_cache(cache_dir: Path, tokenizer: object) -> dict[str, object]:
    root = cache_dir / "conflict-differentiation-arithmetic"
    root.mkdir(parents=True, exist_ok=True)
    train_path = root / "train-tokens.pt"
    validation_path = root / "validation-tokens.pt"
    manifest_path = root / "manifest.json"
    expected = {"train_tokens": 300_000, "validation_tokens": 60_000, "seed": 21021}
    if train_path.is_file() and validation_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train = torch.load(train_path, map_location="cpu")
        validation = torch.load(validation_path, map_location="cpu")
        if (
            all(manifest.get(key) == value for key, value in expected.items())
            and tensor_sha256(train) == manifest.get("train_sha256")
            and tensor_sha256(validation) == manifest.get("validation_sha256")
        ):
            return {"train": train, "validation": validation, "manifest": manifest, "path": manifest_path}
    train, train_examples = arithmetic_stream(tokenizer, target_tokens=expected["train_tokens"], seed=expected["seed"])
    validation, validation_examples = arithmetic_stream(
        tokenizer,
        target_tokens=expected["validation_tokens"],
        seed=expected["seed"] + 1,
    )
    torch.save(train, train_path)
    torch.save(validation, validation_path)
    manifest = {
        "format": "minicells.conflict-arithmetic-corpus.v1",
        **expected,
        "train_examples": train_examples,
        "validation_examples": validation_examples,
        "train_sha256": tensor_sha256(train),
        "validation_sha256": tensor_sha256(validation),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"train": train, "validation": validation, "manifest": manifest, "path": manifest_path}


def deterministic_starts(stream_length: int, *, steps: int, batch_size: int, sequence_length: int, seed: int) -> tuple[tuple[int, ...], ...]:
    if stream_length <= sequence_length + 1:
        raise ValueError("stream too short")
    rng = random.Random(seed)
    high = stream_length - sequence_length - 1
    return tuple(tuple(rng.randrange(high) for _ in range(batch_size)) for _ in range(steps))


def mixed_domain_schedule(*, steps: int, seed: int) -> tuple[str, ...]:
    labels = [DOMAINS[index % 2] for index in range(steps)]
    rng = random.Random(seed)
    rng.shuffle(labels)
    return tuple(labels)


def lr_multiplier(step: int, total_steps: int, warmup: int = 25) -> float:
    if step <= warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
