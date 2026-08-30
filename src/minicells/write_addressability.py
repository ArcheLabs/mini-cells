"""Core Validation 002 synthetic superposition world and frozen protocol config."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F

VariantName = Literal[
    "inferred_address",
    "oracle_address",
    "permuted_address",
    "global_write",
    "dense",
    "moe",
]

_EPS = 1e-12

@dataclass(frozen=True)
class WriteAddressabilityConfig:
    observation_dim: int = 128
    num_features: int = 512
    active_features: int = 4
    output_dim: int = 32
    latent_dim: int = 1024
    latent_topk: int = 8
    coefficient_min_abs: float = 0.5
    coefficient_max_abs: float = 1.5
    edit_scale: float = 0.75
    pretrain_steps: int = 6000
    pretrain_examples: int = 65536
    pretrain_batch_size: int = 512
    pretrain_learning_rate: float = 1e-3
    pretrain_weight_decay: float = 1e-5
    reconstruction_weight: float = 0.5
    gradient_clip_norm: float = 1.0
    validation_examples: int = 4096
    edit_count: int = 100
    edit_examples: int = 8
    affected_examples: int = 512
    invariant_examples: int = 512
    retention_examples_per_edit: int = 16
    repeat_every: int = 5
    previous_target_distractor_every: int = 4
    address_min_shared_fraction: float = 0.75
    address_min_energy: float = 1e-8
    global_edit_steps: int = 32
    global_edit_learning_rate: float = 2e-2
    dense_edit_steps: int = 32
    dense_edit_learning_rate: float = 2e-3
    moe_edit_steps: int = 32
    moe_edit_learning_rate: float = 2e-3
    moe_topk: int = 2
    oracle_probe_examples: int = 8192
    maximum_base_normalized_mse: float = 0.10
    maximum_candidate_update_error: float = 0.10
    maximum_baseline_update_error: float = 0.15
    maximum_leakage_ratio: float = 0.10
    minimum_mechanistic_correlation: float = 0.70
    minimum_permutation_degradation: float = 2.0

    def __post_init__(self) -> None:
        if self.observation_dim < 1 or self.output_dim < 1:
            raise ValueError("observation_dim and output_dim must be positive")
        if self.num_features < 2:
            raise ValueError("num_features must be >= 2")
        if not 1 <= self.active_features < self.num_features:
            raise ValueError("active_features must be in [1, num_features)")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if not 1 <= self.latent_topk <= self.latent_dim:
            raise ValueError("latent_topk must be in [1, latent_dim]")
        if self.coefficient_min_abs <= 0:
            raise ValueError("coefficient_min_abs must be positive")
        if self.coefficient_max_abs < self.coefficient_min_abs:
            raise ValueError("coefficient_max_abs must be >= coefficient_min_abs")
        if self.edit_examples < 2:
            raise ValueError("edit_examples must be >= 2")
        if self.edit_count < 1:
            raise ValueError("edit_count must be positive")
        if not 0 < self.address_min_shared_fraction <= 1:
            raise ValueError("address_min_shared_fraction must be in (0, 1]")
        if self.repeat_every < 0 or self.previous_target_distractor_every < 0:
            raise ValueError("repeat/distractor intervals must be non-negative")

    @property
    def superposition_load(self) -> float:
        return self.num_features / self.observation_dim

    @property
    def recovery_load(self) -> float:
        numerator = self.active_features * math.log(self.num_features / self.active_features)
        return numerator / self.observation_dim

    @classmethod
    def from_protocol(cls, path: str | Path) -> "WriteAddressabilityConfig":
        payload = json.loads(Path(path).read_text())
        world = payload["world"]
        model = payload["model"]
        pretrain = payload["pretraining"]
        editing = payload["editing"]
        evaluation = payload["evaluation"]
        gates = payload["gates"]
        return cls(
            observation_dim=int(world["observation_dim"]),
            num_features=int(world["num_features"]),
            active_features=int(world["active_features"]),
            output_dim=int(world["output_dim"]),
            coefficient_min_abs=float(world["coefficient_min_abs"]),
            coefficient_max_abs=float(world["coefficient_max_abs"]),
            edit_scale=float(world["edit_scale"]),
            latent_dim=int(model["latent_dim"]),
            latent_topk=int(model["latent_topk"]),
            moe_topk=int(model["moe_topk"]),
            pretrain_steps=int(pretrain["steps"]),
            pretrain_examples=int(pretrain["examples"]),
            pretrain_batch_size=int(pretrain["batch_size"]),
            pretrain_learning_rate=float(pretrain["learning_rate"]),
            pretrain_weight_decay=float(pretrain["weight_decay"]),
            reconstruction_weight=float(pretrain["reconstruction_weight"]),
            gradient_clip_norm=float(pretrain["gradient_clip_norm"]),
            validation_examples=int(pretrain["validation_examples"]),
            edit_count=int(editing["edit_count"]),
            edit_examples=int(editing["edit_examples"]),
            affected_examples=int(evaluation["affected_examples"]),
            invariant_examples=int(evaluation["invariant_examples"]),
            retention_examples_per_edit=int(evaluation["retention_examples_per_edit"]),
            repeat_every=int(editing["repeat_every"]),
            previous_target_distractor_every=int(editing["previous_target_distractor_every"]),
            address_min_shared_fraction=float(editing["address_min_shared_fraction"]),
            address_min_energy=float(editing["address_min_energy"]),
            global_edit_steps=int(editing["global_write"]["steps"]),
            global_edit_learning_rate=float(editing["global_write"]["learning_rate"]),
            dense_edit_steps=int(editing["dense"]["steps"]),
            dense_edit_learning_rate=float(editing["dense"]["learning_rate"]),
            moe_edit_steps=int(editing["moe"]["steps"]),
            moe_edit_learning_rate=float(editing["moe"]["learning_rate"]),
            oracle_probe_examples=int(evaluation["oracle_probe_examples"]),
            maximum_base_normalized_mse=float(gates["maximum_base_normalized_mse"]),
            maximum_candidate_update_error=float(gates["maximum_candidate_update_error"]),
            maximum_baseline_update_error=float(gates["maximum_baseline_update_error"]),
            maximum_leakage_ratio=float(gates["maximum_leakage_ratio"]),
            minimum_mechanistic_correlation=float(gates["minimum_mechanistic_correlation"]),
            minimum_permutation_degradation=float(gates["minimum_permutation_degradation"]),
        )


@dataclass
class Batch:
    x: torch.Tensor
    s: torch.Tensor
    y: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(self.x.to(device), self.s.to(device), self.y.to(device))


@dataclass(frozen=True)
class EditTask:
    index: int
    target_feature: int
    delta: torch.Tensor
    forced_distractor: int | None


class SuperpositionWorld:
    """Synthetic world with evaluator-visible sparse ground truth."""

    def __init__(self, config: WriteAddressabilityConfig, *, seed: int) -> None:
        self.config = config
        generator = torch.Generator().manual_seed(seed)
        a = torch.randn(
            config.observation_dim,
            config.num_features,
            generator=generator,
        )
        self.A = F.normalize(a, dim=0)
        v = torch.randn(config.output_dim, config.num_features, generator=generator)
        self.V0 = F.normalize(v, dim=0)
        self.V = self.V0.clone()

    def reset_functions(self) -> None:
        self.V = self.V0.clone()

    def _coefficients(self, count: int, *, generator: torch.Generator) -> torch.Tensor:
        magnitude = torch.empty(count).uniform_(
            self.config.coefficient_min_abs,
            self.config.coefficient_max_abs,
            generator=generator,
        )
        sign = torch.randint(0, 2, (count,), generator=generator, dtype=torch.float32)
        sign = sign.mul_(2).sub_(1)
        return magnitude * sign

    def sample_latents(
        self,
        count: int,
        *,
        generator: torch.Generator,
        include_feature: int | None = None,
        exclude_feature: int | None = None,
        forced_distractor: int | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        if include_feature is not None and exclude_feature == include_feature:
            raise ValueError("a feature cannot be both included and excluded")
        if forced_distractor is not None and forced_distractor == include_feature:
            raise ValueError("forced_distractor must differ from include_feature")
        if forced_distractor is not None and forced_distractor == exclude_feature:
            raise ValueError("forced_distractor cannot be excluded")

        fixed: list[int] = []
        if include_feature is not None:
            fixed.append(int(include_feature))
        if forced_distractor is not None:
            fixed.append(int(forced_distractor))
        remaining = cfg.active_features - len(fixed)
        if remaining < 0:
            raise ValueError("invalid fixed support for configured sparsity")

        if remaining:
            sampled = torch.randint(
                0, cfg.num_features, (count, remaining), generator=generator
            )
            # k is deliberately small. Rejection is much cheaper than drawing a
            # full [batch, F] random ranking and still guarantees exact support.
            while True:
                invalid = torch.zeros(count, dtype=torch.bool)
                if exclude_feature is not None:
                    invalid |= sampled.eq(int(exclude_feature)).any(dim=1)
                for feature in fixed:
                    invalid |= sampled.eq(feature).any(dim=1)
                if remaining > 1:
                    sorted_values = sampled.sort(dim=1).values
                    invalid |= sorted_values[:, 1:].eq(sorted_values[:, :-1]).any(dim=1)
                if not bool(invalid.any()):
                    break
                sampled[invalid] = torch.randint(
                    0, cfg.num_features, (int(invalid.sum().item()), remaining), generator=generator
                )
        else:
            sampled = torch.empty(count, 0, dtype=torch.long)

        if fixed:
            fixed_tensor = torch.tensor(fixed, dtype=torch.long).expand(count, len(fixed))
            support = torch.cat((fixed_tensor, sampled), dim=1)
        else:
            support = sampled
        s = torch.zeros(count, cfg.num_features)
        coefficients = self._coefficients(count * cfg.active_features, generator=generator).reshape(
            count, cfg.active_features
        )
        s.scatter_(1, support, coefficients)
        return s

    def batch_from_latents(self, s: torch.Tensor) -> Batch:
        return Batch(x=s @ self.A.transpose(0, 1), s=s, y=s @ self.V.transpose(0, 1))

    def sample_batch(self, count: int, *, generator: torch.Generator) -> Batch:
        return self.batch_from_latents(self.sample_latents(count, generator=generator))

    def edit_batches(
        self,
        task: EditTask,
        *,
        generator: torch.Generator,
    ) -> tuple[Batch, Batch, Batch, Batch]:
        """Create edit, unseen-affected, invariant, and retention batches.

        The returned y values use the post-edit world.  The world itself is not
        mutated until ``apply_edit`` is called.
        """

        cfg = self.config
        target = task.target_feature
        edit_s = self.sample_latents(
            cfg.edit_examples,
            generator=generator,
            include_feature=target,
            forced_distractor=task.forced_distractor,
        )
        affected_s = self.sample_latents(
            cfg.affected_examples,
            generator=generator,
            include_feature=target,
        )
        invariant_s = self.sample_latents(
            cfg.invariant_examples,
            generator=generator,
            exclude_feature=target,
        )
        retention_s = self.sample_latents(
            cfg.retention_examples_per_edit,
            generator=generator,
            include_feature=target,
        )

        v_after = self.V.clone()
        v_after[:, target] += task.delta

        def build(s: torch.Tensor) -> Batch:
            return Batch(
                x=s @ self.A.transpose(0, 1),
                s=s,
                y=s @ v_after.transpose(0, 1),
            )

        return build(edit_s), build(affected_s), build(invariant_s), build(retention_s)

    def apply_edit(self, task: EditTask) -> None:
        self.V[:, task.target_feature] += task.delta

    def current_targets(self, s: torch.Tensor) -> torch.Tensor:
        return s @ self.V.transpose(0, 1)


