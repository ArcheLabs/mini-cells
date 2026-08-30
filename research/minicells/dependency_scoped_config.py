"""Frozen configuration for Core Validation 003."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_EPS = 1e-8


@dataclass(frozen=True)
class CoreValidation003Config:
    num_contexts: int
    anchor_contexts: int
    content_dim: int
    model_dim: int
    output_dim: int
    num_function_families: int
    basis_hidden: int
    residual_hidden: int
    router_dim: int
    base_experts: int
    base_expert_hidden: int
    topk: int
    granularities: tuple[int, ...]
    pretrain_steps: int
    pretrain_batch_size: int
    pretrain_learning_rate: float
    pretrain_weight_decay: float
    pretrain_validation_examples: int
    transactions: int
    update_train_examples: int
    update_validation_examples: int
    update_steps: int
    update_learning_rate: float
    update_amplitude: float
    historical_examples_per_context: int
    router_drift_noise: float
    maximum_base_normalized_mse: float
    minimum_new_gain_fraction: float
    maximum_local_regression: float
    maximum_false_safe_rate: float
    maximum_dependency_coverage: float
    maximum_structural_escape_rate: float
    minimum_acceptance_rate: float
    maximum_regression_damage_ratio_vs_local_always: float
    minimum_committed_gain_ratio_vs_local_always: float
    maximum_dependency_ratio_vs_coarsest: float

    @classmethod
    def from_protocol(cls, path: str | Path) -> "CoreValidation003Config":
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        world = p["world"]
        model = p["model"]
        pretrain = p["pretraining"]
        continual = p["continual_stream"]
        evaluation = p["evaluation"]
        gates = p["gates"]
        cfg = cls(
            num_contexts=int(world["num_contexts"]),
            anchor_contexts=int(world["anchor_contexts"]),
            content_dim=int(world["content_dim"]),
            model_dim=int(model["model_dim"]),
            output_dim=int(world["output_dim"]),
            num_function_families=int(world["num_function_families"]),
            basis_hidden=int(world["basis_hidden"]),
            residual_hidden=int(world["residual_hidden"]),
            router_dim=int(model["router_dim"]),
            base_experts=int(model["base_experts"]),
            base_expert_hidden=int(model["base_expert_hidden"]),
            topk=int(model["topk"]),
            granularities=tuple(int(x) for x in model["granularities"]),
            pretrain_steps=int(pretrain["steps"]),
            pretrain_batch_size=int(pretrain["batch_size"]),
            pretrain_learning_rate=float(pretrain["learning_rate"]),
            pretrain_weight_decay=float(pretrain["weight_decay"]),
            pretrain_validation_examples=int(pretrain["validation_examples"]),
            transactions=int(continual["transactions"]),
            update_train_examples=int(continual["train_examples"]),
            update_validation_examples=int(continual["validation_examples"]),
            update_steps=int(continual["train_steps"]),
            update_learning_rate=float(continual["learning_rate"]),
            update_amplitude=float(continual["update_amplitude"]),
            historical_examples_per_context=int(evaluation["historical_examples_per_context"]),
            router_drift_noise=float(continual["router_drift_stress"]["noise_scale"]),
            maximum_base_normalized_mse=float(gates["maximum_base_normalized_mse"]),
            minimum_new_gain_fraction=float(gates["minimum_new_gain_fraction"]),
            maximum_local_regression=float(gates["maximum_local_regression"]),
            maximum_false_safe_rate=float(gates["maximum_false_safe_rate"]),
            maximum_dependency_coverage=float(gates["maximum_dependency_coverage"]),
            maximum_structural_escape_rate=float(gates["maximum_structural_escape_rate"]),
            minimum_acceptance_rate=float(gates["minimum_acceptance_rate"]),
            maximum_regression_damage_ratio_vs_local_always=float(
                gates["maximum_regression_damage_ratio_vs_local_always"]
            ),
            minimum_committed_gain_ratio_vs_local_always=float(
                gates["minimum_committed_gain_ratio_vs_local_always"]
            ),
            maximum_dependency_ratio_vs_coarsest=float(
                gates["maximum_dependency_ratio_vs_coarsest"]
            ),
        )
        if cfg.anchor_contexts <= 0 or cfg.anchor_contexts >= cfg.num_contexts:
            raise ValueError("anchor_contexts must be in (0, num_contexts)")
        if cfg.base_expert_hidden % max(cfg.granularities) != 0:
            raise ValueError("base_expert_hidden must be divisible by the maximum granularity")
        if cfg.base_experts != cfg.num_function_families:
            raise ValueError("base_experts must equal num_function_families in frozen v1")
        if cfg.topk != 2:
            raise ValueError("Core Validation 003 v1 requires exactly two routed functional families")
        if not cfg.granularities or cfg.granularities[0] != 1:
            raise ValueError("granularities must begin at 1")
        return cfg
