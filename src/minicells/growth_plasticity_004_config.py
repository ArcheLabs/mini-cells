"""Frozen configuration for Core Validation 004."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path


_EPS = 1e-8


@dataclass(frozen=True)
class CoreValidation004Config:
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
    granularity: int
    growth_hidden: int
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
    growth_steps: int
    growth_learning_rate: float
    historical_examples_per_context: int
    maximum_base_normalized_mse: float
    minimum_new_gain_fraction: float
    maximum_local_regression: float
    maximum_false_safe_rate: float
    maximum_structural_escape_rate: float
    minimum_effective_acceptance_rate: float
    maximum_regression_damage_ratio_vs_local_always: float
    minimum_committed_gain_ratio_vs_local_always: float
    minimum_growth_rescue_rate: float
    minimum_private_cell_reuse_acceptance_rate: float
    maximum_spawned_cells_per_effective_commit: float
    maximum_active_growth_cells_per_input: int
    maximum_final_mutable_nrmse_ratio_vs_local_always: float

    @classmethod
    def from_protocol(cls, path: str | Path) -> "CoreValidation004Config":
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        world = p["world"]
        model = p["model"]
        pre = p["pretraining"]
        stream = p["continual_stream"]
        growth = p["growth"]
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
            granularity=int(model["granularity"]),
            growth_hidden=int(growth["cell_hidden"]),
            pretrain_steps=int(pre["steps"]),
            pretrain_batch_size=int(pre["batch_size"]),
            pretrain_learning_rate=float(pre["learning_rate"]),
            pretrain_weight_decay=float(pre["weight_decay"]),
            pretrain_validation_examples=int(pre["validation_examples"]),
            transactions=int(stream["transactions"]),
            update_train_examples=int(stream["train_examples"]),
            update_validation_examples=int(stream["validation_examples"]),
            update_steps=int(stream["direct_train_steps"]),
            update_learning_rate=float(stream["direct_learning_rate"]),
            update_amplitude=float(stream["update_amplitude"]),
            growth_steps=int(growth["train_steps"]),
            growth_learning_rate=float(growth["learning_rate"]),
            historical_examples_per_context=int(evaluation["historical_examples_per_context"]),
            maximum_base_normalized_mse=float(gates["maximum_base_normalized_mse"]),
            minimum_new_gain_fraction=float(gates["minimum_new_gain_fraction"]),
            maximum_local_regression=float(gates["maximum_local_regression"]),
            maximum_false_safe_rate=float(gates["maximum_false_safe_rate"]),
            maximum_structural_escape_rate=float(gates["maximum_structural_escape_rate"]),
            minimum_effective_acceptance_rate=float(gates["minimum_effective_acceptance_rate"]),
            maximum_regression_damage_ratio_vs_local_always=float(
                gates["maximum_regression_damage_ratio_vs_local_always"]
            ),
            minimum_committed_gain_ratio_vs_local_always=float(
                gates["minimum_committed_gain_ratio_vs_local_always"]
            ),
            minimum_growth_rescue_rate=float(gates["minimum_growth_rescue_rate"]),
            minimum_private_cell_reuse_acceptance_rate=float(
                gates["minimum_private_cell_reuse_acceptance_rate"]
            ),
            maximum_spawned_cells_per_effective_commit=float(
                gates["maximum_spawned_cells_per_effective_commit"]
            ),
            maximum_active_growth_cells_per_input=int(
                gates["maximum_active_growth_cells_per_input"]
            ),
            maximum_final_mutable_nrmse_ratio_vs_local_always=float(
                gates["maximum_final_mutable_nrmse_ratio_vs_local_always"]
            ),
        )
        if cfg.anchor_contexts <= 0 or cfg.anchor_contexts >= cfg.num_contexts:
            raise ValueError("anchor_contexts must be in (0, num_contexts)")
        if cfg.base_experts != cfg.num_function_families:
            raise ValueError("base_experts must equal num_function_families")
        if cfg.topk != 2:
            raise ValueError("v1 requires top-k 2")
        if cfg.base_expert_hidden % cfg.granularity != 0:
            raise ValueError("granularity must divide base_expert_hidden")
        return cfg

