"""Shared task-conditioned allocation and cached-tail branch training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import torch
from torch import Tensor, nn

from .composition import compose_cellular_experts
from .task import TailTrainingCache, answer_token_cross_entropy
from .lora import LoRAConfig, MatchedLoRAExperts, selected_lora_parameters
from .training import Allocation, BranchTrainingConfig, ForkedCellularExperts, allocate_topk, selected_delta_parameters


def _trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _optimizer(parameters: list[nn.Parameter], config: BranchTrainingConfig) -> torch.optim.Optimizer:
    if not parameters:
        raise ValueError("task branch has no trainable parameters")
    if config.optimizer.lower() == "adamw":
        return torch.optim.AdamW(parameters, lr=config.learning_rate)
    if config.optimizer.lower() == "sgd":
        return torch.optim.SGD(parameters, lr=config.learning_rate)
    raise ValueError(f"unsupported optimizer: {config.optimizer}")


def slice_task_cache(cache: TailTrainingCache, start: int, end: int) -> TailTrainingCache:
    """Slice both sequence-shaped and flattened routing caches consistently."""
    rows, width = cache.input_ids.shape
    top_start, top_end = start * width, end * width
    top_k_index = cache.top_k_index[top_start:top_end] if cache.top_k_index.shape[0] == rows * width else cache.top_k_index[start:end]
    top_k_weights = cache.top_k_weights[top_start:top_end] if cache.top_k_weights.shape[0] == rows * width else cache.top_k_weights[start:end]
    return TailTrainingCache(
        mlp_input=cache.mlp_input[start:end],
        pre_mlp_residual=cache.pre_mlp_residual[start:end],
        top_k_index=top_k_index,
        top_k_weights=top_k_weights,
        input_ids=cache.input_ids[start:end],
        attention_mask=cache.attention_mask[start:end],
        labels=cache.labels[start:end],
        loss_mask=cache.loss_mask[start:end],
        sample_ids=cache.sample_ids[start:end],
        split=cache.split,
        identity=cache.identity,
    )


def cached_task_loss(runner: Any, cache: TailTrainingCache, experts: nn.Module) -> Tensor:
    return answer_token_cross_entropy(runner.forward_with_experts(cache, experts), cache.labels)


def task_conditioned_allocation(
    parent_experts: nn.Module,
    runner: Any,
    cache: TailTrainingCache,
    *,
    cells_per_expert: int,
    layer: int = 0,
) -> Allocation:
    """Score all logical Cells from the actual split's answer-token CE."""
    all_cells = {
        expert: tuple(range(cells_per_expert))
        for expert in range(int(parent_experts.num_experts))
    }
    probe = ForkedCellularExperts(parent_experts, all_cells)
    probe.zero_grad(set_to_none=True)
    loss = cached_task_loss(runner, cache, probe)
    loss.backward()
    scores: dict[str, float] = {}
    for expert_index, expert in enumerate(probe.cells):
        for cell_index, cell in enumerate(expert.cells):
            values = [value.grad for name, value in cell.named_parameters() if name.startswith("delta_") and value.grad is not None]
            count = sum(int(value.numel()) for value in values)
            scores[f"L{int(layer)}:E{expert_index}:C{cell_index}"] = (
                sum(float(value.detach().float().pow(2).sum()) for value in values) / max(1, count)
            )
    probe.zero_grad(set_to_none=True)
    return allocate_topk(scores)


def relabel_allocation(allocation: Allocation, layer: int) -> Allocation:
    """Attach the actual layer to allocation IDs without changing ordering."""
    scores = {key.replace("L0:", f"L{int(layer)}:", 1): value for key, value in allocation.scores.items()}
    selected = tuple(key.replace("L0:", f"L{int(layer)}:", 1) for key in allocation.selected)
    return Allocation(scores, selected, allocation.topk_mass, allocation.effective_count)


@dataclass(frozen=True)
class TaskBranchResult:
    branch: str
    selected_cells: tuple[str, ...]
    training_steps: int
    training_tokens: int
    final_loss: float
    runtime: ForkedCellularExperts

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "selected_cells": list(self.selected_cells),
            "training_steps": self.training_steps,
            "training_tokens": self.training_tokens,
            "final_loss": self.final_loss,
        }


def train_cached_branch(
    parent_experts: nn.Module,
    runner: Any,
    cache: TailTrainingCache,
    selected_cells: Iterable[str],
    *,
    layer: int,
    branch: str,
    config: BranchTrainingConfig,
) -> TaskBranchResult:
    """Train a fork only on one task cache under the frozen token/step budget."""
    selected = tuple(str(value) for value in selected_cells)
    if str(branch) in {"A", "B"} and cache.split != f"{branch}_train":
        raise ValueError(f"branch {branch} cannot consume cache split {cache.split}")
    torch.manual_seed(int(config.seed) + sum(ord(char) for char in str(branch)))
    by_expert: dict[int, list[int]] = {}
    prefix = f"L{int(layer)}:E"
    for cell_id in selected:
        if not cell_id.startswith(prefix) or ":C" not in cell_id:
            raise ValueError(f"invalid selected Cell ID: {cell_id}")
        expert_text, cell_text = cell_id[len(prefix):].split(":C", 1)
        by_expert.setdefault(int(expert_text), []).append(int(cell_text))
    runtime = ForkedCellularExperts(parent_experts, by_expert)
    runtime.to(cache.mlp_input.device)
    optimizer = _optimizer(_trainable_parameters(runtime), config)
    rows = 0
    tokens = 0
    final_loss = float("nan")
    while rows < config.max_optimizer_steps and tokens < config.max_training_tokens:
        progressed = False
        for start in range(0, cache.input_ids.shape[0], max(1, config.batch_size)):
            end = min(cache.input_ids.shape[0], start + max(1, config.batch_size))
            batch = slice_task_cache(cache, start, end)
            batch_tokens = int(batch.loss_mask.sum())
            if batch_tokens == 0:
                raise ValueError(f"{branch} task cache contains no answer-token labels")
            if tokens + batch_tokens > config.max_training_tokens:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = cached_task_loss(runner, batch, runtime)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite task loss in branch {branch}")
            loss.backward()
            optimizer.step()
            rows += 1
            tokens += batch_tokens
            final_loss = float(loss.detach())
            progressed = True
            if rows >= config.max_optimizer_steps:
                break
        if not progressed:
            break
    if rows == 0:
        raise RuntimeError(f"branch {branch} did not consume a task batch")
    return TaskBranchResult(branch, selected, rows, tokens, final_loss, runtime)


def train_cached_lora_branch(
    parent_experts: nn.Module,
    runner: Any,
    cache: TailTrainingCache,
    selected_cells: Iterable[str],
    *,
    layer: int,
    branch: str,
    rank: int,
    config: BranchTrainingConfig,
) -> tuple[MatchedLoRAExperts, dict[str, Any]]:
    """Train a matched LoRA branch on the same cached task split as PCU."""
    selected = tuple(str(value) for value in selected_cells)
    if str(branch) in {"A", "B"} and cache.split != f"{branch}_train":
        raise ValueError(f"LoRA branch {branch} cannot consume cache split {cache.split}")
    torch.manual_seed(int(config.seed) + sum(ord(char) for char in str(branch)))
    by_expert: dict[int, list[int]] = {}
    prefix = f"L{int(layer)}:E"
    for cell_id in selected:
        expert_text, cell_text = cell_id[len(prefix):].split(":C", 1)
        by_expert.setdefault(int(expert_text), []).append(int(cell_text))
    runtime = MatchedLoRAExperts(parent_experts, by_expert, LoRAConfig(int(rank)))
    runtime.to(cache.mlp_input.device)
    optimizer = _optimizer(selected_lora_parameters(runtime), config)
    steps = 0
    tokens = 0
    final_loss = float("nan")
    while steps < config.max_optimizer_steps and tokens < config.max_training_tokens:
        progressed = False
        for start in range(0, cache.input_ids.shape[0], max(1, config.batch_size)):
            end = min(cache.input_ids.shape[0], start + max(1, config.batch_size))
            batch = slice_task_cache(cache, start, end)
            batch_tokens = int(batch.loss_mask.sum())
            if tokens + batch_tokens > config.max_training_tokens:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = answer_token_cross_entropy(runner.forward_with_experts(batch, runtime), batch.labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite LoRA task loss in branch {branch}")
            loss.backward()
            optimizer.step()
            steps += 1
            tokens += batch_tokens
            final_loss = float(loss.detach())
            progressed = True
            if steps >= config.max_optimizer_steps:
                break
        if not progressed:
            break
    if steps == 0:
        raise RuntimeError(f"LoRA branch {branch} did not consume a task batch")
    return runtime, {"branch": branch, "selected_cells": list(selected), "rank": int(rank), "training_steps": steps, "training_tokens": tokens, "final_loss": final_loss}


def task_branch_maps(*results: TaskBranchResult) -> dict[str, dict[int, nn.Module]]:
    return {
        result.branch: {index: expert for index, expert in enumerate(result.runtime.cells)}
        for result in results
    }
