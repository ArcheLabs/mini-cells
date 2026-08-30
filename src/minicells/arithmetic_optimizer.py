from __future__ import annotations

import torch

from .arithmetic_tasks import ArithmeticExample, arithmetic_batch
from .continual_learning import (
    MARGIN_Q,
    PERTURBATION_Q,
    STEP_Q,
    TaskBatch,
    candidate,
    delta_vector,
    exact_logits,
    select_indices,
)
from .vocab import CharVocab

OBJECTIVE_SCALE = 1024
ARITHMETIC_WEIGHT = 4


def masked_margin_mean(
    flat: torch.Tensor,
    batch: TaskBatch,
    supervision_mask: torch.Tensor,
    margin_q: int = MARGIN_Q,
) -> int:
    logits = exact_logits(flat, batch.input_ids)
    targets = batch.target_ids.to(dtype=torch.long, device="cpu")
    active = supervision_mask.to(dtype=torch.bool, device="cpu")
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    competitors = logits.clone()
    competitors.scatter_(-1, targets.unsqueeze(-1), -(1 << 60))
    other = competitors.max(dim=-1).values
    loss = (margin_q - (target_logits - other)).clamp_min(0)
    count = int(active.sum().item())
    if count == 0:
        return 0
    total = int(loss[active].sum().item())
    return (total * OBJECTIVE_SCALE + count // 2) // count


def composite_objective(
    flat: torch.Tensor,
    echo_batch: TaskBatch,
    arithmetic: TaskBatch,
    arithmetic_weight: int = ARITHMETIC_WEIGHT,
) -> tuple[int, int, int]:
    echo = masked_margin_mean(flat, echo_batch, echo_batch.mask)
    capability = masked_margin_mean(flat, arithmetic, arithmetic.changed_mask)
    return echo + arithmetic_weight * capability, echo, capability


def guarded_spsa_step(
    flat: torch.Tensor,
    echo_batch: TaskBatch,
    arithmetic: TaskBatch,
    echo_anchor: TaskBatch,
    parent_hash: bytes,
    generation: int,
    block_size: int,
    arithmetic_weight: int = ARITHMETIC_WEIGHT,
    perturbation_q: int = PERTURBATION_Q,
    step_q: int = STEP_Q,
) -> tuple[torch.Tensor, dict[str, int | bool]]:
    delta = delta_vector(parent_hash, generation, block_size)
    base_total, base_echo, base_capability = composite_objective(
        flat, echo_batch, arithmetic, arithmetic_weight
    )
    plus = candidate(flat, delta, 1, perturbation_q)
    minus = candidate(flat, delta, -1, perturbation_q)
    plus_total, _, _ = composite_objective(plus, echo_batch, arithmetic, arithmetic_weight)
    minus_total, _, _ = composite_objective(minus, echo_batch, arithmetic, arithmetic_weight)
    base_anchor = masked_margin_mean(flat, echo_anchor, echo_anchor.mask)

    if plus_total == minus_total:
        return flat, {
            "accepted": False,
            "direction": 0,
            "base_total": base_total,
            "base_echo": base_echo,
            "base_capability": base_capability,
            "plus_total": plus_total,
            "minus_total": minus_total,
            "proposal_total": base_total,
            "base_anchor": base_anchor,
            "proposal_anchor": base_anchor,
        }

    direction = 1 if plus_total < minus_total else -1
    proposal = candidate(flat, delta, direction, step_q)
    proposal_total, _, _ = composite_objective(
        proposal, echo_batch, arithmetic, arithmetic_weight
    )
    proposal_anchor = masked_margin_mean(proposal, echo_anchor, echo_anchor.mask)
    accepted = proposal_total < base_total and proposal_anchor <= base_anchor

    return (proposal if accepted else flat), {
        "accepted": accepted,
        "direction": direction,
        "base_total": base_total,
        "base_echo": base_echo,
        "base_capability": base_capability,
        "plus_total": plus_total,
        "minus_total": minus_total,
        "proposal_total": proposal_total,
        "base_anchor": base_anchor,
        "proposal_anchor": proposal_anchor,
    }


def select_arithmetic_batch(
    vocab: CharVocab,
    examples: list[ArithmeticExample],
    count: int,
    parent_hash: bytes,
    generation: int,
) -> TaskBatch:
    indices = select_indices(len(examples), count, "arithmetic", parent_hash, generation)
    selected = [examples[index] for index in indices]
    return arithmetic_batch(vocab, selected)
