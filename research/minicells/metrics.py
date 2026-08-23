from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    return (losses * mask).sum() / mask.sum().clamp_min(1)


def echo_metrics(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(targets)
    token_accuracy = (correct & mask).sum().float() / mask.sum().clamp_min(1)
    sequence_accuracy = (correct | ~mask).all(dim=1).float().mean()
    return {"token_accuracy": token_accuracy.item(), "exact_sequence_accuracy": sequence_accuracy.item()}


def levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def edit_similarity(prediction: str, target: str) -> float:
    denominator = max(len(prediction), len(target))
    return 1.0 if denominator == 0 else 1.0 - levenshtein(prediction, target) / denominator
