from __future__ import annotations

import torch

from .data import CopyBatch
from .metrics import echo_metrics, masked_cross_entropy


@torch.no_grad()
def evaluate(model, batch: CopyBatch) -> dict[str, float]:
    model.eval()
    logits = model(batch.input_ids)
    result = echo_metrics(logits, batch.target_ids, batch.mask)
    result["loss"] = masked_cross_entropy(logits, batch.target_ids, batch.mask).item()
    return result
