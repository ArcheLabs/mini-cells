"""Shared train and evaluation forward for segment-reset episodes."""
import torch
from torch.nn import functional as F


def forward_episode(model, segments, *, wrapper=None, full_history=False):
    """Return supervised logits and labels with causal next-token alignment."""
    if not segments or any(len(ids) != len(labels) for ids, labels in segments):
        raise ValueError("Expected token-aligned segments")
    if wrapper is not None:
        wrapper.reset_episode()
    if full_history:
        if wrapper is not None:
            raise ValueError("Full-history condition uses native Qwen")
        segments = [(sum((list(ids) for ids, _ in segments), []),
                     sum((list(labels) for _, labels in segments), []))]
    device = model.get_input_embeddings().weight.device
    selected, expected = [], []
    offset = 0
    for ids, labels in segments:
        if not ids or labels[0] != -100:
            raise ValueError("Each segment must begin with unsupervised context")
        tokens = torch.tensor([ids], device=device)
        positions = torch.arange(offset, offset + len(ids), device=device)[None]
        result = model(input_ids=tokens, position_ids=positions, use_cache=False)
        targets = torch.tensor(labels[1:], device=device)
        mask = targets != -100
        if mask.any():
            selected.append(result.logits[0, :-1][mask].float())
            expected.append(targets[mask])
        offset += len(ids)
    if not expected:
        raise ValueError("Episode has no answer tokens")
    return torch.cat(selected), torch.cat(expected)


def answer_loss(logits, targets):
    return F.cross_entropy(logits, targets)
