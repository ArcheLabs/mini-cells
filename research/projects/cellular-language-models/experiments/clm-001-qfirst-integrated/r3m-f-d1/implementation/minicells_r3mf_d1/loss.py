"""Selected-target CE with update-global normalization."""


def selected_target_count(mask, objective):
    if objective not in ("uniform", "answer_eos"):
        raise ValueError("UNKNOWN_D1_OBJECTIVE")
    if objective == "uniform":
        return len(mask)
    return sum(bool(value) for value in mask)


def selected_positions(mask, objective):
    if objective not in ("uniform", "answer_eos"):
        raise ValueError("UNKNOWN_D1_OBJECTIVE")
    return tuple(index for index, include in enumerate(mask) if objective == "uniform" or include)


def normalized_loss_sum(per_token_losses, mask, objective, update_target_count):
    positions = selected_positions(mask, objective)
    if len(per_token_losses) != len(mask):
        raise ValueError("D1_LOGIT_LABEL_MASK_ALIGNMENT")
    if update_target_count <= 0:
        raise ValueError("D1_EMPTY_UPDATE_OBJECTIVE")
    return sum(per_token_losses[index] for index in positions) / update_target_count


def selected_loss(logits, labels, mask, objective, update_denominator):
    """Torch CE sum divided by the total selected labels in the optimizer step.

    ``logits`` and ``labels`` are flat token-position arrays for one fragment.
    Calling this once per fragment and summing the result yields the same global
    normalization regardless of fragment count or empty answer masks.
    """
    import torch.nn.functional as F

    if len(logits) != len(labels) or len(mask) != len(labels):
        raise ValueError("D1_LOGIT_LABEL_MASK_ALIGNMENT")
    if update_denominator <= 0:
        raise ValueError("D1_EMPTY_UPDATE_OBJECTIVE")
    positions = selected_positions(mask, objective)
    if not positions:
        return logits.sum() * 0.0
    import torch

    index = torch.tensor(positions, device=logits.device, dtype=torch.long)
    selected_logits = logits.index_select(0, index)
    selected_labels = labels.index_select(0, index)
    return F.cross_entropy(selected_logits, selected_labels, reduction="sum") / update_denominator


def update_denominator(batches, objective):
    total = 0
    for _, labels, answer_eos_mask in batches:
        total += selected_target_count(answer_eos_mask, objective)
    if total <= 0:
        raise ValueError("D1_EMPTY_UPDATE_OBJECTIVE")
    return total
