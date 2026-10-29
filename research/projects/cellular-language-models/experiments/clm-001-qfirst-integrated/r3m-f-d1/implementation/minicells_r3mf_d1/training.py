"""Later D1 update adapter; importing the module itself never starts training."""

import types

from .data import split_independent


def torch_batches(rows, context, device):
    import torch

    result = []
    for row in rows:
        if context == "PACKED":
            spans = [(row.tokens, row.token_kinds, 0)]
        elif context == "INDEPENDENT_QA":
            spans = [
                (part.tokens, part.target_kinds, part.source_start)
                for part in split_independent(row)
            ]
        else:
            raise ValueError("D1_CONTEXT_UNKNOWN")
        for tokens, kinds, source_start in spans:
            if len(tokens) < 2:
                continue
            x = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(tokens[1:], dtype=torch.long, device=device).unsqueeze(0)
            target_kinds = kinds[1:]
            selected = tuple(i for i, kind in enumerate(target_kinds) if kind in ("answer", "eos"))
            selected_set = set(selected)
            mask = torch.tensor(
                [i in selected_set for i in range(len(target_kinds))],
                dtype=torch.bool,
                device=device,
            )
            result.append(
                {
                    "x": x,
                    "y": y,
                    "answer_eos_mask": mask,
                    "answer_eos_positions": selected,
                    "answer_eos_count": len(selected),
                    "target_kinds": tuple(target_kinds),
                    "source_microbatch": row.microbatch,
                    "source_start": source_start,
                    "input_positions": len(tokens) - 1,
                }
            )
    return result


def selected_count(batches, objective):
    total = 0
    for batch in batches:
        total += batch["y"].numel() if objective == "UNIFORM" else batch["answer_eos_count"]
    if total <= 0:
        raise ValueError("D1_EMPTY_SELECTED_TARGET_SET")
    return total


def update(tr, batches, *, objective, context, d1_config):
    """One bounded update over source-aligned fragments; caller supplies existing state."""
    import time
    import torch
    from torch.nn import functional as F

    if objective not in ("UNIFORM", "ANSWER_EOS") or context not in ("PACKED", "INDEPENDENT_QA"):
        raise ValueError("D1_ARM_CONTRACT_INVALID")
    m = tr.model
    denominator = selected_count(batches, objective)
    tr.cell_opt.zero_grad(set_to_none=True)
    start = time.perf_counter()
    losses = []
    operator_counts = []
    actual_input_positions = 0
    m.active_step = tr.step
    m.microbatch_index = 0
    try:
        for batch in batches:
            x, y = batch["x"], batch["y"]
            if context == "INDEPENDENT_QA":
                out = m.forward_fragment(
                    x,
                    source_context={
                        "absolute_step": tr.step,
                        "source_microbatch": batch["source_microbatch"],
                        "source_start": batch["source_start"],
                    },
                    epsilon=tr.epsilon(),
                    route_rng=tr.route_rng,
                )
            else:
                out = m(x, epsilon=tr.epsilon(), route_rng=tr.route_rng)
            logits = out.logits.reshape(-1, out.logits.shape[-1])
            labels = y.reshape(-1)
            if objective == "ANSWER_EOS":
                positions = torch.tensor(
                    batch["answer_eos_positions"], dtype=torch.long, device=logits.device
                )
                if len(positions):
                    loss_sum = F.cross_entropy(
                        logits.index_select(0, positions),
                        labels.index_select(0, positions),
                        reduction="sum",
                    )
                else:
                    loss_sum = logits.sum() * 0.0
            else:
                loss_sum = F.cross_entropy(logits, labels, reduction="sum")
            loss = loss_sum / denominator
            if not torch.isfinite(loss):
                raise FloatingPointError("D1_NONFINITE_CE")
            loss.backward()
            losses.append(loss_sum.detach())
            if hasattr(out, "operator_rows"):
                operator_counts.append(out.operator_rows.detach())
            actual_input_positions += batch["input_positions"]
    finally:
        m.active_step = None
        m.training_context = None
        m.d1_source_context = None

    gradients = []
    for cell in m.cells:
        sums = [p.grad.square().sum() for p in cell.parameters() if p.grad is not None]
        gradients.append(
            torch.stack(sums).sum().sqrt() if sums else torch.zeros((), device=logits.device)
        )
    gradient_norms = torch.stack(gradients)
    if not torch.isfinite(gradient_norms).all() or not (gradient_norms > 0).any():
        raise FloatingPointError("D1_INVALID_CELL_GRADIENT")
    q = tr.c["engine"]["training"]
    torch.nn.utils.clip_grad_norm_(m.cells.parameters(), q["clip"])
    for group in tr.cell_opt.param_groups:
        group["lr"] = q["learning_rate"] * min(1, (tr.step + 1) / q["warmup_steps"])
    tr.cell_opt.step()
    tr.step += 1
    source_slots = d1_config["source_rows_per_step"] * (d1_config["source_sequence_tokens"] - 1)
    tr.data_cursor += source_slots
    tr.cost["main_updates"] += 1
    tr.cost["main_tokens"] += source_slots
    tr.cost["elapsed_update_seconds"] += time.perf_counter() - start
    tr.cost.setdefault("d1_model_input_positions", 0)
    tr.cost["d1_model_input_positions"] += actual_input_positions
    tr.cost.setdefault("d1_selected_loss_targets", 0)
    tr.cost["d1_selected_loss_targets"] += denominator
    tr.cost.setdefault("d1_source_slots", 0)
    tr.cost["d1_source_slots"] += source_slots
    if operator_counts:
        counts = torch.stack(operator_counts).sum((0, 1)).cpu().tolist()
    else:
        counts = []
    row = {
        "step": tr.step,
        "selected_loss_sum": float(torch.stack(losses).sum()),
        "selected_target_count": denominator,
        "loss": float(torch.stack(losses).sum()) / denominator,
        "cell_gradient_norms": gradient_norms.detach().cpu().tolist(),
        "cell_counts": counts,
        "source_slots": source_slots,
        "model_input_positions": actual_input_positions,
        "context": context,
        "objective": objective,
        "update_seconds": time.perf_counter() - start,
    }
    tr.curve.append(row)
    tr.verify_frozen()
    return row


def advance(tr, batches, *, objective, context, d1_config, schedule, schedule_state):
    """Apply the existing continuous R3m schedule without resetting optimizer state."""
    relative = tr.step - d1_config["anchor_step"]
    lr, epsilon = schedule(
        schedule_state["anchor_lr"],
        schedule_state["anchor_epsilon"],
        relative,
        d1_config,
    )
    engine = tr.c["engine"]["training"]
    previous_lr = engine["learning_rate"]
    previous_epsilon = tr.epsilon

    def fixed_epsilon(self):
        return epsilon

    try:
        engine["learning_rate"] = lr
        tr.epsilon = types.MethodType(fixed_epsilon, tr)
        return update(tr, batches, objective=objective, context=context, d1_config=d1_config)
    finally:
        engine["learning_rate"] = previous_lr
        tr.epsilon = previous_epsilon
