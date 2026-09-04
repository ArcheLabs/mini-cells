from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F


def prompt_for(row: Mapping[str, Any], template: str) -> str:
    return template.format(question=str(row["question"]))


def encode_rows(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    prompt_template: str,
    max_length: int,
    device: str,
    append_eos: bool = True,
) -> dict[str, torch.Tensor]:
    sequences: list[list[int]] = []
    labels: list[list[int]] = []
    eos = tokenizer.eos_token_id
    if append_eos and eos is None:
        raise RuntimeError("tokenizer has no eos token")
    for row in rows:
        prompt = prompt_for(row, prompt_template)
        prompt_ids = list(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        answer_ids = list(tokenizer(str(row["answer"]), add_special_tokens=False)["input_ids"])
        if append_eos:
            answer_ids.append(int(eos))
        available = max_length - len(prompt_ids)
        if available <= 0:
            raise RuntimeError(f"prompt exceeds max length for row {row['id']}")
        answer_ids = answer_ids[:available]
        if not answer_ids:
            raise RuntimeError(f"answer has no supervised token for row {row['id']}")
        ids = prompt_ids + answer_ids
        sequences.append(ids)
        labels.append([-100] * len(prompt_ids) + answer_ids)

    pad = tokenizer.pad_token_id
    if pad is None:
        raise RuntimeError("tokenizer has no pad token")
    width = max(len(row) for row in sequences)
    input_ids = torch.full((len(rows), width), int(pad), dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    label_tensor = torch.full((len(rows), width), -100, dtype=torch.long)
    for index, (ids, target) in enumerate(zip(sequences, labels, strict=True)):
        input_ids[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention[index, : len(ids)] = 1
        label_tensor[index, : len(target)] = torch.tensor(target, dtype=torch.long)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention.to(device),
        "labels": label_tensor.to(device),
    }


def answer_loss_from_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, int, int]:
    shift_logits = logits[:, :-1].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(-100)
    token_count = int(mask.sum().item())
    if token_count == 0:
        raise RuntimeError("batch contains no supervised answer tokens")
    flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
    flat_labels = shift_labels.reshape(-1)
    losses = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="none")
    loss = losses[flat_labels.ne(-100)].mean()
    predictions = flat_logits.argmax(dim=-1)
    correct = int(
        (predictions[flat_labels.ne(-100)] == flat_labels[flat_labels.ne(-100)]).sum().item()
    )
    return loss, token_count, correct


def answer_loss(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    prompt_template: str,
    max_length: int,
    device: str,
) -> torch.Tensor:
    batch = encode_rows(
        tokenizer,
        rows,
        prompt_template=prompt_template,
        max_length=max_length,
        device=device,
    )
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    loss, _count, _correct = answer_loss_from_logits(output.logits, batch["labels"])
    return loss


@torch.no_grad()
def evaluate_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    prompt_template: str,
    max_length: int,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    total_nll = 0.0
    total_tokens = 0
    total_correct = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = encode_rows(
            tokenizer,
            chunk,
            prompt_template=prompt_template,
            max_length=max_length,
            device=device,
        )
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        loss, count, correct = answer_loss_from_logits(output.logits, batch["labels"])
        total_nll += float(loss.item()) * count
        total_tokens += count
        total_correct += correct
    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "mean_reference_nll": mean_nll,
        "reference_answer_token_top1_accuracy": total_correct / max(total_tokens, 1),
        "supervised_tokens": float(total_tokens),
        "mean_sequence_logprob": -mean_nll,
        "perplexity": math.exp(min(mean_nll, 20.0)),
    }


def coordinate_gradient_energy(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    prompt_template: str,
    max_length: int,
    device: str,
    batch_size: int,
    gate_up: tuple[str, torch.nn.Parameter],
    down: tuple[str, torch.nn.Parameter],
    group_size: int,
) -> torch.Tensor:
    gate_param = gate_up[1]
    down_param = down[1]
    intermediate = int(down_param.shape[2])
    if intermediate % group_size:
        raise RuntimeError("group size does not divide expert intermediate width")
    groups = intermediate // group_size
    experts = int(gate_param.shape[0])
    accumulator = torch.zeros((experts, groups), dtype=torch.float64)
    batches = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        model.zero_grad(set_to_none=True)
        loss = answer_loss(
            model,
            tokenizer,
            chunk,
            prompt_template=prompt_template,
            max_length=max_length,
            device=device,
        )
        loss.backward()
        if gate_param.grad is None or down_param.grad is None:
            raise RuntimeError("packed expert tensors did not receive gradients")
        gate = gate_param.grad.detach().float().reshape(experts, 2, intermediate, -1)
        down_grad = down_param.grad.detach().float()
        gate_energy = gate.square().mean(dim=(1, 3))
        down_energy = down_grad.square().mean(dim=1)
        channel_energy = gate_energy + down_energy
        group_energy = channel_energy.reshape(experts, groups, group_size).mean(dim=2)
        accumulator += group_energy.double().cpu()
        batches += 1
    model.zero_grad(set_to_none=True)
    return (accumulator / max(batches, 1)).float()


def selected_gradient_norm(
    gate_grad: torch.Tensor,
    down_grad: torch.Tensor,
    coordinates: Sequence[tuple[int, int]],
    *,
    group_size: int,
) -> float:
    intermediate = int(down_grad.shape[2])
    total = 0.0
    for expert, group in coordinates:
        start = group * group_size
        end = start + group_size
        total += float(gate_grad[expert, start:end].float().square().sum().item())
        total += float(
            gate_grad[expert, intermediate + start : intermediate + end]
            .float()
            .square()
            .sum()
            .item()
        )
        total += float(down_grad[expert, :, start:end].float().square().sum().item())
    return math.sqrt(total)


def apply_selected_gradients_(
    gate_param: torch.nn.Parameter,
    down_param: torch.nn.Parameter,
    coordinates: Sequence[tuple[int, int]],
    *,
    group_size: int,
    learning_rate: float,
    grad_scale: float,
) -> None:
    if gate_param.grad is None or down_param.grad is None:
        raise RuntimeError("packed expert tensors did not receive gradients")
    intermediate = int(down_param.shape[2])
    with torch.no_grad():
        for expert, group in coordinates:
            start = group * group_size
            end = start + group_size
            gate_param[expert, start:end].add_(
                gate_param.grad[expert, start:end], alpha=-learning_rate * grad_scale
            )
            gate_param[expert, intermediate + start : intermediate + end].add_(
                gate_param.grad[expert, intermediate + start : intermediate + end],
                alpha=-learning_rate * grad_scale,
            )
            down_param[expert, :, start:end].add_(
                down_param.grad[expert, :, start:end], alpha=-learning_rate * grad_scale
            )
