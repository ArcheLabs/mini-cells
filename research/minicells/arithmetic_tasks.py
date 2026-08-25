from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .continual_learning import PARAMETER_COUNT, TaskBatch, exact_logits, unpack_flat
from .model import EchoModel
from .quantization_localization import q88_int
from .vocab import CharVocab

RANDOM_DIGIT_BASELINE = 0.10


@dataclass(frozen=True)
class ArithmeticExample:
    operation: str
    left: int
    right: int
    answer: int
    expression: str


def all_arithmetic_examples() -> list[ArithmeticExample]:
    examples: list[ArithmeticExample] = []
    for left in range(10):
        for right in range(10):
            if left + right <= 9:
                examples.append(
                    ArithmeticExample("add", left, right, left + right, f"{left}plus{right}?")
                )
            if left >= right:
                examples.append(
                    ArithmeticExample("sub", left, right, left - right, f"{left}minus{right}?")
                )
    assert len(examples) == 110
    return examples


def split_arithmetic_examples(
    seed: int = 4004,
) -> tuple[list[ArithmeticExample], list[ArithmeticExample]]:
    rng = random.Random(seed)
    train: list[ArithmeticExample] = []
    heldout: list[ArithmeticExample] = []
    source = all_arithmetic_examples()
    for operation in ("add", "sub"):
        group = [item for item in source if item.operation == operation]
        rng.shuffle(group)
        heldout.extend(group[:11])
        train.extend(group[11:])
    rng.shuffle(train)
    rng.shuffle(heldout)
    assert len(train) == 88 and len(heldout) == 22
    return train, heldout


def arithmetic_batch(vocab: CharVocab, examples: list[ArithmeticExample]) -> TaskBatch:
    if not examples:
        raise ValueError("examples must not be empty")
    input_ids = torch.zeros((len(examples), 64), dtype=torch.long)
    target_ids = torch.zeros_like(input_ids)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    answer_mask = torch.zeros_like(mask)
    lengths = torch.empty(len(examples), dtype=torch.long)
    for row, item in enumerate(examples):
        encoded = vocab.encode(item.expression)
        target = encoded.copy()
        answer_position = len(encoded) - 1
        target[answer_position] = vocab.token_to_id[str(item.answer)]
        input_ids[row, : len(encoded)] = torch.tensor(encoded)
        target_ids[row, : len(target)] = torch.tensor(target)
        mask[row, : len(encoded)] = True
        answer_mask[row, answer_position] = True
        lengths[row] = len(encoded)
    return TaskBatch(input_ids, target_ids, mask, lengths, answer_mask)


def arithmetic_metrics(logits: torch.Tensor, batch: TaskBatch) -> dict[str, float]:
    pred = logits.argmax(dim=-1).cpu()
    targets = batch.target_ids.cpu()
    active = batch.changed_mask.cpu().bool()
    correct = int(((pred == targets) & active).sum().item())
    total = int(active.sum().item())
    return {"answer_accuracy": correct / total if total else 0.0}


@torch.no_grad()
def evaluate_integer_arithmetic(flat: torch.Tensor, batch: TaskBatch) -> dict[str, float]:
    return arithmetic_metrics(exact_logits(flat, batch.input_ids), batch)


def load_flat_into_float_model(flat: torch.Tensor, vocab_size: int = 44) -> EchoModel:
    model = EchoModel(vocab_size=vocab_size)
    params = unpack_flat(flat.to(dtype=torch.int64, device="cpu"))
    with torch.no_grad():
        model.embedding.weight.copy_(params["embedding"].float() / 256.0)
        model.update_in.weight.copy_(params["update_in_w"].float() / 256.0)
        model.update_in.bias.copy_(params["update_in_b"].float() / 256.0)
        model.update_out.weight.copy_(params["update_out_w"].float() / 256.0)
        model.update_out.bias.copy_(params["update_out_b"].float() / 256.0)
        model.output.weight.copy_(params["output_w"].float() / 256.0)
        model.output.bias.copy_(params["output_b"].float() / 256.0)
    return model


def quantize_float_model(model: EchoModel) -> torch.Tensor:
    ordered = (
        model.embedding.weight,
        model.update_in.weight,
        model.update_in.bias,
        model.update_out.weight,
        model.update_out.bias,
        model.output.weight,
        model.output.bias,
    )
    flat = torch.cat([q88_int(param.detach().cpu()).reshape(-1) for param in ordered])
    if flat.numel() != PARAMETER_COUNT:
        raise ValueError(f"expected {PARAMETER_COUNT} parameters, got {flat.numel()}")
    return flat


def float_arithmetic_loss(model: EchoModel, batch: TaskBatch, device: torch.device) -> torch.Tensor:
    logits = model(batch.input_ids.to(device))
    target = batch.target_ids.to(device)
    active = batch.changed_mask.to(device)
    return F.cross_entropy(logits[active], target[active])


def float_echo_loss(model: EchoModel, batch: TaskBatch, device: torch.device) -> torch.Tensor:
    logits = model(batch.input_ids.to(device))
    target = batch.target_ids.to(device)
    active = batch.mask.to(device)
    return F.cross_entropy(logits[active], target[active])


@torch.no_grad()
def evaluate_float_arithmetic(
    model: EchoModel, batch: TaskBatch, device: torch.device
) -> dict[str, float]:
    return arithmetic_metrics(model(batch.input_ids.to(device)).cpu(), batch)
