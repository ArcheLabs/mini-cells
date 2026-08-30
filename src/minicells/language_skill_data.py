from __future__ import annotations

from dataclasses import dataclass
import itertools
import random
from typing import Iterable

import torch


PAD = 0
BOS = 1
SEP = 2
EOS = 3
DIGIT_BASE = 16
DIGITS = 10
BASE_TASKS = ("REVERSE", "ROTATE", "INC", "SWAP")
COMPOSITION_MAP = {
    "REVERSE_ROTATE": ("REVERSE", "ROTATE"),
    "REVERSE_INC": ("REVERSE", "INC"),
    "REVERSE_SWAP": ("REVERSE", "SWAP"),
    "ROTATE_INC": ("ROTATE", "INC"),
    "ROTATE_SWAP": ("ROTATE", "SWAP"),
    "INC_SWAP": ("INC", "SWAP"),
}
COMPOSITION_TASKS = tuple(COMPOSITION_MAP)
ALL_TASKS = BASE_TASKS + COMPOSITION_TASKS
TASK_TOKEN_BASE = 4
TASK_TO_TOKEN = {task: TASK_TOKEN_BASE + index for index, task in enumerate(ALL_TASKS)}
TOKEN_TO_TASK = {value: key for key, value in TASK_TO_TOKEN.items()}
VOCAB_SIZE = DIGIT_BASE + DIGITS
DIGIT_COUNT = 6
SEQUENCE_TOKENS = 2 * DIGIT_COUNT + 4
MODEL_LENGTH = SEQUENCE_TOKENS - 1


@dataclass(frozen=True)
class SkillCorpus:
    sequences: torch.Tensor
    task_ids: torch.Tensor
    task_names: tuple[str, ...]
    loss_mask: torch.Tensor

    @property
    def inputs(self) -> torch.Tensor:
        return self.sequences[:, :-1]

    @property
    def targets(self) -> torch.Tensor:
        return self.sequences[:, 1:]


def _reverse(values: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(reversed(values))


def _rotate(values: tuple[int, ...]) -> tuple[int, ...]:
    return values[1:] + values[:1]


def _inc(values: tuple[int, ...]) -> tuple[int, ...]:
    return tuple((value + 1) % DIGITS for value in values)


def _swap(values: tuple[int, ...]) -> tuple[int, ...]:
    out = list(values)
    for index in range(0, len(out) - 1, 2):
        out[index], out[index + 1] = out[index + 1], out[index]
    return tuple(out)


OP = {
    "REVERSE": _reverse,
    "ROTATE": _rotate,
    "INC": _inc,
    "SWAP": _swap,
}


def apply_task(task: str, digits: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in digits)
    if len(values) != DIGIT_COUNT:
        raise ValueError(f"expected {DIGIT_COUNT} digits")
    if any(value < 0 or value >= DIGITS for value in values):
        raise ValueError("digits must be in [0, 9]")
    if task in OP:
        return OP[task](values)
    try:
        left, right = COMPOSITION_MAP[task]
    except KeyError as exc:
        raise ValueError(f"unknown task: {task}") from exc
    return OP[right](OP[left](values))


def encode_example(task: str, digits: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in digits)
    transformed = apply_task(task, values)
    return (
        BOS,
        TASK_TO_TOKEN[task],
        *(DIGIT_BASE + value for value in values),
        SEP,
        *(DIGIT_BASE + value for value in transformed),
        EOS,
    )


def make_loss_mask() -> torch.Tensor:
    sep_index = 2 + DIGIT_COUNT
    target_indices = torch.arange(1, SEQUENCE_TOKENS)
    return target_indices > sep_index


def generate_skill_corpus(
    examples: int,
    *,
    seed: int,
    tasks: tuple[str, ...] = ALL_TASKS,
) -> SkillCorpus:
    if examples < len(tasks):
        raise ValueError("examples must be at least the number of tasks")
    if not tasks or any(task not in TASK_TO_TOKEN for task in tasks):
        raise ValueError("tasks must be a non-empty subset of ALL_TASKS")
    rng = random.Random(seed)
    labels = list(itertools.islice(itertools.cycle(tasks), examples))
    rng.shuffle(labels)
    sequences = []
    task_ids = []
    task_index = {task: index for index, task in enumerate(tasks)}
    for task in labels:
        digits = tuple(rng.randrange(DIGITS) for _ in range(DIGIT_COUNT))
        sequences.append(encode_example(task, digits))
        task_ids.append(task_index[task])
    return SkillCorpus(
        sequences=torch.tensor(sequences, dtype=torch.long),
        task_ids=torch.tensor(task_ids, dtype=torch.long),
        task_names=tuple(tasks),
        loss_mask=make_loss_mask(),
    )


def make_index_schedule(
    examples: int,
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    if examples < 1 or steps < 1 or batch_size < 1:
        raise ValueError("examples, steps, and batch_size must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randint(0, examples, (steps, batch_size), generator=generator)


def batch_from_indices(
    corpus: SkillCorpus,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = corpus.sequences[indices]
    tasks = corpus.task_ids[indices]
    return rows[:, :-1].to(device), rows[:, 1:].to(device), tasks.to(device)
