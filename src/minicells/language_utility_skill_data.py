from __future__ import annotations

from dataclasses import dataclass
import itertools
import random

import torch


PAD = 0
BOS = 1
SEP = 2
EOS = 3
TASK_TOKEN_BASE = 32
DIGIT_BASE = 64
DIGITS = 10
INPUT_DIGITS = 6
OUTPUT_DIGITS = 6
SEQUENCE_TOKENS = 1 + 1 + INPUT_DIGITS + 1 + OUTPUT_DIGITS + 1
MODEL_LENGTH = SEQUENCE_TOKENS - 1

SKILL_FAMILIES = (
    "REVERSE_INC",
    "MOD_ADD",
    "PARITY",
    "LOOKUP",
    "DELAY_COPY",
    "LOCAL_RULE",
)
FAMILY_TO_TOKEN = {name: TASK_TOKEN_BASE + index for index, name in enumerate(SKILL_FAMILIES)}
TOKEN_TO_FAMILY = {value: key for key, value in FAMILY_TO_TOKEN.items()}


@dataclass(frozen=True)
class UtilitySkillCorpus:
    sequences: torch.Tensor
    family_ids: torch.Tensor
    family_names: tuple[str, ...]
    loss_mask: torch.Tensor

    @property
    def inputs(self) -> torch.Tensor:
        return self.sequences[:, :-1]

    @property
    def targets(self) -> torch.Tensor:
        return self.sequences[:, 1:]


def _reverse_inc(values: tuple[int, ...]) -> tuple[int, ...]:
    return tuple((value + 1) % DIGITS for value in reversed(values))


def _mod_add(values: tuple[int, ...]) -> tuple[int, ...]:
    total = 0
    out = []
    for value in values:
        total = (total + value) % DIGITS
        out.append(total)
    return tuple(out)


def _parity(values: tuple[int, ...]) -> tuple[int, ...]:
    parity = 0
    out = []
    for value in values:
        parity ^= value & 1
        out.append(parity)
    return tuple(out)


def _delay_copy(values: tuple[int, ...]) -> tuple[int, ...]:
    return values[:3] + values[:3]


def _local_rule(values: tuple[int, ...]) -> tuple[int, ...]:
    out = [values[0]]
    for index in range(1, len(values)):
        previous = values[index - 1]
        current = values[index]
        out.append((current + (previous & 1)) % DIGITS)
    return tuple(out)


def _lookup(values: tuple[int, ...]) -> tuple[int, ...]:
    key_a, value_a, key_b, value_b, query, selector = values
    del selector
    if key_a == key_b or query not in (key_a, key_b):
        raise ValueError("LOOKUP inputs must contain two distinct keys and query one of them")
    value = value_a if query == key_a else value_b
    return (value,) * OUTPUT_DIGITS


TRANSFORMS = {
    "REVERSE_INC": _reverse_inc,
    "MOD_ADD": _mod_add,
    "PARITY": _parity,
    "LOOKUP": _lookup,
    "DELAY_COPY": _delay_copy,
    "LOCAL_RULE": _local_rule,
}


def apply_family(family: str, digits: tuple[int, ...]) -> tuple[int, ...]:
    if family not in TRANSFORMS:
        raise ValueError(f"unknown skill family: {family}")
    values = tuple(int(value) for value in digits)
    if len(values) != INPUT_DIGITS:
        raise ValueError(f"expected {INPUT_DIGITS} digits")
    if any(value < 0 or value >= DIGITS for value in values):
        raise ValueError("digits must be in [0, 9]")
    return TRANSFORMS[family](values)


def _sample_digits(family: str, rng: random.Random) -> tuple[int, ...]:
    if family != "LOOKUP":
        return tuple(rng.randrange(DIGITS) for _ in range(INPUT_DIGITS))
    key_a = rng.randrange(DIGITS)
    key_b = rng.randrange(DIGITS - 1)
    if key_b >= key_a:
        key_b += 1
    value_a = rng.randrange(DIGITS)
    value_b = rng.randrange(DIGITS)
    choose_a = bool(rng.randrange(2))
    query = key_a if choose_a else key_b
    selector = rng.randrange(DIGITS)
    return key_a, value_a, key_b, value_b, query, selector


def encode_example(family: str, digits: tuple[int, ...]) -> tuple[int, ...]:
    transformed = apply_family(family, digits)
    return (
        BOS,
        FAMILY_TO_TOKEN[family],
        *(DIGIT_BASE + value for value in digits),
        SEP,
        *(DIGIT_BASE + value for value in transformed),
        EOS,
    )


def make_loss_mask() -> torch.Tensor:
    sep_position = 2 + INPUT_DIGITS
    target_positions = torch.arange(1, SEQUENCE_TOKENS)
    return target_positions > sep_position


def generate_utility_skill_corpus(
    examples: int,
    *,
    seed: int,
    families: tuple[str, ...] = SKILL_FAMILIES,
) -> UtilitySkillCorpus:
    if not families or any(family not in FAMILY_TO_TOKEN for family in families):
        raise ValueError("families must be a non-empty subset of SKILL_FAMILIES")
    if examples < len(families):
        raise ValueError("examples must be at least the number of families")
    rng = random.Random(seed)
    labels = list(itertools.islice(itertools.cycle(families), examples))
    rng.shuffle(labels)
    family_index = {family: index for index, family in enumerate(families)}
    sequences = []
    ids = []
    for family in labels:
        digits = _sample_digits(family, rng)
        sequences.append(encode_example(family, digits))
        ids.append(family_index[family])
    return UtilitySkillCorpus(
        sequences=torch.tensor(sequences, dtype=torch.long),
        family_ids=torch.tensor(ids, dtype=torch.long),
        family_names=tuple(families),
        loss_mask=make_loss_mask(),
    )


def make_index_schedule(examples: int, *, steps: int, batch_size: int, seed: int) -> torch.Tensor:
    if examples < 1 or steps < 1 or batch_size < 1:
        raise ValueError("examples, steps, and batch_size must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randint(0, examples, (steps, batch_size), generator=generator)


def batch_from_indices(
    corpus: UtilitySkillCorpus,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = corpus.sequences[indices]
    ids = corpus.family_ids[indices]
    return rows[:, :-1].to(device), rows[:, 1:].to(device), ids.to(device)
