"""Answer-scored token examples for CLM-0.4-mini language validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Iterable

import torch

from .curriculum import TextExample
if TYPE_CHECKING:
    from .tokenizer import TokenizerBundle


@dataclass(frozen=True)
class ScoredTokenExample:
    example_id: str
    address_id: str
    tokens: tuple[int, ...]
    target_mask: tuple[bool, ...]
    knowledge_key: str | None = None
    prompt_text: str | None = None
    answer_text: str | None = None

    def __post_init__(self) -> None:
        if len(self.tokens) < 2:
            raise ValueError("scored example needs at least two tokens")
        if len(self.target_mask) != len(self.tokens) - 1:
            raise ValueError("target_mask must align with next-token targets")
        if not any(self.target_mask):
            raise ValueError("at least one answer/continuation target must be scored")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["tokens"] = list(self.tokens)
        payload["target_mask"] = list(self.target_mask)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "ScoredTokenExample":
        return cls(
            example_id=str(payload["example_id"]),
            address_id=str(payload["address_id"]),
            tokens=tuple(int(x) for x in payload["tokens"]),
            target_mask=tuple(bool(x) for x in payload["target_mask"]),
            knowledge_key=payload.get("knowledge_key"),
            prompt_text=payload.get("prompt_text"),
            answer_text=payload.get("answer_text"),
        )


def tokenize_text_example(
    example: TextExample,
    tokenizer: TokenizerBundle,
    *,
    max_seq_len: int,
) -> ScoredTokenExample:
    """Tokenize while scoring only answer/continuation targets.

    `max_seq_len` is the model input length, so the complete teacher-forced token
    sequence may contain at most `max_seq_len + 1` tokens.
    """
    prompt_ids = tokenizer.encode(example.prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(example.answer, add_special_tokens=False)
    max_total = int(max_seq_len) + 1
    answer_ids = answer_ids[: max(1, max_total - 2)]
    prompt_budget = max(0, max_total - 2 - len(answer_ids))
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
    tokens = [tokenizer.bos_id, *prompt_ids, *answer_ids, tokenizer.eos_id]
    # targets=tokens[1:]; first answer target is at target index len(prompt_ids).
    target_mask = [False] * len(prompt_ids) + [True] * (len(answer_ids) + 1)
    return ScoredTokenExample(
        example_id=example.example_id,
        address_id=example.address_id,
        tokens=tuple(int(x) for x in tokens),
        target_mask=tuple(bool(x) for x in target_mask),
        knowledge_key=example.knowledge_key,
        prompt_text=example.prompt,
        answer_text=example.answer,
    )


def tokenize_examples(
    examples: Iterable[TextExample],
    tokenizer: TokenizerBundle,
    *,
    max_seq_len: int,
) -> list[ScoredTokenExample]:
    return [
        tokenize_text_example(example, tokenizer, max_seq_len=max_seq_len)
        for example in examples
    ]


def collate_scored(
    examples: list[ScoredTokenExample],
    *,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    if not examples:
        raise ValueError("cannot collate an empty scored batch")
    target_length = max(len(item.tokens) - 1 for item in examples)
    x = torch.full(
        (len(examples), target_length), int(pad_id), dtype=torch.long, device=device
    )
    y = torch.full_like(x, int(pad_id))
    mask = torch.zeros((len(examples), target_length), dtype=torch.bool, device=device)
    addresses: list[str] = []
    for row, item in enumerate(examples):
        length = len(item.tokens) - 1
        x[row, :length] = torch.tensor(item.tokens[:-1], dtype=torch.long, device=device)
        y[row, :length] = torch.tensor(item.tokens[1:], dtype=torch.long, device=device)
        mask[row, :length] = torch.tensor(item.target_mask, dtype=torch.bool, device=device)
        addresses.append(item.address_id)
    return x, y, mask, addresses
