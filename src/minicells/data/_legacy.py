from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from ..vocab import CharVocab

WORDS = ("mini", "cells", "jam", "hello", "world", "learn", "echo", "small", "local", "neural")
PUNCTUATION = ("", "", "", ".", ",", "?", "!")


@dataclass(frozen=True)
class CopyBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor
    texts: tuple[str, ...]

    def to(self, device: torch.device | str) -> "CopyBatch":
        return CopyBatch(self.input_ids.to(device), self.target_ids.to(device), self.mask.to(device),
                         self.lengths.to(device), self.texts)


class CopyDataGenerator:
    def __init__(self, vocab: CharVocab, seed: int, min_length: int = 1,
                 max_length: int = 32, num_cells: int = 64,
                 random_fraction: float = 0.7) -> None:
        if not 1 <= min_length <= max_length <= num_cells:
            raise ValueError("require 1 <= min_length <= max_length <= num_cells")
        self.vocab, self.rng = vocab, random.Random(seed)
        self.min_length, self.max_length, self.num_cells = min_length, max_length, num_cells
        self.random_fraction = random_fraction

    def _random_symbols(self, length: int) -> str:
        return "".join(self.rng.choice(self.vocab.SYMBOLS) for _ in range(length))

    def _pseudo_text(self, length: int) -> str:
        parts: list[str] = []
        while len(" ".join(parts)) < length:
            parts.append(self.rng.choice(WORDS))
        text = " ".join(parts)
        if self.rng.random() < 0.35:
            text += self.rng.choice(PUNCTUATION)
        return text[:length]

    def sample_text(self) -> str:
        length = self.rng.randint(self.min_length, self.max_length)
        if self.rng.random() < self.random_fraction:
            return self._random_symbols(length)
        return self._pseudo_text(length)

    def batch(self, batch_size: int, device: torch.device | str = "cpu") -> CopyBatch:
        texts = tuple(self.sample_text() for _ in range(batch_size))
        ids = torch.full((batch_size, self.num_cells), self.vocab.pad_id, dtype=torch.long)
        lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)
        for row, text in enumerate(texts):
            encoded = self.vocab.encode(text)
            ids[row, :len(encoded)] = torch.tensor(encoded)
        mask = torch.arange(self.num_cells).unsqueeze(0) < lengths.unsqueeze(1)
        return CopyBatch(ids.to(device), ids.to(device).clone(), mask.to(device), lengths.to(device), texts)


def fixed_dataset(vocab: CharVocab, seed: int, examples: int, **kwargs) -> CopyBatch:
    return CopyDataGenerator(vocab, seed, **kwargs).batch(examples)
