"""Deterministic multi-token autoregressive evaluation for PCU-KILL-001."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


IDENTIFIER_RE = re.compile(r"[UVW][A-Z2-9]{4}")


@dataclass(frozen=True)
class SampleEvaluation:
    sample_id: str
    expected: str
    generated: str
    exact: bool
    relay_exact: bool | None = None
    terminal_exact: bool | None = None
    both_exact: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "expected": self.expected,
            "generated": self.generated,
            "exact": self.exact,
            "relay_exact": self.relay_exact,
            "terminal_exact": self.terminal_exact,
            "both_exact": self.both_exact,
        }


@dataclass(frozen=True)
class EvaluationSummary:
    split: str
    exact: float
    relay_exact: float | None
    terminal_exact: float | None
    both_exact: float | None
    rows: tuple[SampleEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "exact": self.exact,
            "relay_exact": self.relay_exact,
            "terminal_exact": self.terminal_exact,
            "both_exact": self.both_exact,
            "rows": [row.to_dict() for row in self.rows],
        }


def _encode(tokenizer: Any, text: str) -> list[int]:
    value = tokenizer.encode(text, add_special_tokens=True)
    if hasattr(value, "ids"):
        value = value.ids
    return [int(item) for item in value]


def greedy_generate(model: Any, tokenizer: Any, prompt: str, *, device: str | torch.device, max_new_tokens: int) -> str:
    """Greedy generation with frozen sampling settings and no hidden shortcuts."""
    input_ids = torch.tensor([_encode(tokenizer, prompt)], dtype=torch.long, device=device)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            attention = torch.ones_like(input_ids)
            output = model(input_ids=input_ids, attention_mask=attention)
            logits = getattr(output, "logits", output)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if not isinstance(logits, Tensor):
                raise RuntimeError("model output has no logits")
            next_id = int(logits[0, -1].argmax())
            generated.append(next_id)
            input_ids = torch.cat((input_ids, torch.tensor([[next_id]], device=device)), dim=1)
            if eos_id is not None and next_id == int(eos_id):
                break
    return str(tokenizer.decode(generated, skip_special_tokens=True)).strip()


def evaluate_samples(
    model: Any,
    tokenizer: Any,
    samples: Sequence[Any],
    *,
    split: str,
    device: str | torch.device,
    max_new_tokens: int = 16,
) -> EvaluationSummary:
    rows: list[SampleEvaluation] = []
    is_composition = str(split).startswith("AB")
    for sample in samples:
        generated = greedy_generate(model, tokenizer, sample.prompt, device=device, max_new_tokens=max_new_tokens)
        expected = str(sample.answer).strip()
        identifiers = IDENTIFIER_RE.findall(generated.upper())
        expected_ids = IDENTIFIER_RE.findall(expected.upper())
        if is_composition:
            relay_exact = len(identifiers) >= 1 and len(expected_ids) >= 1 and identifiers[0] == expected_ids[0]
            terminal_exact = len(identifiers) >= 2 and len(expected_ids) >= 2 and identifiers[1] == expected_ids[1]
            both_exact = bool(relay_exact and terminal_exact)
            exact = both_exact
        else:
            relay_exact = terminal_exact = both_exact = None
            # Direct capability is exact identifier recovery, while allowing
            # harmless decoder whitespace or a short textual wrapper.
            exact = bool(expected_ids) and bool(identifiers) and identifiers[0] == expected_ids[0]
        rows.append(SampleEvaluation(str(sample.sample_id), expected, generated, exact, relay_exact, terminal_exact, both_exact))
    n = max(1, len(rows))
    return EvaluationSummary(
        split=str(split),
        exact=sum(row.exact for row in rows) / n,
        relay_exact=(sum(bool(row.relay_exact) for row in rows) / n if is_composition else None),
        terminal_exact=(sum(bool(row.terminal_exact) for row in rows) / n if is_composition else None),
        both_exact=(sum(bool(row.both_exact) for row in rows) / n if is_composition else None),
        rows=tuple(rows),
    )


def evaluate_matrix(
    models: Mapping[str, Any],
    tokenizer: Any,
    splits: Mapping[str, Sequence[Any]],
    *,
    device: str | torch.device,
    max_new_tokens: int = 16,
) -> dict[str, dict[str, EvaluationSummary]]:
    matrix: dict[str, dict[str, EvaluationSummary]] = {}
    for model_name, model in models.items():
        matrix[model_name] = {}
        for split_name in ("A_eval", "B_eval", "AB_eval"):
            matrix[model_name][split_name] = evaluate_samples(
                model, tokenizer, splits.get(split_name, ()), split=split_name, device=device, max_new_tokens=max_new_tokens
            )
    return matrix


def matrix_to_dict(matrix: Mapping[str, Mapping[str, EvaluationSummary]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {model: {split: summary.to_dict() for split, summary in values.items()} for model, values in matrix.items()}


def matrix_accuracy(matrix: Mapping[str, Mapping[str, EvaluationSummary]], model: str, split: str) -> float:
    return float(matrix[model][split].both_exact if split == "AB_eval" else matrix[model][split].exact)
