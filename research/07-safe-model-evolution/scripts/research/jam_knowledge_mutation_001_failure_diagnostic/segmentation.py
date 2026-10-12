from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

SEGMENT_IGNORE = 0
SEGMENT_PREFIX = 1
SEGMENT_CONTENT = 2
SEGMENT_EOS = 3

SEGMENT_NAMES = {
    SEGMENT_PREFIX: "prefix",
    SEGMENT_CONTENT: "canonical_content",
    SEGMENT_EOS: "eos",
}


def _prompt_for(row: Mapping[str, Any], template: str) -> str:
    return template.format(question=str(row["question"]))


def _answer_parts(row: Mapping[str, Any], prefix: str, separator: str) -> tuple[str, str]:
    answer = str(row["answer"])
    expected = prefix + separator
    if not answer.startswith(expected):
        raise RuntimeError(f"row {row['id']} answer does not start with expected diagnostic prefix")
    content = answer[len(expected) :]
    if not content:
        raise RuntimeError(f"row {row['id']} has empty canonical content")
    return answer, content


def encode_segmented_rows(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    prompt_template: str,
    max_length: int,
    device: str,
    prefix: str,
    separator: str = " ",
) -> dict[str, torch.Tensor]:
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("failure diagnostic requires a fast tokenizer with offset_mapping")
    eos = tokenizer.eos_token_id
    bos = getattr(tokenizer, "bos_token_id", None)
    pad = tokenizer.pad_token_id
    if eos is None:
        raise RuntimeError("tokenizer has no eos token")
    if pad is None:
        raise RuntimeError("tokenizer has no pad token")

    sequences: list[list[int]] = []
    labels: list[list[int]] = []
    segments: list[list[int]] = []

    for row in rows:
        prompt = _prompt_for(row, prompt_template)
        prompt_ids: list[int] = []
        if bos is not None:
            prompt_ids.append(int(bos))
        prompt_ids.extend(
            int(value)
            for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )

        answer, _content = _answer_parts(row, prefix, separator)
        encoded = tokenizer(
            answer,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        answer_ids = [int(value) for value in encoded["input_ids"]]
        offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
        if len(answer_ids) != len(offsets):
            raise RuntimeError("tokenizer returned inconsistent offset mapping")
        boundary = len(prefix)

        answer_segments: list[int] = []
        for start, end in offsets:
            if start == end:
                raise RuntimeError(
                    f"zero-width token offset at diagnostic boundary for row {row['id']}"
                )
            if end <= boundary:
                answer_segments.append(SEGMENT_PREFIX)
            elif start >= boundary:
                answer_segments.append(SEGMENT_CONTENT)
            else:
                raise RuntimeError(
                    f"token crosses prefix/content character boundary for row {row['id']}: "
                    f"offset=({start},{end}) boundary={boundary}"
                )

        available = max_length - len(prompt_ids)
        if available <= 1:
            raise RuntimeError(f"prompt exceeds max length for row {row['id']}")
        if len(answer_ids) + 1 > available:
            raise RuntimeError(
                f"diagnostic refuses truncated answer for row {row['id']}; "
                "exact reproduction requires every answer token plus EOS"
            )

        ids = prompt_ids + answer_ids + [int(eos)]
        target = [-100] * len(prompt_ids) + answer_ids + [int(eos)]
        segment = (
            [SEGMENT_IGNORE] * len(prompt_ids)
            + answer_segments
            + [SEGMENT_EOS]
        )
        sequences.append(ids)
        labels.append(target)
        segments.append(segment)

    width = max(len(row) for row in sequences)
    input_ids = torch.full((len(rows), width), int(pad), dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    label_tensor = torch.full((len(rows), width), -100, dtype=torch.long)
    segment_tensor = torch.full((len(rows), width), SEGMENT_IGNORE, dtype=torch.long)
    for index, (ids, target, segment) in enumerate(
        zip(sequences, labels, segments, strict=True)
    ):
        input_ids[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention[index, : len(ids)] = 1
        label_tensor[index, : len(target)] = torch.tensor(target, dtype=torch.long)
        segment_tensor[index, : len(segment)] = torch.tensor(segment, dtype=torch.long)

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention.to(device),
        "labels": label_tensor.to(device),
        "segments": segment_tensor.to(device),
    }


def segmented_metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    segments: torch.Tensor,
) -> tuple[dict[str, dict[str, float]], list[dict[str, dict[str, float]]]]:
    shift_logits = logits[:, :-1].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_segments = segments[:, 1:].contiguous()
    supervised = shift_labels.ne(-100)
    if int(supervised.sum().item()) == 0:
        raise RuntimeError("batch contains no supervised answer tokens")

    flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
    flat_labels = shift_labels.reshape(-1)
    losses = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shift_labels)
    predictions = shift_logits.argmax(dim=-1)

    def summarize(mask: torch.Tensor) -> dict[str, float]:
        count = int(mask.sum().item())
        if count == 0:
            return {"nll_sum": 0.0, "tokens": 0.0, "correct": 0.0}
        return {
            "nll_sum": float(losses[mask].sum().item()),
            "tokens": float(count),
            "correct": float((predictions[mask] == shift_labels[mask]).sum().item()),
        }

    aggregate: dict[str, dict[str, float]] = {}
    for code, name in SEGMENT_NAMES.items():
        aggregate[name] = summarize(supervised & shift_segments.eq(code))
    aggregate["full"] = summarize(supervised)

    per_row: list[dict[str, dict[str, float]]] = []
    for row_index in range(shift_labels.shape[0]):
        row_metrics: dict[str, dict[str, float]] = {}
        for code, name in SEGMENT_NAMES.items():
            mask = torch.zeros_like(supervised)
            mask[row_index] = supervised[row_index] & shift_segments[row_index].eq(code)
            row_metrics[name] = summarize(mask)
        row_mask = torch.zeros_like(supervised)
        row_mask[row_index] = supervised[row_index]
        row_metrics["full"] = summarize(row_mask)
        per_row.append(row_metrics)
    return aggregate, per_row


def finalize_sums(sums: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    finalized: dict[str, dict[str, float]] = {}
    for name, values in sums.items():
        tokens = float(values["tokens"])
        if tokens <= 0:
            raise RuntimeError(f"segment {name} has no tokens")
        mean_nll = float(values["nll_sum"]) / tokens
        finalized[name] = {
            "mean_reference_nll": mean_nll,
            "reference_answer_token_top1_accuracy": float(values["correct"]) / tokens,
            "supervised_tokens": tokens,
            "perplexity": math.exp(min(mean_nll, 20.0)),
        }
    return finalized


def merge_metric_sums(
    target: dict[str, dict[str, float]],
    source: Mapping[str, Mapping[str, float]],
) -> None:
    for name, values in source.items():
        row = target.setdefault(name, {"nll_sum": 0.0, "tokens": 0.0, "correct": 0.0})
        row["nll_sum"] += float(values["nll_sum"])
        row["tokens"] += float(values["tokens"])
        row["correct"] += float(values["correct"])


@torch.no_grad()
def evaluate_segmented_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    prompt_template: str,
    max_length: int,
    device: str,
    batch_size: int,
    prefix: str,
    separator: str = " ",
) -> tuple[dict[str, dict[str, float]], list[dict[str, dict[str, float]]]]:
    totals: dict[str, dict[str, float]] = {}
    all_rows: list[dict[str, dict[str, float]]] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = encode_segmented_rows(
            tokenizer,
            chunk,
            prompt_template=prompt_template,
            max_length=max_length,
            device=device,
            prefix=prefix,
            separator=separator,
        )
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        aggregate, per_row = segmented_metrics_from_logits(
            output.logits,
            batch["labels"],
            batch["segments"],
        )
        merge_metric_sums(totals, aggregate)
        all_rows.extend(per_row)
    return finalize_sums(totals), all_rows


def gain_decomposition(
    base: Mapping[str, Mapping[str, float]],
    mutated: Mapping[str, Mapping[str, float]],
    *,
    original_threshold: float,
) -> dict[str, float]:
    for name in ("prefix", "canonical_content", "eos", "full"):
        if float(base[name]["supervised_tokens"]) != float(mutated[name]["supervised_tokens"]):
            raise RuntimeError(f"token-count mismatch for segment {name}")

    gains = {
        name: float(base[name]["mean_reference_nll"])
        - float(mutated[name]["mean_reference_nll"])
        for name in ("prefix", "canonical_content", "eos", "full")
    }
    prefix_tokens = float(base["prefix"]["supervised_tokens"])
    content_tokens = float(base["canonical_content"]["supervised_tokens"])
    eos_tokens = float(base["eos"]["supervised_tokens"])
    full_tokens = float(base["full"]["supervised_tokens"])
    if abs(prefix_tokens + content_tokens + eos_tokens - full_tokens) > 1e-9:
        raise RuntimeError("segment token counts do not exactly partition full answer")

    content_eos_tokens = content_tokens + eos_tokens
    content_plus_eos_gain = (
        gains["canonical_content"] * content_tokens + gains["eos"] * eos_tokens
    ) / content_eos_tokens
    reconstructed = (
        gains["prefix"] * prefix_tokens
        + gains["canonical_content"] * content_tokens
        + gains["eos"] * eos_tokens
    ) / full_tokens
    required_prefix_gain = (
        original_threshold * full_tokens
        - content_plus_eos_gain * content_eos_tokens
    ) / prefix_tokens

    return {
        "full_reference_nll_gain": gains["full"],
        "prefix_reference_nll_gain": gains["prefix"],
        "canonical_content_reference_nll_gain": gains["canonical_content"],
        "eos_reference_nll_gain": gains["eos"],
        "content_plus_eos_reference_nll_gain": content_plus_eos_gain,
        "prefix_dilution_vs_content_plus_eos": content_plus_eos_gain - gains["full"],
        "reconstructed_full_reference_nll_gain": reconstructed,
        "decomposition_absolute_error": abs(reconstructed - gains["full"]),
        "formal_threshold_reference": float(original_threshold),
        "full_gate_shortfall": float(original_threshold) - gains["full"],
        "content_plus_eos_margin_vs_formal_threshold": content_plus_eos_gain
        - float(original_threshold),
        "required_prefix_gain_to_reach_formal_threshold": required_prefix_gain,
        "prefix_token_fraction": prefix_tokens / full_tokens,
        "canonical_content_token_fraction": content_tokens / full_tokens,
        "eos_token_fraction": eos_tokens / full_tokens,
    }


def classify_capacity_four(
    cases: Sequence[Mapping[str, Any]],
    *,
    original_threshold: float,
) -> str:
    if not cases:
        raise RuntimeError("capacity-four diagnostic has no cases")
    full = [float(row["decomposition"]["full_reference_nll_gain"]) for row in cases]
    content = [
        float(row["decomposition"]["canonical_content_reference_nll_gain"])
        for row in cases
    ]
    content_eos = [
        float(row["decomposition"]["content_plus_eos_reference_nll_gain"])
        for row in cases
    ]
    if all(value < original_threshold for value in full) and all(
        value >= original_threshold for value in content_eos
    ):
        return "FORMULATION_PREFIX_DILUTION_SUFFICIENT_TO_EXPLAIN_CAPACITY4_FORMAL_GATE_FAILURE"
    if all(value < original_threshold for value in full) and all(
        value >= original_threshold for value in content
    ):
        return "NON_CONTENT_TOKENS_DILUTE_ABOVE_THRESHOLD_CANONICAL_CONTENT_AT_CAPACITY4"
    if all(value < original_threshold for value in content):
        return "CANONICAL_CONTENT_GAIN_ALSO_BELOW_ORIGINAL_THRESHOLD_AT_CAPACITY4"
    return "MIXED_CONTENT_AND_FORMULATION_EFFECT_AT_CAPACITY4"
