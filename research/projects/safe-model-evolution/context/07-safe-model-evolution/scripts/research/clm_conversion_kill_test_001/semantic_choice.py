from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F


def summarize_candidate_scores(
    rows: Sequence[dict[str, str]],
    candidates: Sequence[str],
    score_rows: Sequence[Sequence[float]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("candidate-choice evaluation requires at least one row")
    if len(score_rows) != len(rows):
        raise ValueError("candidate score rows do not match evaluation rows")
    candidate_list = [str(value) for value in candidates]
    if len(candidate_list) < 2 or len(set(candidate_list)) != len(candidate_list):
        raise ValueError("candidate set must contain at least two unique values")

    details: list[dict[str, Any]] = []
    margins: list[float] = []
    correct_nlls: list[float] = []
    strict_correct = 0
    for row, scores in zip(rows, score_rows, strict=True):
        values = [float(value) for value in scores]
        if len(values) != len(candidate_list):
            raise ValueError("candidate score width does not match candidate set")
        answer = str(row["answer"])
        if answer not in candidate_list:
            raise ValueError(f"reference answer {answer!r} is outside candidate set")
        correct_index = candidate_list.index(answer)
        correct_nll = values[correct_index]
        incorrect = [
            value for index, value in enumerate(values) if index != correct_index
        ]
        best_incorrect_nll = min(incorrect)
        margin = best_incorrect_nll - correct_nll
        predicted_index = min(range(len(values)), key=values.__getitem__)
        passed = margin > 0.0
        strict_correct += int(passed)
        margins.append(margin)
        correct_nlls.append(correct_nll)
        details.append(
            {
                "row_id": str(row["id"]),
                "reference": answer,
                "predicted": candidate_list[predicted_index],
                "strict_correct": passed,
                "correct_candidate_nll": correct_nll,
                "best_incorrect_candidate_nll": best_incorrect_nll,
                "margin": margin,
                "candidate_nlls": {
                    candidate: values[index]
                    for index, candidate in enumerate(candidate_list)
                },
            }
        )

    return {
        "strict_choice_accuracy": strict_correct / len(rows),
        "mean_correct_candidate_nll": sum(correct_nlls) / len(correct_nlls),
        "mean_choice_margin": sum(margins) / len(margins),
        "minimum_choice_margin": min(margins),
        "rows": details,
    }


@torch.no_grad()
def candidate_choice_metrics(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, str]],
    candidates: Sequence[str],
    *,
    protocol: dict[str, Any],
    device: str,
    overlay: Any | None,
    sequence_module: Any,
) -> dict[str, Any]:
    """Rank nonce semantic candidates without EOS supervision.

    Every candidate is teacher-forced against exactly the same question. The
    correct mapping passes only when its answer-token NLL is strictly lower than
    every incorrect candidate. EOS is intentionally excluded so formatting or
    termination shortcuts cannot satisfy this metric.
    """

    task = protocol["sequence_task"]
    candidate_list = [str(value) for value in candidates]
    expanded: list[dict[str, str]] = []
    for row in rows:
        for candidate in candidate_list:
            candidate_row = dict(row)
            candidate_row["id"] = f"{row['id']}.candidate.{candidate}"
            candidate_row["answer"] = candidate
            expanded.append(candidate_row)

    batch_size = int(protocol["evaluation"]["batch_size"])
    flat_scores: list[float] = []
    for start in range(0, len(expanded), batch_size):
        chunk = expanded[start : start + batch_size]
        batch = sequence_module.encode_rows(
            tokenizer,
            chunk,
            prompt_template=task["prompt_template"],
            max_length=int(task["max_sequence_tokens"]),
            device=device,
            append_eos=False,
        )
        cm = overlay.installed(model) if overlay is not None else contextlib.nullcontext()
        with cm:
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
        shift_logits = output.logits[:, :-1].float().contiguous()
        shift_labels = batch["labels"][:, 1:].contiguous()
        mask = shift_labels.ne(-100)
        losses = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(shift_labels.shape)
        token_counts = mask.sum(dim=1).clamp_min(1)
        per_sequence = (losses * mask).sum(dim=1) / token_counts
        flat_scores.extend(float(value) for value in per_sequence.cpu().tolist())

    width = len(candidate_list)
    score_rows = [
        flat_scores[start : start + width]
        for start in range(0, len(flat_scores), width)
    ]
    return summarize_candidate_scores(rows, candidate_list, score_rows)
