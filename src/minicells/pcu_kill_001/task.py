"""Task-sequence construction and answer-only causal-LM training caches."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor


IGNORE_INDEX = -100
TASK_ENCODING_VERSION = "pcu-kill-001-task-sequences-v1"


@dataclass(frozen=True)
class TaskSequences:
    """Padded teacher-forced ``prompt + answer`` sequences for one split."""

    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    loss_mask: Tensor
    sample_ids: tuple[str, ...]
    split: str
    prompts: tuple[str, ...]
    answers: tuple[str, ...]
    prompt_lengths: tuple[int, ...]
    answer_lengths: tuple[int, ...]
    encoding_version: str = TASK_ENCODING_VERSION

    @property
    def manifest_sha256(self) -> str:
        payload = {
            "split": self.split,
            "sample_ids": list(self.sample_ids),
            "input_ids": self.input_ids.cpu().tolist(),
            "labels": self.labels.cpu().tolist(),
            "encoding_version": self.encoding_version,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": "minicells.pcu-kill-001.task-sequences.v1",
            "split": self.split,
            "encoding_version": self.encoding_version,
            "rows": int(self.input_ids.shape[0]),
            "sequence_length": int(self.input_ids.shape[1]),
            "sample_ids": list(self.sample_ids),
            "prompt_lengths": list(self.prompt_lengths),
            "answer_lengths": list(self.answer_lengths),
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class TailTrainingCache:
    """Cached final-block state needed for exact task loss training."""

    mlp_input: Tensor
    pre_mlp_residual: Tensor
    top_k_index: Tensor
    top_k_weights: Tensor
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    loss_mask: Tensor
    sample_ids: tuple[str, ...]
    split: str
    identity: Mapping[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": "minicells.pcu-kill-001.task-cache.v1",
            "split": self.split,
            "rows": int(self.input_ids.shape[0]),
            "sequence_length": int(self.input_ids.shape[1]),
            "dtype": str(self.mlp_input.dtype),
            "sample_ids": list(self.sample_ids),
            "identity": dict(self.identity),
        }


def save_task_cache(cache: TailTrainingCache, directory: Path, shard_rows: int = 32) -> dict[str, Any]:
    """Serialize a task cache with all state needed for exact replay."""
    if shard_rows <= 0:
        raise ValueError("shard_rows must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    rows, width = cache.input_ids.shape
    shards = []
    for index, start in enumerate(range(0, rows, shard_rows)):
        end = min(rows, start + shard_rows)
        route_start, route_end = start * width, end * width
        route_slice = cache.top_k_index[route_start:route_end] if cache.top_k_index.shape[0] == rows * width else cache.top_k_index[start:end]
        weight_slice = cache.top_k_weights[route_start:route_end] if cache.top_k_weights.shape[0] == rows * width else cache.top_k_weights[start:end]
        payload = {
            "mlp_input": cache.mlp_input[start:end].cpu(),
            "pre_mlp_residual": cache.pre_mlp_residual[start:end].cpu(),
            "top_k_index": route_slice.cpu(),
            "top_k_weights": weight_slice.cpu(),
            "input_ids": cache.input_ids[start:end].cpu(),
            "attention_mask": cache.attention_mask[start:end].cpu(),
            "labels": cache.labels[start:end].cpu(),
            "loss_mask": cache.loss_mask[start:end].cpu(),
        }
        path = directory / f"shard-{index:05d}.pt"
        torch.save(payload, path)
        shards.append({"path": path.name, "start": start, "end": end, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {
        "schema": "minicells.pcu-kill-001.task-cache.v1",
        "split": cache.split,
        "rows": rows,
        "sequence_length": width,
        "sample_ids": list(cache.sample_ids),
        "identity": dict(cache.identity),
        "shards": shards,
    }
    (directory / "CACHE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_task_cache(directory: Path, *, expected_identity: Mapping[str, Any] | None = None, device: str | torch.device = "cpu") -> TailTrainingCache:
    """Reload and verify every task-cache shard before it can be consumed."""
    manifest = json.loads((directory / "CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    identity = dict(manifest.get("identity", {}))
    if expected_identity:
        mismatch = [key for key, value in expected_identity.items() if identity.get(key) != value]
        if mismatch:
            raise ValueError(f"task cache identity mismatch: {mismatch}")
    values = []
    for shard in manifest.get("shards", []):
        path = directory / str(shard["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != shard.get("sha256"):
            raise ValueError(f"task cache shard SHA-256 mismatch: {path}")
        values.append(torch.load(path, map_location=device))
    if not values:
        raise ValueError("task cache has no shards")
    cat = lambda key: torch.cat([value[key] for value in values], dim=0)
    return TailTrainingCache(
        mlp_input=cat("mlp_input"),
        pre_mlp_residual=cat("pre_mlp_residual"),
        top_k_index=cat("top_k_index"),
        top_k_weights=cat("top_k_weights"),
        input_ids=cat("input_ids"),
        attention_mask=cat("attention_mask"),
        labels=cat("labels"),
        loss_mask=cat("loss_mask"),
        sample_ids=tuple(str(value) for value in manifest["sample_ids"]),
        split=str(manifest["split"]),
        identity=identity,
    )


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    return [int(value) for value in encoded]


def build_task_sequences(
    tokenizer: Any,
    samples: Sequence[Any],
    split: str,
    *,
    max_length: int = 128,
) -> TaskSequences:
    """Encode every answer token as a supervised label and mask the prompt."""
    if not samples:
        raise ValueError(f"cannot build an empty task split: {split}")
    rows: list[list[int]] = []
    labels: list[list[int]] = []
    prompt_lengths: list[int] = []
    answer_lengths: list[int] = []
    sample_ids: list[str] = []
    prompts: list[str] = []
    answers: list[str] = []
    for sample in samples:
        prompt_ids = _encode(tokenizer, str(sample.prompt), add_special_tokens=True)
        answer_ids = _encode(tokenizer, str(sample.answer), add_special_tokens=False)
        if not prompt_ids or not answer_ids:
            raise ValueError(f"tokenizer produced an empty prompt or answer for {sample.sample_id}")
        sequence = prompt_ids + answer_ids
        if len(sequence) > max_length:
            raise ValueError(f"task sequence exceeds max_length={max_length}: {sample.sample_id}")
        rows.append(sequence)
        labels.append([IGNORE_INDEX] * len(prompt_ids) + answer_ids)
        prompt_lengths.append(len(prompt_ids))
        answer_lengths.append(len(answer_ids))
        sample_ids.append(str(sample.sample_id))
        prompts.append(str(sample.prompt))
        answers.append(str(sample.answer))
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long)
    label_tensor = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, (row, row_labels) in enumerate(zip(rows, labels)):
        length = len(row)
        input_ids[index, :length] = torch.tensor(row, dtype=torch.long)
        label_tensor[index, :length] = torch.tensor(row_labels, dtype=torch.long)
        attention_mask[index, :length] = 1
    return TaskSequences(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=label_tensor,
        loss_mask=label_tensor.ne(IGNORE_INDEX),
        sample_ids=tuple(sample_ids),
        split=str(split),
        prompts=tuple(prompts),
        answers=tuple(answers),
        prompt_lengths=tuple(prompt_lengths),
        answer_lengths=tuple(answer_lengths),
    )


def answer_token_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    """Causal CE where only answer labels (not prompts or padding) contribute."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must have shapes [N,T,V] and [N,T]")
    if labels[:, 1:].ne(IGNORE_INDEX).sum() == 0:
        raise ValueError("task batch has no supervised next-token labels")
    shifted_logits = logits[:, :-1, :].contiguous().reshape(-1, logits.shape[-1])
    shifted_labels = labels[:, 1:].contiguous().reshape(-1)
    return torch.nn.functional.cross_entropy(shifted_logits, shifted_labels, ignore_index=IGNORE_INDEX)


def validate_answer_only_labels(sequences: TaskSequences) -> None:
    """Fail closed if a prompt or padding position became supervised."""
    for row, prompt_length in enumerate(sequences.prompt_lengths):
        if sequences.labels[row, :prompt_length].ne(IGNORE_INDEX).any():
            raise ValueError(f"prompt position is supervised in row {row}")
        valid_length = int(sequences.attention_mask[row].sum())
        if sequences.labels[row, valid_length:].ne(IGNORE_INDEX).any():
            raise ValueError(f"padding position is supervised in row {row}")
        if sequences.labels[row, prompt_length:valid_length].eq(IGNORE_INDEX).all():
            raise ValueError(f"answer position is not supervised in row {row}")


def cache_task_sequences(
    runner: Any,
    sequences: TaskSequences,
    *,
    identity: Mapping[str, Any],
    device: str | torch.device = "cpu",
) -> TailTrainingCache:
    """Run the frozen prefix once and attach task labels/routes to the cache."""
    validate_answer_only_labels(sequences)
    input_ids = sequences.input_ids.to(device)
    attention_mask = sequences.attention_mask.to(device)
    tail = runner.capture(input_ids, attention_mask, sample_ids=sequences.sample_ids)
    if tail.top_k_index is None or tail.top_k_weights is None:
        raise RuntimeError("task cache is missing inherited parent routing decisions")
    cache_identity = dict(identity)
    cache_identity.update({
        "split": sequences.split,
        "dataset_manifest_sha256": identity.get("dataset_manifest_sha256"),
        "task_sequences_sha256": sequences.manifest_sha256,
        "encoding_version": sequences.encoding_version,
        "dtype": str(tail.mlp_input.dtype),
    })
    return TailTrainingCache(
        mlp_input=tail.mlp_input,
        pre_mlp_residual=tail.pre_mlp_residual,
        top_k_index=tail.top_k_index,
        top_k_weights=tail.top_k_weights,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=sequences.labels.to(device),
        loss_mask=sequences.loss_mask.to(device),
        sample_ids=sequences.sample_ids,
        split=sequences.split,
        identity=cache_identity,
    )
