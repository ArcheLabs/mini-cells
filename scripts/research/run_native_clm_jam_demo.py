#!/usr/bin/env python3
"""Post-train the canonical Native CLM v0 on JAM Knowledge v0.1.

This is an engineering/demo pipeline. It demonstrates that the already-trained
Native CLM can acquire bounded JAM knowledge after its base training. It does not
claim replay-free continual learning or establish a new scientific result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from minicells.native_clm_train import ByteSequenceDataset, evaluate
from minicells.native_clm_v0 import ByteTokenizer, NativeCLM

BASE_CHECKPOINT_SHA256 = "91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f"
DEFAULT_HF_REPO = "archelabs-org/native-clm-v0"
DEFAULT_HF_SUBDIR = "jam-v0.1"
DEFAULT_DATASET = Path("research/datasets/jam-knowledge-v0.1")
DEFAULT_OUTPUT = Path("artifacts/demos/native-clm-jam-v0.1")


@dataclass(frozen=True)
class JamDemoTrainConfig:
    seed: int = 26090521
    steps: int = 1200
    batch_size: int = 8
    base_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    lr_shared: float = 3e-5
    lr_router: float = 6e-5
    lr_cells: float = 1.2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 60
    min_lr_ratio: float = 0.1
    rehearsal_weight: float = 0.20
    eval_interval: int = 100
    base_eval_batches: int = 20
    precision: str = "fp16"

    def validate(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.batch_size < 1 or self.base_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if min(self.lr_shared, self.lr_router, self.lr_cells) <= 0:
            raise ValueError("learning rates must be positive")
        if not 0.0 <= self.rehearsal_weight <= 1.0:
            raise ValueError("rehearsal_weight must be in [0, 1]")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")


class QADataset(Dataset[dict[str, Tensor]]):
    def __init__(self, rows: list[dict[str, Any]], *, max_seq_len: int) -> None:
        if not rows:
            raise ValueError("QA dataset is empty")
        self.samples = [_encode_qa(row, max_seq_len=max_seq_len) for row in rows]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self.samples[index]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_reasoning(dataset: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((dataset / "evaluation" / "reasoning").glob("*.jsonl")):
        rows.extend(_load_jsonl(path))
    return rows


def _run_dataset_build(dataset: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/research/jam_knowledge_v0_1/build_dataset.py",
            "--dataset",
            os.fspath(dataset),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/research/jam_knowledge_v0_1/validate_dataset.py",
            "--dataset",
            os.fspath(dataset),
        ],
        check=True,
    )


def _format_prompt(question: str) -> str:
    return f"Q: {question.strip()}\nA: "


def _encode_qa(row: dict[str, Any], *, max_seq_len: int) -> dict[str, Tensor]:
    prompt = ByteTokenizer.encode(_format_prompt(str(row["question"])))
    answer = ByteTokenizer.encode(str(row["answer"]).strip() + "\n\n")
    max_tokens = max_seq_len + 1

    if len(prompt) >= max_tokens:
        prompt = prompt[: max_tokens - 2]
    room = max_tokens - len(prompt)
    if room < 2:
        prompt = prompt[: max_tokens // 2]
        room = max_tokens - len(prompt)
    answer = answer[:room]

    sequence = prompt + answer
    if len(sequence) < 2:
        raise ValueError(f"QA row {row.get('id')} encoded to fewer than two tokens")

    x = torch.tensor(sequence[:-1], dtype=torch.long)
    y = torch.tensor(sequence[1:], dtype=torch.long)
    labels = torch.full_like(y, -100)
    answer_start = max(0, len(prompt) - 1)
    labels[answer_start:] = y[answer_start:]
    return {"input_ids": x, "labels": labels}


def _collate_qa(samples: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    width = max(int(sample["input_ids"].numel()) for sample in samples)
    inputs = torch.zeros((len(samples), width), dtype=torch.long)
    labels = torch.full((len(samples), width), -100, dtype=torch.long)
    for index, sample in enumerate(samples):
        length = int(sample["input_ids"].numel())
        inputs[index, :length] = sample["input_ids"]
        labels[index, :length] = sample["labels"]
    return {"input_ids": inputs, "labels": labels}


def _cycle(loader: DataLoader[Any]) -> Iterator[Any]:
    while True:
        yield from loader


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _masked_qa_loss(model: NativeCLM, batch: dict[str, Tensor], device: torch.device) -> Tensor:
    x = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    logits = model(x)["logits"]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


def _make_optimizer(model: NativeCLM, config: JamDemoTrainConfig) -> torch.optim.Optimizer:
    groups = model.parameter_groups()
    return torch.optim.AdamW(
        [
            {
                "params": groups["shared"],
                "lr": config.lr_shared,
                "initial_lr": config.lr_shared,
                "group_name": "shared",
            },
            {
                "params": groups["router"],
                "lr": config.lr_router,
                "initial_lr": config.lr_router,
                "group_name": "router",
            },
            {
                "params": groups["cells"],
                "lr": config.lr_cells,
                "initial_lr": config.lr_cells,
                "group_name": "cells",
            },
        ],
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )


def _lr_factor(step: int, config: JamDemoTrainConfig) -> float:
    if step < config.warmup_steps:
        return max(1e-3, (step + 1) / max(1, config.warmup_steps))
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine


def _set_lrs(
    optimizer: torch.optim.Optimizer,
    step: int,
    config: JamDemoTrainConfig,
) -> None:
    factor = _lr_factor(step, config)
    for group in optimizer.param_groups:
        group["lr"] = float(group["initial_lr"]) * factor


@torch.no_grad()
def evaluate_qa(
    model: NativeCLM,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
) -> dict[str, float | int]:
    dataset = QADataset(rows, max_seq_len=model.config.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_qa,
    )
    was_training = model.training
    model.eval()
    total_nll = 0.0
    correct_tokens = 0
    total_tokens = 0
    exact_rows = 0
    total_rows = 0

    for batch in loader:
        x = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with _autocast(device, precision):
            logits = model(x)["logits"]
        flat_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(labels)
        mask = labels != -100
        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels) & mask

        total_nll += float(flat_loss[mask].sum().cpu())
        correct_tokens += int(correct.sum().cpu())
        total_tokens += int(mask.sum().cpu())
        for row_index in range(labels.size(0)):
            row_mask = mask[row_index]
            if bool(row_mask.any()):
                exact_rows += int(bool(correct[row_index][row_mask].all()))
                total_rows += 1

    if was_training:
        model.train()
    mean_nll = total_nll / max(1, total_tokens)
    return {
        "answer_nll": mean_nll,
        "answer_perplexity": float(math.exp(min(20.0, mean_nll))),
        "answer_token_accuracy": correct_tokens / max(1, total_tokens),
        "teacher_forced_exact_rows": exact_rows,
        "rows": total_rows,
        "teacher_forced_exact_rate": exact_rows / max(1, total_rows),
        "answer_tokens": total_tokens,
    }


def _clean_completion(text: str) -> str:
    text = text.replace("\x00", "").strip()
    for marker in ("\n\n", "\nQ:", "\nQuestion:"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


@torch.no_grad()
def generate_answer(
    model: NativeCLM,
    question: str,
    *,
    device: torch.device,
    max_new_tokens: int = 120,
) -> str:
    prompt = _format_prompt(question)
    prompt_tokens = ByteTokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    output = model.generate(
        prompt_tensor,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    completion = ByteTokenizer.decode(output[0, len(prompt_tokens) :])
    return _clean_completion(completion)


def _select_demo_rows(
    factual_rows: list[dict[str, Any]],
    reasoning_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    preferred = [
        "jam.services.refine",
        "jam.services.accumulate",
        "jam.work.refinement",
        "jam.authorization.authorizer",
        "jam.guarantees.guarantee",
        "jam.serialization.erasure_coding",
    ]
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for concept_id in preferred:
        for row in factual_rows:
            if concept_id in row.get("concept_ids", []) and row["id"] not in used_ids:
                selected.append(row)
                used_ids.add(row["id"])
                break
        if len(selected) >= 4:
            break
    for row in reasoning_rows:
        if len(selected) >= 6:
            break
        selected.append(row)
    return selected[:6]


def _qa_log(
    model: NativeCLM,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
) -> list[dict[str, str]]:
    return [
        {
            "id": str(row["id"]),
            "question": str(row["question"]),
            "reference": str(row["answer"]),
            "model_answer": generate_answer(model, str(row["question"]), device=device),
        }
        for row in rows
    ]


def _write_qa_markdown(
    path: Path,
    *,
    before: list[dict[str, str]],
    after: list[dict[str, str]],
) -> None:
    lines = [
        "# Native CLM v0 — JAM Q&A Log",
        "",
        "The same deterministic prompts are shown before and after JAM post-training.",
        "Free-generation output is illustrative; benchmark metrics are recorded separately.",
        "",
    ]
    for label, rows in (("BEFORE JAM TRAINING", before), ("AFTER JAM TRAINING", after)):
        lines.extend([f"## {label}", ""])
        for row in rows:
            lines.extend(
                [
                    f"### {row['id']}",
                    "",
                    f"**Q:** {row['question']}",
                    "",
                    f"**Reference:** {row['reference']}",
                    "",
                    f"**Model:** {row['model_answer']}",
                    "",
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_training_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_from_hf(repo_id: str, token: str | None, output: Path) -> Path:
    from huggingface_hub import hf_hub_download

    resolved = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="final-model.pt",
            token=token,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if resolved.resolve() != output.resolve():
        output.write_bytes(resolved.read_bytes())
    return output


def _hf_publish(
    *,
    repo_id: str,
    subdir: str,
    output_dir: Path,
    token: str,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = {
        "final-model.pt": output_dir / "final-model.pt",
        "provenance.json": output_dir / "provenance.json",
        "benchmarks.json": output_dir / "benchmarks.json",
        "QA_LOG.md": output_dir / "QA_LOG.md",
        "README.md": output_dir / "HF_README.md",
    }
    for remote_name, local_path in files.items():
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{subdir}/{remote_name}",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Native CLM v0: publish JAM post-training artifact ({subdir})",
        )


def _write_hf_readme(
    path: Path,
    *,
    base_sha: str,
    final_sha: str,
    benchmarks: dict[str, Any],
) -> None:
    before = benchmarks["before"]
    after = benchmarks["after"]
    lines = [
        "# Native CLM v0 — JAM v0.1",
        "",
        "This artifact is the canonical Native CLM v0 M1 checkpoint after bounded",
        "post-training on `jam-knowledge-v0.1`.",
        "",
        "It is a demo/engineering artifact. It does **not** claim replay-free continual",
        "learning or replace the untouched root `final-model.pt`.",
        "",
        f"- Base checkpoint SHA-256: `{base_sha}`",
        f"- JAM checkpoint SHA-256: `{final_sha}`",
        f"- JAM validation answer NLL: `{before['validation']['answer_nll']:.4f}` -> "
        f"`{after['validation']['answer_nll']:.4f}`",
        f"- JAM reasoning answer NLL: `{before['reasoning']['answer_nll']:.4f}` -> "
        f"`{after['reasoning']['answer_nll']:.4f}`",
        f"- Base validation perplexity: `{before['base']['perplexity']:.4f}` -> "
        f"`{after['base']['perplexity']:.4f}`",
        "",
        "See `benchmarks.json`, `provenance.json`, and `QA_LOG.md` in this directory.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-checkpoint", type=Path, default=None)
    parser.add_argument("--base-train-file", type=Path, required=True)
    parser.add_argument("--base-validation-file", type=Path, required=True)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-subdir", default=DEFAULT_HF_SUBDIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--publish-hf", action="store_true")
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _run_dataset_build(args.dataset)

    generated = args.dataset / "generated"
    train_rows = _load_jsonl(generated / "train.jsonl")
    validation_rows = _load_jsonl(generated / "validation.jsonl")
    factual_rows = _load_jsonl(generated / "evaluation" / "factual.jsonl")
    relational_rows = _load_jsonl(generated / "evaluation" / "relational.jsonl")
    misconception_rows = _load_jsonl(generated / "evaluation" / "misconceptions.jsonl")
    reasoning_rows = _load_reasoning(args.dataset)

    hf_token = os.environ.get("HF_TOKEN")
    if args.base_checkpoint is None:
        base_path = _checkpoint_from_hf(
            args.hf_repo,
            hf_token,
            output / "base-final-model.pt",
        )
    else:
        base_path = args.base_checkpoint
    base_sha = _sha256_file(base_path)
    if base_sha != BASE_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Refusing JAM post-training: base checkpoint SHA-256 mismatch. "
            f"Expected {BASE_CHECKPOINT_SHA256}, got {base_sha}"
        )

    config = JamDemoTrainConfig(steps=args.steps, precision=args.precision)
    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model, base_extra = NativeCLM.load_checkpoint(base_path, map_location="cpu")
    model.to(device)

    jam_loader = DataLoader(
        QADataset(train_rows, max_seq_len=model.config.max_seq_len),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=_collate_qa,
    )
    base_train_loader = DataLoader(
        ByteSequenceDataset(args.base_train_file, seq_len=model.config.max_seq_len),
        batch_size=config.base_batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(config.seed + 1),
    )
    base_validation_loader = DataLoader(
        ByteSequenceDataset(args.base_validation_file, seq_len=model.config.max_seq_len),
        batch_size=config.base_batch_size,
        shuffle=False,
        drop_last=False,
    )
    jam_iter = _cycle(jam_loader)
    base_iter = _cycle(base_train_loader)

    before = {
        "validation": evaluate_qa(
            model,
            validation_rows,
            device=device,
            precision=config.precision,
            batch_size=config.batch_size,
        ),
        "factual": evaluate_qa(
            model,
            factual_rows,
            device=device,
            precision=config.precision,
            batch_size=config.batch_size,
        ),
        "relational": evaluate_qa(
            model,
            relational_rows,
            device=device,
            precision=config.precision,
            batch_size=config.batch_size,
        ),
        "misconceptions": evaluate_qa(
            model,
            misconception_rows,
            device=device,
            precision=config.precision,
            batch_size=config.batch_size,
        ),
        "reasoning": evaluate_qa(
            model,
            reasoning_rows,
            device=device,
            precision=config.precision,
            batch_size=config.batch_size,
        ),
        "base": evaluate(
            model,
            base_validation_loader,
            device=device,
            batches=config.base_eval_batches,
            precision=config.precision,
        ),
    }
    demo_rows = _select_demo_rows(factual_rows, reasoning_rows)
    before_qa = _qa_log(model, demo_rows, device=device)

    optimizer = _make_optimizer(model, config)
    scaler_enabled = device.type == "cuda" and config.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    training_log: list[dict[str, Any]] = []
    best_validation = float(before["validation"]["answer_nll"])
    best_step = 0
    best_path = output / "best-jam-model.pt"
    start = time.time()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, config.steps + 1):
        _set_lrs(optimizer, step - 1, config)
        total_step_loss = 0.0
        jam_step_loss = 0.0
        base_step_loss = 0.0

        for _ in range(config.gradient_accumulation_steps):
            jam_batch = next(jam_iter)
            base_x, base_y = next(base_iter)
            base_x = base_x.to(device)
            base_y = base_y.to(device)

            with _autocast(device, config.precision):
                jam_loss = _masked_qa_loss(model, jam_batch, device)
                base_loss = model(base_x, base_y)["loss"]
                total_loss = jam_loss + config.rehearsal_weight * base_loss
                scaled_loss = total_loss / config.gradient_accumulation_steps

            scaler.scale(scaled_loss).backward()
            total_step_loss += float(total_loss.detach().cpu())
            jam_step_loss += float(jam_loss.detach().cpu())
            base_step_loss += float(base_loss.detach().cpu())

        if scaler_enabled:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % 20 == 0 or step == config.steps:
            row: dict[str, Any] = {
                "step": step,
                "loss": total_step_loss / config.gradient_accumulation_steps,
                "jam_loss": jam_step_loss / config.gradient_accumulation_steps,
                "base_loss": base_step_loss / config.gradient_accumulation_steps,
                "lr_shared": optimizer.param_groups[0]["lr"],
                "lr_router": optimizer.param_groups[1]["lr"],
                "lr_cells": optimizer.param_groups[2]["lr"],
                "elapsed_seconds": time.time() - start,
            }
            training_log.append(row)
            print(json.dumps(row, sort_keys=True))

        if step % config.eval_interval == 0 or step == config.steps:
            val = evaluate_qa(
                model,
                validation_rows,
                device=device,
                precision=config.precision,
                batch_size=config.batch_size,
            )
            base_eval = evaluate(
                model,
                base_validation_loader,
                device=device,
                batches=config.base_eval_batches,
                precision=config.precision,
            )
            print(
                json.dumps(
                    {
                        "step": step,
                        "jam_validation_answer_nll": val["answer_nll"],
                        "jam_validation_token_accuracy": val["answer_token_accuracy"],
                        "base_validation_perplexity": base_eval["perplexity"],
                    },
                    sort_keys=True,
                )
            )
            if float(val["answer_nll"]) < best_validation:
                best_validation = float(val["answer_nll"])
                best_step = step
                model.save_checkpoint(
                    best_path,
                    extra={
                        "demo": "native-clm-jam-v0.1",
                        "selected_on": "generated/validation.jsonl answer_nll only",
                        "step": step,
                        "base_checkpoint_sha256": base_sha,
                    },
                )

    if best_step == 0:
        model.save_checkpoint(
            best_path,
            extra={
                "demo": "native-clm-jam-v0.1",
                "selected_on": "final fallback; validation did not improve",
                "step": config.steps,
                "base_checkpoint_sha256": base_sha,
            },
        )
        best_step = config.steps

    model, selected_extra = NativeCLM.load_checkpoint(best_path, map_location="cpu")
    model.to(device)

    after = {
        "validation": evaluate_qa(
            model,
            validation_rows,
            device=device,
            precision=config.precision,
            batch_size=config.batch_size,
        ),
        "factual": evaluate_qa(
            model,
            factual_rows,
            device=device,
            precision=config.precision,
            batch_size=config.batch_size,
        ),
        "relational": evaluate_qa(
            model,
            relational_rows,
            device=device,
            precision=config.precision,
            batch_size=config.batch_size,
        ),
        "misconceptions": evaluate_qa(
            model,
            misconception_rows,
            device=device,
            precision=config.precision,
            batch_size=config.batch_size,
        ),
        "reasoning": evaluate_qa(
            model,
            reasoning_rows,
            device=device,
            precision=config.precision,
            batch_size=config.batch_size,
        ),
        "base": evaluate(
            model,
            base_validation_loader,
            device=device,
            batches=config.base_eval_batches,
            precision=config.precision,
        ),
    }
    after_qa = _qa_log(model, demo_rows, device=device)

    final_path = output / "final-model.pt"
    model.save_checkpoint(
        final_path,
        extra={
            "demo": "native-clm-jam-v0.1",
            "scientific_decision": False,
            "base_checkpoint_sha256": base_sha,
            "selected_step": best_step,
            "train_config": asdict(config),
            "base_checkpoint_extra": base_extra,
            "selection_extra": selected_extra,
        },
    )
    final_sha = _sha256_file(final_path)

    dataset_manifest = args.dataset / "manifest.json"
    benchmarks = {
        "format": "minicells.native-clm-jam-demo.benchmarks.v1",
        "scientific_decision": False,
        "claim": "Native CLM v0 can acquire bounded JAM knowledge after initial training.",
        "nonclaims": [
            "replay-free continual learning is solved",
            "JAM semantic reasoning is generally solved",
            "Cell-specific mechanisms are superior to ordinary fine-tuning",
        ],
        "before": before,
        "after": after,
        "delta": {
            "validation_answer_nll": (
                float(after["validation"]["answer_nll"])
                - float(before["validation"]["answer_nll"])
            ),
            "reasoning_answer_nll": (
                float(after["reasoning"]["answer_nll"])
                - float(before["reasoning"]["answer_nll"])
            ),
            "base_perplexity": (
                float(after["base"]["perplexity"]) - float(before["base"]["perplexity"])
            ),
        },
    }
    provenance = {
        "format": "minicells.native-clm-jam-demo.provenance.v1",
        "git_commit": _git_head(),
        "base_hf_repo": args.hf_repo,
        "base_checkpoint_sha256": base_sha,
        "final_checkpoint_sha256": final_sha,
        "dataset": os.fspath(args.dataset),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest),
        "generated_train_sha256": _sha256_file(generated / "train.jsonl"),
        "generated_validation_sha256": _sha256_file(generated / "validation.jsonl"),
        "reasoning_rows": len(reasoning_rows),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_config": asdict(config),
        "selected_step": best_step,
        "hf_target": f"{args.hf_repo}/{args.hf_subdir}",
        "scientific_decision": False,
    }

    (output / "benchmarks.json").write_text(
        json.dumps(benchmarks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_training_csv(output / "training.csv", training_log)
    _write_qa_markdown(output / "QA_LOG.md", before=before_qa, after=after_qa)
    _write_hf_readme(
        output / "HF_README.md",
        base_sha=base_sha,
        final_sha=final_sha,
        benchmarks=benchmarks,
    )

    summary = {
        "base_checkpoint_sha256": base_sha,
        "final_checkpoint_sha256": final_sha,
        "selected_step": best_step,
        "validation_answer_nll": [
            before["validation"]["answer_nll"],
            after["validation"]["answer_nll"],
        ],
        "reasoning_answer_nll": [
            before["reasoning"]["answer_nll"],
            after["reasoning"]["answer_nll"],
        ],
        "base_perplexity": [before["base"]["perplexity"], after["base"]["perplexity"]],
    }
    print(json.dumps(summary, indent=2))

    if args.publish_hf:
        if not hf_token:
            raise RuntimeError("--publish-hf requires HF_TOKEN")
        _hf_publish(
            repo_id=args.hf_repo,
            subdir=args.hf_subdir,
            output_dir=output,
            token=hf_token,
        )
        print(f"Published JAM artifact to {args.hf_repo}/{args.hf_subdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
