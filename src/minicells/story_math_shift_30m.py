"""Experiment 025 — 30M Story→Math developmental shift helpers.

The public comparison is deliberately simple: a matched 30M Transformer
("LLM") versus a CLM upcycled from the retained 30M TextNCA source. Both enter
the shift phase with fresh optimizer state and receive the exact same
deterministic 90% arithmetic / 10% story schedule.

Growth is budgeted rather than exhaustive. Proposal uses recent routing
utilization only; promotion is decided by a short same-data counterfactual
probation. This avoids repeating the expensive 72-shadow CLM-0.3d mechanism
study at 30M scale.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F

from .clm_growth import ProgressiveGrowthCLM
from .clm_release_benchmark import clm_parameter_breakdown, dense_parameter_breakdown
from .language_30m import (
    BATCH_SIZE,
    CONTEXT_LENGTH,
    MODEL_NAME,
    TRAIN_SEQUENCE_LENGTH,
    build_minicells_30m,
    memmap_batch,
)


EXPERIMENT_ID = "025"
FORMAT = "minicells.story-math-shift-30m.v1"
WORKER_FORMAT = "minicells.story-math-shift-30m-worker.v1"
RESULT_DIR_NAME = "experiment-025-story-math-growth"

SOURCE_007_ARTIFACT = Path(
    "artifacts/experiments/007-minicells-30m/minicells-30m-v0-fp16.pt"
)
SOURCE_007_CHECKPOINTS = Path(
    "artifacts/experiments/007-minicells-30m/checkpoints.csv"
)

SHIFT_TOKENS = 50_000_000
TOKENS_PER_STEP = BATCH_SIZE * TRAIN_SEQUENCE_LENGTH
SHIFT_STEPS = SHIFT_TOKENS // TOKENS_PER_STEP
MATH_FRACTION = 0.90
STORY_FRACTION = 0.10
MIXTURE_PERIOD = 10

# Evaluation points include the ends of the two 2M-token probation windows.
EVAL_TOKENS = (
    0,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    12_000_000,
    15_000_000,
    20_000_000,
    25_000_000,
    27_000_000,
    30_000_000,
    35_000_000,
    40_000_000,
    45_000_000,
    50_000_000,
)
GROWTH_DECISION_TOKENS = (10_000_000, 25_000_000)
PROBATION_TOKENS = 2_000_000
MAX_PROMOTIONS = 2
GROWTH_CALIBRATION_BATCHES = 4
GROWTH_PERCEPTION_CAP = 1_024
GROWTH_MIN_ROUTED_SAMPLES = 512

SHIFT_BASE_LR = 1e-4
SHIFT_WARMUP_STEPS = 500
SHIFT_WEIGHT_DECAY = 0.1
SHIFT_BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0

STORY_VALIDATION_BATCHES = 16
MATH_VALIDATION_BATCHES = 16
VALIDATION_BATCH_SIZE = BATCH_SIZE
MATH_EXACT_EXAMPLES = 256
MATH_EXACT_BATCH_SIZE = 32

MATH_TRAIN_STREAM_TOKENS = 12_000_000
MATH_VALIDATION_STREAM_TOKENS = 1_000_000
MATH_CORPUS_SEED = 25025
MATH_EXACT_SEED = 25125

DOMAIN_SEED = 25225
STORY_START_SEED = 25325
MATH_START_SEED = 25425
STORY_VALIDATION_SEED = 25525
MATH_VALIDATION_SEED = 25625
BOOTSTRAP_SEED = 25725
BOOTSTRAP_SAMPLES = 2_000

# Exploratory 30M controller: weaker than the formal 0.3d promotion gate because
# the purpose here is performance under a bounded budget, not re-proving birth.
PROMOTION_STORY_PPL_RATIO_MAX = 1.01
PROMOTION_BOOTSTRAP_CI = 0.80

EXPECTED_SOURCE_MODEL = MODEL_NAME
EXPECTED_SOURCE_TOKENS = 100_000_000


@dataclass(frozen=True)
class MathExample:
    prompt: str
    answer: str

    @property
    def text(self) -> str:
        return f"{self.prompt} Answer {self.answer}."


@dataclass(frozen=True)
class GrowthProposal:
    stage: int
    expert_id: str
    routed_samples: int
    usage: float
    perceptions: torch.Tensor

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "expert_id": self.expert_id,
            "routed_samples": self.routed_samples,
            "usage": self.usage,
            "perception_samples": int(self.perceptions.shape[0]),
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shift_lr(step: int, *, total_steps: int = SHIFT_STEPS) -> float:
    if step <= 0:
        return 0.0
    if step <= SHIFT_WARMUP_STEPS:
        return SHIFT_BASE_LR * step / SHIFT_WARMUP_STEPS
    progress = (step - SHIFT_WARMUP_STEPS) / max(1, total_steps - SHIFT_WARMUP_STEPS)
    progress = min(max(progress, 0.0), 1.0)
    # Keep 10% of the base LR at the end so the late-growth window still learns.
    multiplier = 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return SHIFT_BASE_LR * multiplier


def _math_example(rng: random.Random) -> MathExample:
    kind = rng.randrange(6)
    if kind == 0:
        a, b = rng.randrange(0, 1000), rng.randrange(0, 1000)
        return MathExample(f"Calculate {a} + {b}.", str(a + b))
    if kind == 1:
        a, b = rng.randrange(0, 1000), rng.randrange(0, 1000)
        hi, lo = max(a, b), min(a, b)
        return MathExample(f"Calculate {hi} - {lo}.", str(hi - lo))
    if kind == 2:
        a, b = rng.randrange(0, 100), rng.randrange(0, 100)
        return MathExample(f"Calculate {a} * {b}.", str(a * b))
    if kind == 3:
        x, a = rng.randrange(0, 1000), rng.randrange(0, 1000)
        return MathExample(f"Solve x + {a} = {x + a}.", f"x = {x}")
    if kind == 4:
        x, a = rng.randrange(0, 1000), rng.randrange(0, 1000)
        return MathExample(f"Solve x - {a} = {x - a}.", f"x = {x}")
    x = rng.randrange(0, 200)
    coefficient = rng.randrange(2, 10)
    offset = rng.randrange(0, 100)
    total = coefficient * x + offset
    return MathExample(
        f"Solve {coefficient} * x + {offset} = {total}.",
        f"x = {x}",
    )


def _write_math_stream(
    tokenizer: object,
    *,
    output_path: Path,
    target_tokens: int,
    seed: int,
) -> int:
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise RuntimeError("tokenizer does not contain <eos>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(target_tokens,))
    rng = random.Random(seed)
    cursor = 0
    examples = 0
    try:
        while cursor < target_tokens:
            encoded = tokenizer.encode(_math_example(rng).text).ids
            if not encoded:
                continue
            encoded.append(int(eos_id))
            take = min(len(encoded), target_tokens - cursor)
            values[cursor : cursor + take] = np.asarray(encoded[:take], dtype=np.uint16)
            cursor += take
            examples += 1
        values.flush()
    finally:
        del values
    return examples


def prepare_math_corpus(cache_dir: Path, tokenizer: object) -> dict[str, object]:
    root = cache_dir / "math-30m-shift"
    root.mkdir(parents=True, exist_ok=True)
    train_path = root / "train.u16"
    validation_path = root / "validation.u16"
    manifest_path = root / "manifest.json"
    expected = {
        "format": "minicells.story-math-shift-arithmetic.v1",
        "train_tokens": MATH_TRAIN_STREAM_TOKENS,
        "validation_tokens": MATH_VALIDATION_STREAM_TOKENS,
        "seed": MATH_CORPUS_SEED,
        "generator": "six-family-integer-arithmetic-v1",
    }
    if train_path.is_file() and validation_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        valid = (
            all(manifest.get(key) == value for key, value in expected.items())
            and train_path.stat().st_size == MATH_TRAIN_STREAM_TOKENS * 2
            and validation_path.stat().st_size == MATH_VALIDATION_STREAM_TOKENS * 2
            and sha256_file(train_path) == manifest.get("train_sha256")
            and sha256_file(validation_path) == manifest.get("validation_sha256")
        )
        if valid:
            return {
                "train_path": train_path,
                "validation_path": validation_path,
                "manifest": manifest,
                "manifest_path": manifest_path,
            }

    train_examples = _write_math_stream(
        tokenizer,
        output_path=train_path,
        target_tokens=MATH_TRAIN_STREAM_TOKENS,
        seed=MATH_CORPUS_SEED,
    )
    validation_examples = _write_math_stream(
        tokenizer,
        output_path=validation_path,
        target_tokens=MATH_VALIDATION_STREAM_TOKENS,
        seed=MATH_CORPUS_SEED + 1,
    )
    manifest = {
        **expected,
        "train_examples": train_examples,
        "validation_examples": validation_examples,
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "train_path": train_path,
        "validation_path": validation_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
    }


def _domain_block(block_index: int) -> tuple[str, ...]:
    values = ["math"] * 9 + ["story"]
    rng = random.Random(DOMAIN_SEED + block_index)
    rng.shuffle(values)
    return tuple(values)


def shift_domain(step: int) -> str:
    if step < 0:
        raise ValueError("step must be non-negative")
    block, offset = divmod(step, MIXTURE_PERIOD)
    return _domain_block(block)[offset]


def _starts_for_step(
    step: int,
    *,
    stream_length: int,
    sequence_length: int,
    batch_size: int,
    seed: int,
) -> tuple[int, ...]:
    high = stream_length - sequence_length - 1
    if high <= 0:
        raise ValueError("training stream is too short")
    rng = random.Random(seed + step * 1_000_003)
    return tuple(rng.randrange(high) for _ in range(batch_size))


def shift_batch(
    step: int,
    *,
    story_stream: np.memmap,
    math_stream: np.memmap,
    device: torch.device,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    domain = shift_domain(step)
    if domain == "math":
        stream = math_stream
        seed = MATH_START_SEED
    else:
        stream = story_stream
        seed = STORY_START_SEED
    starts = _starts_for_step(
        step,
        stream_length=len(stream),
        sequence_length=TRAIN_SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        seed=seed,
    )
    inputs, targets = memmap_batch(stream, starts, TRAIN_SEQUENCE_LENGTH, device)
    return domain, inputs, targets


def fixed_validation_starts(
    stream_length: int,
    *,
    batches: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed)
    high = stream_length - CONTEXT_LENGTH - 1
    if high <= 0:
        raise ValueError("validation stream is too short")
    return tuple(
        tuple(rng.randrange(high) for _ in range(VALIDATION_BATCH_SIZE))
        for _ in range(batches)
    )


def _forward_logits(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    clm_backend: str | None,
) -> torch.Tensor:
    if clm_backend is None:
        return model(inputs).logits
    return model(inputs, execution_backend=clm_backend).logits


@torch.no_grad()
def evaluate_stream(
    model: torch.nn.Module,
    stream: np.memmap,
    starts: Iterable[tuple[int, ...]],
    *,
    device: torch.device,
    clm_backend: str | None,
) -> dict[str, object]:
    was_training = model.training
    model.eval()
    total = 0.0
    tokens = 0
    batch_nlls: list[float] = []
    try:
        for batch_starts in starts:
            inputs, targets = memmap_batch(stream, batch_starts, CONTEXT_LENGTH, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = _forward_logits(model, inputs, clm_backend=clm_backend)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum",
                )
            value = float(loss.item())
            count = int(targets.numel())
            total += value
            tokens += count
            batch_nlls.append(value / count)
    finally:
        model.train(was_training)
    nll = total / max(tokens, 1)
    return {
        "nll": nll,
        "ppl": math.exp(min(nll, 20.0)),
        "tokens": tokens,
        "batch_nlls": batch_nlls,
    }


def build_math_exact_batches(
    tokenizer: object,
    *,
    examples: int = MATH_EXACT_EXAMPLES,
    batch_size: int = MATH_EXACT_BATCH_SIZE,
    seed: int = MATH_EXACT_SEED,
) -> list[dict[str, torch.Tensor]]:
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise RuntimeError("tokenizer does not contain <eos>")
    rng = random.Random(seed)
    rows: list[tuple[list[int], list[int], list[bool]]] = []
    while len(rows) < examples:
        example = _math_example(rng)
        prefix = f"{example.prompt} Answer "
        text = prefix + example.answer
        encoding = tokenizer.encode(text)
        ids = list(encoding.ids)
        offsets = list(encoding.offsets)
        if len(ids) < 2 or len(ids) > CONTEXT_LENGTH:
            continue
        answer_start = len(prefix)
        answer_token_indices = [
            index
            for index, (_start, end) in enumerate(offsets)
            if index > 0 and end > answer_start
        ]
        if not answer_token_indices:
            continue
        inputs = ids[:-1]
        targets = ids[1:]
        mask = [False] * len(targets)
        for token_index in answer_token_indices:
            target_index = token_index - 1
            if 0 <= target_index < len(mask):
                mask[target_index] = True
        if not any(mask):
            continue
        rows.append((inputs, targets, mask))

    batches: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        width = max(len(row[0]) for row in chunk)
        input_tensor = torch.full((len(chunk), width), int(eos_id), dtype=torch.long)
        target_tensor = torch.full((len(chunk), width), int(eos_id), dtype=torch.long)
        mask_tensor = torch.zeros((len(chunk), width), dtype=torch.bool)
        for row_index, (inputs, targets, mask) in enumerate(chunk):
            length = len(inputs)
            input_tensor[row_index, :length] = torch.tensor(inputs, dtype=torch.long)
            target_tensor[row_index, :length] = torch.tensor(targets, dtype=torch.long)
            mask_tensor[row_index, :length] = torch.tensor(mask, dtype=torch.bool)
        batches.append(
            {
                "inputs": input_tensor,
                "targets": target_tensor,
                "answer_mask": mask_tensor,
            }
        )
    return batches


@torch.no_grad()
def evaluate_math_exact(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    clm_backend: str | None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    correct_examples = 0
    total_examples = 0
    correct_tokens = 0
    total_tokens = 0
    try:
        for batch in batches:
            inputs = batch["inputs"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            mask = batch["answer_mask"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = _forward_logits(model, inputs, clm_backend=clm_backend)
            predicted = logits.argmax(dim=-1)
            token_correct = predicted.eq(targets)
            correct_tokens += int((token_correct & mask).sum().item())
            total_tokens += int(mask.sum().item())
            per_example = (token_correct | ~mask).all(dim=1)
            correct_examples += int(per_example.sum().item())
            total_examples += int(per_example.numel())
    finally:
        model.train(was_training)
    return {
        "math_exact_answer_accuracy": correct_examples / max(total_examples, 1),
        "math_answer_token_accuracy": correct_tokens / max(total_tokens, 1),
        "math_exact_examples": float(total_examples),
    }


def evaluate_domains(
    model: torch.nn.Module,
    *,
    story_validation: np.memmap,
    math_validation: np.memmap,
    story_starts: tuple[tuple[int, ...], ...],
    math_starts: tuple[tuple[int, ...], ...],
    math_exact_batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    clm_backend: str | None,
) -> dict[str, object]:
    story = evaluate_stream(
        model,
        story_validation,
        story_starts,
        device=device,
        clm_backend=clm_backend,
    )
    arithmetic = evaluate_stream(
        model,
        math_validation,
        math_starts,
        device=device,
        clm_backend=clm_backend,
    )
    exact = evaluate_math_exact(
        model,
        math_exact_batches,
        device=device,
        clm_backend=clm_backend,
    )
    return {
        "story_nll": float(story["nll"]),
        "story_ppl": float(story["ppl"]),
        "story_batch_nlls": list(story["batch_nlls"]),
        "math_nll": float(arithmetic["nll"]),
        "math_ppl": float(arithmetic["ppl"]),
        "math_batch_nlls": list(arithmetic["batch_nlls"]),
        "joint_balanced_nll": 0.5 * (
            float(story["nll"]) + float(arithmetic["nll"])
        ),
        **exact,
    }


def load_30m_textnca_source(
    artifact_path: str | Path,
    *,
    vocab_size: int,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, object]]:
    path = Path(artifact_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "minicells.language-inference.v1":
        raise RuntimeError(f"unexpected 30M source format: {payload.get('format')!r}")
    if payload.get("model_name") != EXPECTED_SOURCE_MODEL:
        raise RuntimeError(f"unexpected 30M source model: {payload.get('model_name')!r}")
    if int(payload.get("consumed_tokens", -1)) != EXPECTED_SOURCE_TOKENS:
        raise RuntimeError("Experiment 025 requires the retained 100M-token 30M source")
    model = build_minicells_30m(vocab_size)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device), {
        "path": str(path),
        "sha256": sha256_file(path),
        "consumed_tokens": EXPECTED_SOURCE_TOKENS,
        "model_name": EXPECTED_SOURCE_MODEL,
    }


def build_30m_clm(
    artifact_path: str | Path,
    *,
    vocab_size: int,
    device: str | torch.device,
) -> tuple[ProgressiveGrowthCLM, dict[str, object]]:
    source, identity = load_30m_textnca_source(
        artifact_path,
        vocab_size=vocab_size,
        device="cpu",
    )
    clm = ProgressiveGrowthCLM(source).to(device)
    return clm, identity


def parameter_snapshot(model: torch.nn.Module) -> dict[str, int]:
    if isinstance(model, ProgressiveGrowthCLM):
        return {
            **clm_parameter_breakdown(model),
            "program_cells": int(model.expert_count),
        }
    return {
        **dense_parameter_breakdown(model),
        "program_cells": 0,
    }


def collect_growth_proposal(
    model: ProgressiveGrowthCLM,
    *,
    story_stream: np.memmap,
    math_stream: np.memmap,
    start_step: int,
    device: torch.device,
    execution_backend: str = "batched_dense",
) -> tuple[GrowthProposal | None, list[dict[str, object]]]:
    banks = [stage.program_bank for stage in model.stages]
    for bank in banks:
        bank.begin_pressure_collection(cap=GROWTH_PERCEPTION_CAP)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for offset in range(GROWTH_CALIBRATION_BATCHES):
                _, inputs, _ = shift_batch(
                    start_step + offset,
                    story_stream=story_stream,
                    math_stream=math_stream,
                    device=device,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    model(inputs, execution_backend=execution_backend)
    finally:
        for bank in banks:
            bank.end_pressure_collection()
        model.train(was_training)

    rows: list[dict[str, object]] = []
    proposals: list[GrowthProposal] = []
    for bank in banks:
        stage_total = sum(
            max(0, int(bank.last_route_counts.get(expert_id, 0)))
            for expert_id in bank.expert_ids
        )
        for expert_id in bank.expert_ids:
            count = int(bank.last_route_counts.get(expert_id, 0))
            pieces = list(bank.last_perceptions.get(expert_id, []))
            perceptions = (
                torch.cat(pieces, dim=0)
                if pieces
                else torch.empty((0, model.stages[0].gru.hidden_size))
            )
            usage = count / max(stage_total, 1)
            eligible = (
                count >= GROWTH_MIN_ROUTED_SAMPLES
                and perceptions.shape[0] >= GROWTH_MIN_ROUTED_SAMPLES
            )
            row = {
                "stage": int(bank.stage),
                "expert_id": expert_id,
                "routed_samples": count,
                "usage": usage,
                "perception_samples": int(perceptions.shape[0]),
                "eligible": bool(eligible),
            }
            rows.append(row)
            if eligible:
                proposals.append(
                    GrowthProposal(
                        int(bank.stage),
                        expert_id,
                        count,
                        usage,
                        perceptions[:GROWTH_PERCEPTION_CAP].contiguous(),
                    )
                )
    proposals.sort(
        key=lambda item: (-item.usage, -item.routed_samples, item.stage, item.expert_id)
    )
    rows.sort(
        key=lambda item: (-float(item["usage"]), int(item["stage"]), str(item["expert_id"]))
    )
    return (proposals[0] if proposals else None), rows


def _bootstrap_balanced_utility(
    control: dict[str, object],
    shadow: dict[str, object],
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, float]:
    control_story = np.asarray(control["story_batch_nlls"], dtype=np.float64)
    shadow_story = np.asarray(shadow["story_batch_nlls"], dtype=np.float64)
    control_math = np.asarray(control["math_batch_nlls"], dtype=np.float64)
    shadow_math = np.asarray(shadow["math_batch_nlls"], dtype=np.float64)
    if (
        control_story.shape != shadow_story.shape
        or control_math.shape != shadow_math.shape
        or control_story.size == 0
        or control_math.size == 0
    ):
        raise ValueError("paired evaluation batches are required for probation")

    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        story_indices = rng.integers(0, control_story.size, size=control_story.size)
        math_indices = rng.integers(0, control_math.size, size=control_math.size)
        story_gain = (
            control_story[story_indices] - shadow_story[story_indices]
        ).mean()
        math_gain = (control_math[math_indices] - shadow_math[math_indices]).mean()
        values[index] = 0.5 * (story_gain + math_gain)

    alpha = 1.0 - PROMOTION_BOOTSTRAP_CI
    mean = 0.5 * (
        float(control_story.mean() - shadow_story.mean())
        + float(control_math.mean() - shadow_math.mean())
    )
    return {
        "mean_balanced_nll_gain": mean,
        "ci_level": PROMOTION_BOOTSTRAP_CI,
        "ci_low": float(np.quantile(values, alpha / 2.0)),
        "ci_high": float(np.quantile(values, 1.0 - alpha / 2.0)),
    }


def promotion_decision(
    control: dict[str, object],
    shadow: dict[str, object],
    *,
    seed: int,
) -> dict[str, object]:
    utility = _bootstrap_balanced_utility(control, shadow, seed=seed)
    story_ratio = float(shadow["story_ppl"]) / max(float(control["story_ppl"]), 1e-12)
    math_nll_gain = float(control["math_nll"]) - float(shadow["math_nll"])
    promote = bool(
        utility["mean_balanced_nll_gain"] > 0.0
        and utility["ci_low"] > 0.0
        and math_nll_gain >= 0.0
        and story_ratio <= PROMOTION_STORY_PPL_RATIO_MAX
    )
    return {
        "promote": promote,
        "rule": (
            "80% paired-bootstrap LCB balanced NLL gain > 0; "
            "math NLL not worse; story PPL ratio <= 1.01"
        ),
        "story_ppl_ratio": story_ratio,
        "math_nll_gain": math_nll_gain,
        "math_exact_accuracy_delta": (
            float(shadow["math_exact_answer_accuracy"])
            - float(control["math_exact_answer_accuracy"])
        ),
        **utility,
    }


def pareto_crossover(
    llm_rows: Iterable[dict[str, object]],
    clm_rows: Iterable[dict[str, object]],
) -> dict[str, object] | None:
    llm = {int(row["shift_tokens"]): row for row in llm_rows}
    clm = {int(row["shift_tokens"]): row for row in clm_rows}
    for tokens in sorted(set(llm) & set(clm)):
        if tokens <= 0:
            continue
        left = llm[tokens]
        right = clm[tokens]
        math_ok = float(right["math_exact_answer_accuracy"]) >= float(
            left["math_exact_answer_accuracy"]
        )
        story_ok = float(right["story_ppl"]) <= float(left["story_ppl"])
        strict = (
            float(right["math_exact_answer_accuracy"])
            > float(left["math_exact_answer_accuracy"]) + 1e-12
            or float(right["story_ppl"]) < float(left["story_ppl"]) - 1e-12
        )
        if math_ok and story_ok and strict:
            return {
                "shift_tokens": tokens,
                "total_experience_tokens": EXPECTED_SOURCE_TOKENS + tokens,
                "llm_story_ppl": float(left["story_ppl"]),
                "clm_story_ppl": float(right["story_ppl"]),
                "llm_math_exact_answer_accuracy": float(
                    left["math_exact_answer_accuracy"]
                ),
                "clm_math_exact_answer_accuracy": float(
                    right["math_exact_answer_accuracy"]
                ),
                "definition": (
                    "CLM math exact-answer accuracy >= LLM and "
                    "CLM story PPL <= LLM, with at least one strict inequality"
                ),
            }
    return None


def schedule_manifest(shift_tokens: int = SHIFT_TOKENS) -> dict[str, object]:
    if shift_tokens % TOKENS_PER_STEP:
        raise ValueError("shift_tokens must align to whole training steps")
    steps = shift_tokens // TOKENS_PER_STEP
    math_steps = sum(1 for step in range(steps) if shift_domain(step) == "math")
    story_steps = steps - math_steps
    digest = hashlib.sha256()
    for step in range(min(steps, 10_000)):
        digest.update(f"{step}:{shift_domain(step)};".encode("utf-8"))
    return {
        "format": FORMAT,
        "shift_tokens": shift_tokens,
        "tokens_per_step": TOKENS_PER_STEP,
        "steps": steps,
        "math_steps": math_steps,
        "story_steps": story_steps,
        "math_fraction": math_steps / max(steps, 1),
        "story_fraction": story_steps / max(steps, 1),
        "schedule_prefix_10k_sha256": digest.hexdigest(),
        "domain_seed": DOMAIN_SEED,
        "story_start_seed": STORY_START_SEED,
        "math_start_seed": MATH_START_SEED,
    }


def experiment_budget() -> dict[str, object]:
    return {
        "hardware": "Tesla T4 x2",
        "available_wall_hours_approx": 10.0,
        "worker_soft_wall_limit_hours": 9.25,
        "llm_pretrain": (
            "reproduce 100M TinyStories Experiment-007 Transformer on GPU0"
        ),
        "clm_pretrain": (
            "reuse retained 100M TinyStories 30M TextNCA artifact on GPU1"
        ),
        "shift_tokens_per_arm": SHIFT_TOKENS,
        "probation_tokens_per_decision": PROBATION_TOKENS,
        "growth_decisions": list(GROWTH_DECISION_TOKENS),
        "max_promotions": MAX_PROMOTIONS,
        "shift_optimizer_reset_both_arms": True,
        "main_training_objective": "standard next-token cross entropy only",
    }
