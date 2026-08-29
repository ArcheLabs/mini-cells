"""Experiment 026 — 30M tissue granularity differentiation helpers.

Every arm starts from the same retained 30M TextNCA checkpoint, receives the
same function-preserving fixed-root CLM upcycle, and partitions every existing
FFN tissue into G micro-cells. Persistent growth is disabled. Experiment 026
combines this structural granularity with one shared local-plasticity rule so
G controls the resolution at which adaptation can be regulated while age-zero
function and trainable parameter count remain fixed.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .developmental_tissue import (
    StressObservation,
    TissueConfig,
    TissueFFN,
    convert_model_experts_to_tissues,
)
from .language_30m import BATCH_SIZE, CONTEXT_LENGTH, TRAIN_SEQUENCE_LENGTH, memmap_batch
from .story_math_shift_30m import (
    EXPECTED_SOURCE_TOKENS,
    SOURCE_007_ARTIFACT,
    build_30m_clm,
    evaluate_stream,
    prepare_math_corpus,
)

EXPERIMENT_ID = "026"
FORMAT = "minicells.cell-granularity-30m.v2"
WORKER_FORMAT = "minicells.cell-granularity-30m-worker.v2"
RESULT_DIR_NAME = "experiment-026-cell-granularity"

GRANULARITIES = (1, 2, 4, 8)
DOMAINS = ("story", "math", "symbolic", "facts")
CONTINUATION_TOKENS = 20_000_000
TOKENS_PER_STEP = BATCH_SIZE * TRAIN_SEQUENCE_LENGTH
CONTINUATION_STEPS = CONTINUATION_TOKENS // TOKENS_PER_STEP
EVAL_TOKENS = (
    0,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    15_000_000,
    20_000_000,
)

BASE_LR = 1e-4
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0

SYNTHETIC_TRAIN_TOKENS = 6_000_000
SYNTHETIC_VALIDATION_TOKENS = 500_000
VALIDATION_BATCHES = 8
DIAGNOSTIC_PERCEPTION_CAP = 256

MIXTURE_SEED = 26026
START_SEEDS = {
    "story": 26126,
    "math": 26226,
    "symbolic": 26326,
    "facts": 26426,
}
VALIDATION_SEEDS = {
    "story": 26526,
    "math": 26626,
    "symbolic": 26726,
    "facts": 26826,
}
SYNTHETIC_SEEDS = {"symbolic": 26926, "facts": 27026}

DIFFERENTIATION_MIN_GAIN_DELTA = 0.05
PERFORMANCE_NLL_RATIO_MAX = 1.02
STABILITY_MIN_COSINE = 0.80


@dataclass(frozen=True)
class SyntheticCorpus:
    train_path: Path
    validation_path: Path
    manifest_path: Path
    manifest: dict[str, object]


@dataclass(frozen=True)
class DiagnosticBatch:
    domain: str
    inputs: torch.Tensor
    targets: torch.Tensor


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def continuation_lr(step: int, *, total_steps: int = CONTINUATION_STEPS) -> float:
    if step <= 0:
        return 0.0
    if step <= WARMUP_STEPS:
        return BASE_LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    progress = min(max(progress, 0.0), 1.0)
    multiplier = 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return BASE_LR * multiplier


def domain_for_step(step: int) -> str:
    if step < 0:
        raise ValueError("step must be non-negative")
    block, offset = divmod(step, len(DOMAINS))
    values = list(DOMAINS)
    random.Random(MIXTURE_SEED + block).shuffle(values)
    return values[offset]


def schedule_manifest(tokens: int = CONTINUATION_TOKENS) -> dict[str, object]:
    if tokens <= 0 or tokens % TOKENS_PER_STEP != 0:
        raise ValueError("tokens must be a positive multiple of TOKENS_PER_STEP")
    steps = tokens // TOKENS_PER_STEP
    counts = {domain: 0 for domain in DOMAINS}
    for step in range(steps):
        counts[domain_for_step(step)] += TOKENS_PER_STEP
    return {
        "format": FORMAT,
        "tokens": tokens,
        "steps": steps,
        "tokens_per_step": TOKENS_PER_STEP,
        "domains": list(DOMAINS),
        "domain_tokens": counts,
        "mixture_rule": (
            "one occurrence from every domain per four-step block; "
            "block order deterministically shuffled"
        ),
        "mixture_seed": MIXTURE_SEED,
    }


def _symbolic_text(rng: random.Random) -> str:
    values = [rng.randrange(0, 100) for _ in range(rng.randrange(4, 8))]
    source = " ".join(str(value) for value in values)
    kind = rng.randrange(3)
    if kind == 0:
        target = " ".join(str(value) for value in reversed(values))
        return f"Sequence {source}. Reverse gives {target}."
    if kind == 1:
        target = " ".join(str(value) for value in sorted(values))
        return f"Sequence {source}. Sort ascending gives {target}."
    shift = rng.randrange(1, 10)
    target = " ".join(str(value + shift) for value in values)
    return f"Sequence {source}. Add {shift} to each item gives {target}."


def _facts_text(rng: random.Random) -> str:
    syllables = ("la", "mi", "no", "ra", "te", "vi", "su", "ke")
    name = "".join(rng.choice(syllables) for _ in range(2)).title()
    key = rng.choice(("code", "score", "year", "count"))
    value = rng.randrange(10, 1000)
    return (
        f"Fact: {name} has {key} {value}. Question: What is {name}'s {key}? "
        f"Answer: {value}."
    )


def _write_synthetic_stream(
    tokenizer: object,
    *,
    domain: str,
    path: Path,
    target_tokens: int,
    seed: int,
) -> int:
    if domain not in {"symbolic", "facts"}:
        raise ValueError(f"unsupported synthetic domain: {domain}")
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise RuntimeError("tokenizer does not contain <eos>")
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.memmap(path, dtype=np.uint16, mode="w+", shape=(target_tokens,))
    rng = random.Random(seed)
    generator = _symbolic_text if domain == "symbolic" else _facts_text
    cursor = 0
    examples = 0
    try:
        while cursor < target_tokens:
            encoded = list(tokenizer.encode(generator(rng)).ids)
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


def prepare_synthetic_corpus(
    cache_dir: Path,
    tokenizer: object,
    domain: str,
) -> SyntheticCorpus:
    if domain not in {"symbolic", "facts"}:
        raise ValueError("Experiment 026 only materializes symbolic/facts corpora")
    root = cache_dir / f"{domain}-granularity-026"
    train_path = root / "train.u16"
    validation_path = root / "validation.u16"
    manifest_path = root / "manifest.json"
    expected = {
        "format": "minicells.cell-granularity-synthetic.v1",
        "domain": domain,
        "train_tokens": SYNTHETIC_TRAIN_TOKENS,
        "validation_tokens": SYNTHETIC_VALIDATION_TOKENS,
        "seed": SYNTHETIC_SEEDS[domain],
    }
    if train_path.is_file() and validation_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        valid = (
            all(manifest.get(key) == value for key, value in expected.items())
            and train_path.stat().st_size == SYNTHETIC_TRAIN_TOKENS * 2
            and validation_path.stat().st_size == SYNTHETIC_VALIDATION_TOKENS * 2
            and sha256_file(train_path) == manifest.get("train_sha256")
            and sha256_file(validation_path) == manifest.get("validation_sha256")
        )
        if valid:
            return SyntheticCorpus(train_path, validation_path, manifest_path, manifest)

    root.mkdir(parents=True, exist_ok=True)
    train_examples = _write_synthetic_stream(
        tokenizer,
        domain=domain,
        path=train_path,
        target_tokens=SYNTHETIC_TRAIN_TOKENS,
        seed=SYNTHETIC_SEEDS[domain],
    )
    validation_examples = _write_synthetic_stream(
        tokenizer,
        domain=domain,
        path=validation_path,
        target_tokens=SYNTHETIC_VALIDATION_TOKENS,
        seed=SYNTHETIC_SEEDS[domain] + 1,
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
    return SyntheticCorpus(train_path, validation_path, manifest_path, manifest)


def prepare_domain_corpora(cache_dir: Path, tokenizer: object) -> dict[str, object]:
    return {
        "math": prepare_math_corpus(cache_dir, tokenizer),
        "symbolic": prepare_synthetic_corpus(cache_dir, tokenizer, "symbolic"),
        "facts": prepare_synthetic_corpus(cache_dir, tokenizer, "facts"),
    }


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


def continuation_batch(
    step: int,
    *,
    streams: Mapping[str, np.memmap],
    device: torch.device,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    domain = domain_for_step(step)
    stream = streams[domain]
    starts = _starts_for_step(
        step,
        stream_length=len(stream),
        sequence_length=TRAIN_SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        seed=START_SEEDS[domain],
    )
    inputs, targets = memmap_batch(stream, starts, TRAIN_SEQUENCE_LENGTH, device)
    return domain, inputs, targets


def fixed_validation_starts(
    stream_length: int,
    domain: str,
) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(VALIDATION_SEEDS[domain])
    high = stream_length - CONTEXT_LENGTH - 1
    if high <= 0:
        raise ValueError("validation stream is too short")
    return tuple(
        tuple(rng.randrange(high) for _ in range(BATCH_SIZE))
        for _ in range(VALIDATION_BATCHES)
    )


def build_granularity_model(
    source_artifact: str | Path,
    *,
    vocab_size: int,
    granularity: int,
    device: str | torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {GRANULARITIES}")
    clm, source_identity = build_30m_clm(
        source_artifact,
        vocab_size=vocab_size,
        device=device,
    )
    model = convert_model_experts_to_tissues(
        clm,
        config=TissueConfig(cells_per_tissue=granularity),
        inplace=True,
    )
    tissues = list(iter_tissues(model))
    if len(tissues) != 12:
        raise RuntimeError(f"expected 12 fixed root tissues, got {len(tissues)}")
    if any(tissue.cell_count != granularity for _, _, tissue in tissues):
        raise RuntimeError("tissue conversion did not preserve requested granularity")
    return model, source_identity


def iter_tissues(model: torch.nn.Module) -> Iterable[tuple[int, str, TissueFFN]]:
    for stage_index, stage in enumerate(model.stages):
        experts = stage.program_bank.experts
        items = experts.items() if hasattr(experts, "items") else enumerate(experts)
        for expert_id, expert in items:
            if not isinstance(expert, TissueFFN):
                raise TypeError("Experiment 026 requires TissueFFN experts")
            yield stage_index, str(expert_id), expert


def model_structure(model: torch.nn.Module) -> dict[str, int]:
    tissues = list(iter_tissues(model))
    return {
        "program_tissues": len(tissues),
        "micro_cells": sum(tissue.cell_count for _, _, tissue in tissues),
        "stored_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


@torch.no_grad()
def evaluate_domains(
    model: torch.nn.Module,
    *,
    validation_streams: Mapping[str, np.memmap],
    validation_starts: Mapping[str, tuple[tuple[int, ...], ...]],
    device: torch.device,
) -> dict[str, float]:
    result: dict[str, float] = {}
    nlls: list[float] = []
    for domain in DOMAINS:
        metrics = evaluate_stream(
            model,
            validation_streams[domain],
            validation_starts[domain],
            device=device,
            clm_backend=None,
        )
        nll = float(metrics["nll"])
        result[f"{domain}_nll"] = nll
        result[f"{domain}_ppl"] = float(metrics["ppl"])
        nlls.append(nll)
    result["balanced_nll"] = float(sum(nlls) / len(nlls))
    result["balanced_ppl_geomean"] = float(
        math.exp(min(result["balanced_nll"], 20.0))
    )
    return result


def diagnostic_batches(
    validation_streams: Mapping[str, np.memmap],
    *,
    device: torch.device,
) -> dict[str, DiagnosticBatch]:
    batches: dict[str, DiagnosticBatch] = {}
    for domain in DOMAINS:
        starts = fixed_validation_starts(len(validation_streams[domain]), domain)[0]
        inputs, targets = memmap_batch(
            validation_streams[domain],
            starts,
            CONTEXT_LENGTH,
            device,
        )
        batches[domain] = DiagnosticBatch(domain, inputs, targets)
    return batches


def _cell_key(stage: int, expert_id: str, cell_index: int) -> str:
    return f"s{stage}/{expert_id}/c{cell_index}"


def _tissue_key(stage: int, expert_id: str) -> str:
    return f"s{stage}/{expert_id}"


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().flatten()
    right = right.float().flatten()
    denom = float(left.norm().item() * right.norm().item())
    if denom <= 1e-12:
        return 0.0
    return float(torch.dot(left, right).item() / denom)


def _profile_specialization(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    total = float(array.sum())
    if total <= 1e-12:
        return 0.0
    probabilities = array / total
    entropy = -float(
        np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)))
    )
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(values))))


def _capture_first_tissue_inputs(
    model: torch.nn.Module,
    batch: DiagnosticBatch,
) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    hooks = []
    for stage, expert_id, tissue in iter_tissues(model):
        key = _tissue_key(stage, expert_id)

        def hook(_module, inputs, *, key=key):
            if key not in captured:
                flat = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                captured[key] = flat[:DIAGNOSTIC_PERCEPTION_CAP].clone()

        hooks.append(tissue.register_forward_pre_hook(hook))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), torch.autocast(
            device_type=batch.inputs.device.type,
            dtype=torch.float16,
            enabled=batch.inputs.device.type == "cuda",
        ):
            model(batch.inputs)
    finally:
        for hook_handle in hooks:
            hook_handle.remove()
        model.train(was_training)
    return captured


def _contribution_profiles(
    model: torch.nn.Module,
    batches: Mapping[str, DiagnosticBatch],
) -> dict[str, dict[str, float]]:
    profiles: dict[str, dict[str, float]] = {}
    tissue_lookup = {
        _tissue_key(stage, expert_id): tissue
        for stage, expert_id, tissue in iter_tissues(model)
    }
    for domain in DOMAINS:
        captured = _capture_first_tissue_inputs(model, batches[domain])
        with torch.no_grad():
            for tissue_key, perceptions in captured.items():
                tissue = tissue_lookup[tissue_key]
                for cell_index, cell in enumerate(tissue.cells):
                    contribution = cell(perceptions)
                    rms = float(
                        contribution.float().square().mean().sqrt().item()
                    )
                    profiles.setdefault(
                        f"{tissue_key}/c{cell_index}", {}
                    )[domain] = rms
    return profiles


def _gradient_profiles(
    model: torch.nn.Module,
    batches: Mapping[str, DiagnosticBatch],
) -> dict[str, dict[str, torch.Tensor]]:
    profiles: dict[str, dict[str, torch.Tensor]] = {}
    was_training = model.training
    model.eval()
    try:
        for domain in DOMAINS:
            model.zero_grad(set_to_none=True)
            batch = batches[domain]
            with torch.autocast(
                device_type=batch.inputs.device.type,
                dtype=torch.float16,
                enabled=batch.inputs.device.type == "cuda",
            ):
                logits = model(batch.inputs).logits
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch.targets.reshape(-1),
                )
            loss.backward()
            for stage, expert_id, tissue in iter_tissues(model):
                for cell_index, cell in enumerate(tissue.cells):
                    gradient = cell.out_proj.weight.grad
                    if gradient is None:
                        signature = torch.zeros(cell.out_proj.out_features)
                    else:
                        signature = gradient.detach().float().mean(dim=1).cpu()
                    key = _cell_key(stage, expert_id, cell_index)
                    profiles.setdefault(key, {})[domain] = signature
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    return profiles


def collect_cell_diagnostics(
    model: torch.nn.Module,
    *,
    batches: Mapping[str, DiagnosticBatch],
    domain_nlls: Mapping[str, float],
    baseline_profiles: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, dict[str, float]],
]:
    contribution = _contribution_profiles(model, batches)
    gradients = _gradient_profiles(model, batches)
    rows: list[dict[str, object]] = []
    tissue_rows: list[dict[str, object]] = []

    for stage, expert_id, tissue in iter_tissues(model):
        tissue_key = _tissue_key(stage, expert_id)
        cell_profiles: list[np.ndarray] = []
        raw_means: list[float] = []
        for cell_index in range(tissue.cell_count):
            key = _cell_key(stage, expert_id, cell_index)
            values = [
                float(contribution.get(key, {}).get(domain, 0.0))
                for domain in DOMAINS
            ]
            cell_profiles.append(np.asarray(values, dtype=np.float64))
            raw_means.append(float(np.mean(values)))
        max_mean = max(raw_means, default=0.0)
        usage_scaled = [value / max(max_mean, 1e-12) for value in raw_means]

        redundancies: list[float] = []
        for left in range(len(cell_profiles)):
            for right in range(left + 1, len(cell_profiles)):
                redundancies.append(
                    _cosine(
                        torch.tensor(cell_profiles[left]),
                        torch.tensor(cell_profiles[right]),
                    )
                )

        specializations: list[float] = []
        stresses: list[float] = []
        stabilities: list[float] = []
        for cell_index, profile in enumerate(cell_profiles):
            cell = tissue.cells[cell_index]
            key = _cell_key(stage, expert_id, cell_index)
            values = profile.tolist()
            specialization = _profile_specialization(values)
            specializations.append(specialization)
            preference_index = (
                int(np.argmax(profile)) if float(profile.sum()) > 0.0 else 0
            )
            preference = DOMAINS[preference_index]

            gradient_vectors = gradients.get(key, {})
            pairwise: list[float] = []
            for left in range(len(DOMAINS)):
                for right in range(left + 1, len(DOMAINS)):
                    left_domain = DOMAINS[left]
                    right_domain = DOMAINS[right]
                    if left_domain in gradient_vectors and right_domain in gradient_vectors:
                        pairwise.append(
                            _cosine(
                                gradient_vectors[left_domain],
                                gradient_vectors[right_domain],
                            )
                        )
            mean_gradient_cosine = float(np.mean(pairwise)) if pairwise else 0.0
            gradient_conflict = max(
                0.0,
                min(1.0, 0.5 * (1.0 - mean_gradient_cosine)),
            )

            baseline = (
                baseline_profiles.get(key)
                if baseline_profiles is not None
                else None
            )
            if baseline:
                baseline_vector = torch.tensor(
                    [float(baseline.get(domain, 0.0)) for domain in DOMAINS]
                )
                stability = _cosine(torch.tensor(profile), baseline_vector)
                novelty = max(0.0, min(1.0, 1.0 - stability))
            else:
                stability = 1.0
                novelty = 0.0
            stabilities.append(stability)

            nll_weights = profile / max(float(profile.sum()), 1e-12)
            weighted_nll = sum(
                float(nll_weights[index]) * float(domain_nlls[domain])
                for index, domain in enumerate(DOMAINS)
            )
            residual_loss = max(
                0.0,
                min(1.0, weighted_nll / (1.0 + weighted_nll)),
            )
            neighbors = tissue.neighbors(cell_index)
            neighbor_capacity = (
                float(np.mean([1.0 - usage_scaled[index] for index in neighbors]))
                if neighbors
                else 0.0
            )
            stress = tissue.instantaneous_stress(
                StressObservation(
                    usage=max(0.0, min(1.0, usage_scaled[cell_index])),
                    residual_loss=residual_loss,
                    novelty=novelty,
                    gradient_conflict=gradient_conflict,
                    neighbor_capacity=max(0.0, min(1.0, neighbor_capacity)),
                )
            )
            stresses.append(stress)
            row = {
                "stage": stage,
                "expert_id": expert_id,
                "cell_index": cell_index,
                "cell_key": key,
                "hidden_width": cell.in_proj.out_features,
                "preferred_domain": preference,
                "specialization": specialization,
                "mean_gradient_cosine": mean_gradient_cosine,
                "gradient_conflict": gradient_conflict,
                "profile_stability_vs_age0": stability,
                "diagnostic_stress": stress,
                "plasticity": float(cell.plasticity.item()),
            }
            row.update(
                {
                    f"contribution_{domain}": values[index]
                    for index, domain in enumerate(DOMAINS)
                }
            )
            rows.append(row)

        tissue_rows.append(
            {
                "stage": stage,
                "expert_id": expert_id,
                "cell_count": tissue.cell_count,
                "mean_cell_specialization": (
                    float(np.mean(specializations)) if specializations else 0.0
                ),
                "mean_pairwise_profile_cosine": (
                    float(np.mean(redundancies)) if redundancies else 0.0
                ),
                "mean_profile_stability_vs_age0": (
                    float(np.mean(stabilities)) if stabilities else 1.0
                ),
                "mean_diagnostic_stress": (
                    float(np.mean(stresses)) if stresses else 0.0
                ),
                "max_diagnostic_stress": (
                    float(np.max(stresses)) if stresses else 0.0
                ),
                "mean_plasticity": float(
                    np.mean([float(cell.plasticity.item()) for cell in tissue.cells])
                ),
                "plasticity_std": float(
                    np.std([float(cell.plasticity.item()) for cell in tissue.cells])
                ),
            }
        )
    return rows, tissue_rows, contribution


def summarize_diagnostics(
    cell_rows: Iterable[Mapping[str, object]],
    tissue_rows: Iterable[Mapping[str, object]],
) -> dict[str, float]:
    cells = list(cell_rows)
    tissues = list(tissue_rows)
    if not cells or not tissues:
        raise ValueError("diagnostics require cell and tissue rows")
    return {
        "mean_cell_specialization": float(
            np.mean([float(row["specialization"]) for row in cells])
        ),
        "median_cell_specialization": float(
            np.median([float(row["specialization"]) for row in cells])
        ),
        "mean_gradient_conflict": float(
            np.mean([float(row["gradient_conflict"]) for row in cells])
        ),
        "mean_profile_stability_vs_age0": float(
            np.mean([float(row["profile_stability_vs_age0"]) for row in cells])
        ),
        "mean_tissue_redundancy": float(
            np.mean([float(row["mean_pairwise_profile_cosine"]) for row in tissues])
        ),
        "mean_diagnostic_stress": float(
            np.mean([float(row["mean_diagnostic_stress"]) for row in tissues])
        ),
        "max_diagnostic_stress": float(
            np.max([float(row["max_diagnostic_stress"]) for row in tissues])
        ),
        "mean_plasticity": float(
            np.mean([float(row["plasticity"]) for row in cells])
        ),
        "plasticity_std": float(
            np.std([float(row["plasticity"]) for row in cells])
        ),
    }


def protocol_manifest() -> dict[str, object]:
    return {
        "format": FORMAT,
        "experiment_id": EXPERIMENT_ID,
        "source_artifact": SOURCE_007_ARTIFACT.as_posix(),
        "source_experience_tokens": EXPECTED_SOURCE_TOKENS,
        "granularities": list(GRANULARITIES),
        "program_tissues": 12,
        "persistent_growth": False,
        "local_plasticity": True,
        "domains": list(DOMAINS),
        "continuation_tokens": CONTINUATION_TOKENS,
        "schedule": schedule_manifest(),
        "eval_tokens": list(EVAL_TOKENS),
        "optimizer": {
            "type": "AdamW",
            "base_lr": BASE_LR,
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "grad_clip": GRAD_CLIP,
        },
        "decision_thresholds": {
            "differentiation_min_gain_delta": DIFFERENTIATION_MIN_GAIN_DELTA,
            "performance_nll_ratio_max": PERFORMANCE_NLL_RATIO_MAX,
            "stability_min_cosine": STABILITY_MIN_COSINE,
        },
        "diagnostic_caveat": (
            "diagnostic stress is recorded but never triggers division in "
            "Experiment 026"
        ),
    }
