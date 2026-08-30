"""Optimized execution for CLM-0.4-mini development calibration.

This module is frozen before seed 90401 is observed. It does not alter the
registered architecture, routes, dependency scope, curriculum, gates, candidate
order, or first-pass-stop rule. Structural/gate evaluation remains FP32.
CUDA AMP is used only for training.
"""

from __future__ import annotations

import copy
import csv
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
from pathlib import Path
import random
import shutil
import time
from typing import Any, Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import engine as engine_module
from . import model as model_module
from .calibration import (
    CALIBRATION_FORMAT,
    CalibrationCandidate,
    _candidate,
    _finite,
    _load,
    _load_base,
    _save_base,
    _summary_csv,
    _write,
    minimum_base_cell_activation,
    verify_calibration_assets,
    verify_committed_plan,
)
from .curriculum import transaction_specs
from .data import base_math_eval_examples, base_story_eval_examples
from .engine import VARIANTS, VariantHarness
from .examples import ScoredTokenExample, collate_scored
from .gates import evaluate_base_prerequisites
from .lock import build_protocol_lock
from .m1 import environment_versions
from .model import SparseCellFFN, TinyCLMDecoder
from .protocol import (
    CandidateOptimizerConfig,
    assert_seed_allowed,
    canonical_json_hash,
    formal_model_config,
    load_protocol,
    m1_thresholds,
)
from .state import model_state_hash
from .tokenizer import TokenizerBundle
from .training import (
    BaseCorpusDataset,
    BaseTrainConfig,
    base_cell_activation_counts,
    collate_base,
    exact_match_accuracy,
)


PERFORMANCE_FORMAT = "minicells.clm-0.4-mini.m1-calibration-performance.v2"
DEFAULT_EVAL_BATCH_SIZE = 64

_ORIGINAL_SPARSE_FORWARD = SparseCellFFN.forward
_ORIGINAL_SCORED_LOGITS = engine_module._scored_logits
_ORIGINAL_TRAIN_ONLY_CELLS = engine_module.train_only_cells


def grouped_sparse_forward(
    self: SparseCellFFN,
    x: torch.Tensor,
    address_ids: list[str | int],
) -> torch.Tensor:
    """Execute samples sharing the same route/private Cell as one tensor batch."""
    if x.size(0) != len(address_ids):
        raise ValueError("address_ids length must equal batch size")
    if x.size(0) == 0:
        return x

    groups: dict[tuple[int, int, str | None], list[int]] = {}
    for row, address_id in enumerate(address_ids):
        left, right = self.base_route(address_id)
        private_key = model_module._private_key(address_id)
        if private_key not in self.private_cells:
            private_key = None
        groups.setdefault((left, right, private_key), []).append(row)

    rows: list[torch.Tensor | None] = [None] * x.size(0)
    for (left, right, private_key), indices in groups.items():
        index = torch.tensor(indices, dtype=torch.long, device=x.device)
        sample = x.index_select(0, index)
        routed = 0.5 * (
            self.base_cells[left](sample) + self.base_cells[right](sample)
        )
        if private_key is not None:
            routed = routed + self.private_cells[private_key](sample)
        for position, row in enumerate(indices):
            rows[row] = routed[position : position + 1]

    if any(row is None for row in rows):
        raise RuntimeError("grouped sparse execution did not fill every batch row")
    return torch.cat([row for row in rows if row is not None], dim=0)


def batched_scored_logits(
    model: TinyCLMDecoder,
    examples: Iterable[ScoredTokenExample],
    *,
    tokenizer: TokenizerBundle,
    device: torch.device,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
) -> dict[str, torch.Tensor]:
    """Compute structural logits in FP32 with one forward per example batch."""
    items = list(examples)
    result: dict[str, torch.Tensor] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(items), int(batch_size)):
            batch = items[start : start + int(batch_size)]
            x, _, mask, addresses = collate_scored(
                batch, pad_id=tokenizer.pad_id, device=device
            )
            logits = model(x, addresses)
            for row, example in enumerate(batch):
                result[example.example_id] = logits[row][mask[row]].detach().cpu()
    return result


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_only_cells_amp(
    model: TinyCLMDecoder,
    *,
    cell_ids: list[str],
    examples: list[ScoredTokenExample],
    tokenizer: TokenizerBundle,
    optimizer_config: CandidateOptimizerConfig,
    device: torch.device,
    rng_seed: int,
) -> dict[str, Any]:
    """Registered Cell-only AdamW update; training autocast only on CUDA."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    params: list[nn.Parameter] = []
    for module in model.modules_for_cell_ids(cell_ids):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            params.append(parameter)
    if not params:
        raise RuntimeError("candidate has no mutable Cell parameters")
    if optimizer_config.optimizer != "AdamW":
        raise ValueError("registered candidate optimizer must be AdamW")

    optimizer = torch.optim.AdamW(
        params,
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
    )
    amp = device.type == "cuda"
    scaler = _grad_scaler(amp)
    rng = random.Random(int(rng_seed))
    started = time.perf_counter()
    model.train()
    training_tokens = 0

    for _ in range(int(optimizer_config.steps)):
        if len(examples) <= optimizer_config.batch_size:
            batch = list(examples)
        else:
            indices = rng.sample(range(len(examples)), int(optimizer_config.batch_size))
            batch = [examples[index] for index in indices]
        x, y, mask, addresses = collate_scored(
            batch, pad_id=tokenizer.pad_id, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if amp else torch.float32,
            enabled=amp,
        ):
            logits = model(x, addresses)
            per_token = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="none",
            ).reshape_as(y)
            selected = per_token[mask]
            if selected.numel() == 0:
                raise RuntimeError("candidate batch has no scored tokens")
            loss = selected.mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        training_tokens += int(mask.sum().item())

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {
        "wall_seconds": time.perf_counter() - started,
        "training_tokens": int(training_tokens),
        "optimizer_steps": int(optimizer_config.steps),
        "training_precision": "fp16-amp" if amp else "fp32",
    }


def install_runtime_patches() -> None:
    SparseCellFFN.forward = grouped_sparse_forward
    engine_module._scored_logits = batched_scored_logits
    engine_module.train_only_cells = train_only_cells_amp


def restore_runtime_patches() -> None:
    SparseCellFFN.forward = _ORIGINAL_SPARSE_FORWARD
    engine_module._scored_logits = _ORIGINAL_SCORED_LOGITS
    engine_module.train_only_cells = _ORIGINAL_TRAIN_ONLY_CELLS


class AddressDataParallel(nn.DataParallel):
    """DataParallel that scatters token rows and out-of-band addresses together."""

    def scatter(self, inputs, kwargs, device_ids):  # type: ignore[override]
        if len(inputs) != 2:
            raise RuntimeError("AddressDataParallel expects (token_ids, address_ids)")
        token_ids, address_ids = inputs
        if not torch.is_tensor(token_ids):
            raise TypeError("token_ids must be a tensor")
        if token_ids.size(0) != len(address_ids):
            raise ValueError("address list must align with token batch")
        target_count = min(len(device_ids), max(1, int(token_ids.size(0))))
        chunks = torch.chunk(token_ids, target_count, dim=0)
        scattered_inputs = []
        cursor = 0
        for device_id, chunk in zip(device_ids[:target_count], chunks, strict=True):
            size = int(chunk.size(0))
            addresses = list(address_ids[cursor : cursor + size])
            cursor += size
            target = torch.device("cuda", int(device_id))
            scattered_inputs.append((chunk.to(target, non_blocking=True), addresses))
        scattered_kwargs = [dict(kwargs) for _ in scattered_inputs]
        return tuple(scattered_inputs), tuple(scattered_kwargs)


def resolve_cuda_devices(
    *,
    requested_device: str | torch.device,
    requested_devices: str | None = None,
) -> list[torch.device]:
    requested = str(requested_device)
    if requested == "cpu":
        return [torch.device("cpu")]
    if requested_devices:
        devices = [
            torch.device(value.strip())
            for value in requested_devices.split(",")
            if value.strip()
        ]
    elif requested.startswith("cuda"):
        devices = [
            torch.device(f"cuda:{index}")
            for index in range(torch.cuda.device_count())
        ]
    else:
        devices = [torch.device(requested)]
    if not devices:
        raise RuntimeError("no calibration devices resolved")
    if any(device.type == "cuda" for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA calibration requested but CUDA is unavailable")
    return devices


def _masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    selected = per_token[mask]
    if selected.numel() == 0:
        raise RuntimeError("base-training batch has no scored tokens")
    return selected.mean()


def train_base_model_parallel(
    model: TinyCLMDecoder,
    *,
    dataset: BaseCorpusDataset,
    tokenizer: TokenizerBundle,
    devices: list[torch.device],
    seed: int,
    config: BaseTrainConfig | None = None,
) -> dict[str, Any]:
    """One corpus pass, using all visible CUDA devices when available."""
    cfg = config or BaseTrainConfig()
    primary = devices[0]
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=primary.type == "cuda",
        collate_fn=lambda batch: collate_base(batch, pad_id=tokenizer.pad_id),
    )

    model.to(primary)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    cuda_devices = [device for device in devices if device.type == "cuda"]
    parallel: nn.Module = model
    if len(cuda_devices) > 1:
        parallel = AddressDataParallel(
            model,
            device_ids=[int(device.index) for device in cuda_devices],
            output_device=int(cuda_devices[0].index),
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
    )
    total_steps = max(1, len(loader))
    warmup_steps = max(1, int(round(total_steps * cfg.warmup_fraction)))

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    amp = primary.type == "cuda"
    scaler = _grad_scaler(amp)
    started = time.perf_counter()
    loss_sum = 0.0
    token_count = 0
    parallel.train()

    for step, (x, y, mask, addresses) in enumerate(loader):
        y = y.to(primary, non_blocking=True)
        mask = mask.to(primary, non_blocking=True)
        if len(cuda_devices) <= 1:
            x = x.to(primary, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=primary.type,
            dtype=torch.float16 if amp else torch.float32,
            enabled=amp,
        ):
            logits = parallel(x, addresses)
            loss = _masked_ce(logits, y, mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
        scale = lr_scale(step)
        for group in optimizer.param_groups:
            group["lr"] = cfg.learning_rate * scale
        scaler.step(optimizer)
        scaler.update()
        scored = int(mask.sum().item())
        loss_sum += float(loss.detach().cpu()) * scored
        token_count += scored

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {
        "optimizer": cfg.to_dict(),
        "steps": total_steps,
        "training_tokens": token_count,
        "mean_nll": loss_sum / float(max(1, token_count)),
        "wall_seconds": time.perf_counter() - started,
        "training_precision": "fp16-amp" if amp else "fp32",
        "data_parallel_devices": [str(device) for device in devices],
    }


def prepare_or_load_base_parallel(
    *,
    protocol: Mapping[str, Any],
    data_dir: Path,
    out_dir: Path,
    assets: Mapping[str, Any],
    seed: int,
    devices: list[torch.device],
) -> tuple[TinyCLMDecoder, TokenizerBundle, dict[str, Any]]:
    primary = devices[0]
    cfg = formal_model_config(protocol, routing_salt=assets["identity"]["routing_salt"])
    tokenizer = TokenizerBundle.load(data_dir / "tokenizer" / "tokenizer.json")
    dataset = BaseCorpusDataset(data_dir / "base-corpus")
    model = TinyCLMDecoder(cfg).to(primary)
    checkpoint = out_dir / "base" / "checkpoint.pt"

    if checkpoint.is_file():
        saved = _load_base(checkpoint, model, seed, assets["identity"], primary)
        base_train, source = saved["base_train"], "resumed"
    else:
        base_train = train_base_model_parallel(
            model,
            dataset=dataset,
            tokenizer=tokenizer,
            devices=devices,
            seed=seed,
        )
        _save_base(checkpoint, model, seed, assets["identity"], base_train)
        source = "trained-once"

    counts = base_cell_activation_counts(model, dataset)
    threshold = minimum_base_cell_activation(protocol, base_sequences=len(dataset))
    math_acc = exact_match_accuracy(
        model, base_math_eval_examples(64), tokenizer=tokenizer, device=primary
    )
    story_acc = exact_match_accuracy(
        model, base_story_eval_examples(64), tokenizer=tokenizer, device=primary
    )
    prerequisites = evaluate_base_prerequisites(
        protocol=protocol,
        math_exact_match=math_acc,
        story_exact_match=story_acc,
        cell_activation_counts=counts,
        locked_minimum_activation=threshold,
        numeric_finite=_finite(model),
        hashes_match_lock=bool(assets["verified"]),
    )
    metrics = {
        "checkpoint": str(checkpoint),
        "checkpoint_source": source,
        "state_hash": model_state_hash(model),
        "base_sequences": len(dataset),
        "minimum_base_cell_activation": threshold,
        "math_exact_match": math_acc,
        "story_exact_match": story_acc,
        "base_train": base_train,
        "prerequisites": prerequisites,
    }
    _write(out_dir / "base" / "metrics.json", metrics)
    with (out_dir / "base" / "activation-counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", "activation_count"])
        writer.writerows((cell_id, count) for cell_id, count in sorted(counts.items()))
    return model, tokenizer, metrics


def materialize_tokenized_curriculum(
    *,
    curriculum_manifest: Mapping[str, Any],
    tokenizer: TokenizerBundle,
    max_seq_len: int,
) -> dict[int, dict[str, list[ScoredTokenExample]]]:
    from .m1 import _tokenize_transaction

    cache: dict[int, dict[str, list[ScoredTokenExample]]] = {}
    for spec in transaction_specs(dict(curriculum_manifest)):
        cache[int(spec.transaction_id)] = _tokenize_transaction(
            spec,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            smoke=False,
        )
    return cache


def materialize_base_probes(
    *, tokenizer: TokenizerBundle, max_seq_len: int
) -> list[ScoredTokenExample]:
    from .m1 import _base_probes

    return _base_probes(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        smoke=False,
    )


def run_single_variant(
    *,
    variant: str,
    protocol: Mapping[str, Any],
    base_model: TinyCLMDecoder,
    tokenizer: TokenizerBundle,
    curriculum_manifest: Mapping[str, Any],
    tokenized_transactions: Mapping[int, Mapping[str, list[ScoredTokenExample]]],
    base_probes: list[ScoredTokenExample],
    direct_optimizer: CandidateOptimizerConfig,
    growth_optimizer: CandidateOptimizerConfig,
    seed: int,
    device: torch.device,
) -> VariantHarness:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    harness = VariantHarness(
        variant=variant,
        model=copy.deepcopy(base_model).to(device),
        tokenizer=tokenizer,
        device=device,
        thresholds=m1_thresholds(protocol),
    )
    harness.admit_probes(base_probes)
    variant_index = {"local_always": 0, "local_tx": 1, "local_tx_growth": 2}[variant]

    for spec in transaction_specs(dict(curriculum_manifest)):
        data = tokenized_transactions[int(spec.transaction_id)]
        transaction_seed = (
            int(seed) * 1_000_003 + int(spec.transaction_id) * 97 + variant_index * 13
        ) & 0x7FFFFFFF
        harness.execute(
            transaction_id=spec.transaction_id,
            operation=spec.operation,
            address_id=spec.address_id,
            knowledge_key=spec.knowledge_key,
            supersedes_key=spec.supersedes_key,
            train_examples=list(data["train"]),
            validation_examples=list(data["validation"]),
            probe_examples=list(data["probe"]),
            direct_optimizer=direct_optimizer,
            growth_optimizer=growth_optimizer,
            rng_seed=transaction_seed,
        )
    return harness


def _maximum_active_private_from_records(harness: VariantHarness) -> int:
    maximum = 0
    for record in harness.records:
        for cells in record["active_cells_by_layer"].values():
            maximum = max(
                maximum,
                sum(1 for cell in cells if str(cell).startswith("growth:")),
            )
    return maximum


def _ratio(value: float, reference: float) -> float:
    if reference > 0.0:
        return value / reference
    return 0.0 if value <= 0.0 else math.inf


def evaluate_gate_summaries(
    *,
    protocol: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, Any]],
    growth_harness: VariantHarness,
) -> dict[str, Any]:
    always = summaries["local_always"]
    local_tx = summaries["local_tx"]
    growth = summaries["local_tx_growth"]
    registered = protocol["m1_gates"]
    damage_ratio = _ratio(
        float(growth["positive_global_regression_damage"]),
        float(always["positive_global_regression_damage"]),
    )
    gain_ratio = _ratio(
        float(growth["committed_new_gain"]),
        float(always["committed_new_gain"]),
    )
    active_private = _maximum_active_private_from_records(growth_harness)
    gates = {
        "false_safe_rate": {"value": growth["false_safe_rate"], "threshold": registered["maximum_false_safe_rate"], "pass": growth["false_safe_rate"] <= registered["maximum_false_safe_rate"]},
        "structural_escape_rate": {"value": growth["maximum_structural_escape_rate"], "threshold": registered["maximum_structural_escape_rate"], "pass": growth["maximum_structural_escape_rate"] <= registered["maximum_structural_escape_rate"]},
        "regression_damage_ratio_vs_local_always": {"value": damage_ratio, "threshold": registered["maximum_regression_damage_ratio_vs_local_always"], "pass": damage_ratio <= registered["maximum_regression_damage_ratio_vs_local_always"]},
        "effective_acceptance_rate": {"value": growth["effective_acceptance_rate"], "threshold": registered["minimum_effective_acceptance_rate"], "pass": growth["effective_acceptance_rate"] >= registered["minimum_effective_acceptance_rate"]},
        "committed_gain_ratio_vs_local_always": {"value": gain_ratio, "threshold": registered["minimum_committed_gain_ratio_vs_local_always"], "pass": gain_ratio >= registered["minimum_committed_gain_ratio_vs_local_always"]},
        "final_protected_retention_ratio": {"value": growth["final_protected_retention_ratio"], "threshold": registered["minimum_final_protected_retention_ratio"], "pass": growth["final_protected_retention_ratio"] >= registered["minimum_final_protected_retention_ratio"]},
        "growth_exceeds_local_tx_gain": {"value": growth["committed_new_gain"] - local_tx["committed_new_gain"], "threshold": ">0", "pass": growth["committed_new_gain"] > local_tx["committed_new_gain"]},
        "growth_rescue_rate": {"value": growth["growth_rescue_rate"], "threshold": registered["minimum_growth_rescue_rate"], "pass": growth["growth_rescue_rate"] >= registered["minimum_growth_rescue_rate"]},
        "private_reuse_acceptance_rate": {"value": growth["private_reuse_acceptance_rate"], "threshold": registered["minimum_private_reuse_acceptance_rate"], "pass": growth["private_reuse_acceptance_rate"] >= registered["minimum_private_reuse_acceptance_rate"]},
        "spawned_bundles_per_effective_commit": {"value": growth["spawned_bundles_per_effective_commit"], "threshold": registered["maximum_spawned_bundles_per_effective_commit"], "pass": growth["spawned_bundles_per_effective_commit"] <= registered["maximum_spawned_bundles_per_effective_commit"]},
        "growth_parameter_overhead_ratio": {"value": growth["growth_parameter_overhead_ratio"], "threshold": registered["maximum_growth_parameter_overhead_ratio"], "pass": growth["growth_parameter_overhead_ratio"] <= registered["maximum_growth_parameter_overhead_ratio"]},
        "active_private_cells_per_layer_per_input": {"value": active_private, "threshold": registered["maximum_active_private_cells_per_growth_layer_per_input"], "pass": active_private <= registered["maximum_active_private_cells_per_growth_layer_per_input"]},
        "mean_direct_dependency_coverage": {"value": growth["mean_direct_dependency_coverage"], "threshold": registered["maximum_mean_direct_dependency_coverage"], "pass": growth["mean_direct_dependency_coverage"] <= registered["maximum_mean_direct_dependency_coverage"]},
    }
    return {
        "variant_summaries": {name: dict(value) for name, value in summaries.items()},
        "gates": gates,
        "pass": all(item["pass"] for item in gates.values()),
    }


def _write_harness_evidence(
    root: Path,
    harness: VariantHarness,
    *,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    resolved_summary = dict(summary) if summary is not None else harness.summary()
    _write(root / "summary.json", resolved_summary)
    with (root / "transactions.jsonl").open("w", encoding="utf-8") as handle:
        for record in harness.records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    with (root / "cell-registry.jsonl").open("w", encoding="utf-8") as handle:
        for entry in harness.registry.snapshot(harness.model, harness.dependency_index):
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return resolved_summary


def _baseline_key(config: CandidateOptimizerConfig) -> str:
    return canonical_json_hash({"direct_optimizer": config.to_dict()})[:20]


def _load_cached_baselines(
    root: Path,
    *,
    direct: CandidateOptimizerConfig,
) -> dict[str, dict[str, Any]] | None:
    metadata = root / "cache.json"
    if not metadata.is_file():
        return None
    cache = _load(metadata)
    if cache.get("performance_format") != PERFORMANCE_FORMAT:
        return None
    if cache.get("direct_optimizer") != direct.to_dict():
        raise RuntimeError("baseline-cache direct optimizer mismatch")
    result: dict[str, dict[str, Any]] = {}
    for variant in ("local_always", "local_tx"):
        path = root / variant / "summary.json"
        if not path.is_file():
            return None
        result[variant] = _load(path)
    return result


def _run_baseline_pair(
    *,
    cache_root: Path,
    protocol: Mapping[str, Any],
    base_model: TinyCLMDecoder,
    tokenizer: TokenizerBundle,
    curriculum_manifest: Mapping[str, Any],
    tokenized_transactions: Mapping[int, Mapping[str, list[ScoredTokenExample]]],
    base_probes: list[ScoredTokenExample],
    direct: CandidateOptimizerConfig,
    growth: CandidateOptimizerConfig,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for variant in ("local_always", "local_tx"):
        harness = run_single_variant(
            variant=variant,
            protocol=protocol,
            base_model=base_model,
            tokenizer=tokenizer,
            curriculum_manifest=curriculum_manifest,
            tokenized_transactions=tokenized_transactions,
            base_probes=base_probes,
            direct_optimizer=direct,
            growth_optimizer=growth,
            seed=seed,
            device=device,
        )
        summary = harness.summary()
        _write_harness_evidence(cache_root / variant, harness, summary=summary)
        summaries[variant] = summary
        del harness
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _write(
        cache_root / "cache.json",
        {
            "performance_format": PERFORMANCE_FORMAT,
            "direct_optimizer": direct.to_dict(),
            "variants": ["local_always", "local_tx"],
        },
    )
    return summaries


def _copy_selected_baselines(cache_root: Path, selected_root: Path) -> None:
    for variant in ("local_always", "local_tx"):
        destination = selected_root / variant
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(cache_root / variant, destination)


def _performance_environment(devices: list[torch.device]) -> dict[str, Any]:
    primary = devices[0]
    environment = environment_versions(primary)
    environment.update(
        {
            "execution_engine": PERFORMANCE_FORMAT,
            "training_precision": "fp16-amp" if primary.type == "cuda" else "fp32",
            "validation_precision": "fp32",
            "eval_batch_size": DEFAULT_EVAL_BATCH_SIZE,
            "devices": [str(device) for device in devices],
            "gpu_names": [
                torch.cuda.get_device_name(device)
                for device in devices
                if device.type == "cuda"
            ],
            "direct_baseline_cache": True,
            "route_grouped_sparse_execution": True,
            "tokenized_curriculum_cache": True,
        }
    )
    return environment


def run_calibration_optimized(
    *,
    protocol_path: str | Path,
    expected_assets_path: str | Path,
    committed_plan_path: str | Path,
    protocol_lock_template_path: str | Path,
    data_dir: str | Path,
    out_dir: str | Path,
    seed: int,
    device: str | torch.device,
    devices: str | None,
    code_commit: str,
    code_tree: str,
    tracked_tree_dirty: bool,
) -> dict[str, Any]:
    install_runtime_patches()
    protocol = load_protocol(protocol_path)
    assert_seed_allowed(protocol, mode="calibration", seed=seed)
    if tracked_tree_dirty or not code_commit or not code_tree:
        raise RuntimeError("calibration requires a clean committed source tree")

    out_dir, data_dir = Path(out_dir), Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = verify_committed_plan(protocol, committed_plan_path)
    _write(out_dir / "calibration-plan.json", plan)
    assets = verify_calibration_assets(
        protocol=protocol,
        data_dir=data_dir,
        expected_assets_path=expected_assets_path,
    )
    _write(
        out_dir / "asset-verification.json",
        {
            "verified": True,
            "identity": assets["identity"],
            "identity_sha256": assets["identity_sha256"],
        },
    )

    resolved = resolve_cuda_devices(
        requested_device=device,
        requested_devices=devices,
    )
    primary = resolved[0]
    torch.manual_seed(seed)
    random.seed(seed)
    if primary.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model, tokenizer, base = prepare_or_load_base_parallel(
        protocol=protocol,
        data_dir=data_dir,
        out_dir=out_dir,
        assets=assets,
        seed=seed,
        devices=resolved,
    )
    environment = _performance_environment(resolved)
    common = {
        "format": CALIBRATION_FORMAT,
        "performance_format": PERFORMANCE_FORMAT,
        "seed": seed,
        "scientific_decision": False,
        "development_seed_observed": True,
        "formal_seeds_observed": False,
        "asset_identity": assets["identity"],
        "plan_sha256": plan["plan_sha256"],
        "base": base,
        "code_commit": code_commit,
        "code_tree": code_tree,
        "environment": environment,
    }

    if not base["prerequisites"]["pass"]:
        decision = {
            "status": "CALIBRATION_BASE_PREREQUISITES_FAILED",
            "scientific_decision": False,
            "development_seed_observed": True,
            "formal_seeds_observed": False,
            "selected_candidate": None,
        }
        _write(out_dir / "decision.json", decision)
        _write(out_dir / "summary.json", {**common, "decision": decision})
        return decision

    tokenized_transactions = materialize_tokenized_curriculum(
        curriculum_manifest=assets["curriculum_manifest"],
        tokenizer=tokenizer,
        max_seq_len=model.cfg.max_seq_len,
    )
    base_probes = materialize_base_probes(
        tokenizer=tokenizer,
        max_seq_len=model.cfg.max_seq_len,
    )

    model = model.cpu()
    immutable_hash = model_state_hash(model)
    base_snapshot = copy.deepcopy(model).cpu()
    rows: list[dict[str, Any]] = []
    selected: CalibrationCandidate | None = None
    selected_gates: dict[str, Any] | None = None

    for payload in plan["candidates"]:
        candidate = _candidate(payload)
        result_path = out_dir / "candidates" / candidate.candidate_id / "candidate.json"
        if result_path.is_file():
            row = _load(result_path)
            if row["candidate"] != candidate.to_dict():
                raise RuntimeError(f"resume candidate drift: {candidate.candidate_id}")
            if row.get("performance_format") != PERFORMANCE_FORMAT:
                raise RuntimeError("cannot resume candidate from a different execution engine")
            rows.append(row)
            if row["pass"]:
                selected, selected_gates = candidate, row["gate_snapshot"]
                break
            continue

        if model_state_hash(model) != immutable_hash:
            raise RuntimeError("immutable base model changed before candidate")

        started = time.perf_counter()
        cache_root = out_dir / "baseline-cache" / _baseline_key(candidate.direct)
        baseline_summaries = _load_cached_baselines(
            cache_root, direct=candidate.direct
        )
        cache_hit = baseline_summaries is not None
        growth_device = resolved[1] if len(resolved) > 1 else primary

        if baseline_summaries is None and len(resolved) > 1:
            with ThreadPoolExecutor(max_workers=2) as pool:
                baseline_future = pool.submit(
                    _run_baseline_pair,
                    cache_root=cache_root,
                    protocol=protocol,
                    base_model=base_snapshot,
                    tokenizer=tokenizer,
                    curriculum_manifest=assets["curriculum_manifest"],
                    tokenized_transactions=tokenized_transactions,
                    base_probes=base_probes,
                    direct=candidate.direct,
                    growth=candidate.growth_private,
                    seed=seed,
                    device=primary,
                )
                growth_future = pool.submit(
                    run_single_variant,
                    variant="local_tx_growth",
                    protocol=protocol,
                    base_model=base_snapshot,
                    tokenizer=tokenizer,
                    curriculum_manifest=assets["curriculum_manifest"],
                    tokenized_transactions=tokenized_transactions,
                    base_probes=base_probes,
                    direct_optimizer=candidate.direct,
                    growth_optimizer=candidate.growth_private,
                    seed=seed,
                    device=growth_device,
                )
                baseline_summaries = baseline_future.result()
                growth_harness = growth_future.result()
        else:
            if baseline_summaries is None:
                baseline_summaries = _run_baseline_pair(
                    cache_root=cache_root,
                    protocol=protocol,
                    base_model=base_snapshot,
                    tokenizer=tokenizer,
                    curriculum_manifest=assets["curriculum_manifest"],
                    tokenized_transactions=tokenized_transactions,
                    base_probes=base_probes,
                    direct=candidate.direct,
                    growth=candidate.growth_private,
                    seed=seed,
                    device=primary,
                )
            growth_harness = run_single_variant(
                variant="local_tx_growth",
                protocol=protocol,
                base_model=base_snapshot,
                tokenizer=tokenizer,
                curriculum_manifest=assets["curriculum_manifest"],
                tokenized_transactions=tokenized_transactions,
                base_probes=base_probes,
                direct_optimizer=candidate.direct,
                growth_optimizer=candidate.growth_private,
                seed=seed,
                device=growth_device,
            )

        growth_summary = growth_harness.summary()
        summaries = {
            **baseline_summaries,
            "local_tx_growth": growth_summary,
        }
        gate_snapshot = evaluate_gate_summaries(
            protocol=protocol,
            summaries=summaries,
            growth_harness=growth_harness,
        )
        if model_state_hash(model) != immutable_hash:
            raise RuntimeError("candidate mutated immutable base model")

        row = {
            "candidate": candidate.to_dict(),
            "pass": bool(gate_snapshot["pass"]),
            "gate_snapshot": gate_snapshot,
            "base_state_hash_before_and_after": immutable_hash,
            "wall_seconds": time.perf_counter() - started,
            "performance_format": PERFORMANCE_FORMAT,
            "baseline_cache_key": _baseline_key(candidate.direct),
            "baseline_cache_hit": cache_hit,
            "devices": [str(value) for value in resolved],
        }
        _write(result_path, row)
        rows.append(row)

        if row["pass"]:
            selected, selected_gates = candidate, gate_snapshot
            selected_root = out_dir / "selected"
            _copy_selected_baselines(cache_root, selected_root)
            _write_harness_evidence(
                selected_root / "local_tx_growth",
                growth_harness,
                summary=growth_summary,
            )
            del growth_harness
            gc.collect()
            if primary.type == "cuda":
                torch.cuda.empty_cache()
            break

        del growth_harness
        gc.collect()
        if primary.type == "cuda":
            torch.cuda.empty_cache()

    _summary_csv(out_dir, rows)
    if selected is None:
        decision = {
            "status": "CALIBRATION_NO_CONFIGURATION_PASSED",
            "scientific_decision": False,
            "development_seed_observed": True,
            "formal_seeds_observed": False,
            "selected_candidate": None,
            "candidates_evaluated": len(rows),
        }
        _write(out_dir / "decision.json", decision)
        _write(out_dir / "summary.json", {**common, "decision": decision})
        return decision

    selected_payload = {
        "candidate": selected.to_dict(),
        "selection_rule": plan["selection_rule"],
        "first_passing_ordinal": selected.ordinal,
        "gate_snapshot": selected_gates,
        "candidates_evaluated": len(rows),
    }
    _write(out_dir / "selected.json", selected_payload)
    protocol_lock = build_protocol_lock(
        protocol=protocol,
        template=_load(protocol_lock_template_path),
        protocol_path=protocol_path,
        direct_optimizer=selected.direct,
        growth_optimizer=selected.growth_private,
        tokenizer_manifest=assets["tokenizer_manifest"],
        base_corpus_manifest=assets["base_corpus_manifest"],
        curriculum_manifest=assets["curriculum_manifest"],
        dataset_revision=assets["identity"]["dataset_revision"],
        routing_salt=assets["identity"]["routing_salt"],
        minimum_base_cell_activation=base["minimum_base_cell_activation"],
        code_commit=code_commit,
        code_tree=code_tree,
        environment=environment,
    )
    _write(out_dir / "protocol-lock.candidate.json", protocol_lock)

    decision = {
        "status": "CALIBRATION_CONFIGURATION_SELECTED",
        "scientific_decision": False,
        "development_seed_observed": True,
        "formal_seeds_observed": False,
        "selected_candidate": selected.candidate_id,
        "candidates_evaluated": len(rows),
        "formal_execution_authorized": False,
        "next_required_action": (
            "commit protocol-lock.candidate.json as canonical protocol-lock.json "
            "before any formal seed is opened"
        ),
    }
    _write(out_dir / "decision.json", decision)
    _write(
        out_dir / "summary.json",
        {
            **common,
            "decision": decision,
            "selected": selected_payload,
            "protocol_lock_candidate": "protocol-lock.candidate.json",
        },
    )
    return decision


def equivalence_probe(
    *,
    model: TinyCLMDecoder,
    examples: list[ScoredTokenExample],
    tokenizer: TokenizerBundle,
    device: torch.device,
) -> dict[str, Any]:
    restore_runtime_patches()
    legacy = _ORIGINAL_SCORED_LOGITS(
        model, examples, tokenizer=tokenizer, device=device
    )
    install_runtime_patches()
    optimized = batched_scored_logits(
        model, examples, tokenizer=tokenizer, device=device
    )
    maximum = max(
        [
            float((legacy[key] - optimized[key]).abs().max())
            for key in legacy
        ]
        or [0.0]
    )
    return {
        "examples": len(examples),
        "legacy_forward_calls": len(examples),
        "batched_forward_calls": math.ceil(len(examples) / DEFAULT_EVAL_BATCH_SIZE),
        "maximum_absolute_logit_delta": maximum,
    }
