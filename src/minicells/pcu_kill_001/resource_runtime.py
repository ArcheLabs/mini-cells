"""Resource-bounded Granite execution for PCU-KILL-001.

The scientific protocol requires FP32 equivalence, but it does not require two
resident copies of the 1.3B foundation.  This module keeps one full model and
materializes only the final-MoE expert representation twice: the exact parent
experts and their exact CellularExpert decomposition.  Full-model reference
calls use a temporary expert overlay and restore the cellular foundation in a
``finally`` block.

No scientific threshold, dataset, routing decision, optimizer, Cell allocation
rule, or merge rule is changed here.
"""

from __future__ import annotations

from dataclasses import asdict
import gc
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from . import experiment as _experiment
from .cache import CachedTailRunner
from .cellular import GraniteArchitectureInspector, patch_moe_block
from .equivalence import compare_values, verify_end_to_end, verify_expert_algebra
from .governance import set_deterministic_seeds
from .model import MODEL_ID, target_module
from .overlay import ExpertsOverlayModel
from .registry import module_tensor_hash
from .synthetic import audit_dataset, generate_world


@torch.no_grad()
def inference_logits(model: Any, inputs: Mapping[str, Tensor]) -> Tensor:
    """Return detached logits for evaluation-only full-model forwards."""
    value = getattr(model(**dict(inputs)), "logits", None)
    if not isinstance(value, Tensor):
        raise RuntimeError("model output has no logits")
    return value.detach()


def cellularize_in_place(
    model: Any,
    inspector: GraniteArchitectureInspector,
) -> tuple[Any, Any]:
    """Patch only the final experts and keep the rest of the foundation singular.

    Returns ``(cellular_model, exact_parent_experts)``.  Both the pretrained
    foundation and the exact cellular decomposition are frozen.  Fork/LoRA
    modules created later own the only trainable parameters.
    """
    model.requires_grad_(False)
    block = target_module(model, inspector.target_path)
    parent_experts = block.experts
    cellular_experts = patch_moe_block(block, inspector.partition)
    cellular_experts.requires_grad_(False)
    model.eval()
    return model, parent_experts


@torch.no_grad()
def full_moe_overlay_equivalence(
    block: Any,
    reference_experts: Any,
    candidate_experts: Any,
    layer_input: Tensor,
    tolerance: float = 2e-5,
):
    """Compare one MoE block under parent vs cellular experts without cloning it."""
    if layer_input.ndim == 2:
        layer_input = layer_input.unsqueeze(0)
    if layer_input.ndim != 3:
        raise ValueError(
            "full MoE equivalence requires [batch, sequence, hidden] or [tokens, hidden]"
        )
    resident = block.experts
    try:
        block.experts = reference_experts
        reference = block(layer_input)
        block.experts = candidate_experts
        candidate = block(layer_input)
    finally:
        block.experts = resident
    if isinstance(reference, tuple):
        reference = reference[0]
    if isinstance(candidate, tuple):
        candidate = candidate[0]
    return compare_values(reference, candidate, tolerance=tolerance)


def _prepare_single_foundation(
    *,
    model_repo: str,
    revision: str | None,
    device: str,
) -> tuple[Any, Any, Any, Any, GraniteArchitectureInspector, dict[str, Any]]:
    tokenizer, model, manifest = _experiment.load_granite(
        model_repo, revision=revision, device=device
    )
    inspector = GraniteArchitectureInspector.inspect(model, require_granite=True)
    # load_granite computes the immutable foundation identity before any
    # cellular representation is installed.  Preserve that identity verbatim.
    foundation_hash = str(manifest.get("foundation_tensor_sha256") or module_tensor_hash(model))
    cellular, parent_experts = cellularize_in_place(model, inspector)
    target = target_module(cellular, inspector.target_path)
    base_view = ExpertsOverlayModel(cellular, inspector.target_path, parent_experts).eval()
    manifest = {
        **manifest,
        "architecture": asdict(inspector),
        "foundation_tensor_sha256": foundation_hash,
        "execution_memory_model": "single_full_foundation_plus_expert_overlay",
    }
    return tokenizer, cellular, base_view, parent_experts, inspector, manifest


def g0_preflight(
    *,
    seed: int,
    model_repo: str,
    revision: str | None,
    device: str,
    probe_prefix: str,
) -> dict[str, Any]:
    """Resource-bounded version of the fail-fast G0 preflight."""
    set_deterministic_seeds(seed)
    tokenizer, cellular, base_view, parent_experts, inspector, manifest = (
        _prepare_single_foundation(
            model_repo=model_repo,
            revision=revision,
            device=device,
        )
    )
    target = target_module(cellular, inspector.target_path)
    probe_texts = [f"{probe_prefix} {index:03d}." for index in range(128)]
    probe_inputs = _experiment._token_batch(tokenizer, probe_texts, device)
    g0_expert = [
        verify_expert_algebra(
            parent_experts,
            index,
            inspector.partition,
            vectors=1024,
            seed=seed,
        )
        for index in range(inspector.local_experts)
    ]
    moe_probe = torch.randn(
        128,
        inspector.hidden_size,
        generator=torch.Generator(device="cpu").manual_seed(seed + 1),
    ).to(device)
    g0_full_moe = full_moe_overlay_equivalence(
        target, parent_experts, target.experts, moe_probe
    )
    g0_e2e = verify_end_to_end(base_view, cellular, probe_inputs)
    passed = bool(
        all(item.passed for item in g0_expert)
        and g0_full_moe.passed
        and g0_e2e.passed
    )
    # Do not return parent_experts/base_view: the preflight caller releases the
    # one resident model before the shared worker reloads the pinned revision.
    return {
        "passed": passed,
        "tokenizer": tokenizer,
        "model": cellular,
        "cellular": cellular,
        "manifest": manifest,
        "inspector": inspector,
        "metrics": {
            "g0_expert": {
                str(index): item.to_dict() for index, item in enumerate(g0_expert)
            },
            "g0_full_moe": g0_full_moe.to_dict(),
            "g0_end_to_end": g0_e2e.to_dict(),
            "g0_exact_embedding": passed,
            "execution_memory_model": "single_full_foundation_plus_expert_overlay",
        },
    }


def run_granite_engineering(seed: int, output: Path, device: str) -> dict[str, Any]:
    """Run E0 with a single full Granite foundation and exact expert overlays."""
    set_deterministic_seeds(seed)
    tokenizer, cellular, base_view, parent_experts, inspector, manifest = (
        _prepare_single_foundation(model_repo=MODEL_ID, revision=None, device=device)
    )
    target = target_module(cellular, inspector.target_path)
    probe_texts = [
        f"PCU-KILL-001 immutable engineering probe {index:03d}." for index in range(128)
    ]
    probe_inputs = _experiment._token_batch(tokenizer, probe_texts, device)
    original_logits = inference_logits(base_view, probe_inputs)
    g0 = [
        verify_expert_algebra(
            parent_experts,
            index,
            inspector.partition,
            vectors=1024,
            seed=seed,
        )
        for index in range(inspector.local_experts)
    ]
    moe_probe = torch.randn(
        128,
        inspector.hidden_size,
        generator=torch.Generator(device="cpu").manual_seed(seed + 1),
    ).to(device)
    g0_full_moe = full_moe_overlay_equivalence(
        target, parent_experts, target.experts, moe_probe
    )
    g0_e2e = verify_end_to_end(base_view, cellular, probe_inputs)
    cache_runner = CachedTailRunner(cellular, inspector.decoder_layer_path)
    cache = cache_runner.capture(
        probe_inputs["input_ids"],
        probe_inputs.get("attention_mask"),
        tuple(f"probe-{index:03d}" for index in range(128)),
    )
    cache_gate = cache_runner.verify(cache, full_logits=original_logits)
    world = generate_world(seed, count=128, tokenizer=tokenizer)
    audit = audit_dataset(world)
    if not audit.passed:
        raise RuntimeError(f"DATASET_LEAKAGE_AUDIT failed: {audit.errors}")
    return _experiment._run_shared_scientific_pipeline(
        phase="engineering",
        seed=seed,
        output=output,
        device=device,
        tokenizer=tokenizer,
        original=base_view,
        cellular=cellular,
        manifest=manifest,
        inspector=inspector,
        g0=g0,
        g0_full_moe=g0_full_moe,
        g0_e2e=g0_e2e,
        cache_gate=cache_gate,
        world=world,
        audit=audit,
        allow_search=True,
    )


def run_formal_execution(
    seed: int,
    protocol_path: Path,
    output: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Formal worker with identical science and the resource-bounded model layout."""
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_BEFORE_FORMAL":
        raise RuntimeError("formal worker requires FROZEN_BEFORE_FORMAL protocol")
    if int(seed) not in tuple(int(value) for value in payload.get("formal_seeds", [])):
        raise ValueError(f"seed {seed} is not listed in the frozen protocol")
    set_deterministic_seeds(seed)
    model_info = payload["model"]
    tokenizer, cellular, base_view, parent_experts, inspector, manifest = (
        _prepare_single_foundation(
            model_repo=str(model_info["model_repo"]),
            revision=str(model_info["model_revision"]),
            device=device,
        )
    )
    for key in (
        "model_repo",
        "model_revision",
        "config_sha256",
        "weight_file_sha256",
        "tokenizer_sha256",
    ):
        if manifest.get(key) != model_info.get(key):
            raise RuntimeError(f"formal foundation identity mismatch: {key}")
    if (
        model_info.get("foundation_tensor_sha256")
        and manifest.get("foundation_tensor_sha256")
        != model_info["foundation_tensor_sha256"]
    ):
        raise RuntimeError("formal foundation identity mismatch: foundation_tensor_sha256")
    expected_architecture = payload.get("architecture", model_info)
    actual_architecture = asdict(inspector)
    for key in (
        "target_layer",
        "target_path",
        "hidden_size",
        "intermediate_size",
        "local_experts",
        "experts_per_token",
        "cells",
        "fused_order",
    ):
        if expected_architecture.get(key) != actual_architecture.get(key):
            raise RuntimeError(f"formal architecture mismatch: {key}")

    target = target_module(cellular, inspector.target_path)
    prompts = [f"PCU-KILL-001 formal seed {seed} probe {index:03d}." for index in range(128)]
    inputs = _experiment._token_batch(tokenizer, prompts, device)
    original_logits = inference_logits(base_view, inputs)
    g0 = [
        verify_expert_algebra(
            parent_experts,
            index,
            inspector.partition,
            vectors=1024,
            seed=seed,
        )
        for index in range(inspector.local_experts)
    ]
    moe_probe = torch.randn(
        128,
        inspector.hidden_size,
        generator=torch.Generator(device="cpu").manual_seed(seed + 1),
    ).to(device)
    g0_full_moe = full_moe_overlay_equivalence(
        target, parent_experts, target.experts, moe_probe
    )
    g0_e2e = verify_end_to_end(base_view, cellular, inputs)
    cache_runner = CachedTailRunner(cellular, inspector.decoder_layer_path)
    cache = cache_runner.capture(inputs["input_ids"], inputs.get("attention_mask"))
    cache_gate = cache_runner.verify(cache, full_logits=original_logits)
    world = generate_world(seed, count=128, tokenizer=tokenizer)
    audit = audit_dataset(world)
    if not audit.passed:
        raise RuntimeError(f"DATASET_LEAKAGE_AUDIT failed: {audit.errors}")

    training = payload["training"]
    allocation = payload.get("allocation", {})
    if (
        allocation.get("method") != "task-conditioned-gradient-l2-per-parameter"
        or allocation.get("calibration_sample_rule") != "first_64_samples"
    ):
        raise RuntimeError("formal protocol has no supported frozen allocation policy")
    if int(allocation.get("selected_k", -1)) != int(training["selected_k"]):
        raise RuntimeError("formal protocol allocation K does not match training K")
    frozen = {
        "optimizer": training["optimizer"],
        "learning_rate": training["learning_rate"],
        "max_optimizer_steps": training["max_optimizer_steps"],
        "max_training_tokens": training["max_training_tokens"],
        "selected_k": training["selected_k"],
        "lora_rank": training["lora_rank"],
        "generation": payload.get("evaluation", {}).get(
            "generation", dict(_experiment.GENERATION_CONFIG)
        ),
    }
    return _experiment._run_shared_scientific_pipeline(
        phase="formal",
        seed=seed,
        output=output,
        device=device,
        tokenizer=tokenizer,
        original=base_view,
        cellular=cellular,
        manifest=manifest,
        inspector=inspector,
        g0=g0,
        g0_full_moe=g0_full_moe,
        g0_e2e=g0_e2e,
        cache_gate=cache_gate,
        world=world,
        audit=audit,
        allow_search=False,
        frozen_config=frozen,
    )


def release_cuda() -> None:
    """Best-effort allocator cleanup between the preflight and shared worker."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
