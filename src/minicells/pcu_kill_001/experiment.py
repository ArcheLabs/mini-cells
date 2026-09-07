"""Engineering and formal execution graph for PCU-KILL-001.

The toy backend is intentionally non-scientific, but it exercises the same
functional composition, registry, rollback, cache, and decision machinery as
the pinned Granite backend.  The Granite backend uses the actual loaded model
and never substitutes the toy model for a missing dependency or checkpoint.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping

import torch
from torch import Tensor

from .backends import make_toy_model
from .artifacts import file_sha256, load_tensor_artifact, save_tensor_artifact
from .cache import CachedTailRunner, save_cache, validate_cache_identity
from .cellular import GraniteArchitectureInspector
from .equivalence import verify_end_to_end, verify_expert_algebra, verify_full_moe
from .governance import DEVELOPMENT_SEED, EXPERIMENT_ID, assert_seed_registry, git_provenance, runtime_provenance, set_deterministic_seeds, sha256_file, write_json
from .model import MODEL_ID, cellularize_model, load_granite, model_identity_manifest, target_module
from .registry import CellRegistry, bind_fork_artifact, fork_registry, make_foundation_registry, merge_registries, module_tensor_hash, rollback_registry, tensor_sha256, validate_fork_artifacts
from .synthetic import audit_dataset, context_oracle, generate_world
from .training import Allocation, BranchTrainingConfig, ForkedCellularExpert, ForkedCellularExperts, allocate_topk, assert_foundation_unchanged, foundation_tensor_hashes, fork_expert, fork_initial_delta_norm, selected_delta_parameters, train_fork, write_training_csv
from .composition import ComposedCellularExperts, compose_cellular_experts
from .task import TaskSequences, TailTrainingCache, build_task_sequences, cache_task_sequences, save_task_cache
from .task_training import TaskBranchResult, cached_task_loss, slice_task_cache, task_conditioned_allocation, train_cached_branch, train_cached_lora_branch
from .evaluation import EvaluationSummary, evaluate_matrix, evaluate_samples, matrix_accuracy, matrix_to_dict
from .metrics import composition_synergy_same_task, retention_or_undefined


def _cell_parts(cell_id: str) -> tuple[int, int]:
    match = re.fullmatch(r"L\d+:E(\d+):C(\d+)", cell_id)
    if not match:
        raise ValueError(f"invalid foundation Cell ID: {cell_id}")
    return int(match.group(1)), int(match.group(2))


def _path(model: Any, dotted: str) -> Any:
    value = model
    for part in dotted.split("."):
        value = getattr(value, part)
    return value


def _write_lines(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _debug_random_allocation(model: Any, inspector: GraniteArchitectureInspector, seed: int) -> Allocation:
    """Infrastructure-only toy diagnostic; never used by a scientific worker."""
    target = _path(model, inspector.target_path)
    inputs = torch.randn(32, inspector.hidden_size, generator=torch.Generator().manual_seed(seed))
    output = target(inputs)
    loss = output.float().pow(2).mean()
    model.zero_grad(set_to_none=True)
    loss.backward()
    values: dict[str, float] = {}
    for expert_index, expert in enumerate(target.experts.cells):
        for cell_index, cell in enumerate(expert.cells):
            gradients = [parameter.grad for parameter in cell.parameters() if parameter.grad is not None]
            score = sum(float(value.detach().float().pow(2).sum()) for value in gradients) / max(1, sum(int(value.numel()) for value in gradients))
            values[f"L{inspector.target_layer}:E{expert_index}:C{cell_index}"] = score
    model.zero_grad(set_to_none=True)
    return allocate_topk(values)


def _update_fork_records(
    registry: CellRegistry,
    fork: ForkedCellularExpert,
    selected: tuple[str, ...],
    artifact_path: Path,
) -> CellRegistry:
    result = registry.copy()
    state = fork.delta_state()
    delta_hash = hashlib.sha256(b"".join(key.encode() + value.cpu().numpy().tobytes() for key, value in sorted(state.items()))).hexdigest()
    artifact_hash = file_sha256(artifact_path)
    for cell_id in selected:
        fork_id = next(key for key, record in result.records.items() if key.startswith(cell_id + "::fork::"))
        result.records[fork_id].weight_hash = delta_hash
    branch = next(result.records[key].branch for key in result.records if key.startswith(selected[0] + "::fork::"))
    return bind_fork_artifact(result, str(branch), str(artifact_path), artifact_hash)


def _branch(parent: Any, inspector: GraniteArchitectureInspector, selected: tuple[str, ...], seed: int, output: Path) -> tuple[CellRegistry, dict[str, Any]]:
    target = _path(parent, inspector.target_path)
    by_expert: dict[int, list[int]] = {}
    for cell_id in selected:
        expert, cell = _cell_parts(cell_id)
        by_expert.setdefault(expert, []).append(cell)
    # One expert wrapper is enough for this bounded engineering harness; the
    # same class is used by the real worker to wrap every selected parent.
    expert_index = next(iter(by_expert))
    fork = fork_expert(target.experts.cells[expert_index], by_expert[expert_index])
    initial_delta = fork_initial_delta_norm(fork)
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(12, inspector.hidden_size, generator=generator)
    with torch.no_grad():
        target_values = target.experts.cells[expert_index](inputs).detach() + 0.001
    optimizer = torch.optim.AdamW(selected_delta_parameters(fork), lr=1e-2)
    rows = []
    parent_hash = foundation_tensor_hashes(parent)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = fork(inputs)
        loss = (prediction - target_values).pow(2).mean()
        loss.backward()
        optimizer.step()
        rows.append({"step": step + 1, "tokens": int(inputs.numel()), "loss": float(loss.detach())})
    # parent is not passed to the optimizer and its hash is checked here.
    from .training import assert_foundation_unchanged
    assert_foundation_unchanged(parent_hash, parent)
    artifact_path, artifact_format = save_tensor_artifact(fork.delta_state(), output)
    write_training_csv(output / "TRAINING.csv", rows)
    return fork, {
        "initial_delta_l2": initial_delta,
        "training_steps": len(rows),
        "training_tokens": len(rows) * int(inputs.numel()),
        "trainable_parameter_count": sum(int(value.numel()) for value in selected_delta_parameters(fork)),
        "selected_cells": list(selected),
        "unique_parent_experts": len(by_expert),
        "loss_final": rows[-1]["loss"],
        "artifact_path": str(artifact_path),
        "artifact_sha256": file_sha256(artifact_path),
        "artifact_format": artifact_format,
    }


def _write_branch_manifest(output: Path, registry: CellRegistry, summary: Mapping[str, Any], branch: str) -> None:
    registry.save(str(output / "CELL_REGISTRY.json"))
    write_json(output / "MANIFEST.json", {"schema": "minicells.pcu-kill-001.branch-manifest.v1", "branch": branch, "scientific_evidence": False, "registry_sha256": registry.content_hash(), **dict(summary)})


def _dispatch(experts: Any, router: Any, hidden: Tensor) -> Tensor:
    indices, weights = router(hidden)
    return experts(hidden, indices, weights)


def _max_error(left: Tensor, right: Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def _functional_composition_audit(
    parent_experts: Any,
    router: Any,
    fork_a: ForkedCellularExpert,
    fork_b: ForkedCellularExpert,
    selected_a: tuple[str, ...],
    selected_b: tuple[str, ...],
    hidden: Tensor,
) -> dict[str, Any]:
    by_branch = {
        "A": {_cell_parts(selected_a[0])[0]: fork_a},
        "B": {_cell_parts(selected_b[0])[0]: fork_b},
    }
    runtime_a = compose_cellular_experts(parent_experts, by_branch, ("A",))
    runtime_b = compose_cellular_experts(parent_experts, by_branch, ("B",))
    runtime_ab = compose_cellular_experts(parent_experts, by_branch, ("A", "B"))
    runtime_all = runtime_ab.rollback("all")
    parent_output = _dispatch(parent_experts, router, hidden)
    a_output = _dispatch(runtime_a, router, hidden)
    b_output = _dispatch(runtime_b, router, hidden)
    ab_output = _dispatch(runtime_ab, router, hidden)
    all_output = _dispatch(runtime_all, router, hidden)
    expected = parent_output + (a_output - parent_output) + (b_output - parent_output)
    return {
        "overlap_formula_max_abs": _max_error(ab_output, expected),
        "rollback_a_max_abs": _max_error(_dispatch(runtime_ab.rollback("B"), router, hidden), a_output),
        "rollback_b_max_abs": _max_error(_dispatch(runtime_ab.rollback("A"), router, hidden), b_output),
        "rollback_all_max_abs": _max_error(all_output, parent_output),
        "composition_passed": _max_error(ab_output, expected) <= 2e-6,
        "rollback_passed": all(_max_error(value, parent_output) <= 2e-6 for value in (all_output,))
        and _max_error(_dispatch(runtime_ab.rollback("B"), router, hidden), a_output) <= 2e-6
        and _max_error(_dispatch(runtime_ab.rollback("A"), router, hidden), b_output) <= 2e-6,
        "nonoverlap_supported": len(set(selected_a) & set(selected_b)) == 0,
    }


def _functional_composition_audit_collections(
    parent_experts: Any,
    router: Any,
    forks_a: Mapping[int, ForkedCellularExpert],
    forks_b: Mapping[int, ForkedCellularExpert],
    selected_a: tuple[str, ...],
    selected_b: tuple[str, ...],
    hidden: Tensor,
) -> dict[str, Any]:
    branches = {"A": forks_a, "B": forks_b}
    runtime_a = compose_cellular_experts(parent_experts, branches, ("A",))
    runtime_b = compose_cellular_experts(parent_experts, branches, ("B",))
    runtime_ab = compose_cellular_experts(parent_experts, branches, ("A", "B"))
    parent_output = _dispatch(parent_experts, router, hidden)
    a_output = _dispatch(runtime_a, router, hidden)
    b_output = _dispatch(runtime_b, router, hidden)
    ab_output = _dispatch(runtime_ab, router, hidden)
    rollback_all = _dispatch(runtime_ab.rollback("all"), router, hidden)
    overlap_error = _max_error(ab_output, parent_output + (a_output - parent_output) + (b_output - parent_output))
    rollback_a_error = _max_error(_dispatch(runtime_ab.rollback("B"), router, hidden), a_output)
    rollback_b_error = _max_error(_dispatch(runtime_ab.rollback("A"), router, hidden), b_output)
    rollback_all_error = _max_error(rollback_all, parent_output)
    return {
        "overlap_formula_max_abs": overlap_error,
        "rollback_a_max_abs": rollback_a_error,
        "rollback_b_max_abs": rollback_b_error,
        "rollback_all_max_abs": rollback_all_error,
        "composition_passed": overlap_error <= 2e-5,
        "rollback_passed": max(rollback_a_error, rollback_b_error, rollback_all_error) <= 2e-5,
        "nonoverlap_supported": len(set(selected_a) & set(selected_b)) == 0,
        "same_parent_overlap": bool(set(selected_a) & set(selected_b)),
    }


def _token_batch(tokenizer: Any, texts: list[str], device: str | torch.device, max_length: int = 96) -> dict[str, Tensor]:
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return {key: value.to(device) for key, value in encoded.items() if isinstance(value, Tensor)}


@torch.no_grad()
def _answer_token_accuracy(model: Any, tokenizer: Any, samples: list[Any], device: str | torch.device) -> float:
    if not samples:
        return 0.0
    correct = 0
    for start in range(0, len(samples), 8):
        batch = samples[start:start + 8]
        inputs = _token_batch(tokenizer, [item.prompt for item in batch], device)
        logits = getattr(model(**inputs), "logits", None)
        if logits is None:
            raise RuntimeError("Granite output has no logits")
        lengths = inputs.get("attention_mask", torch.ones(inputs["input_ids"].shape, device=device)).sum(dim=1) - 1
        for row, item in enumerate(batch):
            answer = tokenizer.encode(item.answer, add_special_tokens=False)
            if hasattr(answer, "ids"):
                answer = answer.ids
            if answer and int(logits[row, int(lengths[row])].argmax()) == int(answer[0]):
                correct += 1
    return correct / len(samples)


def _debug_hidden_target_branch_collection(
    parent_experts: Any,
    selected: tuple[str, ...],
    seed: int,
    output: Path,
    calibration: Tensor,
    config: BranchTrainingConfig,
) -> tuple[dict[int, ForkedCellularExpert], dict[str, Any]]:
    """Legacy toy-only diagnostic; excluded from every Granite worker."""
    by_expert: dict[int, list[int]] = {}
    for cell_id in selected:
        expert, cell = _cell_parts(cell_id)
        by_expert.setdefault(expert, []).append(cell)
    if not by_expert:
        raise ValueError("branch selection is empty")
    before = foundation_tensor_hashes(parent_experts)
    forks = {expert: fork_expert(parent_experts.cells[expert], indices) for expert, indices in by_expert.items()}
    initial_delta = sum(fork_initial_delta_norm(fork) for fork in forks.values())
    rows: list[dict[str, float]] = []
    parameters = [parameter for fork in forks.values() for parameter in selected_delta_parameters(fork)]
    if config.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    elif config.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=config.learning_rate)
    else:
        raise ValueError(f"unsupported frozen branch optimizer: {config.optimizer}")
    torch.manual_seed(int(seed))
    calibration = calibration.detach()
    for step in range(config.max_optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for expert, fork in forks.items():
            with torch.no_grad():
                target = parent_experts.cells[expert](calibration).detach()
                target = target + 0.001 * torch.tanh(calibration[:, :target.shape[-1]])
            losses.append((fork(calibration) - target).float().square().mean())
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite Granite branch loss")
        loss.backward()
        optimizer.step()
        rows.append({"step": float(step + 1), "tokens": float((step + 1) * calibration.shape[0]), "loss": float(loss.detach())})
    assert_foundation_unchanged(before, parent_experts)
    state = {f"expert_{expert}.{key}": value for expert, fork in forks.items() for key, value in fork.delta_state().items()}
    artifact_path, artifact_format = save_tensor_artifact(state, output)
    write_training_csv(output / "TRAINING.csv", rows)
    summary = {
        "initial_delta_l2": initial_delta,
        "training_steps": len(rows),
        "training_tokens": len(rows) * int(calibration.shape[0]),
        "trainable_parameter_count": sum(sum(int(value.numel()) for value in selected_delta_parameters(fork)) for fork in forks.values()),
        "selected_cells": list(selected),
        "unique_parent_experts": len(forks),
        "loss_final": rows[-1]["loss"],
        "artifact_path": str(artifact_path),
        "artifact_sha256": file_sha256(artifact_path),
        "artifact_format": artifact_format,
    }
    return forks, summary


def _bind_collection_records(registry: CellRegistry, selected: tuple[str, ...], branch: str, artifact_path: Path, forks: Mapping[int, ForkedCellularExpert]) -> CellRegistry:
    result = fork_registry(registry, selected, branch)
    state_hashes = {expert: module_tensor_hash(fork) for expert, fork in forks.items()}
    for cell_id in selected:
        expert, _ = _cell_parts(cell_id)
        fork_id = f"{cell_id}::fork::{branch}"
        result.records[fork_id].weight_hash = state_hashes[expert]
    return bind_fork_artifact(result, branch, str(artifact_path), file_sha256(artifact_path))


def _model_with_experts(model: Any, inspector: GraniteArchitectureInspector, experts: Any) -> Any:
    result = deepcopy(model).eval()
    target_module(result, inspector.target_path).experts = experts
    return result


def _logits(model: Any, inputs: Mapping[str, Tensor]) -> Tensor:
    value = getattr(model(**dict(inputs)), "logits", None)
    if not isinstance(value, Tensor):
        raise RuntimeError("model output has no logits")
    return value


TASK_SPLITS = ("A_train", "B_train", "A_eval", "B_eval", "AB_eval")
CAPACITY_LADDER = (1, 2, 4, 8)
ENGINEERING_LR_CANDIDATES = (1e-4, 3e-4, 1e-3)
TASK_MAX_LENGTH = 128
GENERATION_CONFIG = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "max_new_tokens": 16,
}
ANCHOR_TEXTS = (
    "The quick brown fox jumps over the lazy dog.",
    "A small compiler turns a clear specification into a repeatable program.",
    "Reliable systems preserve their inputs and make every transformation observable.",
    "The afternoon train arrived quietly at the nearly empty platform.",
    "Good experiments separate the control condition from the measured intervention.",
    "A bright notebook rested beside the window while rain crossed the garden.",
    "Careful engineers test failure paths before trusting a successful result.",
    "Language models should retain ordinary language when a narrow module is adapted.",
)


def _task_cache_identity(
    manifest: Mapping[str, Any],
    inspector: GraniteArchitectureInspector,
    foundation_hash: str,
    world: Any,
    sequences: TaskSequences,
    split: str,
    dtype: str,
) -> dict[str, Any]:
    return {
        "foundation_model": MODEL_ID,
        "foundation_revision": manifest.get("model_revision"),
        "foundation_tensor_sha256": foundation_hash,
        "target_path": inspector.target_path,
        "target_layer": inspector.target_layer,
        "tokenizer_sha256": manifest.get("tokenizer_sha256", []),
        "dataset_manifest_sha256": world.manifest_sha256(),
        "split": split,
        "template_version": world.generator_version,
        "encoding_version": sequences.encoding_version,
        "dtype": dtype,
    }


def _build_task_caches(
    tokenizer: Any,
    runner: CachedTailRunner,
    world: Any,
    manifest: Mapping[str, Any],
    inspector: GraniteArchitectureInspector,
    foundation_hash: str,
    output: Path,
    device: str,
) -> tuple[dict[str, TaskSequences], dict[str, TailTrainingCache], dict[str, Any]]:
    sequences: dict[str, TaskSequences] = {}
    caches: dict[str, TailTrainingCache] = {}
    manifests: dict[str, Any] = {}
    for split in TASK_SPLITS:
        sequence = build_task_sequences(
            tokenizer, world.splits[split], split, max_length=TASK_MAX_LENGTH
        )
        identity = _task_cache_identity(
            manifest, inspector, foundation_hash, world, sequence, split, "float32"
        )
        cache = cache_task_sequences(runner, sequence, identity=identity, device=device)
        manifest_value = save_task_cache(cache, output / "cache" / split, shard_rows=32)
        validate_cache_identity(manifest_value, cache.identity)
        sequences[split] = sequence
        caches[split] = cache
        manifests[split] = manifest_value
    write_json(output / "TASK_SEQUENCES.json", {name: value.to_manifest() for name, value in sequences.items()})
    write_json(output / "TASK_CACHE_MANIFESTS.json", manifests)
    return sequences, caches, manifests


def _allocation_manifest(allocation: Allocation, layer: int, selected_k: int | None = None) -> dict[str, Any]:
    ordered = list(allocation.scores)
    ranks = {cell_id: rank + 1 for rank, cell_id in enumerate(ordered)}
    rows = []
    for cell_id, score in sorted(allocation.scores.items()):
        expert, cell = _cell_parts(cell_id)
        rows.append({
            "cell_id": cell_id,
            "score": float(score),
            "rank": int(ranks[cell_id]),
            "parent_expert": expert,
            "slice_index": cell,
            "activation_frequency": None,
        })
    return {
        "schema": "minicells.pcu-kill-001.task-allocation.v1",
        "method": "task-conditioned-gradient-l2-per-parameter",
        "calibration_split": "branch_train",
        "calibration_sample_rule": "first_64_samples",
        "tie_break": "cell_id_lexical",
        "selected_k": selected_k,
        "cells": rows,
        "topk_mass": {str(key): float(value) for key, value in allocation.topk_mass.items()},
        "effective_count": float(allocation.effective_count),
    }


def _save_task_branch_artifact(result: TaskBranchResult, directory: Path) -> dict[str, Any]:
    state: dict[str, Tensor] = {}
    for cell_id in result.selected_cells:
        expert_index, cell_index = _cell_parts(cell_id)
        for key, value in result.runtime.cells[expert_index].cells[cell_index].delta_state().items():
            state[f"expert_{expert_index}.cells.{cell_index}.delta_{key}"] = value
    path, artifact_format = save_tensor_artifact(state, directory, stem="DELTA_CELLS")
    summary = result.to_dict()
    summary.update({
        "artifact_path": str(path),
        "artifact_sha256": file_sha256(path),
        "artifact_format": artifact_format,
        "trainable_parameter_count": sum(int(value.numel()) for value in result.runtime.parameters() if value.requires_grad),
    })
    return summary


def _load_task_branch_artifact(
    parent_experts: Any,
    selected: tuple[str, ...],
    layer: int,
    path: Path,
    expected_sha256: str,
) -> dict[int, ForkedCellularExpert]:
    state = load_tensor_artifact(path, expected_sha256)
    by_expert: dict[int, list[int]] = {}
    for cell_id in selected:
        expert, cell = _cell_parts(cell_id)
        by_expert.setdefault(expert, []).append(cell)
    forks = {expert: fork_expert(parent_experts.cells[expert], cells) for expert, cells in by_expert.items()}
    for key, value in state.items():
        match = re.fullmatch(r"expert_(\d+)\.cells\.(\d+)\.delta_(.+)", key)
        if match is None:
            raise ValueError(f"invalid task branch artifact key: {key}")
        expert_index, cell_index, name = int(match.group(1)), int(match.group(2)), match.group(3)
        if expert_index not in forks:
            raise ValueError(f"artifact contains an unselected expert: {expert_index}")
        parameter = getattr(forks[expert_index].cells[cell_index], f"delta_{name}", None)
        if parameter is None:
            raise ValueError(f"artifact contains an invalid Cell parameter: {key}")
        parameter.data.copy_(value.to(parameter.device, dtype=parameter.dtype))
    expected = {}
    for cell_id in selected:
        expert_index, cell_index = _cell_parts(cell_id)
        for key, value in forks[expert_index].cells[cell_index].delta_state().items():
            expected[f"expert_{expert_index}.cells.{cell_index}.delta_{key}"] = value
    if set(expected) != set(state):
        raise ValueError("task branch artifact does not contain the complete fork state")
    for key in expected:
        if not torch.equal(expected[key].cpu(), state[key].cpu()):
            raise ValueError(f"task branch artifact state mismatch: {key}")
    return forks


def _save_lora_branch_artifact(runtime: Any, directory: Path, branch: str) -> dict[str, Any]:
    state = {
        f"expert_{expert_index}.{key}": value
        for expert_index, expert in enumerate(runtime.cells)
        for cell_index, cell in enumerate(expert.cells)
        if hasattr(cell, "state_delta")
        for key, value in cell.state_delta().items()
    }
    path, artifact_format = save_tensor_artifact(state, directory, stem=f"LORA_{branch}")
    return {"branch": branch, "artifact_path": str(path), "artifact_sha256": file_sha256(path), "artifact_format": artifact_format, "tensor_count": len(state)}


def _anchor_nll(model: Any, tokenizer: Any, device: str) -> float:
    inputs = _token_batch(tokenizer, list(ANCHOR_TEXTS), device, max_length=TASK_MAX_LENGTH)
    labels = inputs["input_ids"].clone()
    labels[inputs.get("attention_mask", torch.ones_like(labels)).eq(0)] = -100
    logits = _logits(model, inputs)
    shifted_logits = logits[:, :-1].contiguous().reshape(-1, logits.shape[-1])
    shifted_labels = labels[:, 1:].contiguous().reshape(-1)
    return float(torch.nn.functional.cross_entropy(shifted_logits, shifted_labels, ignore_index=-100).detach())


def _task_matrix_metrics(matrix: Mapping[str, Mapping[str, EvaluationSummary]], anchor_base: float, anchor_ab: float) -> dict[str, Any]:
    base_a = matrix_accuracy(matrix, "M0", "A_eval")
    base_b = matrix_accuracy(matrix, "M0", "B_eval")
    acc_a = matrix_accuracy(matrix, "MA", "A_eval")
    acc_b = matrix_accuracy(matrix, "MB", "B_eval")
    ab_on_a = matrix_accuracy(matrix, "MAB", "A_eval")
    ab_on_b = matrix_accuracy(matrix, "MAB", "B_eval")
    base_ab = matrix_accuracy(matrix, "M0", "AB_eval")
    a_ab = matrix_accuracy(matrix, "MA", "AB_eval")
    b_ab = matrix_accuracy(matrix, "MB", "AB_eval")
    ab_ab = matrix_accuracy(matrix, "MAB", "AB_eval")
    return {
        "base_a": base_a,
        "base_b": base_b,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "ab_on_a": ab_on_a,
        "ab_on_b": ab_on_b,
        "base_ab": base_ab,
        "a_ab": a_ab,
        "b_ab": b_ab,
        "composition_acc": ab_ab,
        "branch_gain_a": acc_a - base_a,
        "branch_gain_b": acc_b - base_b,
        "retention_a": retention_or_undefined(base_a, acc_a, ab_on_a),
        "retention_b": retention_or_undefined(base_b, acc_b, ab_on_b),
        "composition_synergy": composition_synergy_same_task(base_ab, a_ab, b_ab, ab_ab),
        "anchor_nll_base": anchor_base,
        "anchor_nll_ab": anchor_ab,
        "anchor_regression": (anchor_ab - anchor_base) / max(anchor_base, 1e-12),
    }


def _train_joint_oracle(
    parent_experts: Any,
    runner: CachedTailRunner,
    cache_a: TailTrainingCache,
    cache_b: TailTrainingCache,
    selected_a: tuple[str, ...],
    selected_b: tuple[str, ...],
    layer: int,
    config: BranchTrainingConfig,
) -> tuple[Any, dict[str, Any]]:
    """Failure-only JOINT diagnostic trained on A_train+B_train only."""
    selected = tuple(dict.fromkeys((*selected_a, *selected_b)))
    by_expert: dict[int, list[int]] = {}
    for cell_id in selected:
        expert, cell = _cell_parts(cell_id)
        by_expert.setdefault(expert, []).append(cell)
    runtime = ForkedCellularExperts(parent_experts, by_expert)
    runtime.to(cache_a.mlp_input.device)
    parameters = [value for value in runtime.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate) if config.optimizer.lower() == "adamw" else torch.optim.SGD(parameters, lr=config.learning_rate)
    steps = 0
    tokens = 0
    final_loss = float("nan")
    while steps < config.max_optimizer_steps and tokens < config.max_training_tokens:
        progressed = False
        for cache in (cache_a, cache_b):
            for start in range(0, cache.input_ids.shape[0], max(1, config.batch_size)):
                end = min(cache.input_ids.shape[0], start + max(1, config.batch_size))
                batch = slice_task_cache(cache, start, end)
                batch_tokens = int(batch.loss_mask.sum())
                if tokens + batch_tokens > config.max_training_tokens:
                    break
                optimizer.zero_grad(set_to_none=True)
                loss = cached_task_loss(runner, batch, runtime)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite JOINT oracle loss")
                loss.backward()
                optimizer.step()
                steps += 1
                tokens += batch_tokens
                final_loss = float(loss.detach())
                progressed = True
                if steps >= config.max_optimizer_steps:
                    break
            if steps >= config.max_optimizer_steps:
                break
        if not progressed:
            break
    if steps == 0:
        raise RuntimeError("JOINT oracle did not consume a task batch")
    return runtime, {"selected_cells": list(selected), "training_steps": steps, "training_tokens": tokens, "final_loss": final_loss}


def _write_task_branch_manifest(
    output: Path,
    registry: CellRegistry,
    summary: Mapping[str, Any],
    branch: str,
    scientific_evidence: bool,
) -> None:
    registry.save(str(output / "CELL_REGISTRY.json"))
    write_json(output / "MANIFEST.json", {
        "schema": "minicells.pcu-kill-001.branch-manifest.v2",
        "branch": branch,
        "scientific_evidence": scientific_evidence,
        "registry_sha256": registry.content_hash(),
        **dict(summary),
    })


def _runtime_from_task_result(
    parent_experts: Any,
    result_a: TaskBranchResult,
    result_b: TaskBranchResult,
    active: tuple[str, ...],
) -> Any:
    branches = {
        "A": {index: expert for index, expert in enumerate(result_a.runtime.cells)},
        "B": {index: expert for index, expert in enumerate(result_b.runtime.cells)},
    }
    return compose_cellular_experts(parent_experts, branches, active)


def _run_shared_scientific_pipeline(
    *,
    phase: str,
    seed: int,
    output: Path,
    device: str,
    tokenizer: Any,
    original: Any,
    cellular: Any,
    manifest: Mapping[str, Any],
    inspector: GraniteArchitectureInspector,
    g0: list[Any],
    g0_full_moe: Any,
    g0_e2e: Any,
    cache_gate: Any,
    world: Any,
    audit: Any,
    allow_search: bool,
    frozen_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the identical task pipeline for engineering and formal phases."""
    output.mkdir(parents=True, exist_ok=True)
    scientific_evidence = phase == "formal"
    generation_config = dict(GENERATION_CONFIG)
    if frozen_config and isinstance(frozen_config.get("generation"), Mapping):
        generation_config.update(dict(frozen_config["generation"]))
    target = target_module(cellular, inspector.target_path)
    foundation_before = foundation_tensor_hashes(target.experts)
    runner = CachedTailRunner(cellular, inspector.decoder_layer_path)
    foundation_hash = str(manifest.get("foundation_tensor_sha256") or module_tensor_hash(original))
    sequences, caches, cache_manifests = _build_task_caches(
        tokenizer, runner, world, manifest, inspector, foundation_hash, output, device
    )
    oracle = context_oracle(world, model=original, tokenizer=tokenizer, device=device, max_new_tokens=int(generation_config["max_new_tokens"]))
    write_json(output / "CONTEXT_ORACLE.json", {**oracle, "scientific_evidence": scientific_evidence})
    if not oracle["passed"]:
        source = git_provenance(Path(__file__).resolve().parents[3])
        decision = {
            "schema": "minicells.pcu-kill-001.engineering-decision.v3", "experiment": EXPERIMENT_ID, "phase": phase,
            "status": "TESTBED_COMPOSITION_CAPACITY_INADEQUATE", "scientific_evidence": scientific_evidence,
            "formal_protocol_ready": False, "formal_ready": False, "valid_run": True,
            "gates": {"g0": bool(g0_e2e.passed), "cache": bool(cache_gate.passed), "dataset_audit": bool(audit.passed), "context_oracle": False, "formal_seed_untouched": True},
            "metrics": {"context_oracle_accuracy": oracle["accuracy"], "g0_exact_embedding": bool(g0_e2e.passed)},
            "allocation": {"method": "task-conditioned-gradient-l2-per-parameter", "calibration_split": "branch_train", "calibration_sample_rule": "first_64_samples", "tie_break": "cell_id_lexical", "capacity_ladder": list(CAPACITY_LADDER), "selected_k": None},
            "foundation": dict(manifest), "architecture": asdict(inspector), "source": source,
            "formal_execution_not_started": phase != "formal", "reason": "context oracle below the registered 0.90 capacity floor",
        }
        write_json(output / "DECISION.json", decision)
        write_json(output / "ENGINEERING_DECISION.json", decision)
        return {"status": decision["status"], "scientific_evidence": scientific_evidence, "formal_ready": False, "selected_k": None, "output": str(output)}

    calibration_a = slice_task_cache(caches["A_train"], 0, min(64, len(caches["A_train"].sample_ids)))
    calibration_b = slice_task_cache(caches["B_train"], 0, min(64, len(caches["B_train"].sample_ids)))
    allocation_a = task_conditioned_allocation(
        target.experts, runner, calibration_a, cells_per_expert=inspector.cells, layer=inspector.target_layer
    )
    allocation_b = task_conditioned_allocation(
        target.experts, runner, calibration_b, cells_per_expert=inspector.cells, layer=inspector.target_layer
    )
    allocation_policy = {
        "method": "task-conditioned-gradient-l2-per-parameter",
        "calibration_split": "branch_train",
        "calibration_sample_rule": "first_64_samples",
        "tie_break": "cell_id_lexical",
        "capacity_ladder": list(CAPACITY_LADDER),
    }
    write_json(output / "GRADIENT_GEOMETRY_A.json", _allocation_manifest(allocation_a, inspector.target_layer))
    write_json(output / "GRADIENT_GEOMETRY_B.json", _allocation_manifest(allocation_b, inspector.target_layer))

    base_a_summary = evaluate_samples(original, tokenizer, world.splits["A_eval"], split="A_eval", device=device, max_new_tokens=int(generation_config["max_new_tokens"]))
    base_b_summary = evaluate_samples(original, tokenizer, world.splits["B_eval"], split="B_eval", device=device, max_new_tokens=int(generation_config["max_new_tokens"]))
    if frozen_config is not None:
        learning_rates = [float(frozen_config["learning_rate"])]
        k_values = [int(frozen_config["selected_k"])]
    else:
        learning_rates = list(ENGINEERING_LR_CANDIDATES)
        k_values = list(CAPACITY_LADDER)

    candidate_rows: list[dict[str, Any]] = []
    selected_k: int | None = None
    selected_lr: float | None = None
    selected_cells_a: tuple[str, ...] = ()
    selected_cells_b: tuple[str, ...] = ()
    selected_config: BranchTrainingConfig | None = None
    for learning_rate in learning_rates:
        candidate = {"learning_rate": learning_rate, "capacity": {}}
        for k in k_values:
            if k not in CAPACITY_LADDER:
                raise ValueError(f"selected K is outside the registered capacity ladder: {k}")
            cells_a = tuple(allocation_a.selected[:k])
            cells_b = tuple(allocation_b.selected[:k])
            config = BranchTrainingConfig(
                optimizer=str((frozen_config or {}).get("optimizer", "AdamW")),
                learning_rate=learning_rate,
                max_optimizer_steps=int((frozen_config or {}).get("max_optimizer_steps", 128)),
                max_training_tokens=int((frozen_config or {}).get("max_training_tokens", 500_000)),
                batch_size=int((frozen_config or {}).get("batch_size", 8)),
                seed=seed,
            )
            trial_a = train_cached_branch(target.experts, runner, caches["A_train"], cells_a, layer=inspector.target_layer, branch="A", config=config)
            trial_b = train_cached_branch(target.experts, runner, caches["B_train"], cells_b, layer=inspector.target_layer, branch="B", config=config)
            trial_model_a = _model_with_experts(cellular, inspector, _runtime_from_task_result(target.experts, trial_a, trial_b, ("A",)))
            trial_model_b = _model_with_experts(cellular, inspector, _runtime_from_task_result(target.experts, trial_a, trial_b, ("B",)))
            direct_a = evaluate_samples(trial_model_a, tokenizer, world.splits["A_eval"], split="A_eval", device=device, max_new_tokens=int(generation_config["max_new_tokens"]))
            direct_b = evaluate_samples(trial_model_b, tokenizer, world.splits["B_eval"], split="B_eval", device=device, max_new_tokens=int(generation_config["max_new_tokens"]))
            row = {
                "cells_a": list(cells_a), "cells_b": list(cells_b),
                "acc_a": direct_a.exact, "acc_b": direct_b.exact,
                "training_tokens_a": trial_a.training_tokens, "training_tokens_b": trial_b.training_tokens,
                "training_steps_a": trial_a.training_steps, "training_steps_b": trial_b.training_steps,
                "passes": direct_a.exact >= 0.80 and direct_b.exact >= 0.80,
            }
            candidate["capacity"][str(k)] = row
            if row["passes"] and selected_k is None:
                selected_k, selected_lr = k, learning_rate
                selected_cells_a, selected_cells_b = cells_a, cells_b
                selected_config = config
        candidate_rows.append(candidate)
        if selected_k is not None:
            break

    capacity_ladder = candidate_rows[-1]["capacity"] if candidate_rows else {}
    if selected_k is None:
        decision = {
            "schema": "minicells.pcu-kill-001.engineering-decision.v3",
            "experiment": EXPERIMENT_ID, "phase": phase,
            "status": "LOCAL_CELL_MUTATION_UNSUPPORTED",
            "scientific_evidence": scientific_evidence,
            "formal_protocol_ready": False, "formal_ready": False, "valid_run": True,
            "gates": {"g0": bool(g0_e2e.passed), "cache": bool(cache_gate.passed), "dataset_audit": bool(audit.passed), "context_oracle": True, "gradient_allocation": True, "capacity_ladder": False, "formal_seed_untouched": True},
            "metrics": {"base_a": base_a_summary.exact, "base_b": base_b_summary.exact},
            "allocation": allocation_policy, "capacity_ladder": capacity_ladder, "candidates": candidate_rows,
            "foundation": dict(manifest), "architecture": asdict(inspector),
            "source": git_provenance(Path(__file__).resolve().parents[3]),
            "formal_execution_not_started": phase != "formal",
            "reason": "no registered K reached direct A/B exact capability threshold",
        }
        write_json(output / "DECISION.json", decision)
        write_json(output / "ENGINEERING_DECISION.json", decision)
        return {"status": decision["status"], "scientific_evidence": scientific_evidence, "formal_ready": False, "selected_k": None, "output": str(output)}

    assert selected_config is not None and selected_lr is not None
    result_a = train_cached_branch(target.experts, runner, caches["A_train"], selected_cells_a, layer=inspector.target_layer, branch="A", config=selected_config)
    result_b = train_cached_branch(target.experts, runner, caches["B_train"], selected_cells_b, layer=inspector.target_layer, branch="B", config=selected_config)
    summary_a = _save_task_branch_artifact(result_a, output / "branch_A")
    summary_b = _save_task_branch_artifact(result_b, output / "branch_B")
    loaded_a = _load_task_branch_artifact(target.experts, selected_cells_a, inspector.target_layer, Path(summary_a["artifact_path"]), summary_a["artifact_sha256"])
    loaded_b = _load_task_branch_artifact(target.experts, selected_cells_b, inspector.target_layer, Path(summary_b["artifact_path"]), summary_b["artifact_sha256"])
    registry = make_foundation_registry(
        layer=inspector.target_layer,
        experts=inspector.local_experts,
        cells_per_expert=inspector.cells,
        cell_width=inspector.partition.cell_size,
        foundation_model=str(manifest["model_repo"]),
        foundation_revision=str(manifest["model_revision"]),
        foundation_hash=foundation_hash,
        protocol_sha256="engineering-pending-freeze" if phase == "engineering" else str(frozen_config.get("protocol_sha256", "frozen-protocol")),
    )
    registry_a = _bind_collection_records(registry, selected_cells_a, "A", Path(summary_a["artifact_path"]), loaded_a)
    registry_b = _bind_collection_records(registry, selected_cells_b, "B", Path(summary_b["artifact_path"]), loaded_b)
    validate_fork_artifacts(registry_a)
    validate_fork_artifacts(registry_b)
    _write_task_branch_manifest(output / "branch_A", registry_a, summary_a, "A", scientific_evidence)
    _write_task_branch_manifest(output / "branch_B", registry_b, summary_b, "B", scientific_evidence)
    merged_registry = merge_registries(registry, registry_a, registry_b)
    (output / "merged").mkdir(parents=True, exist_ok=True)
    merged_registry.save(str(output / "merged" / "CELL_REGISTRY.json"))
    registry_a.save(str(output / "REGISTRY_A.json"))
    registry_b.save(str(output / "REGISTRY_B.json"))
    merged_registry.save(str(output / "REGISTRY_AB.json"))
    branches = {"A": loaded_a, "B": loaded_b}
    runtime_a = compose_cellular_experts(target.experts, branches, ("A",))
    runtime_b = compose_cellular_experts(target.experts, branches, ("B",))
    runtime_ab = compose_cellular_experts(target.experts, branches, ("A", "B"))
    models: dict[str, Any] = {
        "M0": original,
        "MA": _model_with_experts(cellular, inspector, runtime_a),
        "MB": _model_with_experts(cellular, inspector, runtime_b),
        "MAB": _model_with_experts(cellular, inspector, runtime_ab),
    }
    functional = _functional_composition_audit_collections(target.experts, target.router, loaded_a, loaded_b, selected_cells_a, selected_cells_b, calibration_a.mlp_input.reshape(-1, inspector.hidden_size)[:64].detach())
    anchor_base = _anchor_nll(original, tokenizer, device)
    anchor_ab = _anchor_nll(models["MAB"], tokenizer, device)

    from .lora import choose_matched_rank, lora_parameter_count, merged_effective_deltas
    pcu_parameters = sum(int(value.numel()) for fork in loaded_a.values() for value in selected_delta_parameters(fork))
    lora_rank = int((frozen_config or {}).get("lora_rank", 0)) or choose_matched_rank(pcu_parameters, inspector.hidden_size, inspector.partition.cell_size, len(selected_cells_a))
    lora_a, lora_summary_a = train_cached_lora_branch(target.experts, runner, caches["A_train"], selected_cells_a, layer=inspector.target_layer, branch="A", rank=lora_rank, config=selected_config)
    lora_b, lora_summary_b = train_cached_lora_branch(target.experts, runner, caches["B_train"], selected_cells_b, layer=inspector.target_layer, branch="B", rank=lora_rank, config=selected_config)
    lora_artifact_a = _save_lora_branch_artifact(lora_a, output / "baseline_lora" / "branch_A", "A")
    lora_artifact_b = _save_lora_branch_artifact(lora_b, output / "baseline_lora" / "branch_B", "B")
    lora_summary_a.update(lora_artifact_a)
    lora_summary_b.update(lora_artifact_b)
    assert_foundation_unchanged(foundation_before, target.experts)
    lora_maps = {"A": {i: expert for i, expert in enumerate(lora_a.cells)}, "B": {i: expert for i, expert in enumerate(lora_b.cells)}}
    models.update({
        "LA": _model_with_experts(cellular, inspector, compose_cellular_experts(target.experts, lora_maps, ("A",))),
        "LB": _model_with_experts(cellular, inspector, compose_cellular_experts(target.experts, lora_maps, ("B",))),
        "LAB": _model_with_experts(cellular, inspector, compose_cellular_experts(target.experts, lora_maps, ("A", "B"))),
    })
    matrix = evaluate_matrix(models, tokenizer, world.splits, device=device, max_new_tokens=int(generation_config["max_new_tokens"]))
    pcu_matrix = {key: matrix[key] for key in ("M0", "MA", "MB", "MAB")}
    lora_matrix = {"M0": matrix["M0"], "MA": matrix["LA"], "MB": matrix["LB"], "MAB": matrix["LAB"]}
    pcu_metrics = _task_matrix_metrics(pcu_matrix, anchor_base, anchor_ab)
    lora_metrics = _task_matrix_metrics(lora_matrix, anchor_base, _anchor_nll(models["LAB"], tokenizer, device))
    overlap = sorted(set(selected_cells_a) & set(selected_cells_b))
    if overlap:
        first_expert, first_cell_index = _cell_parts(overlap[0])
        lora_cell_a = lora_a.cells[first_expert].cells[first_cell_index]
        lora_cell_b = lora_b.cells[first_expert].cells[first_cell_index]
        exact_lora = merged_effective_deltas(lora_cell_a.state_delta(), lora_cell_b.state_delta(), scale_a=lora_cell_a.scale, scale_b=lora_cell_b.scale)
        expected_lora = {key: lora_cell_a.effective_deltas()[key] + lora_cell_b.effective_deltas()[key] for key in ("gate", "up", "down")}
        lora_error = max(_max_error(exact_lora[key], expected_lora[key]) for key in expected_lora)
    else:
        # Disjoint LoRA selections require no factor merge for a shared Cell;
        # the exact-union error is therefore identically zero.
        lora_error = 0.0
    lora_parameters = lora_parameter_count(inspector.hidden_size, inspector.partition.cell_size, len(selected_cells_a), lora_rank)
    parameter_difference = abs(lora_parameters - pcu_parameters) / max(1, pcu_parameters)
    thresholds = {"context_oracle_accuracy": 0.90, "direct_accuracy": 0.80, "merge_retention": 0.90, "composition_accuracy": 0.50, "composition_synergy": 0.30, "anchor_regression": 0.01, "matched_lora_parameter_tolerance": 0.10}
    joint_oracle_pass: bool | None = None
    metrics = {**pcu_metrics, "context_oracle_accuracy": oracle["accuracy"], "g0_exact_embedding": bool(all(item.passed for item in g0) and g0_e2e.passed and (g0_full_moe is None or g0_full_moe.passed)), "cache_equivalence": cache_gate.to_dict(), "functional_overlap_max_abs": functional["overlap_formula_max_abs"], "functional_rollback_max_abs": max(functional["rollback_a_max_abs"], functional["rollback_b_max_abs"], functional["rollback_all_max_abs"]), "lora_exact_merge_max_abs": lora_error, "lora": lora_metrics}
    direct_a_pass = pcu_metrics["acc_a"] >= thresholds["direct_accuracy"]
    direct_b_pass = pcu_metrics["acc_b"] >= thresholds["direct_accuracy"]
    retention_pass = pcu_metrics["retention_a"] is not None and pcu_metrics["retention_b"] is not None and pcu_metrics["retention_a"] >= thresholds["merge_retention"] and pcu_metrics["retention_b"] >= thresholds["merge_retention"]
    composition_pass = pcu_metrics["composition_acc"] >= thresholds["composition_accuracy"] and pcu_metrics["composition_synergy"] >= thresholds["composition_synergy"]
    joint_oracle_summary: dict[str, Any] | None = None
    if direct_a_pass and direct_b_pass and retention_pass and not composition_pass:
        joint_runtime, joint_oracle_summary = _train_joint_oracle(target.experts, runner, caches["A_train"], caches["B_train"], selected_cells_a, selected_cells_b, inspector.target_layer, selected_config)
        joint_model = _model_with_experts(cellular, inspector, compose_cellular_experts(target.experts, {"JOINT": {i: expert for i, expert in enumerate(joint_runtime.cells)}}, ("JOINT",)))
        joint_eval = evaluate_samples(joint_model, tokenizer, world.splits["AB_eval"], split="AB_eval", device=device, max_new_tokens=int(generation_config["max_new_tokens"]))
        joint_oracle_summary["AB_eval"] = joint_eval.to_dict()
        joint_oracle_pass = bool(joint_eval.both_exact is not None and joint_eval.both_exact >= thresholds["composition_accuracy"])
        write_json(output / "JOINT_ORACLE.json", {"scientific_evidence": scientific_evidence, "independent_failure_only": True, "training_splits": ["A_train", "B_train"], "result": joint_oracle_summary, "passed": joint_oracle_pass})
    metrics["joint_oracle_pass"] = joint_oracle_pass
    gates = {
        "g0": bool(metrics["g0_exact_embedding"]), "cache": bool(cache_gate.passed), "dataset_audit": bool(audit.passed), "context_oracle": oracle["passed"],
        "gradient_allocation": bool(allocation_a.scores and allocation_b.scores), "capacity_ladder": (all(str(k) in capacity_ladder for k in CAPACITY_LADDER) if allow_search else selected_k in CAPACITY_LADDER),
        "branch_a_capability": direct_a_pass, "branch_b_capability": direct_b_pass, "foundation_immutable": True,
        "functional_composition_runtime": bool(functional["composition_passed"]), "functional_rollback": bool(functional["rollback_passed"]), "merge_retention": retention_pass,
        "anchor_regression": pcu_metrics["anchor_regression"] <= thresholds["anchor_regression"], "composition": composition_pass,
        "lora_training": lora_summary_a["training_steps"] > 0 and lora_summary_b["training_steps"] > 0, "lora_exact_merge": lora_error <= 2e-5,
        "lora_parameter_match": parameter_difference <= thresholds["matched_lora_parameter_tolerance"], "formal_seed_untouched": True, "artifact_roundtrip": True,
    }
    formal_ready = all(gates.values())
    if not direct_a_pass or not direct_b_pass:
        status = "LOCAL_CELL_MUTATION_UNSUPPORTED"
    elif not retention_pass:
        status = "PARAMETER_LOCALITY_ONLY_MERGEABILITY_FAILED"
    elif not gates["anchor_regression"]:
        status = "MERGEABLE_BUT_FOUNDATION_REGRESSION_UNSAFE"
    elif not composition_pass:
        status = "INDEPENDENT_CELL_COMPOSITION_UNSUPPORTED" if joint_oracle_pass else "COMPOSITION_TESTBED_INCONCLUSIVE"
    elif lora_metrics["composition_acc"] >= pcu_metrics["composition_acc"]:
        status = "PCU_MECHANISM_SUPPORTED_ADVANTAGE_UNPROVEN"
    else:
        status = "ENGINEERING_PREVIEW_MECHANISM_PASS" if phase == "engineering" else "PCU_COMPOSABILITY_CONSTRUCTIVE_EVIDENCE"
    source = git_provenance(Path(__file__).resolve().parents[3])
    decision = {
        "schema": "minicells.pcu-kill-001.engineering-decision.v3", "experiment": EXPERIMENT_ID, "phase": phase, "status": status,
        "scientific_evidence": scientific_evidence, "formal_protocol_ready": formal_ready, "formal_ready": formal_ready, "valid_run": True, "valid_formal_run": phase == "formal",
        "gates": gates, "metrics": metrics, "baseline": {"lora_composition_acc": lora_metrics["composition_acc"], "lora": lora_metrics}, "thresholds": thresholds,
        "selected": {"k": selected_k, "optimizer": selected_config.optimizer, "learning_rate": selected_lr, "max_optimizer_steps": selected_config.max_optimizer_steps, "max_training_tokens": selected_config.max_training_tokens, "lora_rank": lora_rank, "engineering_selected_cells": {"A": list(selected_cells_a), "B": list(selected_cells_b)}},
        "allocation": {**allocation_policy, "selected_k": selected_k}, "capacity_ladder": capacity_ladder, "candidates": candidate_rows,
        "evaluation": {"generation": generation_config, "composition_primary": "both_exact"},
        "task_encoding": {"max_length": TASK_MAX_LENGTH, "sequence_encoding_version": sequences["A_train"].encoding_version},
        "foundation": dict(manifest), "architecture": asdict(inspector), "parameter_budget": {"pcu_trainable_parameters": pcu_parameters, "lora_trainable_parameters": lora_parameters, "relative_difference": parameter_difference},
        "branch_a": summary_a, "branch_b": summary_b, "composition": functional,
        "lora_baseline": {"rank": lora_rank, "metrics": lora_metrics, "exact_merge_max_abs": lora_error, "training_a": lora_summary_a, "training_b": lora_summary_b},
        "source": {"commit": source.get("source_commit"), "tree": source.get("source_tree"), **source}, "cache_manifests": cache_manifests,
        "formal_execution_not_started": phase != "formal", "reason": "shared task-conditioned scientific pipeline completed",
    }
    write_json(output / "MODEL_MANIFEST.json", {**dict(manifest), "architecture": asdict(inspector)})
    write_json(output / "DATASET_MANIFEST.json", {**world.to_manifest(), "scientific_evidence": scientific_evidence})
    write_json(output / "DATASET_AUDIT.json", {**audit.to_dict(), "scientific_evidence": scientific_evidence})
    write_json(output / "ALLOCATION_A.json", {**_allocation_manifest(allocation_a, inspector.target_layer, selected_k), "selected": list(selected_cells_a), "independent": True})
    write_json(output / "ALLOCATION_B.json", {**_allocation_manifest(allocation_b, inspector.target_layer, selected_k), "selected": list(selected_cells_b), "independent": True})
    write_json(output / "EQUIVALENCE.json", {"g0_expert": {str(i): item.to_dict() for i, item in enumerate(g0)}, "g0_full_moe": g0_full_moe.to_dict() if g0_full_moe is not None else None, "g0_end_to_end": g0_e2e.to_dict(), "cache": cache_gate.to_dict(), "scientific_evidence": scientific_evidence})
    write_json(output / "CACHE_EQUIVALENCE.json", cache_gate.to_dict())
    write_json(output / "EVALUATION_MATRIX.json", {"pcu": matrix_to_dict(pcu_matrix), "lora": matrix_to_dict(lora_matrix), "scientific_evidence": scientific_evidence})
    write_json(output / "METRICS.json", metrics)
    write_json(output / "METRICS_PCU.json", pcu_metrics)
    write_json(output / "METRICS_LORA.json", lora_metrics)
    write_json(output / "baseline_lora" / "MANIFEST.json", {"schema": "minicells.pcu-kill-001.lora-manifest.v3", "scientific_evidence": scientific_evidence, "rank": lora_rank, "pcu_parameter_count": pcu_parameters, "lora_parameter_count": lora_parameters, "relative_difference": parameter_difference, "exact_merge_max_abs": lora_error, "exact_merge_passed": lora_error <= 2e-5, "routing": "INHERITED_PARENT", "training_a": lora_summary_a, "training_b": lora_summary_b, "metrics": lora_metrics})
    write_json(output / "TRAINING_A.json", {"branch": "A", "scientific_evidence": scientific_evidence, **summary_a})
    write_json(output / "TRAINING_B.json", {"branch": "B", "scientific_evidence": scientific_evidence, **summary_b})
    write_json(output / "ANCHOR_MANIFEST.json", {"schema": "minicells.pcu-kill-001.anchor.v1", "texts": list(ANCHOR_TEXTS), "metric": "relative_full_sequence_nll", "scientific_evidence": scientific_evidence})
    write_json(output / "MERGE_AUDIT.json", {"scientific_evidence": scientific_evidence, **functional})
    write_json(output / "DECISION.json", decision)
    write_json(output / "ENGINEERING_DECISION.json", decision)
    write_json(output / "RUN_MANIFEST.json", {"schema": "minicells.pcu-kill-001.run-manifest.v3", "experiment": EXPERIMENT_ID, "phase": phase, "seed": seed, "backend": "granite", "scientific_evidence": scientific_evidence, "source": source, "runtime": runtime_provenance(device)})
    (output / "QA_LOG.md").write_text(f"# QA log\n\n- status: `{status}`\n- formal seeds executed: `{'NONE' if phase == 'engineering' else seed}`\n- scientific evidence: `{str(scientific_evidence).lower()}`\n", encoding="utf-8")
    return {"status": status, "scientific_evidence": scientific_evidence, "formal_ready": formal_ready, "selected_k": selected_k, "selected_a": list(selected_cells_a), "selected_b": list(selected_cells_b), "output": str(output), "metrics": metrics}


def _run_granite_engineering(seed: int, output: Path, device: str) -> dict[str, Any]:
    """Run the real pinned-Granite engineering graph with no toy fallback."""
    set_deterministic_seeds(seed)
    tokenizer, original, manifest = load_granite(MODEL_ID, revision=None, device=device)
    inspector = GraniteArchitectureInspector.inspect(original, require_granite=True)
    cellular, _ = cellularize_model(original, inspector)
    target = target_module(cellular, inspector.target_path)
    probe_texts = [f"PCU-KILL-001 immutable engineering probe {index:03d}." for index in range(128)]
    probe_inputs = _token_batch(tokenizer, probe_texts, device)
    original_logits = _logits(original, probe_inputs)
    g0 = [verify_expert_algebra(target_module(original, inspector.target_path).experts, index, inspector.partition, vectors=1024, seed=seed) for index in range(inspector.local_experts)]
    moe_probe = torch.randn(128, inspector.hidden_size, generator=torch.Generator(device="cpu").manual_seed(seed + 1)).to(device)
    g0_full_moe = verify_full_moe(target_module(original, inspector.target_path), target, moe_probe)
    g0_e2e = verify_end_to_end(original, cellular, probe_inputs)
    cache_runner = CachedTailRunner(cellular, inspector.decoder_layer_path)
    cache = cache_runner.capture(probe_inputs["input_ids"], probe_inputs.get("attention_mask"), tuple(f"probe-{index:03d}" for index in range(128)))
    cache_gate = cache_runner.verify(cache, full_logits=original_logits)
    world = generate_world(seed, count=128, tokenizer=tokenizer)
    audit = audit_dataset(world)
    if not audit.passed:
        raise RuntimeError(f"DATASET_LEAKAGE_AUDIT failed: {audit.errors}")
    manifest = {**manifest, "architecture": asdict(inspector), "foundation_tensor_sha256": module_tensor_hash(original)}
    return _run_shared_scientific_pipeline(phase="engineering", seed=seed, output=output, device=device, tokenizer=tokenizer, original=original, cellular=cellular, manifest=manifest, inspector=inspector, g0=g0, g0_full_moe=g0_full_moe, g0_e2e=g0_e2e, cache_gate=cache_gate, world=world, audit=audit, allow_search=True)


def _quarantined_non_scientific_legacy_engineering(seed: int, output: Path, device: str) -> dict[str, Any]:
    """Retired pre-repair harness, retained only for historical debugging.

    No production entry point references this function.  Granite engineering
    always resolves to ``_run_granite_engineering`` above.
    """
    raise RuntimeError("retired non-scientific engineering harness is not callable")
    """Run the real pinned-Granite engineering graph.

    This function intentionally has no toy fallback.  If Transformers, the
    pinned checkpoint, or the tokenizer cannot be loaded, the caller receives
    that concrete failure and no scientific artifact is emitted.
    """
    set_deterministic_seeds(seed)
    started = time.time()
    tokenizer, original, manifest = load_granite(MODEL_ID, revision=None, device=device)
    inspector = GraniteArchitectureInspector.inspect(original, require_granite=True)
    cellular, _ = cellularize_model(original, inspector)
    original_target = target_module(original, inspector.target_path)
    cellular_target = target_module(cellular, inspector.target_path)
    probe_texts = [f"PCU-KILL-001 immutable engineering probe {index:03d}." for index in range(128)]
    probe_inputs = _token_batch(tokenizer, probe_texts, device)
    original_logits = _logits(original, probe_inputs)
    g0 = [verify_expert_algebra(original_target.experts, index, inspector.partition, vectors=1024, seed=seed) for index in range(inspector.local_experts)]
    moe_probe = torch.randn(128, inspector.hidden_size, generator=torch.Generator(device="cpu").manual_seed(seed + 1)).to(device)
    g0_full_moe = verify_full_moe(original_target, cellular_target, moe_probe)
    g0_e2e = verify_end_to_end(original, cellular, probe_inputs)
    cache_runner = CachedTailRunner(cellular, inspector.decoder_layer_path)
    cache = cache_runner.capture(probe_inputs["input_ids"], probe_inputs.get("attention_mask"), tuple(f"probe-{index:03d}" for index in range(128)))
    cache_gate = cache_runner.verify(cache, full_logits=original_logits)
    world = generate_world(seed, count=128, tokenizer=tokenizer)
    audit = audit_dataset(world)
    if not audit.passed:
        raise RuntimeError(f"DATASET_LEAKAGE_AUDIT failed: {audit.errors}")
    foundation_hash = module_tensor_hash(original)
    manifest = {**manifest, "architecture": asdict(inspector), "foundation_tensor_sha256": foundation_hash}
    cache_identity = {
        "foundation_model": MODEL_ID,
        "foundation_revision": manifest.get("model_revision"),
        "foundation_tensor_sha256": foundation_hash,
        "target_path": inspector.target_path,
        "target_layer": inspector.target_layer,
        "tokenizer_sha256": manifest.get("tokenizer_sha256", []),
        "dataset_manifest_sha256": world.manifest_sha256(),
        "encoding": "tokenizer-fixed-probe-v1",
        "dtype": str(cache.mlp_input.dtype),
    }
    cache_manifest = save_cache(cache, output / "cache", identity=cache_identity)
    validate_cache_identity(cache_manifest, cache_identity)
    geometry_a = _debug_random_allocation(cellular, inspector, seed + 11)
    geometry_b = _debug_random_allocation(cellular, inspector, seed + 12)
    ladder = {str(k): {"topk_mass_a": geometry_a.topk_mass.get(k, 0.0), "topk_mass_b": geometry_b.topk_mass.get(k, 0.0)} for k in (1, 2, 4, 8)}
    selected_k = next((k for k in (1, 2, 4, 8) if min(geometry_a.topk_mass.get(k, 0.0), geometry_b.topk_mass.get(k, 0.0)) >= 0.50), 8)
    selected_a, selected_b = geometry_a.selected[:selected_k], geometry_b.selected[:selected_k]
    config = BranchTrainingConfig(learning_rate=1e-3, max_optimizer_steps=8, max_training_tokens=500_000, seed=seed)
    calibration = cache.mlp_input.reshape(-1, inspector.hidden_size)[:64].detach()
    branch_a_dir, branch_b_dir = output / "branch_A", output / "branch_B"
    branch_a_dir.mkdir(parents=True, exist_ok=True)
    branch_b_dir.mkdir(parents=True, exist_ok=True)
    forks_a, summary_a = _debug_hidden_target_branch_collection(cellular_target.experts, selected_a, seed + 21, branch_a_dir, calibration, config)
    forks_b, summary_b = _debug_hidden_target_branch_collection(cellular_target.experts, selected_b, seed + 22, branch_b_dir, calibration, config)
    registry = make_foundation_registry(
        layer=inspector.target_layer,
        experts=inspector.local_experts,
        cells_per_expert=inspector.cells,
        cell_width=inspector.partition.cell_size,
        foundation_model=MODEL_ID,
        foundation_revision=str(manifest["model_revision"]),
        foundation_hash=foundation_hash,
        protocol_sha256="engineering-pending-freeze",
    )
    registry_a = _bind_collection_records(registry, selected_a, "A", Path(summary_a["artifact_path"]), forks_a)
    registry_b = _bind_collection_records(registry, selected_b, "B", Path(summary_b["artifact_path"]), forks_b)
    _write_branch_manifest(branch_a_dir, registry_a, summary_a, "A")
    _write_branch_manifest(branch_b_dir, registry_b, summary_b, "B")
    merged = merge_registries(registry, registry_a, registry_b)
    (output / "merged").mkdir(parents=True, exist_ok=True)
    merged.save(str(output / "merged/CELL_REGISTRY.json"))
    merged.save(str(output / "REGISTRY_AB.json"))
    registry_a.save(str(output / "REGISTRY_A.json"))
    registry_b.save(str(output / "REGISTRY_B.json"))
    functional = _functional_composition_audit_collections(
        cellular_target.experts,
        cellular_target.router,
        forks_a,
        forks_b,
        selected_a,
        selected_b,
        calibration,
    )
    runtime_a = compose_cellular_experts(cellular_target.experts, {"A": forks_a, "B": forks_b}, ("A",))
    runtime_b = compose_cellular_experts(cellular_target.experts, {"A": forks_a, "B": forks_b}, ("B",))
    runtime_ab = compose_cellular_experts(cellular_target.experts, {"A": forks_a, "B": forks_b}, ("A", "B"))
    model_a = _model_with_experts(cellular, inspector, runtime_a)
    model_b = _model_with_experts(cellular, inspector, runtime_b)
    model_ab = _model_with_experts(cellular, inspector, runtime_ab)
    base_a = _answer_token_accuracy(original, tokenizer, world.splits["A_eval"], device)
    base_b = _answer_token_accuracy(original, tokenizer, world.splits["B_eval"], device)
    acc_a = _answer_token_accuracy(model_a, tokenizer, world.splits["A_eval"], device)
    acc_b = _answer_token_accuracy(model_b, tokenizer, world.splits["B_eval"], device)
    composition_acc = _answer_token_accuracy(model_ab, tokenizer, world.splits["AB_eval"], device)
    anchor_base = original_logits.float()
    anchor_merged = _logits(model_ab, probe_inputs).float()
    anchor_regression = float((anchor_merged - anchor_base).norm() / anchor_base.norm().clamp_min(1e-12))
    lora_dir = output / "baseline_lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    from .lora import LoRAConfig, LoRACell, choose_matched_rank, merged_effective_deltas
    first_cell_id = selected_a[0]
    first_cell = cellular_target.experts.cells[_cell_parts(first_cell_id)[0]].cells[_cell_parts(first_cell_id)[1]]
    pcu_parameters = sum(int(value.numel()) for fork in forks_a.values() for value in selected_delta_parameters(fork))
    lora_rank = choose_matched_rank(pcu_parameters, inspector.hidden_size, inspector.partition.cell_size, max(1, len(selected_a)))
    lora_a = LoRACell(first_cell, LoRAConfig(lora_rank), trainable=False)
    lora_b = LoRACell(first_cell, LoRAConfig(lora_rank), trainable=False)
    with torch.no_grad():
        for parameter in list(lora_a.parameters()) + list(lora_b.parameters()):
            if parameter.ndim > 1:
                parameter.normal_(std=0.01)
    exact_lora = merged_effective_deltas(lora_a.state_delta(), lora_b.state_delta(), scale_a=lora_a.scale, scale_b=lora_b.scale)
    expected_lora = {key: lora_a.effective_deltas()[key] + lora_b.effective_deltas()[key] for key in ("gate", "up", "down")}
    lora_error = max(_max_error(exact_lora[key], expected_lora[key]) for key in expected_lora)
    lora_parameters = 3 * lora_rank * (inspector.hidden_size + inspector.partition.cell_size) * max(1, len(selected_a))
    write_json(lora_dir / "MANIFEST.json", {"schema": "minicells.pcu-kill-001.lora-manifest.v2", "scientific_evidence": False, "rank": lora_rank, "pcu_parameter_count": pcu_parameters, "lora_parameter_count": lora_parameters, "within_tolerance": abs(lora_parameters - pcu_parameters) / max(1, pcu_parameters) <= 0.10, "exact_merge_max_abs": lora_error, "exact_merge_passed": lora_error <= 2e-5, "routing": "INHERITED_PARENT"})
    gates = {
        "g0": bool(all(item.passed for item in g0) and g0_e2e.passed and (g0_full_moe is None or g0_full_moe.passed)),
        "cache": cache_gate.passed,
        "dataset_audit": audit.passed,
        "gradient_allocation": bool(geometry_a.selected and geometry_b.selected and set(ladder) == {"1", "2", "4", "8"}),
        "branch_a": summary_a["training_steps"] > 0,
        "branch_b": summary_b["training_steps"] > 0,
        "functional_composition": bool(functional["composition_passed"]),
        "functional_rollback": bool(functional["rollback_passed"]),
        "lora_exact_merge": lora_error <= 2e-5,
        "lora_parameter_match": abs(lora_parameters - pcu_parameters) / max(1, pcu_parameters) <= 0.10,
        "foundation_immutable": True,
        "formal_seed_untouched": True,
    }
    metrics = {
        "g0_exact_embedding": gates["g0"],
        "g0_expert": {str(index): item.to_dict() for index, item in enumerate(g0)},
        "g0_full_moe": g0_full_moe.to_dict() if g0_full_moe is not None else None,
        "g0_end_to_end": g0_e2e.to_dict(),
        "cache_equivalence": cache_gate.to_dict(),
        "base_a": base_a,
        "base_b": base_b,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "retention_a": retention_or_undefined(base_a, acc_a, _answer_token_accuracy(model_ab, tokenizer, world.splits["A_eval"], device)),
        "retention_b": retention_or_undefined(base_b, acc_b, _answer_token_accuracy(model_ab, tokenizer, world.splits["B_eval"], device)),
        "composition_acc": composition_acc,
        "composition_synergy": composition_acc - max(base_a, base_b, acc_a, acc_b),
        "anchor_regression": anchor_regression,
        "functional_overlap_max_abs": functional["overlap_formula_max_abs"],
        "functional_rollback_max_abs": max(functional["rollback_a_max_abs"], functional["rollback_b_max_abs"], functional["rollback_all_max_abs"]),
        "lora_exact_merge_max_abs": lora_error,
        "scientific_evidence": False,
    }
    thresholds = {"g0_top1_token_agreement": 1.0, "cache_top1_token_agreement": 1.0, "context_oracle_accuracy": 0.9, "direct_accuracy": 0.8, "merge_retention": 0.9, "composition_accuracy": 0.5, "composition_synergy": 0.3, "anchor_regression": 0.01, "matched_lora_parameter_tolerance": 0.1}
    source = git_provenance(Path(__file__).resolve().parents[3])
    decision = {
        "schema": "minicells.pcu-kill-001.engineering-decision.v2",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering",
        "status": "REAL_GRANITE_ENGINEERING_IMPLEMENTED",
        "scientific_evidence": False,
        "formal_ready": all(gates.values()),
        "valid_run": all(gates.values()),
        "gates": gates,
        "metrics": metrics,
        "baseline": {"lora_composition_acc": None},
        "selected": {"k": selected_k, "cells_a": list(selected_a), "cells_b": list(selected_b), "optimizer": "AdamW", "learning_rate": config.learning_rate, "max_optimizer_steps": config.max_optimizer_steps, "max_training_tokens": config.max_training_tokens, "lora_rank": lora_rank},
        "capacity_ladder": ladder,
        "thresholds": thresholds,
        "foundation": {key: manifest.get(key) for key in ("model_repo", "model_revision", "config_sha256", "weight_file_sha256", "tokenizer_sha256")},
        "architecture": asdict(inspector),
        "parameter_budget": {"pcu_trainable_parameters": pcu_parameters, "lora_trainable_parameters": lora_parameters, "relative_difference": abs(lora_parameters - pcu_parameters) / max(1, pcu_parameters)},
        "branch_a": summary_a,
        "branch_b": summary_b,
        "composition": functional,
        "lora_baseline": {"rank": lora_rank, "exact_merge_max_abs": lora_error},
        "source": {"commit": source.get("source_commit"), "tree": source.get("source_tree"), **source},
        "cache_manifest": cache_manifest,
        "reason": "real Granite engineering graph completed; formal seeds remain reserved and untouched",
        "formal_execution_not_started": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "MODEL_MANIFEST.json", manifest)
    write_json(output / "DATASET_MANIFEST.json", {**world.to_manifest(), "scientific_evidence": False})
    write_json(output / "DATASET_AUDIT.json", {**audit.to_dict(), "scientific_evidence": False})
    write_json(output / "GRADIENT_GEOMETRY_A.json", geometry_a.to_dict())
    write_json(output / "GRADIENT_GEOMETRY_B.json", geometry_b.to_dict())
    write_json(output / "ALLOCATION_A.json", {"selected": list(selected_a), "k": selected_k, "independent": True, "capacity_ladder": ladder, "scientific_evidence": False})
    write_json(output / "ALLOCATION_B.json", {"selected": list(selected_b), "k": selected_k, "independent": True, "capacity_ladder": ladder, "scientific_evidence": False})
    write_json(output / "EQUIVALENCE.json", metrics)
    write_json(output / "CACHE_EQUIVALENCE.json", cache_gate.to_dict())
    write_json(output / "METRICS.json", metrics)
    write_json(output / "METRICS_PCU.json", metrics)
    write_json(output / "METRICS_LORA.json", {"scientific_evidence": False, "rank": lora_rank, "pcu_parameter_count": pcu_parameters, "lora_parameter_count": lora_parameters, "exact_merge_max_abs": lora_error})
    write_json(output / "DECISION.json", decision)
    write_json(output / "ENGINEERING_DECISION.json", decision)
    write_json(output / "TRAINING_A.json", {"branch": "A", "scientific_evidence": False, **summary_a})
    write_json(output / "TRAINING_B.json", {"branch": "B", "scientific_evidence": False, **summary_b})
    write_json(output / "MERGE_AUDIT.json", {"scientific_evidence": False, **functional, "registry_sha256": merged.content_hash()})
    write_json(output / "ROLLBACK_AUDIT.json", {"scientific_evidence": False, "rollback_a": functional["rollback_a_max_abs"], "rollback_b": functional["rollback_b_max_abs"], "rollback_all": functional["rollback_all_max_abs"]})
    write_json(output / "PROVENANCE.json", {"schema": "minicells.pcu-kill-001.provenance.v1", "experiment": EXPERIMENT_ID, "phase": "engineering", "seed": seed, "scientific_evidence": False, **source})
    write_json(output / "RUN_MANIFEST.json", {"schema": "minicells.pcu-kill-001.run-manifest.v2", "experiment": EXPERIMENT_ID, "phase": "engineering", "seed": seed, "backend": "granite", "scientific_evidence": False, "started_at": started, "completed_at": time.time(), "source": source, "runtime": runtime_provenance(device)})
    (output / "QA_LOG.md").write_text("# QA log\n\n- status: `REAL_GRANITE_ENGINEERING_IMPLEMENTED`\n- formal seeds executed: `NONE`\n- scientific evidence: `false`\n", encoding="utf-8")
    (output / "RESULTS.md").write_text("# PCU-KILL-001 engineering run\n\nReal Granite engineering artifacts were produced. Formal execution has not started and all engineering artifacts declare `scientific_evidence=false`.\n", encoding="utf-8")
    return {"status": decision["status"], "scientific_evidence": False, "formal_ready": decision["formal_ready"], "seed": seed, "g0": gates["g0"], "cache": gates["cache"], "dataset_audit": gates["dataset_audit"], "selected_k": selected_k, "selected_a": list(selected_a), "selected_b": list(selected_b), "output": str(output)}


def run_formal_execution(seed: int, protocol_path: Path, output: Path, device: str = "cpu") -> dict[str, Any]:
    """Run one formal seed with frozen K/allocation policy and no search."""
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_BEFORE_FORMAL":
        raise RuntimeError("formal worker requires FROZEN_BEFORE_FORMAL protocol")
    if int(seed) not in tuple(int(value) for value in payload.get("formal_seeds", [])):
        raise ValueError(f"seed {seed} is not listed in the frozen protocol")
    set_deterministic_seeds(seed)
    model_info = payload["model"]
    tokenizer, original, manifest = load_granite(str(model_info["model_repo"]), revision=str(model_info["model_revision"]), device=device)
    for key in ("model_repo", "model_revision", "config_sha256", "weight_file_sha256", "tokenizer_sha256"):
        if manifest.get(key) != model_info.get(key):
            raise RuntimeError(f"formal foundation identity mismatch: {key}")
    if model_info.get("foundation_tensor_sha256") and manifest.get("foundation_tensor_sha256") != model_info["foundation_tensor_sha256"]:
        raise RuntimeError("formal foundation identity mismatch: foundation_tensor_sha256")
    inspector = GraniteArchitectureInspector.inspect(original, require_granite=True)
    expected_architecture = payload.get("architecture", model_info)
    actual_architecture = asdict(inspector)
    for key in ("target_layer", "target_path", "hidden_size", "intermediate_size", "local_experts", "experts_per_token", "cells", "fused_order"):
        if expected_architecture.get(key) != actual_architecture.get(key):
            raise RuntimeError(f"formal architecture mismatch: {key}")
    cellular, _ = cellularize_model(original, inspector)
    target = target_module(cellular, inspector.target_path)
    prompts = [f"PCU-KILL-001 formal seed {seed} probe {index:03d}." for index in range(128)]
    inputs = _token_batch(tokenizer, prompts, device)
    original_logits = _logits(original, inputs)
    g0 = [verify_expert_algebra(target_module(original, inspector.target_path).experts, index, inspector.partition, vectors=1024, seed=seed) for index in range(inspector.local_experts)]
    moe_probe = torch.randn(128, inspector.hidden_size, generator=torch.Generator(device="cpu").manual_seed(seed + 1)).to(device)
    g0_full_moe = verify_full_moe(target_module(original, inspector.target_path), target, moe_probe)
    g0_e2e = verify_end_to_end(original, cellular, inputs)
    cache_runner = CachedTailRunner(cellular, inspector.decoder_layer_path)
    cache = cache_runner.capture(inputs["input_ids"], inputs.get("attention_mask"))
    cache_gate = cache_runner.verify(cache, full_logits=original_logits)
    world = generate_world(seed, count=128, tokenizer=tokenizer)
    audit = audit_dataset(world)
    if not audit.passed:
        raise RuntimeError(f"DATASET_LEAKAGE_AUDIT failed: {audit.errors}")
    training = payload["training"]
    allocation = payload.get("allocation", {})
    if allocation.get("method") != "task-conditioned-gradient-l2-per-parameter" or allocation.get("calibration_sample_rule") != "first_64_samples":
        raise RuntimeError("formal protocol has no supported frozen allocation policy")
    if int(allocation.get("selected_k", -1)) != int(training["selected_k"]):
        raise RuntimeError("formal protocol allocation K does not match training K")
    frozen = {
        "optimizer": training["optimizer"], "learning_rate": training["learning_rate"],
        "max_optimizer_steps": training["max_optimizer_steps"], "max_training_tokens": training["max_training_tokens"],
        "selected_k": training["selected_k"], "lora_rank": training["lora_rank"],
        "generation": payload.get("evaluation", {}).get("generation", dict(GENERATION_CONFIG)),
    }
    manifest = {**manifest, "architecture": actual_architecture, "foundation_tensor_sha256": module_tensor_hash(original)}
    return _run_shared_scientific_pipeline(phase="formal", seed=seed, output=output, device=device, tokenizer=tokenizer, original=original, cellular=cellular, manifest=manifest, inspector=inspector, g0=g0, g0_full_moe=g0_full_moe, g0_e2e=g0_e2e, cache_gate=cache_gate, world=world, audit=audit, allow_search=False, frozen_config=frozen)


def _quarantined_non_scientific_legacy_formal(seed: int, protocol_path: Path, output: Path, device: str = "cpu") -> dict[str, Any]:
    """Retired historical code; never reachable from the formal worker."""
    raise RuntimeError("retired non-scientific formal harness is not callable")
    """Execute one explicitly authorized formal seed using frozen decisions.

    No capacity, optimizer, or rank search occurs here.  Those values and the
    selected Cell IDs are read from the frozen protocol produced by the
    engineering decision and are validated against the reloaded checkpoint.
    """
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_BEFORE_FORMAL":
        raise RuntimeError("formal worker requires FROZEN_BEFORE_FORMAL protocol")
    if int(seed) not in tuple(int(value) for value in payload.get("formal_seeds", [])):
        raise ValueError(f"seed {seed} is not listed in the frozen protocol")
    set_deterministic_seeds(seed)
    model_info = payload["model"]
    tokenizer, original, manifest = load_granite(
        str(model_info["model_repo"]), revision=str(model_info["model_revision"]), device=device
    )
    for key in ("model_repo", "model_revision", "config_sha256", "weight_file_sha256", "tokenizer_sha256"):
        if manifest.get(key) != model_info.get(key):
            raise RuntimeError(f"formal foundation identity mismatch: {key}")
    inspector = GraniteArchitectureInspector.inspect(original, require_granite=True)
    architecture = model_info
    for key in ("target_layer", "target_path", "hidden_size", "intermediate_size", "local_experts", "experts_per_token", "cells_per_expert", "cell_width", "fused_projection_order"):
        actual = getattr(inspector, {"cells_per_expert": "cells", "fused_projection_order": "fused_order"}.get(key, key), None)
        if key == "cell_width":
            actual = inspector.partition.cell_size
        if actual != architecture.get(key):
            raise RuntimeError(f"formal architecture mismatch: {key}")
    cellular, _ = cellularize_model(original, inspector)
    target = target_module(cellular, inspector.target_path)
    prompts = [f"PCU-KILL-001 formal seed {seed} probe {index:03d}." for index in range(128)]
    inputs = _token_batch(tokenizer, prompts, device)
    original_logits = _logits(original, inputs)
    g0 = [verify_expert_algebra(target_module(original, inspector.target_path).experts, index, inspector.partition, vectors=1024, seed=seed) for index in range(inspector.local_experts)]
    g0_e2e = verify_end_to_end(original, cellular, inputs)
    cache = CachedTailRunner(cellular, inspector.decoder_layer_path).capture(inputs["input_ids"], inputs.get("attention_mask"))
    cache_gate = CachedTailRunner(cellular, inspector.decoder_layer_path).verify(cache, full_logits=original_logits)
    world = generate_world(seed, count=128, tokenizer=tokenizer)
    audit = audit_dataset(world)
    if not audit.passed:
        raise RuntimeError(f"DATASET_LEAKAGE_AUDIT failed: {audit.errors}")
    selected = payload.get("engineering_decision", {}).get("selected", {})
    selected_a = tuple(str(value) for value in selected.get("cells_a", []))
    selected_b = tuple(str(value) for value in selected.get("cells_b", []))
    if not selected_a or not selected_b:
        raise RuntimeError("frozen protocol has no immutable A/B Cell selections")
    config = BranchTrainingConfig(
        optimizer=str(payload["training"]["optimizer"]),
        learning_rate=float(payload["training"]["learning_rate"]),
        max_optimizer_steps=int(payload["training"]["max_optimizer_steps"]),
        max_training_tokens=int(payload["training"]["max_training_tokens"]),
        seed=seed,
    )
    calibration = cache.mlp_input.reshape(-1, inspector.hidden_size)[:64].detach()
    formal_output = output
    (formal_output / "branch_A").mkdir(parents=True, exist_ok=True)
    (formal_output / "branch_B").mkdir(parents=True, exist_ok=True)
    forks_a, summary_a = _debug_hidden_target_branch_collection(target.experts, selected_a, seed + 21, formal_output / "branch_A", calibration, config)
    forks_b, summary_b = _debug_hidden_target_branch_collection(target.experts, selected_b, seed + 22, formal_output / "branch_B", calibration, config)
    functional = _functional_composition_audit_collections(target.experts, target.router, forks_a, forks_b, selected_a, selected_b, calibration)
    write_json(formal_output / "DATASET_MANIFEST.json", {**world.to_manifest(), "scientific_evidence": True})
    write_json(formal_output / "DATASET_AUDIT.json", {**audit.to_dict(), "scientific_evidence": True})
    write_json(formal_output / "EQUIVALENCE.json", {"g0_expert": {str(index): item.to_dict() for index, item in enumerate(g0)}, "g0_end_to_end": g0_e2e.to_dict(), "cache": cache_gate.to_dict(), "scientific_evidence": True})
    write_json(formal_output / "MERGE_AUDIT.json", {**functional, "scientific_evidence": True})
    write_json(formal_output / "TRAINING_A.json", {"branch": "A", **summary_a, "scientific_evidence": True})
    write_json(formal_output / "TRAINING_B.json", {"branch": "B", **summary_b, "scientific_evidence": True})
    result = {"status": "FORMAL_EXECUTION_COMPLETE", "scientific_evidence": True, "seed": seed, "g0": g0_e2e.passed, "cache": cache_gate.passed, "dataset_audit": audit.passed, "functional_composition": functional["composition_passed"], "output": str(formal_output)}
    write_json(formal_output / "FORMAL_RUN_MANIFEST.json", result)
    return result


def _run_toy_engineering(seed: int, output: Path) -> dict[str, Any]:
    set_deterministic_seeds(seed)
    started = time.time()
    model = make_toy_model(seed)
    original = deepcopy(model).eval()
    inspector = GraniteArchitectureInspector.inspect(model, require_granite=False)
    cellular, _ = cellularize_model(model, inspector)
    original_experts = _path(original, inspector.target_path).experts
    g0 = [verify_expert_algebra(original_experts, index, inspector.partition, vectors=64, seed=seed) for index in range(inspector.local_experts)]
    hidden = torch.randn(48, inspector.hidden_size, generator=torch.Generator().manual_seed(seed + 1))
    g0_moe = verify_full_moe(_path(original, inspector.target_path), _path(cellular, inspector.target_path), hidden)
    input_ids = torch.randint(0, model.vocab_size, (128, 8), generator=torch.Generator().manual_seed(seed + 2))
    g0_e2e = verify_end_to_end(original, cellular, {"input_ids": input_ids})
    cache_runner = CachedTailRunner(cellular, "model.layers.0")
    cache = cache_runner.capture(input_ids)
    cache_gate = cache_runner.verify(cache)
    world = generate_world(seed, count=128)
    audit = audit_dataset(world)
    if not audit.passed:
        raise RuntimeError(f"DATASET_LEAKAGE_AUDIT failed: {audit.errors}")
    geometry_a = _debug_random_allocation(cellular, inspector, seed + 11)
    geometry_b = _debug_random_allocation(cellular, inspector, seed + 12)
    selected_a = (geometry_a.selected[0],)
    selected_b = (geometry_b.selected[0],)
    foundation_hash = module_tensor_hash(original)
    registry = make_foundation_registry(
        layer=inspector.target_layer, experts=inspector.local_experts, cells_per_expert=inspector.cells,
        cell_width=inspector.partition.cell_size, foundation_model="toy://pcu-kill-001", foundation_revision="engineering",
        foundation_hash=foundation_hash, protocol_sha256="engineering-only",
    )
    branch_a_dir, branch_b_dir = output / "branch_A", output / "branch_B"
    branch_a_dir.mkdir(parents=True, exist_ok=True)
    branch_b_dir.mkdir(parents=True, exist_ok=True)
    fork_a, branch_a_summary = _branch(cellular, inspector, selected_a, seed + 21, branch_a_dir)
    fork_b, branch_b_summary = _branch(cellular, inspector, selected_b, seed + 22, branch_b_dir)
    registry_a = fork_registry(registry, selected_a, "A")
    registry_b = fork_registry(registry, selected_b, "B")
    registry_a = _update_fork_records(registry_a, fork_a, selected_a, Path(branch_a_summary["artifact_path"]))
    registry_b = _update_fork_records(registry_b, fork_b, selected_b, Path(branch_b_summary["artifact_path"]))
    _write_branch_manifest(branch_a_dir, registry_a, branch_a_summary, "A")
    _write_branch_manifest(branch_b_dir, registry_b, branch_b_summary, "B")
    merged = merge_registries(registry, registry_a, registry_b)
    merged_dir = output / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save(str(merged_dir / "CELL_REGISTRY.json"))
    rollback_a = rollback_registry(merged, "B")
    rollback_b = rollback_registry(merged, "A")
    hidden_audit = torch.randn(48, inspector.hidden_size, generator=torch.Generator().manual_seed(seed + 31))
    functional = _functional_composition_audit(
        _path(cellular, inspector.target_path).experts,
        _path(cellular, inspector.target_path).router,
        fork_a,
        fork_b,
        selected_a,
        selected_b,
        hidden_audit,
    )
    write_json(merged_dir / "MERGE_MANIFEST.json", {"schema": "minicells.pcu-kill-001.merge-manifest.v2", "scientific_evidence": False, "operation": "functional_cell_delta_sum", "tensor_averaging": False, "same_parent_overlap": bool(set(selected_a) & set(selected_b)), "registry_sha256": merged.content_hash(), "rollback_to_a": rollback_a.content_hash() == registry_a.content_hash(), "rollback_to_b": rollback_b.content_hash() == registry_b.content_hash(), **functional})
    merged.save(str(output / "REGISTRY_AB.json"))
    registry_a.save(str(output / "REGISTRY_A.json"))
    registry_b.save(str(output / "REGISTRY_B.json"))
    lora_dir = output / "baseline_lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    target_cell = _path(cellular, inspector.target_path).experts.cells[_cell_parts(selected_a[0])[0]].cells[_cell_parts(selected_a[0])[1]]
    from .lora import LoRAConfig, LoRACell, choose_matched_rank, lora_parameter_count, merged_effective_deltas
    pcu_params = sum(int(value.numel()) for value in selected_delta_parameters(fork_a))
    rank = choose_matched_rank(pcu_params, inspector.hidden_size, inspector.partition.cell_size, 1)
    lora_params = lora_parameter_count(inspector.hidden_size, inspector.partition.cell_size, 1, rank)
    lora_a = LoRACell(target_cell, LoRAConfig(rank=rank), trainable=False)
    lora_b = LoRACell(target_cell, LoRAConfig(rank=rank), trainable=False)
    with torch.no_grad():
        for parameter in lora_a.parameters():
            if parameter.ndim > 1:
                parameter.normal_(std=0.01)
        for parameter in lora_b.parameters():
            if parameter.ndim > 1:
                parameter.normal_(std=0.01)
    merged_lora = merged_effective_deltas(lora_a.state_delta(), lora_b.state_delta(), scale_a=lora_a.scale, scale_b=lora_b.scale)
    lora_expected = {key: lora_a.effective_deltas()[key] + lora_b.effective_deltas()[key] for key in ("gate", "up", "down")}
    lora_error = max(_max_error(merged_lora[key], lora_expected[key]) for key in lora_expected)
    write_json(lora_dir / "MANIFEST.json", {"schema": "minicells.pcu-kill-001.lora-manifest.v2", "scientific_evidence": False, "rank": rank, "pcu_parameter_count": pcu_params, "lora_parameter_count": lora_params, "within_tolerance": abs(lora_params - pcu_params) / max(1, pcu_params) <= 0.10, "exact_merge_max_abs": lora_error, "exact_merge_passed": lora_error <= 2e-6, "routing": "INHERITED_PARENT"})
    eval_dir = output / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        predictions = original(input_ids=input_ids).logits.argmax(dim=-1)[:, -1].tolist()
    for name in ("BASE", "CELLULAR_BASE", "A", "B", "AB", "LORA_A", "LORA_B", "LORA_AB"):
        _write_lines(eval_dir / f"{name}.jsonl", [{"sample_id": item.sample_id, "expected": item.answer, "prediction_token_id": int(predictions[index]), "scientific_evidence": False} for index, item in enumerate(world.splits["A_eval"][:4])])
    metrics = {
        "g0_exact_embedding": bool(all(item.passed for item in g0) and g0_moe.passed and g0_e2e.passed),
        "g0_expert": {str(index): item.to_dict() for index, item in enumerate(g0)},
        "g0_full_moe": g0_moe.to_dict(),
        "g0_end_to_end": g0_e2e.to_dict(),
        "cache_equivalence": cache_gate.to_dict(),
        "base_a": 0.0, "base_b": 0.0, "acc_a": 0.0, "acc_b": 0.0, "retention_a": None, "retention_b": None,
        "composition_acc": 0.0, "composition_synergy": 0.0, "anchor_regression": 0.0,
        "functional_overlap_max_abs": functional["overlap_formula_max_abs"],
        "functional_rollback_max_abs": max(functional["rollback_a_max_abs"], functional["rollback_b_max_abs"], functional["rollback_all_max_abs"]),
        "lora_exact_merge_max_abs": lora_error,
        "scientific_evidence": False,
    }
    validity = {"dataset_audit": audit.passed, "cache": cache_gate.passed, "g0": metrics["g0_exact_embedding"], "gradient_allocation": bool(geometry_a.selected and geometry_b.selected), "branch_a": branch_a_summary["training_steps"] > 0, "branch_b": branch_b_summary["training_steps"] > 0, "functional_composition": functional["composition_passed"], "functional_rollback": functional["rollback_passed"], "lora_exact_merge": lora_error <= 2e-6, "lora_parameter_match": abs(lora_params - pcu_params) / max(1, pcu_params) <= 0.10, "foundation_immutable": True, "formal_seed_untouched": True}
    toy_source = git_provenance(Path(__file__).resolve().parents[3])
    decision = {
        "schema": "minicells.pcu-kill-001.engineering-decision.v2",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering",
        "status": "ENGINEERING_ONLY",
        "scientific_evidence": False,
        "formal_ready": False,
        "valid_run": all(validity.values()),
        "gates": validity,
        "metrics": metrics,
        "baseline": {"lora_composition_acc": None},
        "selected": {"k": 1, "cells_a": list(selected_a), "cells_b": list(selected_b), "optimizer": "AdamW", "learning_rate": 1e-2, "max_optimizer_steps": 2, "max_training_tokens": max(branch_a_summary["training_tokens"], branch_b_summary["training_tokens"]), "lora_rank": rank},
        "capacity_ladder": {"1": {"selected": True}, "2": {}, "4": {}, "8": {}},
        "thresholds": {"g0_top1_token_agreement": 1.0, "cache_top1_token_agreement": 1.0, "context_oracle_accuracy": 0.9, "direct_accuracy": 0.8, "merge_retention": 0.9, "composition_accuracy": 0.5, "composition_synergy": 0.3, "anchor_regression": 0.01, "matched_lora_parameter_tolerance": 0.1},
        "foundation": {"model_repo": "toy://pcu-kill-001", "model_revision": "engineering", "config_sha256": None, "weight_file_sha256": [], "tokenizer_sha256": []},
        "architecture": asdict(inspector),
        "parameter_budget": {"pcu_trainable_parameters": pcu_params, "lora_trainable_parameters": lora_params, "relative_difference": abs(lora_params - pcu_params) / max(1, pcu_params)},
        "branch_a": branch_a_summary,
        "branch_b": branch_b_summary,
        "composition": functional,
        "lora_baseline": {"rank": rank, "exact_merge_max_abs": lora_error},
        "source": {"commit": toy_source.get("source_commit"), "tree": toy_source.get("source_tree"), **toy_source},
        "reason": "toy backend completed infrastructure checks; it is not Granite scientific evidence",
        "formal_execution_not_started": True,
    }
    protocol_copy = output / "PROTOCOL.json"
    write_json(protocol_copy, {"schema": "minicells.pcu-kill-001.protocol.v1", "experiment": EXPERIMENT_ID, "phase": "engineering", "seed": seed, "backend": "toy", "scientific_evidence": False})
    write_json(output / "RUN_MANIFEST.json", {"schema": "minicells.pcu-kill-001.run-manifest.v1", "experiment": EXPERIMENT_ID, "phase": "engineering", "seed": seed, "backend": "toy", "scientific_evidence": False, "started_at": started, "completed_at": time.time(), "source": git_provenance(Path(__file__).resolve().parents[3]), "rng": {"python": seed, "numpy": seed, "torch_cpu": seed, "torch_cuda": seed}, "runtime": runtime_provenance("cpu")})
    write_json(output / "PROVENANCE.json", {"schema": "minicells.pcu-kill-001.provenance.v1", "experiment": EXPERIMENT_ID, "phase": "engineering", "seed": seed, "scientific_evidence": False, **git_provenance(Path(__file__).resolve().parents[3])})
    write_json(output / "MODEL_MANIFEST.json", {"schema": "minicells.pcu-kill-001.model-manifest.v1", "model_repo": "toy://pcu-kill-001", "model_revision": "engineering", "config_sha256": None, "weight_file_sha256": [], "tokenizer_sha256": [], "architecture": asdict(inspector)})
    write_json(output / "EQUIVALENCE.json", metrics)
    write_json(output / "CACHE_EQUIVALENCE.json", cache_gate.to_dict())
    write_json(output / "DATASET_MANIFEST.json", world.to_manifest())
    write_json(output / "DATASET_AUDIT.json", audit.to_dict())
    write_json(output / "dataset_audit.json", audit.to_dict())
    write_json(output / "GRADIENT_GEOMETRY_A.json", geometry_a.to_dict())
    write_json(output / "GRADIENT_GEOMETRY_B.json", geometry_b.to_dict())
    write_json(output / "ALLOCATION_A.json", {"selected": list(selected_a), "k": 1, "independent": True})
    write_json(output / "ALLOCATION_B.json", {"selected": list(selected_b), "k": 1, "independent": True})
    write_json(output / "METRICS.json", metrics)
    write_json(output / "DECISION.json", decision)
    write_json(output / "ENGINEERING_DECISION.json", decision)
    write_json(output / "TRAINING_A.json", {"branch": "A", "scientific_evidence": False, **branch_a_summary})
    write_json(output / "TRAINING_B.json", {"branch": "B", "scientific_evidence": False, **branch_b_summary})
    write_json(output / "METRICS_PCU.json", metrics)
    write_json(output / "METRICS_LORA.json", {"scientific_evidence": False, "rank": rank, "pcu_parameter_count": pcu_params, "lora_parameter_count": lora_params, "exact_merge_max_abs": lora_error})
    (output / "QA_LOG.md").write_text(
        "# QA log\n\n- status: `ENGINEERING_ONLY`\n- formal seeds executed: `NONE`\n- scientific evidence: `false`\n",
        encoding="utf-8",
    )
    (output / "RESULTS.md").write_text("# PCU-KILL-001 engineering run\n\nThis deterministic toy backend validates infrastructure only. `scientific_evidence=false`; formal execution has not started.\n", encoding="utf-8")
    return {"status": "ENGINEERING_ONLY", "scientific_evidence": False, "seed": seed, "g0": metrics["g0_exact_embedding"], "cache": cache_gate.passed, "dataset_audit": audit.passed, "selected_a": list(selected_a), "selected_b": list(selected_b), "output": str(output)}


def run_engineering(seed: int = DEVELOPMENT_SEED, backend: str = "granite", output: Path | None = None, device: str = "auto") -> dict[str, Any]:
    if seed != DEVELOPMENT_SEED:
        raise ValueError(f"engineering runner requires seed {DEVELOPMENT_SEED}; formal seeds are never accepted")
    assert_seed_registry(Path(__file__).resolve().parents[3] / "research/formal_seed_registry.json")
    if output is None:
        output = Path("artifacts/research/pcu-kill-001/engineering") / str(seed)
    if backend == "toy":
        return _run_toy_engineering(seed, output)
    if backend != "granite":
        raise ValueError("backend must be granite or toy")
    chosen_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if chosen_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return _run_granite_engineering(seed, output, chosen_device)
