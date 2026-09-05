"""Engineering-only end-to-end harness for PCU-KILL-001.

The real Granite path is intentionally explicit.  The toy backend exercises
the same tensor partition, router dispatch, fork, registry union, rollback,
cache, and governance code without producing scientific evidence.
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
from .cache import CachedTailRunner
from .cellular import GraniteArchitectureInspector
from .equivalence import verify_end_to_end, verify_expert_algebra, verify_full_moe
from .governance import DEVELOPMENT_SEED, EXPERIMENT_ID, git_provenance, runtime_provenance, set_deterministic_seeds, sha256_file, write_json
from .model import MODEL_ID, cellularize_model, load_granite, model_identity_manifest, target_module
from .registry import CellRegistry, make_foundation_registry, merge_registries, module_tensor_hash, rollback_registry, tensor_sha256
from .synthetic import audit_dataset, generate_world
from .training import Allocation, ForkedCellularExpert, allocate_topk, foundation_tensor_hashes, fork_expert, fork_initial_delta_norm, selected_delta_parameters, write_training_csv


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


def _allocation(model: Any, inspector: GraniteArchitectureInspector, seed: int) -> Allocation:
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


def _update_fork_records(registry: CellRegistry, fork: ForkedCellularExpert, selected: tuple[str, ...]) -> CellRegistry:
    result = registry.copy()
    state = fork.delta_state()
    delta_hash = hashlib.sha256(b"".join(key.encode() + value.cpu().numpy().tobytes() for key, value in sorted(state.items()))).hexdigest()
    for cell_id in selected:
        fork_id = next(key for key, record in result.records.items() if key.startswith(cell_id + "::fork::"))
        result.records[fork_id].weight_hash = delta_hash
    return result


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
    torch.save(fork.delta_state(), output / "DELTA_CELLS.safetensors")
    write_training_csv(output / "TRAINING.csv", rows)
    return fork, {"initial_delta_l2": initial_delta, "training_steps": len(rows), "training_tokens": len(rows) * int(inputs.numel()), "trainable_parameter_count": sum(int(value.numel()) for value in selected_delta_parameters(fork)), "selected_cells": list(selected), "unique_parent_experts": len(by_expert), "loss_final": rows[-1]["loss"]}


def _write_branch_manifest(output: Path, registry: CellRegistry, summary: Mapping[str, Any], branch: str) -> None:
    registry.save(str(output / "CELL_REGISTRY.json"))
    write_json(output / "MANIFEST.json", {"schema": "minicells.pcu-kill-001.branch-manifest.v1", "branch": branch, "scientific_evidence": False, "registry_sha256": registry.content_hash(), **dict(summary)})


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
    geometry_a = _allocation(cellular, inspector, seed + 11)
    geometry_b = _allocation(cellular, inspector, seed + 12)
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
    registry_a = __import__("minicells.pcu_kill_001.registry", fromlist=["fork_registry"]).fork_registry(registry, selected_a, "A")
    registry_b = __import__("minicells.pcu_kill_001.registry", fromlist=["fork_registry"]).fork_registry(registry, selected_b, "B")
    registry_a = _update_fork_records(registry_a, fork_a, selected_a)
    registry_b = _update_fork_records(registry_b, fork_b, selected_b)
    _write_branch_manifest(branch_a_dir, registry_a, branch_a_summary, "A")
    _write_branch_manifest(branch_b_dir, registry_b, branch_b_summary, "B")
    merged = merge_registries(registry, registry_a, registry_b)
    merged_dir = output / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save(str(merged_dir / "CELL_REGISTRY.json"))
    rollback_a = rollback_registry(merged, "B")
    rollback_b = rollback_registry(merged, "A")
    write_json(merged_dir / "MERGE_MANIFEST.json", {"schema": "minicells.pcu-kill-001.merge-manifest.v1", "scientific_evidence": False, "operation": "registry_union_only", "tensor_averaging": False, "same_parent_overlap": bool(set(selected_a) & set(selected_b)), "registry_sha256": merged.content_hash(), "rollback_to_a": rollback_a.content_hash() == registry_a.content_hash(), "rollback_to_b": rollback_b.content_hash() == registry_b.content_hash()})
    lora_dir = output / "baseline_lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    target_cell = _path(cellular, inspector.target_path).experts.cells[_cell_parts(selected_a[0])[0]].cells[_cell_parts(selected_a[0])[1]]
    from .lora import choose_matched_rank, lora_parameter_count
    pcu_params = sum(int(value.numel()) for value in selected_delta_parameters(fork_a))
    rank = choose_matched_rank(pcu_params, inspector.hidden_size, inspector.partition.cell_size, 1)
    lora_params = lora_parameter_count(inspector.hidden_size, inspector.partition.cell_size, 1, rank)
    write_json(lora_dir / "MANIFEST.json", {"schema": "minicells.pcu-kill-001.lora-manifest.v1", "scientific_evidence": False, "rank": rank, "pcu_parameter_count": pcu_params, "lora_parameter_count": lora_params, "within_tolerance": abs(lora_params - pcu_params) / max(1, pcu_params) <= 0.10, "routing": "INHERITED_PARENT"})
    eval_dir = output / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for name in ("BASE", "CELLULAR_BASE", "A", "B", "AB", "LORA_A", "LORA_B", "LORA_AB"):
        _write_lines(eval_dir / f"{name}.jsonl", [{"sample_id": item.sample_id, "expected": item.answer, "prediction": "engineering-placeholder", "scientific_evidence": False} for item in world.splits["A_eval"][:4]])
    metrics = {
        "g0_exact_embedding": bool(all(item.passed for item in g0) and g0_moe.passed and g0_e2e.passed),
        "g0_expert": {str(index): item.to_dict() for index, item in enumerate(g0)},
        "g0_full_moe": g0_moe.to_dict(),
        "g0_end_to_end": g0_e2e.to_dict(),
        "cache_equivalence": cache_gate.to_dict(),
        "base_a": 0.0, "base_b": 0.0, "acc_a": 0.0, "acc_b": 0.0, "retention_a": 0.0, "retention_b": 0.0,
        "composition_acc": 0.0, "composition_synergy": 0.0, "anchor_regression": 0.0,
        "scientific_evidence": False,
    }
    validity = {"dataset_audit": audit.passed, "cache_equivalence": cache_gate.passed, "foundation_immutable": True, "formal_seed_untouched": True}
    decision = {"schema": "minicells.pcu-kill-001.decision.v1", "status": "ENGINEERING_ONLY", "scientific_decision": False, "valid_run": True, "gates": validity, "metrics": metrics, "baseline": {"lora_composition_acc": 0.0}, "reason": "engineering backend completed; no toy result is scientific evidence", "formal_execution_not_started": True}
    protocol_copy = output / "PROTOCOL.json"
    write_json(protocol_copy, {"schema": "minicells.pcu-kill-001.protocol.v1", "experiment": EXPERIMENT_ID, "phase": "engineering", "seed": seed, "backend": "toy", "scientific_evidence": False})
    write_json(output / "RUN_MANIFEST.json", {"schema": "minicells.pcu-kill-001.run-manifest.v1", "experiment": EXPERIMENT_ID, "phase": "engineering", "seed": seed, "backend": "toy", "scientific_evidence": False, "started_at": started, "completed_at": time.time(), "source": git_provenance(Path(__file__).resolve().parents[3]), "rng": {"python": seed, "numpy": seed, "torch_cpu": seed, "torch_cuda": seed}, "runtime": runtime_provenance("cpu")})
    write_json(output / "PROVENANCE.json", {"schema": "minicells.pcu-kill-001.provenance.v1", "experiment": EXPERIMENT_ID, "phase": "engineering", "seed": seed, "scientific_evidence": False, **git_provenance(Path(__file__).resolve().parents[3])})
    write_json(output / "MODEL_MANIFEST.json", {"schema": "minicells.pcu-kill-001.model-manifest.v1", "model_repo": "toy://pcu-kill-001", "model_revision": "engineering", "architecture": asdict(inspector)})
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
    (output / "QA_LOG.md").write_text(
        "# QA log\n\n- status: `ENGINEERING_ONLY`\n- formal seeds executed: `NONE`\n- scientific evidence: `false`\n",
        encoding="utf-8",
    )
    (output / "RESULTS.md").write_text("# PCU-KILL-001 engineering run\n\nThis deterministic toy backend validates infrastructure only. `scientific_evidence=false`; formal execution has not started.\n", encoding="utf-8")
    return {"status": "ENGINEERING_ONLY", "scientific_evidence": False, "seed": seed, "g0": metrics["g0_exact_embedding"], "cache": cache_gate.passed, "dataset_audit": audit.passed, "selected_a": list(selected_a), "selected_b": list(selected_b), "output": str(output)}


def run_engineering(seed: int = DEVELOPMENT_SEED, backend: str = "toy", output: Path | None = None, device: str = "auto") -> dict[str, Any]:
    if seed != DEVELOPMENT_SEED:
        raise ValueError(f"engineering runner requires seed {DEVELOPMENT_SEED}; formal seeds are never accepted")
    if output is None:
        output = Path("artifacts/research/pcu-kill-001/engineering") / str(seed)
    if backend == "toy":
        return _run_toy_engineering(seed, output)
    if backend != "granite":
        raise ValueError("backend must be granite or toy")
    chosen_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if chosen_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    tokenizer, model, manifest = load_granite(MODEL_ID, revision=None, device=chosen_device)
    inspector = GraniteArchitectureInspector.inspect(model, require_granite=True)
    raise RuntimeError("Granite model preflight succeeded, but this environment has no approved engineering dataset/cache executor; refusing to emit a partial scientific run")
