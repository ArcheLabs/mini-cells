"""Engineering-only layer placement diagnostic for PCU-KILL-001.

PCU-LAYER-PLACEMENT-001 changes exactly one causal variable relative to the
published E0 failure: the decoder layer that owns the mutated Cells.  The
training objective, optimizer, learning rate, K, batch size, step budget,
dataset, seed, Cell parameterization, inherited parent routing, and direct
A-evaluation are held fixed.

The published L23 result is reused.  Two additional A-only branches are run at
deterministically selected early/mid MoE layers.  This module is diagnostic
only and never touches formal seeds or the frozen/formal PCU protocol.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .cellular import CellPartition, extract_expert_projections, patch_moe_block
from .evaluation import evaluate_samples
from .governance import git_provenance, write_json
from .model import MODEL_ID, load_granite, target_module
from .synthetic import audit_dataset, generate_world
from .task import TaskSequences, answer_token_cross_entropy, build_task_sequences, validate_answer_only_labels
from .training import Allocation, BranchTrainingConfig, ForkedCellularExperts, allocate_topk, selected_delta_parameters


EXPERIMENT_ID = "PCU-LAYER-PLACEMENT-001"
ENGINEERING_SEED = 26090501
K = 8
LEARNING_RATE = 1e-3
MAX_OPTIMIZER_STEPS = 128
MAX_TRAINING_TOKENS = 500_000
BATCH_SIZE = 8
DIRECT_CAPABILITY_FLOOR = 0.80
BASELINE_ROOT = Path("artifacts/research/pcu-kill-001/engineering/26090501-oracle-v2")
DEFAULT_OUTPUT = Path("artifacts/research/pcu-layer-placement-001/engineering/26090501-layer-only")


@dataclass(frozen=True)
class MoeLayer:
    layer: int
    path: str
    hidden_size: int
    intermediate_size: int
    local_experts: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_moe_layers(model: nn.Module) -> tuple[MoeLayer, ...]:
    """Enumerate router+expert decoder blocks without assuming every layer is MoE."""
    found: dict[int, MoeLayer] = {}
    for name, module in model.named_modules():
        experts = getattr(module, "experts", None)
        router = getattr(module, "router", None)
        if experts is None or router is None:
            continue
        numbers = [int(value) for value in re.findall(r"layers\.(\d+)", name)]
        if not numbers:
            continue
        try:
            projection = extract_expert_projections(experts, 0)
            count = int(getattr(experts, "num_experts"))
        except Exception:
            continue
        layer = numbers[-1]
        candidate = MoeLayer(
            layer=layer,
            path=name,
            hidden_size=projection.hidden_size,
            intermediate_size=projection.intermediate_size,
            local_experts=count,
        )
        previous = found.get(layer)
        if previous is not None and previous.path != candidate.path:
            raise RuntimeError(f"multiple MoE blocks resolved for decoder layer {layer}")
        found[layer] = candidate
    if len(found) < 3:
        raise RuntimeError("layer placement diagnostic requires at least three MoE decoder layers")
    return tuple(found[index] for index in sorted(found))


def choose_layer_targets(layers: Sequence[int]) -> dict[str, int]:
    """Choose early/mid probes by decoder-depth fractions, with deterministic ties."""
    values = sorted({int(value) for value in layers})
    if len(values) < 3:
        raise ValueError("at least three distinct layers are required")
    late = values[-1]

    def nearest(target: int, forbidden: set[int]) -> int:
        candidates = [value for value in values if value not in forbidden]
        if not candidates:
            raise ValueError("no distinct layer remains for placement probe")
        return min(candidates, key=lambda value: (abs(value - target), value))

    early = nearest(late // 3, {late})
    mid = nearest((2 * late) // 3, {late, early})
    return {"early": early, "mid": mid, "late_baseline": late}


def _slice_sequences(sequences: TaskSequences, start: int, end: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        sequences.input_ids[start:end],
        sequences.attention_mask[start:end],
        sequences.labels[start:end],
        sequences.loss_mask[start:end],
    )


def _full_task_loss(model: nn.Module, input_ids: Tensor, attention_mask: Tensor, labels: Tensor) -> Tensor:
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        raise RuntimeError("layer-placement model output has no logits tensor")
    return answer_token_cross_entropy(logits, labels)


def _selected_map(selected_cells: Sequence[str], layer: int) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    prefix = f"L{int(layer)}:E"
    for value in selected_cells:
        text = str(value)
        if not text.startswith(prefix) or ":C" not in text:
            raise ValueError(f"invalid Cell ID for layer {layer}: {text}")
        expert_text, cell_text = text[len(prefix):].split(":C", 1)
        result.setdefault(int(expert_text), []).append(int(cell_text))
    return result


def full_model_task_conditioned_allocation(
    model: nn.Module,
    block: nn.Module,
    parent_experts: nn.Module,
    sequences: TaskSequences,
    *,
    layer: int,
    calibration_rows: int = 64,
    device: str,
) -> Allocation:
    """Apply the same answer-token CE gradient score at an arbitrary MoE layer."""
    rows = min(int(calibration_rows), int(sequences.input_ids.shape[0]))
    all_cells = {
        expert: tuple(range(int(parent_experts.partition.cells)))
        for expert in range(int(parent_experts.num_experts))
    }
    probe = ForkedCellularExperts(parent_experts, all_cells).to(device)
    resident = block.experts
    block.experts = probe
    try:
        input_ids, attention, labels, _ = _slice_sequences(sequences, 0, rows)
        probe.zero_grad(set_to_none=True)
        loss = _full_task_loss(
            model,
            input_ids.to(device),
            attention.to(device),
            labels.to(device),
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite layer-placement allocation loss")
        loss.backward()
        scores: dict[str, float] = {}
        for expert_index, expert in enumerate(probe.cells):
            for cell_index, cell in enumerate(expert.cells):
                gradients = [
                    value.grad
                    for name, value in cell.named_parameters()
                    if name.startswith("delta_") and value.grad is not None
                ]
                count = sum(int(value.numel()) for value in gradients)
                scores[f"L{int(layer)}:E{expert_index}:C{cell_index}"] = (
                    sum(float(value.detach().float().pow(2).sum()) for value in gradients)
                    / max(1, count)
                )
        if not scores or max(scores.values(), default=0.0) <= 0.0:
            raise RuntimeError("layer-placement allocation produced no positive gradient scores")
        return allocate_topk(scores)
    finally:
        block.experts = resident
        probe.zero_grad(set_to_none=True)


def _train_full_model_branch(
    model: nn.Module,
    block: nn.Module,
    parent_experts: nn.Module,
    sequences: TaskSequences,
    selected_cells: Sequence[str],
    *,
    layer: int,
    device: str,
    config: BranchTrainingConfig,
) -> tuple[ForkedCellularExperts, dict[str, Any]]:
    """Train only selected zero-delta Cells through the real frozen model tail."""
    runtime = ForkedCellularExperts(parent_experts, _selected_map(selected_cells, layer)).to(device)
    parameters = selected_delta_parameters(runtime)
    if not parameters:
        raise RuntimeError("layer-placement branch has no selected trainable deltas")
    optimizer = torch.optim.AdamW(parameters, lr=float(config.learning_rate))
    block.experts = runtime
    steps = 0
    tokens = 0
    final_loss = float("nan")
    while steps < int(config.max_optimizer_steps) and tokens < int(config.max_training_tokens):
        progressed = False
        for start in range(0, int(sequences.input_ids.shape[0]), max(1, int(config.batch_size))):
            end = min(int(sequences.input_ids.shape[0]), start + max(1, int(config.batch_size)))
            input_ids, attention, labels, loss_mask = _slice_sequences(sequences, start, end)
            batch_tokens = int(loss_mask.sum())
            if batch_tokens <= 0:
                raise RuntimeError("layer-placement training batch has no answer-token labels")
            if tokens + batch_tokens > int(config.max_training_tokens):
                break
            optimizer.zero_grad(set_to_none=True)
            loss = _full_task_loss(
                model,
                input_ids.to(device),
                attention.to(device),
                labels.to(device),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite layer-placement training loss")
            loss.backward()
            optimizer.step()
            steps += 1
            tokens += batch_tokens
            final_loss = float(loss.detach())
            progressed = True
            if steps >= int(config.max_optimizer_steps):
                break
        if not progressed:
            break
    if steps != int(config.max_optimizer_steps):
        raise RuntimeError(f"layer-placement branch stopped at {steps} steps, expected {config.max_optimizer_steps}")
    return runtime, {
        "optimizer": config.optimizer,
        "learning_rate": float(config.learning_rate),
        "max_optimizer_steps": int(config.max_optimizer_steps),
        "max_training_tokens": int(config.max_training_tokens),
        "batch_size": int(config.batch_size),
        "training_steps": steps,
        "training_tokens": tokens,
        "final_loss": final_loss,
        "selected_cells": list(selected_cells),
    }


def _validate_foundation_manifest(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key in (
        "model_repo",
        "model_revision",
        "config_sha256",
        "foundation_tensor_sha256",
        "weight_file_sha256",
        "tokenizer_sha256",
    ):
        if actual.get(key) != expected.get(key):
            raise RuntimeError(f"layer-placement foundation identity mismatch: {key}")


def _run_one_layer(
    *,
    layer: int,
    path: str,
    device: str,
    expected_foundation: Mapping[str, Any],
    expected_dataset_sha256: str,
    seed: int,
) -> dict[str, Any]:
    tokenizer, model, manifest = load_granite(
        str(expected_foundation["model_repo"]),
        revision=str(expected_foundation["model_revision"]),
        device=device,
    )
    try:
        _validate_foundation_manifest(manifest, expected_foundation)
        model.requires_grad_(False)
        block = target_module(model, path)
        original_experts = block.experts
        projection = extract_expert_projections(original_experts, 0)
        partition = CellPartition(projection.intermediate_size, 4)
        cellular_experts = patch_moe_block(block, partition)
        model.requires_grad_(False)
        cellular_experts.requires_grad_(False)

        world = generate_world(seed, count=128, tokenizer=tokenizer)
        audit = audit_dataset(world)
        if not audit.passed:
            raise RuntimeError(f"layer-placement dataset audit failed: {audit.errors}")
        if world.manifest_sha256() != expected_dataset_sha256:
            raise RuntimeError("layer-placement dataset identity differs from published E0")
        train = build_task_sequences(tokenizer, world.splits["A_train"], "A_train", max_length=128)
        validate_answer_only_labels(train)

        allocation = full_model_task_conditioned_allocation(
            model,
            block,
            cellular_experts,
            train,
            layer=layer,
            calibration_rows=64,
            device=device,
        )
        selected = tuple(allocation.selected[:K])
        config = BranchTrainingConfig(
            optimizer="AdamW",
            learning_rate=LEARNING_RATE,
            max_optimizer_steps=MAX_OPTIMIZER_STEPS,
            max_training_tokens=MAX_TRAINING_TOKENS,
            batch_size=BATCH_SIZE,
            seed=seed,
        )
        runtime, training = _train_full_model_branch(
            model,
            block,
            cellular_experts,
            train,
            selected,
            layer=layer,
            device=device,
            config=config,
        )
        evaluation = evaluate_samples(
            model,
            tokenizer,
            world.splits["A_eval"],
            split="A_eval",
            device=device,
            max_new_tokens=16,
            batch_size=16,
        )
        # The original raw experts and the exact cellular parent remain frozen;
        # only runtime deltas were handed to AdamW.
        if any(parameter.requires_grad for parameter in model.parameters() if parameter not in set(parameters := selected_delta_parameters(runtime))):
            raise RuntimeError("non-Cell parameter unexpectedly became trainable")
        return {
            "schema": "minicells.pcu-layer-placement-001.layer-result.v1",
            "experiment": EXPERIMENT_ID,
            "layer": int(layer),
            "target_path": path,
            "device": device,
            "dataset_manifest_sha256": world.manifest_sha256(),
            "allocation": {
                "method": "task-conditioned-gradient-l2-per-parameter",
                "calibration_split": "A_train",
                "calibration_sample_rule": "first_64_samples",
                "selected_k": K,
                "selected": list(selected),
                "topk_mass": {str(key): value for key, value in allocation.topk_mass.items()},
                "effective_count": allocation.effective_count,
            },
            "training": training,
            "evaluation": evaluation.to_dict(),
            "direct_accuracy": evaluation.exact,
            "capability_floor": DIRECT_CAPABILITY_FLOOR,
            "passes": bool(evaluation.exact >= DIRECT_CAPABILITY_FLOOR),
        }
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_baseline(root: Path) -> dict[str, Any]:
    decision = json.loads((root / "ENGINEERING_DECISION.json").read_text(encoding="utf-8"))
    oracle = json.loads((root / "CONTEXT_ORACLE.json").read_text(encoding="utf-8"))
    identity = json.loads((root / "RUN_IDENTITY.json").read_text(encoding="utf-8"))
    acceleration = json.loads((root / "EXECUTION_ACCELERATION.json").read_text(encoding="utf-8"))
    if decision.get("status") != "LOCAL_CELL_MUTATION_UNSUPPORTED":
        raise RuntimeError("published baseline is not the expected local-mutation failure")
    if oracle.get("passed") is not True:
        raise RuntimeError("published baseline context oracle did not pass")
    if identity.get("source", {}).get("source_dirty") is not False:
        raise RuntimeError("published baseline source identity is dirty")
    target_row = None
    for candidate in decision.get("candidates", []):
        if float(candidate.get("learning_rate", -1.0)) == LEARNING_RATE:
            target_row = candidate.get("capacity", {}).get(str(K))
            break
    if not isinstance(target_row, Mapping):
        raise RuntimeError("published baseline has no lr=1e-3/K=8 capacity row")
    foundation = dict(decision["foundation"])
    dataset_sha = str(acceleration["identity"]["dataset_manifest_sha256"])
    return {
        "decision": decision,
        "foundation": foundation,
        "dataset_manifest_sha256": dataset_sha,
        "late_layer": int(decision["architecture"]["target_layer"]),
        "late_target_path": str(decision["architecture"]["target_path"]),
        "direct_accuracy": float(target_row["acc_a"]),
        "selected_cells": list(target_row["cells_a"]),
        "training_steps": int(target_row["training_steps_a"]),
        "training_tokens": int(target_row["training_tokens_a"]),
        "source": identity.get("source", {}),
    }


def run_layer_placement_diagnostic(
    *,
    output: Path = DEFAULT_OUTPUT,
    baseline_root: Path = BASELINE_ROOT,
    devices: tuple[str, str] = ("cuda:0", "cuda:1"),
    seed: int = ENGINEERING_SEED,
) -> dict[str, Any]:
    """Run only the two new A-only layer probes and compare with published L23."""
    if int(seed) != ENGINEERING_SEED:
        raise ValueError("PCU-LAYER-PLACEMENT-001 is engineering-seed only")
    if len(devices) != 2 or devices[0] == devices[1]:
        raise ValueError("two distinct devices are required")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("PCU-LAYER-PLACEMENT-001 requires two CUDA devices")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = _load_baseline(Path(baseline_root))

    # Discover the exact MoE topology from the same pinned checkpoint before
    # launching two independent layer workers.
    tokenizer, probe_model, probe_manifest = load_granite(
        str(baseline["foundation"]["model_repo"]),
        revision=str(baseline["foundation"]["model_revision"]),
        device=devices[0],
    )
    try:
        _validate_foundation_manifest(probe_manifest, baseline["foundation"])
        available = discover_moe_layers(probe_model)
    finally:
        del probe_model
        del tokenizer
        torch.cuda.empty_cache()
    choices = choose_layer_targets([item.layer for item in available])
    if choices["late_baseline"] != baseline["late_layer"]:
        raise RuntimeError("discovered final MoE layer does not match published L23 baseline")
    by_layer = {item.layer: item for item in available}
    early = by_layer[choices["early"]]
    mid = by_layer[choices["mid"]]

    source = git_provenance(Path(__file__).resolve().parents[3])
    if source.get("source_dirty") is not False:
        raise RuntimeError("layer-placement diagnostic requires a clean source tree")
    design = {
        "schema": "minicells.pcu-layer-placement-001.design.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": seed,
        "causal_variable": "target_moe_decoder_layer_only",
        "fixed": {
            "task": "A_only_U_to_V",
            "dataset_manifest_sha256": baseline["dataset_manifest_sha256"],
            "loss": "answer-token-causal-cross-entropy",
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "selected_k": K,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "max_training_tokens": MAX_TRAINING_TOKENS,
            "batch_size": BATCH_SIZE,
            "allocation": "task-conditioned-gradient-l2-per-parameter:first_64_A_train",
            "routing": "inherited_parent_router",
            "evaluation": "A_eval_greedy_exact",
            "capability_floor": DIRECT_CAPABILITY_FLOOR,
        },
        "layer_selection_rule": "nearest_available(last//3), nearest_available(2*last//3), reuse_last_baseline",
        "available_moe_layers": [item.to_dict() for item in available],
        "targets": {
            "early": early.to_dict(),
            "mid": mid.to_dict(),
            "late_baseline": {
                "layer": baseline["late_layer"],
                "path": baseline["late_target_path"],
                "direct_accuracy": baseline["direct_accuracy"],
                "source": "published_PCU_KILL_001_E0",
            },
        },
        "formal_execution_not_started": True,
        "scientific_evidence": False,
    }
    write_json(output / "DESIGN.json", design)
    write_json(output / "RUN_IDENTITY.json", {
        "schema": "minicells.pcu-layer-placement-001.run-identity.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": seed,
        "run_id": output.name,
        "source": source,
        "baseline_source": baseline["source"],
        "formal_execution_not_started": True,
    })

    jobs = (("early", early, devices[0]), ("mid", mid, devices[1]))
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            name: pool.submit(
                _run_one_layer,
                layer=item.layer,
                path=item.path,
                device=device,
                expected_foundation=baseline["foundation"],
                expected_dataset_sha256=baseline["dataset_manifest_sha256"],
                seed=seed,
            )
            for name, item, device in jobs
        }
        for name, future in futures.items():
            results[name] = future.result()
            write_json(output / f"LAYER_{results[name]['layer']:02d}.json", results[name])

    comparison = {
        "late_baseline": {
            "layer": baseline["late_layer"],
            "direct_accuracy": baseline["direct_accuracy"],
            "selected_cells": baseline["selected_cells"],
            "training_steps": baseline["training_steps"],
            "training_tokens": baseline["training_tokens"],
        },
        "early": {"layer": results["early"]["layer"], "direct_accuracy": results["early"]["direct_accuracy"]},
        "mid": {"layer": results["mid"]["layer"], "direct_accuracy": results["mid"]["direct_accuracy"]},
    }
    best_name, best = max(
        ((name, row) for name, row in comparison.items()),
        key=lambda item: float(item[1]["direct_accuracy"]),
    )
    rescued = bool(float(best["direct_accuracy"]) >= DIRECT_CAPABILITY_FLOOR and best_name != "late_baseline")
    status = "LAYER_PLACEMENT_RESCUES_LOCAL_CELL_MUTATION" if rescued else "LAYER_PLACEMENT_DID_NOT_RESCUE"
    decision = {
        "schema": "minicells.pcu-layer-placement-001.decision.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "status": status,
        "valid_run": True,
        "scientific_evidence": False,
        "formal_execution_not_started": True,
        "comparison": comparison,
        "best": {"condition": best_name, **best},
        "capability_floor": DIRECT_CAPABILITY_FLOOR,
        "rescued": rescued,
        "interpretation": (
            "changing only Cell layer placement reached the inherited direct-capability floor"
            if rescued
            else "changing only Cell layer placement did not reach the inherited direct-capability floor"
        ),
        "source": source,
    }
    write_json(output / "DECISION.json", decision)
    return decision


__all__ = [
    "EXPERIMENT_ID",
    "ENGINEERING_SEED",
    "K",
    "LEARNING_RATE",
    "MoeLayer",
    "choose_layer_targets",
    "discover_moe_layers",
    "full_model_task_conditioned_allocation",
    "run_layer_placement_diagnostic",
]
