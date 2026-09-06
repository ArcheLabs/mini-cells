"""Joint L15/L23 coordination diagnostic for PCU sparse Cell mutation.

PCU-JOINT-CROSS-LAYER-001 changes exactly one scientific mechanism relative to
published PCU-SPARSE-PATH-DEPTH-001 depth-3: L15 and L23 are allowed to adapt
jointly instead of L15 being trained/frozen before L23.

The exact published depth-3 L15/K16 and L23/K16 Cell IDs are reused; there is no
reallocation. The published L7/K64 hybrid mutation is replayed exactly and then
frozen. Two joint arms are intentionally separated:

* joint128 (primary): both L15 and L23 receive 128 optimizer updates, matching
  the per-parameter update count of the sequential depth-3 control.
* joint256 (secondary): both receive 256 updates, an extra-joint-training upper
  diagnostic that must not be interpreted as pure coordination evidence.

Engineering-only evidence; formal PCU-KILL-001 seeds are never consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .cellular import CellPartition, extract_expert_projections, patch_moe_block
from .cross_layer_readout import (
    ASSOCIATION_K,
    ASSOCIATION_LAYER,
    HYBRID_BASELINE_ROOT,
    READOUT_BASELINE_ROOT,
    _load_published_baselines,
)
from .evaluation import evaluate_samples
from .governance import git_provenance, write_json
from .hybrid_objective import OBJECTIVE_BASELINE_ROOT, _load_baselines, _train_hybrid_branch
from .layer_placement import (
    BATCH_SIZE,
    DIRECT_CAPABILITY_FLOOR,
    LEARNING_RATE,
    MAX_TRAINING_TOKENS,
    _full_task_loss,
    _selected_map,
    _slice_sequences,
    _validate_foundation_manifest,
)
from .locality_width import ENGINEERING_SEED
from .model import load_granite, target_module
from .objective_alignment import ASSOCIATION_FLOOR, evaluate_candidate_ranking
from .readout_localization import gold_prefix_token_readout
from .sparse_path_depth import _freeze_l7_runtime
from .synthetic import audit_dataset, generate_world
from .task import build_task_sequences, validate_answer_only_labels
from .training import ForkedCellularExperts, selected_delta_parameters


EXPERIMENT_ID = "PCU-JOINT-CROSS-LAYER-001"
TRANSPORT_LAYER = 15
READOUT_LAYER = 23
TRANSPORT_K = 16
READOUT_K = 16
PRIMARY_STEPS = 128
SECONDARY_STEPS = 256
SEQUENTIAL_DIRECT = 0.140625
SEQUENTIAL_RANKING = 0.7890625
SEQUENTIAL_FIRST_TOKEN_TOP1 = 0.6328125
SEQUENTIAL_LATER_TOKEN_TOP1 = 0.5764331210191083
DEPTH3_ROOT = Path(
    "artifacts/research/pcu-sparse-path-depth-001/engineering/26090501-depth3-4-5"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research/pcu-joint-cross-layer-001/engineering/26090501-l15k16-l23k16-joint"
)


@dataclass(frozen=True)
class PublishedDepth3:
    selected_l15: tuple[str, ...]
    selected_l23: tuple[str, ...]
    dataset_manifest_sha256: str
    source_commit: str
    source_tree: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_l15": list(self.selected_l15),
            "selected_l23": list(self.selected_l23),
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "sequential_metrics": {
                "direct_accuracy": SEQUENTIAL_DIRECT,
                "ranking_eval_accuracy": SEQUENTIAL_RANKING,
                "first_token_top1_accuracy": SEQUENTIAL_FIRST_TOKEN_TOP1,
                "later_token_top1_accuracy": SEQUENTIAL_LATER_TOKEN_TOP1,
            },
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_published_depth3(root: Path = DEPTH3_ROOT) -> PublishedDepth3:
    root = Path(root)
    worker_path = root / "DEPTH_3.json"
    decision_path = root / "DECISION.json"
    if not worker_path.is_file() or not decision_path.is_file():
        raise RuntimeError("joint diagnostic requires published sparse-path depth-3 evidence")
    payload = json.loads(worker_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "DEEPER_SPARSE_PATH_DID_NOT_IMPROVE":
        raise RuntimeError("unexpected sparse-path depth sweep decision")
    if payload.get("experiment") != "PCU-SPARSE-PATH-DEPTH-001" or int(payload.get("depth", -1)) != 3:
        raise RuntimeError("published depth-3 identity mismatch")
    topology = payload.get("topology", {})
    if topology.get("layers") != [7, 15, 23]:
        raise RuntimeError("published depth-3 topology changed")
    if topology.get("transport_k") != [16] or int(topology.get("readout_k", -1)) != 16:
        raise RuntimeError("published depth-3 K budget changed")
    metrics = payload.get("metrics", {})
    expected = {
        "direct_accuracy": SEQUENTIAL_DIRECT,
        "ranking_eval_accuracy": SEQUENTIAL_RANKING,
        "first_token_top1_accuracy": SEQUENTIAL_FIRST_TOKEN_TOP1,
        "later_token_top1_accuracy": SEQUENTIAL_LATER_TOKEN_TOP1,
    }
    for key, value in expected.items():
        if abs(float(metrics.get(key, -1.0)) - float(value)) > 1e-12:
            raise RuntimeError(f"published depth-3 metric changed: {key}")
    stages = payload.get("stages", [])
    if len(stages) != 2:
        raise RuntimeError("published depth-3 must contain exactly L15 and L23 stages")
    stage15, stage23 = stages
    if int(stage15.get("layer", -1)) != TRANSPORT_LAYER or int(stage15.get("selected_k", -1)) != TRANSPORT_K:
        raise RuntimeError("published depth-3 L15 stage changed")
    if int(stage23.get("layer", -1)) != READOUT_LAYER or int(stage23.get("selected_k", -1)) != READOUT_K:
        raise RuntimeError("published depth-3 L23 stage changed")
    selected_l15 = tuple(str(value) for value in stage15.get("selected_cells", ()))
    selected_l23 = tuple(str(value) for value in stage23.get("selected_cells", ()))
    if len(selected_l15) != TRANSPORT_K or len(selected_l23) != READOUT_K:
        raise RuntimeError("published depth-3 selected Cell count changed")
    source = payload.get("source", {})
    if source.get("source_dirty") is not False or not source.get("source_commit") or not source.get("source_tree"):
        raise RuntimeError("published depth-3 lacks clean source provenance")
    if payload.get("formal_execution_not_started") is not True:
        raise RuntimeError("published depth-3 crossed formal boundary")
    return PublishedDepth3(
        selected_l15=selected_l15,
        selected_l23=selected_l23,
        dataset_manifest_sha256=str(payload["dataset_manifest_sha256"]),
        source_commit=str(source["source_commit"]),
        source_tree=str(source["source_tree"]),
    )


def _assert_joint_trainable_only(model: nn.Module, runtimes: Sequence[nn.Module]) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    allowed: set[int] = set()
    for runtime in runtimes:
        current = selected_delta_parameters(runtime)
        parameters.extend(current)
        allowed.update(id(value) for value in current)
    if not parameters:
        raise RuntimeError("joint diagnostic has no selected delta parameters")
    if len(allowed) != len(parameters):
        raise RuntimeError("joint diagnostic contains duplicate trainable parameter identities")
    unexpected = [
        name
        for name, value in model.named_parameters()
        if value.requires_grad and id(value) not in allowed
    ]
    if unexpected:
        raise RuntimeError(f"non-joint parameter unexpectedly trainable: {unexpected[:8]}")
    observed = {id(value) for value in model.parameters() if value.requires_grad}
    if observed != allowed:
        raise RuntimeError("joint trainable parameter identity set mismatch")
    return parameters


def _train_joint(
    model: nn.Module,
    train_sequences: Any,
    runtime15: nn.Module,
    runtime23: nn.Module,
    *,
    steps: int,
    device: str,
) -> dict[str, Any]:
    parameters = _assert_joint_trainable_only(model, (runtime15, runtime23))
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE)
    completed = 0
    tokens = 0
    final_loss = float("nan")
    while completed < int(steps) and tokens < MAX_TRAINING_TOKENS:
        progressed = False
        for start in range(0, int(train_sequences.input_ids.shape[0]), BATCH_SIZE):
            end = min(int(train_sequences.input_ids.shape[0]), start + BATCH_SIZE)
            input_ids, attention, labels, loss_mask = _slice_sequences(train_sequences, start, end)
            batch_tokens = int(loss_mask.sum())
            if batch_tokens <= 0:
                raise RuntimeError("joint training batch has no answer-token labels")
            if tokens + batch_tokens > MAX_TRAINING_TOKENS:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = _full_task_loss(
                model,
                input_ids.to(device),
                attention.to(device),
                labels.to(device),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite joint training loss")
            loss.backward()
            optimizer.step()
            completed += 1
            tokens += batch_tokens
            final_loss = float(loss.detach())
            progressed = True
            if completed % 32 == 0 or completed == int(steps):
                print(
                    f"[pcu-joint] steps={completed}/{steps} loss={final_loss:.6f} tokens={tokens}",
                    flush=True,
                )
            if completed >= int(steps):
                break
        if not progressed:
            break
    if completed != int(steps):
        raise RuntimeError(f"joint training stopped at {completed}, expected {steps}")
    return {
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "optimizer_steps": completed,
        "training_tokens": tokens,
        "final_loss": final_loss,
        "jointly_trainable_layers": [TRANSPORT_LAYER, READOUT_LAYER],
        "selected_l15": list(load_published_depth3().selected_l15),
        "selected_l23": list(load_published_depth3().selected_l23),
    }


def _evaluate(
    model: nn.Module,
    tokenizer: Any,
    eval_samples: Sequence[Any],
    candidate_universe: Sequence[str],
    *,
    device: str,
) -> dict[str, Any]:
    ranking = evaluate_candidate_ranking(model, tokenizer, eval_samples, candidate_universe, device=device)
    direct = evaluate_samples(
        model,
        tokenizer,
        eval_samples,
        split="A_eval",
        device=device,
        max_new_tokens=16,
        batch_size=16,
    )
    eval_sequences = build_task_sequences(tokenizer, eval_samples, "A_eval", max_length=128)
    validate_answer_only_labels(eval_sequences)
    gold = gold_prefix_token_readout(model, tokenizer, eval_sequences, device=device, batch_size=8)
    return {
        "ranking_eval_accuracy": float(ranking.accuracy),
        "direct_accuracy": float(direct.exact),
        "first_token_top1_accuracy": float(gold["first_token_top1_accuracy"]),
        "later_token_top1_accuracy": float(gold["later_token_top1_accuracy"]),
        "all_token_top1_accuracy": float(gold["all_token_top1_accuracy"]),
        "sequence_all_tokens_top1_accuracy": float(gold["sequence_all_tokens_top1_accuracy"]),
        "ranking": ranking.to_dict(),
        "direct_evaluation": direct.to_dict(),
        "gold_prefix": gold,
    }


def run_joint_arm(
    *,
    steps: int,
    output: Path,
    device: str,
    objective_root: Path = OBJECTIVE_BASELINE_ROOT,
    hybrid_root: Path = HYBRID_BASELINE_ROOT,
    readout_root: Path = READOUT_BASELINE_ROOT,
    depth3_root: Path = DEPTH3_ROOT,
    seed: int = ENGINEERING_SEED,
) -> dict[str, Any]:
    if int(seed) != ENGINEERING_SEED:
        raise ValueError("PCU-JOINT-CROSS-LAYER-001 is engineering-seed only")
    if int(steps) not in (PRIMARY_STEPS, SECONDARY_STEPS):
        raise ValueError(f"steps must be {PRIMARY_STEPS} or {SECONDARY_STEPS}")
    parsed = torch.device(str(device))
    if parsed.type != "cuda" or parsed.index is None:
        raise ValueError("joint arm requires an explicit CUDA device")
    if not torch.cuda.is_available() or parsed.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device unavailable: {device}")
    torch.cuda.set_device(parsed.index)

    depth3 = load_published_depth3(Path(depth3_root))
    published = _load_published_baselines(Path(hybrid_root), Path(readout_root))
    baseline = _load_baselines(Path(objective_root))
    if baseline.selected_cells != published.selected_l7:
        raise RuntimeError("joint diagnostic L7 Cell identity drifted")
    if baseline.dataset_manifest_sha256 != depth3.dataset_manifest_sha256:
        raise RuntimeError("joint diagnostic dataset identity drifted")
    source = git_provenance(_repo_root())
    if source.get("source_dirty") is not False:
        raise RuntimeError("joint arm requires clean source tree")

    tokenizer, model, manifest = load_granite(
        str(baseline.foundation["model_repo"]),
        revision=str(baseline.foundation["model_revision"]),
        device=device,
    )
    try:
        _validate_foundation_manifest(manifest, baseline.foundation)
        world = generate_world(ENGINEERING_SEED, count=128, tokenizer=tokenizer)
        audit = audit_dataset(world)
        if not audit.passed or world.manifest_sha256() != depth3.dataset_manifest_sha256:
            raise RuntimeError("joint diagnostic dataset audit/identity failure")
        train_samples = list(world.splits["A_train"])
        eval_samples = list(world.splits["A_eval"])
        candidate_universe = tuple(item.v for item in world.triples)
        train_sequences = build_task_sequences(tokenizer, train_samples, "A_train", max_length=128)
        validate_answer_only_labels(train_sequences)

        model.requires_grad_(False)
        block7 = target_module(model, f"model.layers.{ASSOCIATION_LAYER}.block_sparse_moe")
        projection7 = extract_expert_projections(block7.experts, 0)
        cellular7 = patch_moe_block(block7, CellPartition(projection7.intermediate_size, 4))
        model.requires_grad_(False)
        cellular7.requires_grad_(False)
        l7_config = type("JointL7Config", (), {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "max_optimizer_steps": 128,
            "max_training_tokens": MAX_TRAINING_TOKENS,
            "batch_size": BATCH_SIZE,
            "seed": ENGINEERING_SEED,
        })()
        runtime7, l7_training = _train_hybrid_branch(
            model,
            block7,
            cellular7,
            tokenizer,
            train_samples,
            published.selected_l7,
            device=device,
            config=l7_config,
        )
        l7_ranking = evaluate_candidate_ranking(model, tokenizer, eval_samples, candidate_universe, device=device)
        l7_direct = evaluate_samples(
            model, tokenizer, eval_samples, split="A_eval", device=device, max_new_tokens=16, batch_size=16
        )
        if abs(float(l7_ranking.accuracy) - published.l7_ranking_accuracy) > 1e-12:
            raise RuntimeError("JOINT_L7_REPRODUCTION_MISMATCH ranking")
        if abs(float(l7_direct.exact) - published.l7_direct_accuracy) > 1e-12:
            raise RuntimeError("JOINT_L7_REPRODUCTION_MISMATCH direct")
        _freeze_l7_runtime(model, runtime7)

        runtimes: dict[int, ForkedCellularExperts] = {}
        for layer, selected in (
            (TRANSPORT_LAYER, depth3.selected_l15),
            (READOUT_LAYER, depth3.selected_l23),
        ):
            block = target_module(model, f"model.layers.{layer}.block_sparse_moe")
            projection = extract_expert_projections(block.experts, 0)
            cellular = patch_moe_block(block, CellPartition(projection.intermediate_size, 4))
            model.requires_grad_(False)
            cellular.requires_grad_(False)
            runtime = ForkedCellularExperts(cellular, _selected_map(selected, layer)).to(device)
            block.experts = runtime
            runtimes[layer] = runtime

        training = _train_joint(
            model,
            train_sequences,
            runtimes[TRANSPORT_LAYER],
            runtimes[READOUT_LAYER],
            steps=int(steps),
            device=device,
        )
        if tuple(training["selected_l15"]) != depth3.selected_l15:
            raise RuntimeError("joint L15 Cell identity drift")
        if tuple(training["selected_l23"]) != depth3.selected_l23:
            raise RuntimeError("joint L23 Cell identity drift")
        metrics = _evaluate(model, tokenizer, eval_samples, candidate_universe, device=device)
        result = {
            "schema": "minicells.pcu-joint-cross-layer-001.arm.v1",
            "experiment": EXPERIMENT_ID,
            "phase": "engineering_diagnostic",
            "seed": ENGINEERING_SEED,
            "arm": f"joint_{int(steps)}",
            "optimizer_steps": int(steps),
            "primary_coordination_arm": int(steps) == PRIMARY_STEPS,
            "valid_run": True,
            "scientific_evidence": False,
            "formal_execution_not_started": True,
            "source": source,
            "foundation": dict(manifest),
            "dataset_manifest_sha256": world.manifest_sha256(),
            "published_depth3": depth3.to_dict(),
            "l7_reproduction": {
                "ranking_eval_accuracy": float(l7_ranking.accuracy),
                "direct_accuracy": float(l7_direct.exact),
                "exact": True,
                "training": l7_training,
            },
            "joint_topology": {
                "layers": [ASSOCIATION_LAYER, TRANSPORT_LAYER, READOUT_LAYER],
                "frozen_association_layer": ASSOCIATION_LAYER,
                "joint_layers": [TRANSPORT_LAYER, READOUT_LAYER],
                "selected_l15": list(depth3.selected_l15),
                "selected_l23": list(depth3.selected_l23),
                "selected_k_each": [TRANSPORT_K, READOUT_K],
                "selection": "reuse_exact_published_depth3_cells_no_reallocation",
            },
            "training": training,
            "metrics": metrics,
        }
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)
        return result
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except torch.AcceleratorError:
                pass


def classify_joint(primary: Mapping[str, Any], secondary: Mapping[str, Any]) -> str:
    p = primary["metrics"]
    s = secondary["metrics"]
    p_direct = float(p["direct_accuracy"])
    p_ranking = float(p["ranking_eval_accuracy"])
    s_direct = float(s["direct_accuracy"])
    s_ranking = float(s["ranking_eval_accuracy"])
    if p_direct >= DIRECT_CAPABILITY_FLOOR and p_ranking >= ASSOCIATION_FLOOR:
        return "JOINT_COORDINATION_RESCUES_NATIVE_GENERATION"
    if p_direct > SEQUENTIAL_DIRECT and p_ranking >= ASSOCIATION_FLOOR:
        return "JOINT_COORDINATION_IMPROVES_BUT_DOES_NOT_RESCUE"
    if s_direct >= DIRECT_CAPABILITY_FLOOR and s_ranking >= ASSOCIATION_FLOOR:
        return "EXTRA_JOINT_UPDATES_RESCUE_COORDINATION_ALONE_UNPROVEN"
    if s_direct > SEQUENTIAL_DIRECT and s_ranking >= ASSOCIATION_FLOOR:
        return "EXTRA_JOINT_UPDATES_IMPROVE_COORDINATION_ALONE_UNPROVEN"
    if p_direct > SEQUENTIAL_DIRECT or s_direct > SEQUENTIAL_DIRECT:
        return "JOINT_GENERATION_IMPROVES_ASSOCIATION_REGRESSED"
    return "JOINT_COORDINATION_DID_NOT_IMPROVE"


def aggregate_joint(
    *,
    primary_file: Path,
    secondary_file: Path,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    depth3 = load_published_depth3(DEPTH3_ROOT)
    primary = json.loads(Path(primary_file).read_text(encoding="utf-8"))
    secondary = json.loads(Path(secondary_file).read_text(encoding="utf-8"))
    for payload, steps in ((primary, PRIMARY_STEPS), (secondary, SECONDARY_STEPS)):
        if payload.get("experiment") != EXPERIMENT_ID or int(payload.get("optimizer_steps", -1)) != steps:
            raise RuntimeError("joint worker identity mismatch")
        if payload.get("valid_run") is not True or payload.get("formal_execution_not_started") is not True:
            raise RuntimeError("joint worker invalid/formal")
        if payload.get("l7_reproduction", {}).get("exact") is not True:
            raise RuntimeError("joint worker failed exact L7 reproduction")
        topology = payload.get("joint_topology", {})
        if topology.get("layers") != [7, 15, 23] or topology.get("joint_layers") != [15, 23]:
            raise RuntimeError("joint worker topology drifted")
        if tuple(topology.get("selected_l15", ())) != depth3.selected_l15:
            raise RuntimeError("joint worker L15 identity drifted")
        if tuple(topology.get("selected_l23", ())) != depth3.selected_l23:
            raise RuntimeError("joint worker L23 identity drifted")
    if primary.get("source") != secondary.get("source"):
        raise RuntimeError("joint workers did not use identical source provenance")
    status = classify_joint(primary, secondary)
    summary = {
        "sequential_depth3": {
            "direct_accuracy": SEQUENTIAL_DIRECT,
            "ranking_eval_accuracy": SEQUENTIAL_RANKING,
            "first_token_top1_accuracy": SEQUENTIAL_FIRST_TOKEN_TOP1,
            "later_token_top1_accuracy": SEQUENTIAL_LATER_TOKEN_TOP1,
            "training_mode": "L15_128_then_freeze_then_L23_128",
        },
        "joint128": {key: primary["metrics"][key] for key in (
            "direct_accuracy", "ranking_eval_accuracy", "first_token_top1_accuracy",
            "later_token_top1_accuracy", "sequence_all_tokens_top1_accuracy"
        )},
        "joint256": {key: secondary["metrics"][key] for key in (
            "direct_accuracy", "ranking_eval_accuracy", "first_token_top1_accuracy",
            "later_token_top1_accuracy", "sequence_all_tokens_top1_accuracy"
        )},
    }
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "minicells.pcu-joint-cross-layer-001.result.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "status": status,
        "valid_run": True,
        "scientific_evidence": False,
        "formal_execution_not_started": True,
        "causal_question": "does_joint_L15_L23_optimization_outperform_identical_sequential_depth3_cells",
        "primary_arm": "joint128_per_parameter_update_matched",
        "secondary_arm": "joint256_extra_joint_updates_diagnostic",
        "published_depth3": depth3.to_dict(),
        "summary": summary,
        "worker_files_external": {
            "joint128": str(primary_file),
            "joint256": str(secondary_file),
        },
    }
    write_json(output_root / "RESULT.json", result)
    write_json(output_root / "DECISION.json", {
        "schema": "minicells.pcu-joint-cross-layer-001.decision.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "status": status,
        "valid_run": True,
        "scientific_evidence": False,
        "formal_execution_not_started": True,
        "association_floor": ASSOCIATION_FLOOR,
        "direct_capability_floor": DIRECT_CAPABILITY_FLOOR,
        "sequential_direct_accuracy": SEQUENTIAL_DIRECT,
        "sequential_ranking_accuracy": SEQUENTIAL_RANKING,
        "joint128_direct_accuracy": primary["metrics"]["direct_accuracy"],
        "joint128_ranking_accuracy": primary["metrics"]["ranking_eval_accuracy"],
        "joint128_first_token_top1_accuracy": primary["metrics"]["first_token_top1_accuracy"],
        "joint128_later_token_top1_accuracy": primary["metrics"]["later_token_top1_accuracy"],
        "joint256_direct_accuracy": secondary["metrics"]["direct_accuracy"],
        "joint256_ranking_accuracy": secondary["metrics"]["ranking_eval_accuracy"],
        "exact_published_depth3_cells_reused": True,
        "no_reallocation": True,
        "l7_frozen_before_joint_training": True,
        "dual_gpu_execution_required": True,
        "primary_coordination_steps": PRIMARY_STEPS,
        "secondary_extra_joint_steps": SECONDARY_STEPS,
    })
    return result


__all__ = [
    "EXPERIMENT_ID",
    "TRANSPORT_LAYER",
    "READOUT_LAYER",
    "TRANSPORT_K",
    "READOUT_K",
    "PRIMARY_STEPS",
    "SECONDARY_STEPS",
    "SEQUENTIAL_DIRECT",
    "SEQUENTIAL_RANKING",
    "DEPTH3_ROOT",
    "DEFAULT_OUTPUT",
    "load_published_depth3",
    "run_joint_arm",
    "aggregate_joint",
    "classify_joint",
]
