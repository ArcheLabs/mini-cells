"""Protocol-v3 causal reattachment and amplitude-sweep experiment.

Protocol v3 is a post-observation amendment to PCU-HYBRID-REATTACHMENT-001.
The first engineering execution on source commit
cadaf6c397000c55deb35db67a3b266003cb3004 compared untouched fused Granite
against a cellularized graph under the strict zero-state gate. Cellularization
changes floating-point reduction order, so v3 keeps that native->cellular
comparison as a G0 numerical diagnostic and applies the unchanged 1e-5 gate to
the matched cellular path PARENT_ZERO_DELTA <-> CELL_OFF.

GPU0 executes the corrected causal ON/OFF arm. GPU1 independently reconstructs
the same published ranking-only L7/K64 mutation and evaluates a frozen alpha
sweep without any additional training, bridge, readout, or router.

Engineering evidence only. Formal PCU seeds remain RESERVED_UNTOUCHED.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterator

import matplotlib.pyplot as plt
import torch
from torch import nn

from .cellular import CellPartition, extract_expert_projections, patch_moe_block
from .evaluation import evaluate_samples
from .governance import git_provenance, write_json
from .hybrid_objective import OBJECTIVE_BASELINE_ROOT, _load_baselines
from .hybrid_reattachment import (
    AnswerMetrics,
    _capture_logits,
    _load_published_source,
    compare_logits,
    delta_sha256,
    evaluate_answer_metrics,
    frozen_parent_sha256,
    temporarily_zero_cell_deltas,
)
from .layer_placement import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_OPTIMIZER_STEPS,
    MAX_TRAINING_TOKENS,
    _assert_only_selected_deltas_trainable,
    _validate_foundation_manifest,
)
from .locality_width import ENGINEERING_SEED
from .model import load_granite, target_module
from .objective_alignment import (
    ASSOCIATION_FLOOR,
    _train_ranking_branch,
    evaluate_candidate_ranking,
)
from .synthetic import audit_dataset, generate_world
from .task import TaskSequences, build_task_sequences
from .training import BranchTrainingConfig


EXPERIMENT_ID = "PCU-HYBRID-REATTACHMENT-001"
PROTOCOL_VERSION = 3
TARGET_K = 64
PUBLISHED_SOURCE_ROOT = OBJECTIVE_BASELINE_ROOT
DEFAULT_OUTPUT = Path(
    "artifacts/research/pcu-hybrid-reattachment-001/engineering/"
    "26090501-l7-k64-ranking-causal-reattach-v3"
)
FIRST_ENGINEERING_SOURCE_COMMIT = "cadaf6c397000c55deb35db67a3b266003cb3004"
EXPECTED_PUBLISHED_RANKING_ACCURACY = 0.8203125
EXPECTED_PUBLISHED_DIRECT_ACCURACY = 0.0

# Frozen before protocol-v3 GPU execution. None were relaxed after v2.
EQUIVALENCE_MAX_ABS_LOGIT_DIFF = 1e-5
RESTORATION_MAX_ABS_LOGIT_DIFF = 1e-5
MIN_CAUSAL_RANKING_GAIN = 0.50
MIN_CAUSAL_MARGIN_GAIN = 0.0
MAX_CONTROL_ANSWER_NLL_INCREASE = 0.10
ALPHA_SWEEP = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)


@dataclass
class ReplayState:
    tokenizer: Any
    model: nn.Module
    runtime: nn.Module
    baseline: Any
    published: dict[str, Any]
    source: dict[str, Any]
    world: Any
    a_eval: list[Any]
    b_eval: list[Any]
    a_sequences: TaskSequences
    b_sequences: TaskSequences
    candidate_universe: tuple[str, ...]
    training: dict[str, Any]
    trained_delta_hash: str
    parent_hash_before: str
    base_a_logits: torch.Tensor | None = None
    base_b_logits: torch.Tensor | None = None
    parent_a_logits: torch.Tensor | None = None
    parent_b_logits: torch.Tensor | None = None
    base_a_answer: AnswerMetrics | None = None
    base_b_answer: AnswerMetrics | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_cuda(device: str) -> None:
    parsed = torch.device(str(device))
    if parsed.type != "cuda" or parsed.index is None:
        raise ValueError("protocol-v3 worker requires an explicit CUDA device")
    if not torch.cuda.is_available() or parsed.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device unavailable: {device}")
    torch.cuda.set_device(parsed.index)


def _delta_parameters(runtime: nn.Module) -> list[tuple[str, nn.Parameter]]:
    values = [
        (name, parameter)
        for name, parameter in runtime.named_parameters()
        if name.startswith("delta_") or ".delta_" in name
    ]
    if not values:
        raise RuntimeError("protocol-v3 runtime exposes no Cell delta parameters")
    return values


@contextmanager
def temporarily_scale_cell_deltas(runtime: nn.Module, alpha: float) -> Iterator[dict[str, Any]]:
    """Scale the already-trained mutation only, then restore byte-for-byte."""
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    parameters = _delta_parameters(runtime)
    snapshots = [(name, parameter, parameter.detach().clone()) for name, parameter in parameters]
    trained_sha = delta_sha256(runtime)
    with torch.no_grad():
        for _, parameter, snapshot in snapshots:
            parameter.copy_(snapshot * alpha)
    scaled_sha = delta_sha256(runtime)
    try:
        yield {"alpha": alpha, "trained_sha256": trained_sha, "scaled_sha256": scaled_sha}
    finally:
        with torch.no_grad():
            for _, parameter, snapshot in snapshots:
                parameter.copy_(snapshot)
        if delta_sha256(runtime) != trained_sha:
            raise RuntimeError("ALPHA_SWEEP_DELTA_RESTORATION_MISMATCH")


@contextmanager
def replay_published_mutation(
    *,
    device: str,
    source_root: Path = PUBLISHED_SOURCE_ROOT,
    capture_native_and_parent: bool,
) -> Iterator[ReplayState]:
    """Deterministically reconstruct the exact ranking-only L7/K64 mutation."""
    _validate_cuda(device)
    baseline = _load_baselines(Path(source_root))
    published = _load_published_source(Path(source_root))
    source = git_provenance(_repo_root())
    if source.get("source_dirty") is not False:
        raise RuntimeError("protocol-v3 worker requires a clean source tree")

    tokenizer, model, manifest = load_granite(
        str(baseline.foundation["model_repo"]),
        revision=str(baseline.foundation["model_revision"]),
        device=str(device),
    )
    try:
        _validate_foundation_manifest(manifest, baseline.foundation)
        model.requires_grad_(False)
        model.eval()
        world = generate_world(ENGINEERING_SEED, count=128, tokenizer=tokenizer)
        audit = audit_dataset(world)
        if not audit.passed:
            raise RuntimeError(f"protocol-v3 dataset audit failed: {audit.errors}")
        if world.manifest_sha256() != baseline.dataset_manifest_sha256:
            raise RuntimeError("protocol-v3 dataset differs from published mutation source")

        train_samples = list(world.splits["A_train"])
        a_eval = list(world.splits["A_eval"])
        b_eval = list(world.splits["B_eval"])
        a_sequences = build_task_sequences(tokenizer, a_eval, "A_eval", max_length=128)
        b_sequences = build_task_sequences(tokenizer, b_eval, "B_eval", max_length=128)
        candidate_universe = tuple(item.v for item in world.triples)

        base_a_logits = base_b_logits = None
        base_a_answer = base_b_answer = None
        if capture_native_and_parent:
            base_a_logits = _capture_logits(model, a_sequences, device=str(device))
            base_b_logits = _capture_logits(model, b_sequences, device=str(device))
            base_a_answer = evaluate_answer_metrics(model, a_sequences, device=str(device))
            base_b_answer = evaluate_answer_metrics(model, b_sequences, device=str(device))

        block = target_module(model, baseline.target_path)
        projection = extract_expert_projections(block.experts, 0)
        cellular_experts = patch_moe_block(block, CellPartition(projection.intermediate_size, 4))
        model.requires_grad_(False)
        cellular_experts.requires_grad_(False)

        parent_a_logits = parent_b_logits = None
        if capture_native_and_parent:
            parent_a_logits = _capture_logits(model, a_sequences, device=str(device))
            parent_b_logits = _capture_logits(model, b_sequences, device=str(device))

        config = BranchTrainingConfig(
            optimizer="AdamW",
            learning_rate=LEARNING_RATE,
            max_optimizer_steps=MAX_OPTIMIZER_STEPS,
            max_training_tokens=MAX_TRAINING_TOKENS,
            batch_size=BATCH_SIZE,
            seed=ENGINEERING_SEED,
        )
        print("[pcu-reattach-v3] replaying published ranking-only L7/K64 mutation", flush=True)
        runtime, training = _train_ranking_branch(
            model,
            block,
            cellular_experts,
            tokenizer,
            train_samples,
            candidate_universe,
            baseline.selected_cells,
            device=str(device),
            config=config,
        )
        _assert_only_selected_deltas_trainable(model, runtime)
        if tuple(training["selected_cells"]) != baseline.selected_cells:
            raise RuntimeError("REATTACHMENT_V3_ALLOCATION_DRIFT")
        model.eval()
        state = ReplayState(
            tokenizer=tokenizer,
            model=model,
            runtime=runtime,
            baseline=baseline,
            published=published,
            source=source,
            world=world,
            a_eval=a_eval,
            b_eval=b_eval,
            a_sequences=a_sequences,
            b_sequences=b_sequences,
            candidate_universe=candidate_universe,
            training=training,
            trained_delta_hash=delta_sha256(runtime),
            parent_hash_before=frozen_parent_sha256(runtime),
            base_a_logits=base_a_logits,
            base_b_logits=base_b_logits,
            parent_a_logits=parent_a_logits,
            parent_b_logits=parent_b_logits,
            base_a_answer=base_a_answer,
            base_b_answer=base_b_answer,
        )
        yield state
        if frozen_parent_sha256(runtime) != state.parent_hash_before:
            raise RuntimeError("FROZEN_PARENT_MUTATION_DETECTED")
        if delta_sha256(runtime) != state.trained_delta_hash:
            raise RuntimeError("TRAINED_DELTA_NOT_RESTORED")
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except torch.AcceleratorError:
                pass


def classify_primary_v3(
    *,
    replay_matches: bool,
    same_graph_equivalence_max_abs: float,
    restoration_max_abs: float,
    ranking_on: float,
    ranking_off: float,
    margin_gain: float,
    control_nll_increase: float,
) -> str:
    """Engineering classifier. Native G0 drift is intentionally non-gating."""
    if not replay_matches:
        return "REPLAY_DID_NOT_MATCH_PUBLISHED_MUTATION"
    if same_graph_equivalence_max_abs > EQUIVALENCE_MAX_ABS_LOGIT_DIFF:
        return "SAME_GRAPH_ZERO_STATE_EQUIVALENCE_FAILED"
    if restoration_max_abs > RESTORATION_MAX_ABS_LOGIT_DIFF:
        return "REVERSIBILITY_FAILED"
    ranking_gain = float(ranking_on) - float(ranking_off)
    causal_pass = (
        ranking_on >= ASSOCIATION_FLOOR
        and ranking_gain >= MIN_CAUSAL_RANKING_GAIN
        and margin_gain > MIN_CAUSAL_MARGIN_GAIN
    )
    if causal_pass and control_nll_increase <= MAX_CONTROL_ANSWER_NLL_INCREASE:
        return "ENGINEERING_SIGNAL_HYBRID_REATTACHMENT_SUPPORTED"
    if causal_pass:
        return "CAUSAL_HYBRID_CONSUMPTION_SUPPORTED_LOCALITY_FAILED"
    if ranking_gain > 0.0 or margin_gain > 0.0:
        return "CAUSAL_EXPRESSION_PRESENT_GATES_UNRESOLVED"
    return "NO_CAUSAL_EXPRESSION_ENGINEERING"


def run_primary_arm(*, output: Path, device: str = "cuda:0") -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with replay_published_mutation(
        device=device,
        source_root=PUBLISHED_SOURCE_ROOT,
        capture_native_and_parent=True,
    ) as state:
        assert state.base_a_logits is not None and state.base_b_logits is not None
        assert state.parent_a_logits is not None and state.parent_b_logits is not None
        assert state.base_a_answer is not None and state.base_b_answer is not None

        on_a_logits = _capture_logits(state.model, state.a_sequences, device=device)
        on_b_logits = _capture_logits(state.model, state.b_sequences, device=device)
        on_a_answer = evaluate_answer_metrics(state.model, state.a_sequences, device=device)
        on_b_answer = evaluate_answer_metrics(state.model, state.b_sequences, device=device)
        on_ranking = evaluate_candidate_ranking(
            state.model,
            state.tokenizer,
            state.a_eval,
            state.candidate_universe,
            device=device,
        )
        on_direct = evaluate_samples(
            state.model,
            state.tokenizer,
            state.a_eval,
            split="A_eval",
            device=device,
            max_new_tokens=16,
            batch_size=16,
        )

        with temporarily_zero_cell_deltas(state.runtime) as intervention:
            off_a_logits = _capture_logits(state.model, state.a_sequences, device=device)
            off_b_logits = _capture_logits(state.model, state.b_sequences, device=device)
            off_a_answer = evaluate_answer_metrics(state.model, state.a_sequences, device=device)
            off_b_answer = evaluate_answer_metrics(state.model, state.b_sequences, device=device)
            off_ranking = evaluate_candidate_ranking(
                state.model,
                state.tokenizer,
                state.a_eval,
                state.candidate_universe,
                device=device,
            )
            off_direct = evaluate_samples(
                state.model,
                state.tokenizer,
                state.a_eval,
                split="A_eval",
                device=device,
                max_new_tokens=16,
                batch_size=16,
            )

        restored_a_logits = _capture_logits(state.model, state.a_sequences, device=device)
        restored_a_answer = evaluate_answer_metrics(state.model, state.a_sequences, device=device)
        restored_delta_hash = delta_sha256(state.runtime)
        parent_hash_after = frozen_parent_sha256(state.runtime)

        diffs = {
            "base_vs_parent_A": compare_logits(state.base_a_logits, state.parent_a_logits),
            "base_vs_parent_B": compare_logits(state.base_b_logits, state.parent_b_logits),
            "base_vs_off_A": compare_logits(state.base_a_logits, off_a_logits),
            "base_vs_off_B": compare_logits(state.base_b_logits, off_b_logits),
            "parent_vs_off_A": compare_logits(state.parent_a_logits, off_a_logits),
            "parent_vs_off_B": compare_logits(state.parent_b_logits, off_b_logits),
            "on_vs_off_A": compare_logits(on_a_logits, off_a_logits),
            "on_vs_off_B": compare_logits(on_b_logits, off_b_logits),
            "on_vs_restored_A": compare_logits(on_a_logits, restored_a_logits),
        }
        ranking_gain = float(on_ranking.accuracy) - float(off_ranking.accuracy)
        margin_gain = float(on_a_answer.mean_target_margin) - float(off_a_answer.mean_target_margin)
        target_nll_gain = float(off_a_answer.answer_nll) - float(on_a_answer.answer_nll)
        control_nll_increase = float(on_b_answer.answer_nll) - float(off_b_answer.answer_nll)
        same_graph_equivalence = max(
            diffs["parent_vs_off_A"].max_abs,
            diffs["parent_vs_off_B"].max_abs,
        )
        native_g0_drift = max(
            diffs["base_vs_parent_A"].max_abs,
            diffs["base_vs_parent_B"].max_abs,
        )
        restoration = diffs["on_vs_restored_A"].max_abs
        replay_ranking_matches = (
            abs(float(on_ranking.accuracy) - EXPECTED_PUBLISHED_RANKING_ACCURACY) <= 1e-12
        )
        replay_direct_matches = (
            abs(float(on_direct.exact) - EXPECTED_PUBLISHED_DIRECT_ACCURACY) <= 1e-12
        )
        replay_matches = bool(replay_ranking_matches and replay_direct_matches)
        status = classify_primary_v3(
            replay_matches=replay_matches,
            same_graph_equivalence_max_abs=same_graph_equivalence,
            restoration_max_abs=restoration,
            ranking_on=float(on_ranking.accuracy),
            ranking_off=float(off_ranking.accuracy),
            margin_gain=margin_gain,
            control_nll_increase=control_nll_increase,
        )

        result = {
            "schema": "minicells.pcu-hybrid-reattachment-001.primary.v3",
            "experiment": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "arm": "primary_causal_reattachment",
            "phase": "engineering_diagnostic",
            "status": status,
            "valid_run": replay_matches,
            "scientific_evidence": False,
            "formal_execution_not_started": True,
            "source": state.source,
            "foundation": dict(state.baseline.foundation),
            "dataset_manifest_sha256": state.world.manifest_sha256(),
            "selected_k": TARGET_K,
            "selected_cells": list(state.baseline.selected_cells),
            "training": state.training,
            "mutation": {
                "trained_delta_sha256": state.trained_delta_hash,
                "restored_delta_sha256": restored_delta_hash,
                "zero_intervention_sha256": intervention["zero_sha256"],
                "frozen_parent_sha256_before": state.parent_hash_before,
                "frozen_parent_sha256_after": parent_hash_after,
                "restoration_exact": restored_delta_hash == state.trained_delta_hash,
                "parent_unchanged": parent_hash_after == state.parent_hash_before,
            },
            "logit_differences": {name: value.to_dict() for name, value in diffs.items()},
            "equivalence": {
                "gate_definition": "PARENT_ZERO_DELTA_vs_CELL_OFF_same_cellular_graph",
                "same_graph_max_abs": same_graph_equivalence,
                "same_graph_passes": same_graph_equivalence <= EQUIVALENCE_MAX_ABS_LOGIT_DIFF,
                "native_G0_max_abs": native_g0_drift,
                "native_G0_is_diagnostic_only": True,
                "threshold_unchanged_from_v2": EQUIVALENCE_MAX_ABS_LOGIT_DIFF,
            },
            "answer_metrics": {
                "A_base": state.base_a_answer.to_dict(),
                "B_base": state.base_b_answer.to_dict(),
                "A_on": on_a_answer.to_dict(),
                "B_on": on_b_answer.to_dict(),
                "A_off": off_a_answer.to_dict(),
                "B_off": off_b_answer.to_dict(),
                "A_restored": restored_a_answer.to_dict(),
            },
            "causal_effect": {
                "ranking_on": float(on_ranking.accuracy),
                "ranking_off": float(off_ranking.accuracy),
                "ranking_gain": ranking_gain,
                "answer_margin_gain": margin_gain,
                "answer_nll_gain": target_nll_gain,
                "B_control_answer_nll_increase": control_nll_increase,
                "direct_on": float(on_direct.exact),
                "direct_off": float(off_direct.exact),
                "direct_gain": float(on_direct.exact) - float(off_direct.exact),
            },
            "ranking": {"on": on_ranking.to_dict(), "off": off_ranking.to_dict()},
            "direct_generation": {
                "on": on_direct.to_dict(),
                "off": off_direct.to_dict(),
                "primary_success_metric": False,
            },
            "published_replay": {
                "expected_ranking_accuracy": EXPECTED_PUBLISHED_RANKING_ACCURACY,
                "observed_ranking_accuracy": float(on_ranking.accuracy),
                "ranking_matches": replay_ranking_matches,
                "expected_direct_accuracy": EXPECTED_PUBLISHED_DIRECT_ACCURACY,
                "observed_direct_accuracy": float(on_direct.exact),
                "direct_matches": replay_direct_matches,
                "replay_matches": replay_matches,
            },
        }
        decision = {
            "schema": "minicells.pcu-hybrid-reattachment-001.primary-decision.v3",
            "experiment": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "status": status,
            "valid_run": replay_matches,
            "scientific_evidence": False,
            "formal_execution_not_started": True,
            "formal_decision": "RESERVED_UNRUN",
            "same_graph_zero_state_equivalence_passes": (
                same_graph_equivalence <= EQUIVALENCE_MAX_ABS_LOGIT_DIFF
            ),
            "native_G0_equivalence_is_diagnostic_only": True,
            "native_G0_max_abs_logit_diff": native_g0_drift,
            "reversibility_passes": restoration <= RESTORATION_MAX_ABS_LOGIT_DIFF,
            "association_on_passes": float(on_ranking.accuracy) >= ASSOCIATION_FLOOR,
            "causal_ranking_gain_passes": ranking_gain >= MIN_CAUSAL_RANKING_GAIN,
            "causal_margin_gain_passes": margin_gain > MIN_CAUSAL_MARGIN_GAIN,
            "control_locality_passes": control_nll_increase <= MAX_CONTROL_ANSWER_NLL_INCREASE,
            "ranking_on": float(on_ranking.accuracy),
            "ranking_off": float(off_ranking.accuracy),
            "ranking_gain": ranking_gain,
            "answer_margin_gain": margin_gain,
            "answer_nll_gain": target_nll_gain,
            "B_control_answer_nll_increase": control_nll_increase,
            "cell_alone_takeover_required": False,
            "new_bridge_used": False,
            "new_router_used": False,
            "ce_readout_regularizer_used": False,
            "source": state.source,
        }
        write_json(output / "PRIMARY_RESULT.json", result)
        write_json(output / "PRIMARY_DECISION.json", decision)
        return result


def classify_sweep(
    rows: list[dict[str, Any]],
    *,
    replay_matches: bool,
    restoration_exact: bool,
) -> str:
    if not replay_matches:
        return "AMPLITUDE_SWEEP_REPLAY_DID_NOT_MATCH"
    if not restoration_exact:
        return "AMPLITUDE_SWEEP_REVERSIBILITY_FAILED"
    compatible = [row for row in rows if float(row["alpha"]) > 0.0 and bool(row["joint_pass"])]
    if compatible:
        return "AMPLITUDE_SWEEP_FINDS_LOCALITY_COMPATIBLE_POINT"
    return "AMPLITUDE_SWEEP_NO_LOCALITY_COMPATIBLE_POINT"


def run_amplitude_sweep_arm(*, output: Path, device: str = "cuda:1") -> dict[str, Any]:
    """Evaluate alpha*delta without any additional optimizer step or learned gate."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with replay_published_mutation(
        device=device,
        source_root=PUBLISHED_SOURCE_ROOT,
        capture_native_and_parent=False,
    ) as state:
        rows: list[dict[str, Any]] = []
        for alpha in ALPHA_SWEEP:
            with temporarily_scale_cell_deltas(state.runtime, alpha) as intervention:
                a_ranking = evaluate_candidate_ranking(
                    state.model,
                    state.tokenizer,
                    state.a_eval,
                    state.candidate_universe,
                    device=device,
                )
                a_answer = evaluate_answer_metrics(state.model, state.a_sequences, device=device)
                b_answer = evaluate_answer_metrics(state.model, state.b_sequences, device=device)
                rows.append({
                    "alpha": float(alpha),
                    "scaled_delta_sha256": intervention["scaled_sha256"],
                    "A_ranking_accuracy": float(a_ranking.accuracy),
                    "A_answer_nll": float(a_answer.answer_nll),
                    "A_mean_target_margin": float(a_answer.mean_target_margin),
                    "B_answer_nll": float(b_answer.answer_nll),
                })

        restoration_exact = delta_sha256(state.runtime) == state.trained_delta_hash
        zero = rows[0]
        for row in rows:
            row["ranking_gain_vs_zero"] = float(row["A_ranking_accuracy"]) - float(
                zero["A_ranking_accuracy"]
            )
            row["A_answer_nll_gain_vs_zero"] = float(zero["A_answer_nll"]) - float(
                row["A_answer_nll"]
            )
            row["A_margin_gain_vs_zero"] = float(row["A_mean_target_margin"]) - float(
                zero["A_mean_target_margin"]
            )
            row["B_control_answer_nll_increase_vs_zero"] = float(row["B_answer_nll"]) - float(
                zero["B_answer_nll"]
            )
            row["association_pass"] = float(row["A_ranking_accuracy"]) >= ASSOCIATION_FLOOR
            row["locality_pass"] = (
                float(row["B_control_answer_nll_increase_vs_zero"])
                <= MAX_CONTROL_ANSWER_NLL_INCREASE
            )
            row["joint_pass"] = bool(row["association_pass"] and row["locality_pass"])

        alpha1 = next(row for row in rows if abs(float(row["alpha"]) - 1.0) <= 1e-12)
        replay_matches = (
            abs(float(alpha1["A_ranking_accuracy"]) - EXPECTED_PUBLISHED_RANKING_ACCURACY)
            <= 1e-12
        )
        status = classify_sweep(
            rows,
            replay_matches=replay_matches,
            restoration_exact=restoration_exact,
        )
        compatible = [row for row in rows if float(row["alpha"]) > 0.0 and bool(row["joint_pass"])]
        selected = None
        if compatible:
            selected = sorted(
                compatible,
                key=lambda row: (
                    -float(row["A_ranking_accuracy"]),
                    float(row["B_control_answer_nll_increase_vs_zero"]),
                    float(row["alpha"]),
                ),
            )[0]

        result = {
            "schema": "minicells.pcu-hybrid-reattachment-001.alpha-sweep.v3",
            "experiment": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "arm": "amplitude_sweep_no_retraining",
            "phase": "engineering_diagnostic",
            "status": status,
            "valid_run": bool(replay_matches and restoration_exact),
            "scientific_evidence": False,
            "formal_execution_not_started": True,
            "source": state.source,
            "dataset_manifest_sha256": state.world.manifest_sha256(),
            "selected_k": TARGET_K,
            "selected_cells": list(state.baseline.selected_cells),
            "training": state.training,
            "additional_training_after_replay": False,
            "new_bridge_used": False,
            "new_router_used": False,
            "alpha_grid": list(ALPHA_SWEEP),
            "thresholds": {
                "association_floor": ASSOCIATION_FLOOR,
                "maximum_B_control_answer_nll_increase": MAX_CONTROL_ANSWER_NLL_INCREASE,
            },
            "rows": rows,
            "selected_locality_compatible_point": selected,
            "alpha1_replay_matches": replay_matches,
            "restoration_exact": restoration_exact,
            "trained_delta_sha256": state.trained_delta_hash,
        }
        write_json(output / "SWEEP_RESULT.json", result)
        write_json(output / "SWEEP_DECISION.json", {
            "schema": "minicells.pcu-hybrid-reattachment-001.alpha-sweep-decision.v3",
            "experiment": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "status": status,
            "valid_run": result["valid_run"],
            "scientific_evidence": False,
            "formal_execution_not_started": True,
            "formal_decision": "RESERVED_UNRUN",
            "additional_training_after_replay": False,
            "selected_locality_compatible_point": selected,
            "source": state.source,
        })
        fieldnames = [
            "alpha",
            "A_ranking_accuracy",
            "ranking_gain_vs_zero",
            "A_answer_nll",
            "A_answer_nll_gain_vs_zero",
            "A_mean_target_margin",
            "A_margin_gain_vs_zero",
            "B_answer_nll",
            "B_control_answer_nll_increase_vs_zero",
            "association_pass",
            "locality_pass",
            "joint_pass",
        ]
        with (output / "AMPLITUDE_SWEEP.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row[key] for key in fieldnames})
        return result


def _same_source(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("source_commit") == right.get("source_commit")
        and left.get("source_tree") == right.get("source_tree")
        and left.get("source_dirty") is False
        and right.get("source_dirty") is False
    )


def render_visualizations(
    primary: dict[str, Any],
    sweep: dict[str, Any],
    output: Path,
) -> list[str]:
    output = Path(output)
    files: list[str] = []
    diffs = primary["logit_differences"]

    labels = [
        "Native→parent A",
        "Native→parent B",
        "Parent→off A",
        "Parent→off B",
        "On→restored A",
    ]
    values = [
        float(diffs["base_vs_parent_A"]["max_abs"]),
        float(diffs["base_vs_parent_B"]["max_abs"]),
        float(diffs["parent_vs_off_A"]["max_abs"]),
        float(diffs["parent_vs_off_B"]["max_abs"]),
        float(diffs["on_vs_restored_A"]["max_abs"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, [max(value, 1e-12) for value in values])
    ax.axhline(EQUIVALENCE_MAX_ABS_LOGIT_DIFF, linestyle="--", label="1e-5 gate")
    ax.set_yscale("log")
    ax.set_ylabel("Max absolute logit difference")
    ax.set_title("Numerical equivalence: diagnostic vs protocol-v3 gates")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    path = output / "equivalence_diffs.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    files.append(path.name)

    causal = primary["causal_effect"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(
        ["CELL_OFF", "CELL_ON"],
        [float(causal["ranking_off"]), float(causal["ranking_on"])],
    )
    ax.axhline(ASSOCIATION_FLOOR, linestyle="--", label="Association floor 0.80")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("A_eval 16-way ranking accuracy")
    ax.set_title("Causal PCU mutation expression through frozen Granite")
    ax.legend()
    fig.tight_layout()
    path = output / "causal_ranking_on_off.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    files.append(path.name)

    rows = sweep["rows"]
    alphas = [float(row["alpha"]) for row in rows]
    rankings = [float(row["A_ranking_accuracy"]) for row in rows]
    harms = [float(row["B_control_answer_nll_increase_vs_zero"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(alphas, rankings, marker="o", label="A ranking accuracy")
    ax.axhline(ASSOCIATION_FLOOR, linestyle="--", label="Association floor")
    ax.set_xlabel("Mutation amplitude α")
    ax.set_ylabel("A ranking accuracy")
    ax.set_ylim(0.0, 1.0)
    second = ax.twinx()
    second.plot(alphas, harms, marker="s", label="B NLL increase")
    second.axhline(MAX_CONTROL_ANSWER_NLL_INCREASE, linestyle=":", label="Locality ceiling")
    second.set_ylabel("B-control answer NLL increase vs α=0")
    left_lines, left_labels = ax.get_legend_handles_labels()
    right_lines, right_labels = second.get_legend_handles_labels()
    ax.legend(left_lines + right_lines, left_labels + right_labels, loc="best")
    ax.set_title("Amplitude sweep: association–locality trade-off")
    fig.tight_layout()
    path = output / "alpha_sweep_tradeoff.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    files.append(path.name)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(harms, rankings)
    for alpha, x_value, y_value in zip(alphas, harms, rankings):
        ax.annotate(
            f"α={alpha:g}",
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.axvline(MAX_CONTROL_ANSWER_NLL_INCREASE, linestyle="--", label="Locality ceiling")
    ax.axhline(ASSOCIATION_FLOOR, linestyle=":", label="Association floor")
    ax.set_xlabel("B-control answer NLL increase vs α=0")
    ax.set_ylabel("A_eval ranking accuracy")
    ax.set_title("Association–locality Pareto view")
    ax.legend()
    fig.tight_layout()
    path = output / "association_locality_pareto.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    files.append(path.name)
    return files


def aggregate_dual_gpu(
    *,
    primary_root: Path,
    sweep_root: Path,
    output: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    primary_root = Path(primary_root)
    sweep_root = Path(sweep_root)
    output = Path(output)
    primary = json.loads((primary_root / "PRIMARY_RESULT.json").read_text(encoding="utf-8"))
    primary_decision = json.loads(
        (primary_root / "PRIMARY_DECISION.json").read_text(encoding="utf-8")
    )
    sweep = json.loads((sweep_root / "SWEEP_RESULT.json").read_text(encoding="utf-8"))
    sweep_decision = json.loads(
        (sweep_root / "SWEEP_DECISION.json").read_text(encoding="utf-8")
    )
    if sweep_decision.get("status") != sweep.get("status"):
        raise RuntimeError("sweep result/decision status mismatch")
    if not _same_source(primary["source"], sweep["source"]):
        raise RuntimeError("dual-GPU worker source provenance differs")
    if not _same_source(primary["source"], source):
        raise RuntimeError("dual-GPU worker source differs from orchestrator source")
    if primary["dataset_manifest_sha256"] != sweep["dataset_manifest_sha256"]:
        raise RuntimeError("dual-GPU workers used different datasets")
    if primary["selected_cells"] != sweep["selected_cells"]:
        raise RuntimeError("dual-GPU workers used different Cell identities")

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(primary_root / "PRIMARY_RESULT.json", output / "PRIMARY_RESULT.json")
    shutil.copy2(primary_root / "PRIMARY_DECISION.json", output / "PRIMARY_DECISION.json")
    shutil.copy2(sweep_root / "SWEEP_RESULT.json", output / "SWEEP_RESULT.json")
    shutil.copy2(sweep_root / "SWEEP_DECISION.json", output / "SWEEP_DECISION.json")
    shutil.copy2(sweep_root / "AMPLITUDE_SWEEP.csv", output / "AMPLITUDE_SWEEP.csv")

    primary_protocol_pass = bool(
        primary_decision["same_graph_zero_state_equivalence_passes"]
        and primary_decision["reversibility_passes"]
        and primary_decision["association_on_passes"]
        and primary_decision["causal_ranking_gain_passes"]
        and primary_decision["causal_margin_gain_passes"]
    )
    sweep_rescues_locality = (
        sweep["status"] == "AMPLITUDE_SWEEP_FINDS_LOCALITY_COMPATIBLE_POINT"
    )
    if not bool(primary["valid_run"] and sweep["valid_run"]):
        combined_status = "DUAL_GPU_PROTOCOL_INVALID"
    elif not primary_protocol_pass:
        combined_status = str(primary["status"])
    elif primary_decision["control_locality_passes"]:
        combined_status = "HYBRID_REATTACHMENT_SUPPORTED_AT_ALPHA_1"
    elif sweep_rescues_locality:
        combined_status = "HYBRID_REATTACHMENT_SUPPORTED_WITH_BOUNDED_AMPLITUDE"
    else:
        combined_status = "HYBRID_CAUSAL_CONSUMPTION_SUPPORTED_LOCALITY_UNRESOLVED"

    result = {
        "schema": "minicells.pcu-hybrid-reattachment-001.result.v3",
        "experiment": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "phase": "engineering_diagnostic",
        "status": combined_status,
        "valid_run": bool(primary["valid_run"] and sweep["valid_run"]),
        "scientific_evidence": False,
        "formal_execution_not_started": True,
        "source": source,
        "primary_status": primary["status"],
        "sweep_status": sweep["status"],
        "primary_causal_effect": primary["causal_effect"],
        "primary_equivalence": primary["equivalence"],
        "selected_locality_compatible_point": sweep["selected_locality_compatible_point"],
        "alpha_grid": sweep["alpha_grid"],
        "dual_gpu_execution": {
            "cuda:0": "primary_causal_reattachment",
            "cuda:1": "amplitude_sweep",
        },
    }
    decision = {
        "schema": "minicells.pcu-hybrid-reattachment-001.decision.v3",
        "experiment": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "phase": "engineering_diagnostic",
        "status": combined_status,
        "valid_run": result["valid_run"],
        "scientific_evidence": False,
        "formal_execution_not_started": True,
        "formal_decision": "RESERVED_UNRUN",
        "protocol_amendment": {
            "triggered_by_source_commit": FIRST_ENGINEERING_SOURCE_COMMIT,
            "thresholds_changed": False,
            "v2_native_G0_result_reclassified_as_diagnostic": True,
            "v3_zero_state_gate": "PARENT_ZERO_DELTA_vs_CELL_OFF_same_cellular_graph",
        },
        "same_graph_zero_state_equivalence_passes": primary_decision[
            "same_graph_zero_state_equivalence_passes"
        ],
        "native_G0_equivalence_is_diagnostic_only": True,
        "native_G0_max_abs_logit_diff": primary_decision["native_G0_max_abs_logit_diff"],
        "causal_hybrid_consumption_passes": primary_protocol_pass,
        "alpha1_locality_passes": primary_decision["control_locality_passes"],
        "amplitude_sweep_locality_rescue_passes": sweep_rescues_locality,
        "selected_locality_compatible_point": sweep["selected_locality_compatible_point"],
        "dual_gpu_execution_required": True,
        "worker_devices": {"primary": "cuda:0", "sweep": "cuda:1"},
        "new_bridge_used": False,
        "new_router_used": False,
        "additional_training_for_amplitude_sweep": False,
        "source": source,
    }
    write_json(output / "RESULT.json", result)
    write_json(output / "DECISION.json", decision)
    visualizations = render_visualizations(primary, sweep, output)
    result["visualizations"] = visualizations
    write_json(output / "RESULT.json", result)

    report = f"""# PCU-HYBRID-REATTACHMENT-001 — Protocol v3 engineering report

Status: `{combined_status}`

This run keeps all v2 numeric thresholds unchanged. The strict zero-state gate
is applied to `PARENT_ZERO_DELTA ↔ CELL_OFF`; native Granite ↔ cellularized
Granite remains a G0 numerical diagnostic.

## Primary causal arm

- Ranking OFF: {primary['causal_effect']['ranking_off']:.6f}
- Ranking ON: {primary['causal_effect']['ranking_on']:.6f}
- Ranking gain: {primary['causal_effect']['ranking_gain']:.6f}
- Answer margin gain: {primary['causal_effect']['answer_margin_gain']:.6f}
- A answer NLL gain: {primary['causal_effect']['answer_nll_gain']:.6f}
- B-control NLL increase: {primary['causal_effect']['B_control_answer_nll_increase']:.6f}
- Same-graph zero-state max logit diff: {primary['equivalence']['same_graph_max_abs']:.9g}
- Native G0 max logit diff (diagnostic): {primary['equivalence']['native_G0_max_abs']:.9g}

## Amplitude sweep

- Grid: {', '.join(str(value) for value in sweep['alpha_grid'])}
- Status: `{sweep['status']}`
- Selected locality-compatible point: `{sweep['selected_locality_compatible_point']}`
- No additional training, bridge, readout, or router was introduced.

## Visualizations

![Equivalence](equivalence_diffs.png)

![Causal ranking](causal_ranking_on_off.png)

![Amplitude trade-off](alpha_sweep_tradeoff.png)

![Pareto](association_locality_pareto.png)

Formal seeds remain `RESERVED_UNTOUCHED`; this artifact is engineering evidence only.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return result


__all__ = [
    "EXPERIMENT_ID",
    "PROTOCOL_VERSION",
    "DEFAULT_OUTPUT",
    "ALPHA_SWEEP",
    "EQUIVALENCE_MAX_ABS_LOGIT_DIFF",
    "RESTORATION_MAX_ABS_LOGIT_DIFF",
    "MIN_CAUSAL_RANKING_GAIN",
    "MAX_CONTROL_ANSWER_NLL_INCREASE",
    "temporarily_scale_cell_deltas",
    "classify_primary_v3",
    "classify_sweep",
    "run_primary_arm",
    "run_amplitude_sweep_arm",
    "render_visualizations",
    "aggregate_dual_gpu",
]
