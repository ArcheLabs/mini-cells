"""Shared constants and decision helpers for the CLM-0.3 public release benchmark."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .clm_growth import ProgressiveGrowthCLM
from .language_scaling import build_minicells_v2


RELEASE_BENCHMARK_FORMAT = "minicells.clm-0.3-release-benchmark.v1"
BRIDGE_WORKER_FORMAT = "minicells.clm-0.3-release-bridge-worker.v1"
BRIDGE_SUMMARY_FORMAT = "minicells.clm-0.3-release-bridge-summary.v1"

SOURCE_006_CHECKPOINT = Path(
    "artifacts/experiments/006-consumer-language-scaling/minicells-v2-10m.pt"
)
SOURCE_006_CHECKPOINT_SHA256 = (
    "b76e2dd28b31470c1ce8bcd265c56e1b306191631e304161ded55b4e763f9e9e"
)
SOURCE_006_DECISION = Path("artifacts/experiments/006-consumer-language-scaling/decision.json")
SOURCE_007_DECISION = Path("artifacts/experiments/007-minicells-30m/decision.json")

CAPABILITY_RESULTS_REF = "kaggle/clm-0.3d-probationary-mitosis-results"
CAPABILITY_ARTIFACT_ROOT = "artifacts/experiments/clm-0.3d-probationary-mitosis"
CAPABILITY_DECISION_FORMAT = "minicells.clm-0.3d-probationary-mitosis.decision.v1"
CAPABILITY_EXPECTED_TRAINING_COMMIT = "af1eed85ac674495b684c22db49e839cf433bbe0"
CAPABILITY_EXPECTED_TRAINING_TREE = "dbe4c7ff609105cdeb2083f0269de0af17289cdb"

BRIDGE_ARMS = ("textnca_continuation", "clm_fixed4")
BRIDGE_TRAIN_PREFIX_OFFSET = 15_000_000
BRIDGE_MATERIALIZED_TRAIN_TOKENS = 18_000_000
BRIDGE_BUDGET_TOKENS = 1_000_000
BRIDGE_CHECKPOINT_TOKENS = (0, 100_000, 250_000, 500_000, 1_000_000)
BRIDGE_BATCH_SIZE = 8
BRIDGE_SEQUENCE_LENGTH = 125
BRIDGE_TOKENS_PER_STEP = BRIDGE_BATCH_SIZE * BRIDGE_SEQUENCE_LENGTH
BRIDGE_SCHEDULE_SEED = 57003
BRIDGE_MODEL_SEED = 57005
BRIDGE_VALIDATION_SEED = 57007
BRIDGE_VALIDATION_BATCHES = 48
BRIDGE_VALIDATION_BATCH_SIZE = 8
BRIDGE_VALIDATION_SEQUENCE_LENGTH = 128
BRIDGE_KL_BETA = 0.5
BRIDGE_BASE_LR = 3e-4
BRIDGE_WEIGHT_DECAY = 0.1
BRIDGE_WARMUP_STEPS = 100
BRIDGE_GRAD_CLIP = 1.0
BRIDGE_BETAS = (0.9, 0.95)
BRIDGE_STATE_INTERVAL_TOKENS = 250_000

QUALITY_COMPETITIVE_RATIO = 1.03
QUALITY_MODEST_RATIO = 1.05
REFERENCE_RUNTIME_MIN_THROUGHPUT_RATIO = 0.50
REFERENCE_RUNTIME_MAX_VRAM_RATIO = 2.50
AGE_ZERO_PPL_RATIO_TOLERANCE = 1e-5
AGE_ZERO_MAX_LOGITS_DIFF = 2e-5


@dataclass(frozen=True)
class BridgeRuntime:
    train_tokens_per_second: float
    train_peak_vram_bytes: int
    inference_tokens_per_second: float
    inference_peak_vram_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "train_tokens_per_second": self.train_tokens_per_second,
            "train_peak_vram_bytes": self.train_peak_vram_bytes,
            "inference_tokens_per_second": self.inference_tokens_per_second,
            "inference_peak_vram_bytes": self.inference_peak_vram_bytes,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_source_checkpoint(path: str | Path) -> str:
    observed = sha256_file(path)
    if observed != SOURCE_006_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Experiment-006 source checkpoint hash mismatch: "
            f"expected {SOURCE_006_CHECKPOINT_SHA256}, observed {observed}"
        )
    return observed


def load_source_textnca(
    checkpoint_path: str | Path,
    *,
    vocab_size: int,
    device: str | torch.device = "cpu",
) -> nn.Module:
    verify_source_checkpoint(checkpoint_path)
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if payload.get("format") != "minicells.language-checkpoint.v1":
        raise RuntimeError(f"unexpected source checkpoint format: {payload.get('format')!r}")
    if payload.get("model_name") != "minicells-v2":
        raise RuntimeError(f"unexpected source model: {payload.get('model_name')!r}")
    if int(payload.get("consumed_tokens", -1)) != 10_000_000:
        raise RuntimeError("release bridge requires the 10M Experiment-006 checkpoint")
    model = build_minicells_v2(vocab_size)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device)


def build_bridge_model(
    arm: str,
    checkpoint_path: str | Path,
    *,
    vocab_size: int,
    device: str | torch.device = "cpu",
) -> nn.Module:
    if arm not in BRIDGE_ARMS:
        raise ValueError(f"unknown bridge arm: {arm!r}")
    torch.manual_seed(BRIDGE_MODEL_SEED)
    source = load_source_textnca(checkpoint_path, vocab_size=vocab_size, device="cpu")
    if arm == "textnca_continuation":
        return source.to(device)
    return ProgressiveGrowthCLM(source).to(device)


def bridge_lr(step: int, total_steps: int) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step <= BRIDGE_WARMUP_STEPS:
        multiplier = step / max(BRIDGE_WARMUP_STEPS, 1)
    else:
        progress = (step - BRIDGE_WARMUP_STEPS) / max(
            total_steps - BRIDGE_WARMUP_STEPS, 1
        )
        multiplier = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return BRIDGE_BASE_LR * multiplier


def clm_parameter_breakdown(model: ProgressiveGrowthCLM) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    expert_total = 0
    active_expert = 0
    router_total = 0
    for stage in model.stages:
        bank = stage.program_bank
        ids = bank.expert_ids
        if not ids:
            raise RuntimeError("CLM stage contains no experts")
        expert_total += sum(
            parameter.numel()
            for expert_id in ids
            for parameter in bank.experts[expert_id].parameters()
        )
        active_expert += sum(
            parameter.numel() for parameter in bank.experts[ids[0]].parameters()
        )
        router_total += sum(parameter.numel() for parameter in bank.router.parameters())
    shared = total - expert_total - router_total
    if min(total, expert_total, active_expert, router_total, shared) < 0:
        raise RuntimeError("invalid CLM parameter accounting")
    active_proxy = shared + active_expert + router_total
    return {
        "total_parameters": int(total),
        "expert_parameters_total": int(expert_total),
        "active_expert_parameters": int(active_expert),
        "router_parameters": int(router_total),
        "shared_parameters": int(shared),
        "active_parameter_proxy": int(active_proxy),
    }


def dense_parameter_breakdown(model: nn.Module) -> dict[str, int]:
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    return {
        "total_parameters": total,
        "expert_parameters_total": 0,
        "active_expert_parameters": 0,
        "router_parameters": 0,
        "shared_parameters": total,
        "active_parameter_proxy": total,
    }


def quality_status(ppl_ratio: float) -> str:
    if not math.isfinite(ppl_ratio) or ppl_ratio <= 0:
        raise ValueError("ppl_ratio must be finite and positive")
    if ppl_ratio <= QUALITY_COMPETITIVE_RATIO:
        return "CLM_RELEASE_LM_QUALITY_COMPETITIVE"
    if ppl_ratio <= QUALITY_MODEST_RATIO:
        return "CLM_RELEASE_LM_QUALITY_MODEST_OVERHEAD"
    return "CLM_RELEASE_LM_QUALITY_HOLD"


def runtime_status(
    *,
    clm_inference_tokens_per_second: float,
    textnca_inference_tokens_per_second: float,
    clm_inference_vram_bytes: int,
    textnca_inference_vram_bytes: int,
) -> tuple[str, dict[str, float]]:
    if min(clm_inference_tokens_per_second, textnca_inference_tokens_per_second) <= 0:
        raise ValueError("inference throughput must be positive")
    if min(clm_inference_vram_bytes, textnca_inference_vram_bytes) <= 0:
        raise ValueError("inference VRAM must be positive")
    throughput_ratio = clm_inference_tokens_per_second / textnca_inference_tokens_per_second
    vram_ratio = clm_inference_vram_bytes / textnca_inference_vram_bytes
    status = (
        "CLM_RELEASE_REFERENCE_RUNTIME_ACCEPTABLE"
        if throughput_ratio >= REFERENCE_RUNTIME_MIN_THROUGHPUT_RATIO
        and vram_ratio <= REFERENCE_RUNTIME_MAX_VRAM_RATIO
        else "CLM_RELEASE_REFERENCE_RUNTIME_OPTIMIZATION_REQUIRED"
    )
    return status, {
        "inference_throughput_ratio_clm_over_textnca": throughput_ratio,
        "inference_time_per_token_ratio_clm_over_textnca": 1.0 / throughput_ratio,
        "inference_vram_ratio_clm_over_textnca": vram_ratio,
    }


def validate_historical_evidence(
    experiment_006: dict[str, Any], experiment_007: dict[str, Any]
) -> dict[str, object]:
    if experiment_006.get("format") != "minicells.consumer-language-scaling.v1":
        raise RuntimeError("unexpected Experiment-006 decision format")
    if experiment_007.get("format") != "minicells.language-30m.v1":
        raise RuntimeError("unexpected Experiment-007 decision format")
    if experiment_006.get("status") != "GREEN" or experiment_007.get("status") != "GREEN":
        raise RuntimeError("historical TextNCA language evidence is not GREEN")
    ratio_006 = float(experiment_006["comparison"]["ppl_ratio_10m"])
    ratio_007 = float(experiment_007["comparison"]["ppl_ratio_100m"])
    return {
        "status": "CLM_RELEASE_TEXTNCA_LANGUAGE_FOUNDATION_CONFIRMED",
        "experiment_006": {
            "parameters_textnca": int(experiment_006["parameter_matching"]["minicells_parameters"]),
            "parameters_transformer": int(experiment_006["parameter_matching"]["transformer_parameters"]),
            "training_tokens": 10_000_000,
            "textnca_ppl": float(experiment_006["candidate"]["ppl_10m"]),
            "transformer_ppl": float(experiment_006["transformer"]["ppl_10m"]),
            "ppl_ratio_textnca_over_transformer": ratio_006,
        },
        "experiment_007": {
            "parameters_textnca": int(experiment_007["parameter_matching"]["minicells_parameters"]),
            "parameters_transformer": int(experiment_007["parameter_matching"]["transformer_parameters"]),
            "training_tokens": 100_000_000,
            "textnca_ppl": float(experiment_007["candidate"]["ppl_100m"]),
            "transformer_ppl": float(experiment_007["transformer"]["ppl_100m"]),
            "ppl_ratio_textnca_over_transformer": ratio_007,
        },
    }


def normalize_capability_evidence(
    decision: dict[str, Any],
    replicate_summary: Iterable[dict[str, Any]],
    *,
    source_ref: str,
    source_commit: str,
) -> dict[str, object]:
    if decision.get("format") != CAPABILITY_DECISION_FORMAT:
        raise RuntimeError("unexpected CLM-0.3d capability decision format")
    if decision.get("formal_gpu_experiment_run") is not True:
        raise RuntimeError("CLM-0.3d capability evidence is not a formal GPU run")
    if decision.get("training_code_commit") != CAPABILITY_EXPECTED_TRAINING_COMMIT:
        raise RuntimeError("CLM-0.3d training commit does not match the release preregistration")
    if decision.get("training_code_tree_sha") != CAPABILITY_EXPECTED_TRAINING_TREE:
        raise RuntimeError("CLM-0.3d training tree does not match the release preregistration")
    if decision.get("overall", {}).get("status") != "CLM_PROBATIONARY_MITOSIS_SIGNAL":
        raise RuntimeError("CLM-0.3d did not establish the required probationary-mitosis signal")
    growth = decision.get("growth_equivalence", {})
    if int(growth.get("births_checked", -1)) != 72 or int(growth.get("births_equivalent", -1)) != 72:
        raise RuntimeError("CLM-0.3d growth-equivalence matrix is incomplete")

    rows = list(replicate_summary)
    if len(rows) != 3:
        raise RuntimeError("CLM-0.3d release evidence requires three replicates")
    stationary_rejected = 0
    shift_promoted = 0
    gains: list[dict[str, object]] = []
    for row in rows:
        conditions = row.get("conditions", {})
        stationary = conditions.get("stationary_story", {})
        shift = conditions.get("story_arithmetic_shift", {})
        stationary_rejected += int(stationary.get("action") == "REJECT")
        promoted = shift.get("action") == "PROMOTE" and shift.get("independent_confirmed") is True
        shift_promoted += int(promoted)
        if promoted:
            ratio = float(shift["final_ppl_ratio"])
            gains.append(
                {
                    "replicate": int(row["replicate"]),
                    "selected_expert": shift.get("selected_expert"),
                    "ppl_ratio": ratio,
                    "ppl_improvement_percent": 100.0 * (1.0 - ratio),
                }
            )
    if stationary_rejected < 2 or shift_promoted < 2:
        raise RuntimeError("CLM-0.3d replicate summary no longer satisfies the release capability gate")
    return {
        "status": "CLM_RELEASE_DEVELOPMENTAL_CAPABILITY_CONFIRMED",
        "source_ref": source_ref,
        "source_commit": source_commit,
        "training_code_commit": decision["training_code_commit"],
        "training_code_tree_sha": decision["training_code_tree_sha"],
        "births_equivalent": 72,
        "births_checked": 72,
        "stationary_rejected": stationary_rejected,
        "stationary_total": 3,
        "shift_promoted": shift_promoted,
        "shift_total": 3,
        "promoted_replicates": gains,
    }


def make_release_decision(
    *,
    historical: dict[str, object],
    bridge: dict[str, Any],
    capability: dict[str, object],
) -> dict[str, object]:
    arms = bridge.get("arms", {})
    dense = arms.get("textnca_continuation")
    clm = arms.get("clm_fixed4")
    if dense is None or clm is None:
        raise RuntimeError("release bridge is missing one or more arms")
    if bridge.get("age_zero_equivalence", {}).get("status") != "CLM_RELEASE_BRIDGE_EQUIVALENCE":
        raise RuntimeError("release bridge did not preserve the source function at age zero")

    final_ratio = float(clm["final_ppl"]) / float(dense["final_ppl"])
    lm_status = quality_status(final_ratio)
    runtime_gate, runtime_ratios = runtime_status(
        clm_inference_tokens_per_second=float(clm["runtime"]["inference_tokens_per_second"]),
        textnca_inference_tokens_per_second=float(
            dense["runtime"]["inference_tokens_per_second"]
        ),
        clm_inference_vram_bytes=int(clm["runtime"]["inference_peak_vram_bytes"]),
        textnca_inference_vram_bytes=int(dense["runtime"]["inference_peak_vram_bytes"]),
    )
    train_time_ratio = float(dense["runtime"]["train_tokens_per_second"]) / float(
        clm["runtime"]["train_tokens_per_second"]
    )
    clm_params = clm["parameters"]
    dense_params = dense["parameters"]
    structural = {
        "active_parameter_proxy_ratio_clm_over_textnca": float(
            clm_params["active_parameter_proxy"]
        )
        / float(dense_params["active_parameter_proxy"]),
        "stored_parameter_ratio_clm_over_textnca": float(clm_params["total_parameters"])
        / float(dense_params["total_parameters"]),
        "train_time_per_token_ratio_clm_over_textnca": train_time_ratio,
    }

    capability_valid = capability.get("status") == "CLM_RELEASE_DEVELOPMENTAL_CAPABILITY_CONFIRMED"
    if not capability_valid or lm_status == "CLM_RELEASE_LM_QUALITY_HOLD":
        overall = "CLM_0_3_PUBLIC_RELEASE_HOLD"
    elif lm_status == "CLM_RELEASE_LM_QUALITY_COMPETITIVE" and runtime_gate == "CLM_RELEASE_REFERENCE_RUNTIME_ACCEPTABLE":
        overall = "CLM_0_3_PUBLIC_RELEASE_READY"
    else:
        overall = "CLM_0_3_PUBLIC_RESEARCH_RELEASE_READY"

    return {
        "format": RELEASE_BENCHMARK_FORMAT,
        "overall": {"status": overall},
        "language_foundation": historical,
        "language_quality": {
            "status": lm_status,
            "final_ppl_ratio_clm_over_textnca": final_ratio,
            "competitive_threshold": QUALITY_COMPETITIVE_RATIO,
            "modest_overhead_threshold": QUALITY_MODEST_RATIO,
        },
        "reference_runtime": {
            "status": runtime_gate,
            **runtime_ratios,
            **structural,
            "minimum_inference_throughput_ratio": REFERENCE_RUNTIME_MIN_THROUGHPUT_RATIO,
            "maximum_inference_vram_ratio": REFERENCE_RUNTIME_MAX_VRAM_RATIO,
        },
        "developmental_capability": capability,
        "bridge": {
            "training_commit": bridge.get("training_commit"),
            "training_tree_sha": bridge.get("training_tree_sha"),
            "source_checkpoint_sha256": bridge.get("source_checkpoint_sha256"),
            "final_textnca_ppl": float(dense["final_ppl"]),
            "final_clm_ppl": float(clm["final_ppl"]),
        },
    }
