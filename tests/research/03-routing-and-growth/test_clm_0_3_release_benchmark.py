from __future__ import annotations

import math

from minicells.clm_growth import ProgressiveGrowthCLM
from minicells.clm_release_benchmark import (
    BRIDGE_BUDGET_TOKENS,
    BRIDGE_WARMUP_STEPS,
    bridge_lr,
    clm_parameter_breakdown,
    dense_parameter_breakdown,
    make_release_decision,
    normalize_capability_evidence,
    quality_status,
    runtime_status,
    validate_historical_evidence,
)
from minicells.language_models import TextNCALM


def _historical() -> tuple[dict, dict]:
    exp006 = {
        "format": "minicells.consumer-language-scaling.v1",
        "status": "GREEN",
        "candidate": {"ppl_10m": 19.44867213327214},
        "transformer": {"ppl_10m": 18.963440556993035},
        "comparison": {"ppl_ratio_10m": 1.0255877394621922},
        "parameter_matching": {"minicells_parameters": 1170816, "transformer_parameters": 1183936},
    }
    exp007 = {
        "format": "minicells.language-30m.v1",
        "status": "GREEN",
        "candidate": {"ppl_100m": 5.3532357975464855},
        "transformer": {"ppl_100m": 5.328995153085013},
        "comparison": {"ppl_ratio_100m": 1.00454882088745},
        "parameter_matching": {"minicells_parameters": 29602800, "transformer_parameters": 29458432},
    }
    return exp006, exp007


def _capability() -> tuple[dict, list[dict]]:
    decision = {
        "format": "minicells.clm-0.3d-probationary-mitosis.decision.v1",
        "formal_gpu_experiment_run": True,
        "training_code_commit": "af1eed85ac674495b684c22db49e839cf433bbe0",
        "training_code_tree_sha": "dbe4c7ff609105cdeb2083f0269de0af17289cdb",
        "overall": {"status": "CLM_PROBATIONARY_MITOSIS_SIGNAL"},
        "growth_equivalence": {"births_checked": 72, "births_equivalent": 72},
    }
    rows = [
        {
            "replicate": 0,
            "conditions": {
                "stationary_story": {"action": "REJECT"},
                "story_arithmetic_shift": {
                    "action": "PROMOTE",
                    "independent_confirmed": True,
                    "final_ppl_ratio": 0.986,
                    "selected_expert": "s1-e3",
                },
            },
        },
        {
            "replicate": 1,
            "conditions": {
                "stationary_story": {"action": "REJECT"},
                "story_arithmetic_shift": {
                    "action": "PROMOTE",
                    "independent_confirmed": True,
                    "final_ppl_ratio": 0.991,
                    "selected_expert": "s1-e0",
                },
            },
        },
        {
            "replicate": 2,
            "conditions": {
                "stationary_story": {"action": "REJECT"},
                "story_arithmetic_shift": {"action": "REJECT", "independent_confirmed": False},
            },
        },
    ]
    return decision, rows


def _bridge(*, clm_ppl: float = 10.2, dense_ppl: float = 10.0, clm_tps: float = 6000.0) -> dict:
    dense_parameters = {
        "total_parameters": 1_000_000,
        "active_parameter_proxy": 1_000_000,
    }
    clm_parameters = {
        "total_parameters": 2_000_000,
        "active_parameter_proxy": 1_010_000,
    }
    return {
        "training_commit": "abc",
        "training_tree_sha": "def",
        "source_checkpoint_sha256": "source",
        "age_zero_equivalence": {"status": "CLM_RELEASE_BRIDGE_EQUIVALENCE"},
        "arms": {
            "textnca_continuation": {
                "final_ppl": dense_ppl,
                "parameters": dense_parameters,
                "runtime": {
                    "train_tokens_per_second": 10000.0,
                    "inference_tokens_per_second": 10000.0,
                    "inference_peak_vram_bytes": 1000,
                },
            },
            "clm_fixed4": {
                "final_ppl": clm_ppl,
                "parameters": clm_parameters,
                "runtime": {
                    "train_tokens_per_second": 5000.0,
                    "inference_tokens_per_second": clm_tps,
                    "inference_peak_vram_bytes": 1800,
                },
            },
        },
    }


def test_historical_language_evidence_is_normalized_without_composing_ratios() -> None:
    exp006, exp007 = _historical()
    result = validate_historical_evidence(exp006, exp007)
    assert result["status"] == "CLM_RELEASE_TEXTNCA_LANGUAGE_FOUNDATION_CONFIRMED"
    assert math.isclose(result["experiment_006"]["ppl_ratio_textnca_over_transformer"], 1.0255877394621922)
    assert math.isclose(result["experiment_007"]["ppl_ratio_textnca_over_transformer"], 1.00454882088745)


def test_capability_release_gate_requires_selective_formal_result() -> None:
    decision, rows = _capability()
    result = normalize_capability_evidence(
        decision,
        rows,
        source_ref="kaggle/clm-0.3d-probationary-mitosis-results",
        source_commit="results-commit",
    )
    assert result["stationary_rejected"] == 3
    assert result["shift_promoted"] == 2
    assert len(result["promoted_replicates"]) == 2
    assert result["status"] == "CLM_RELEASE_DEVELOPMENTAL_CAPABILITY_CONFIRMED"


def test_release_quality_thresholds_are_preregistered() -> None:
    assert quality_status(1.0299) == "CLM_RELEASE_LM_QUALITY_COMPETITIVE"
    assert quality_status(1.04) == "CLM_RELEASE_LM_QUALITY_MODEST_OVERHEAD"
    assert quality_status(1.051) == "CLM_RELEASE_LM_QUALITY_HOLD"


def test_reference_runtime_gate_uses_measured_throughput_and_vram() -> None:
    status, ratios = runtime_status(
        clm_inference_tokens_per_second=6000,
        textnca_inference_tokens_per_second=10000,
        clm_inference_vram_bytes=1800,
        textnca_inference_vram_bytes=1000,
    )
    assert status == "CLM_RELEASE_REFERENCE_RUNTIME_ACCEPTABLE"
    assert math.isclose(ratios["inference_time_per_token_ratio_clm_over_textnca"], 1 / 0.6)
    status, _ = runtime_status(
        clm_inference_tokens_per_second=4000,
        textnca_inference_tokens_per_second=10000,
        clm_inference_vram_bytes=1800,
        textnca_inference_vram_bytes=1000,
    )
    assert status == "CLM_RELEASE_REFERENCE_RUNTIME_OPTIMIZATION_REQUIRED"


def test_release_decision_separates_science_from_reference_runtime() -> None:
    exp006, exp007 = _historical()
    historical = validate_historical_evidence(exp006, exp007)
    cap_decision, cap_rows = _capability()
    capability = normalize_capability_evidence(
        cap_decision, cap_rows, source_ref="results", source_commit="commit"
    )
    ready = make_release_decision(
        historical=historical,
        bridge=_bridge(clm_ppl=10.2, clm_tps=6000),
        capability=capability,
    )
    assert ready["overall"]["status"] == "CLM_0_3_PUBLIC_RELEASE_READY"

    research = make_release_decision(
        historical=historical,
        bridge=_bridge(clm_ppl=10.2, clm_tps=3000),
        capability=capability,
    )
    assert research["overall"]["status"] == "CLM_0_3_PUBLIC_RESEARCH_RELEASE_READY"

    hold = make_release_decision(
        historical=historical,
        bridge=_bridge(clm_ppl=10.6, clm_tps=6000),
        capability=capability,
    )
    assert hold["overall"]["status"] == "CLM_0_3_PUBLIC_RELEASE_HOLD"


def test_clm_active_parameter_proxy_counts_one_expert_per_stage() -> None:
    dense = TextNCALM(
        vocab_size=31,
        max_context=8,
        dim=8,
        heads=2,
        ffn_dim=12,
        windows=(2, 3, 4),
        iterations=(1, 1, 1),
        carry_bias=2.0,
    )
    clm = ProgressiveGrowthCLM(dense)
    dense_counts = dense_parameter_breakdown(dense)
    clm_counts = clm_parameter_breakdown(clm)
    assert clm_counts["total_parameters"] > dense_counts["total_parameters"]
    assert clm_counts["active_parameter_proxy"] < clm_counts["total_parameters"]
    assert clm_counts["active_expert_parameters"] > 0
    assert clm_counts["router_parameters"] > 0


def test_bridge_learning_rate_warms_up_and_decays() -> None:
    total_steps = BRIDGE_BUDGET_TOKENS // 1000
    assert bridge_lr(1, total_steps) < bridge_lr(BRIDGE_WARMUP_STEPS, total_steps)
    assert math.isclose(bridge_lr(BRIDGE_WARMUP_STEPS, total_steps), 3e-4)
    assert bridge_lr(total_steps, total_steps) <= 1e-12
