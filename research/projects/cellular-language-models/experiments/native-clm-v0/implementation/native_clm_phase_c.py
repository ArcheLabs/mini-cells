from __future__ import annotations

import numpy as np

import native_clm_runtime as base
from native_clm_candidates import C1_DIM, C1_HEADS, VERIFIED_DATASET_REVISION
from native_clm_phase_b import N1_ALPHA, PhaseBLM

base.DATASET_REVISION = VERIFIED_DATASET_REVISION

X1_NAME = "X1-n1-swiglu"
X2_NAME = "X2-n1-rope"
X3_NAME = "X3-n1-swiglu-rope"
X4_NAME = "X4-n1-swiglu-rope-rmsnorm"
PHASE_C_NAMES = (X1_NAME, X2_NAME, X3_NAME, X4_NAME)
PARAMETER_TOLERANCE = 0.01

# Spend capacity released by removing C1's learned gate on useful FFN width.
X1_SWIGLU_FFN = 624
X2_GELU_FFN = 954
X3_SWIGLU_FFN = 638
X4_SWIGLU_FFN = 639


def build_phase_c(name: str, vocab_size: int = base.VOCAB_SIZE):
    common = dict(dim=C1_DIM, heads=C1_HEADS, vocab_size=vocab_size, update="residual-only")
    if name == X1_NAME:
        return PhaseBLM(**common, ffn_dim=X1_SWIGLU_FFN, activation="swiglu", use_rope=False, norm_kind="layernorm")
    if name == X2_NAME:
        return PhaseBLM(**common, ffn_dim=X2_GELU_FFN, activation="gelu", use_rope=True, norm_kind="layernorm")
    if name == X3_NAME:
        return PhaseBLM(**common, ffn_dim=X3_SWIGLU_FFN, activation="swiglu", use_rope=True, norm_kind="layernorm")
    if name == X4_NAME:
        return PhaseBLM(**common, ffn_dim=X4_SWIGLU_FFN, activation="swiglu", use_rope=True, norm_kind="rmsnorm")
    raise ValueError(name)


def phase_c_config(name: str) -> dict[str, object]:
    common: dict[str, object] = {
        "name": name,
        "search_phase": "C-winner-cross-over",
        "parent_native": "N1-residual-only",
        "parent_modernization": "M-series",
        "update": "fixed-scaled-residual",
        "residual_alpha": N1_ALPHA,
        "windows": list(base.C0_WINDOWS),
        "iterations": [4, 4, 4],
    }
    if name == X1_NAME:
        return {**common, "factor": "N1+SwiGLU", "normalization": "LayerNorm", "activation": "SwiGLU", "ffn_dim": X1_SWIGLU_FFN, "position": "learned-absolute"}
    if name == X2_NAME:
        return {**common, "factor": "N1+RoPE", "normalization": "LayerNorm", "activation": "GELU", "ffn_dim": X2_GELU_FFN, "position": "RoPE", "learned_position_embedding": False}
    if name == X3_NAME:
        return {**common, "factor": "N1+SwiGLU+RoPE", "normalization": "LayerNorm", "activation": "SwiGLU", "ffn_dim": X3_SWIGLU_FFN, "position": "RoPE", "learned_position_embedding": False}
    if name == X4_NAME:
        return {**common, "factor": "N1+SwiGLU+RoPE+RMSNorm", "normalization": "RMSNorm", "activation": "SwiGLU", "ffn_dim": X4_SWIGLU_FFN, "position": "RoPE", "learned_position_embedding": False}
    raise ValueError(name)


def parameter_summary() -> dict[str, object]:
    t1 = base.count_parameters(base.build_model(base.T1_NAME))
    result: dict[str, object] = {"T1_anchor": t1, "tolerance": PARAMETER_TOLERANCE}
    for name in PHASE_C_NAMES:
        params = base.count_parameters(build_phase_c(name))
        result[name] = {"parameters": params, "over_t1": params / t1, "relative_error": abs(params / t1 - 1.0)}
    return result


def _attn_cost(tokens: int, dim: int, window: int) -> int:
    return 8 * tokens * dim * dim + 4 * dim * base._local_pairs(tokens, window)


def estimate_forward_flops(name: str, sequence_length: int = base.TRAIN_SEQUENCE_LENGTH) -> int:
    t = sequence_length
    d = C1_DIM
    if name == X1_NAME:
        ffn_cost = 6 * t * d * X1_SWIGLU_FFN
    elif name == X2_NAME:
        ffn_cost = 4 * t * d * X2_GELU_FFN
    elif name == X3_NAME:
        ffn_cost = 6 * t * d * X3_SWIGLU_FFN
    elif name == X4_NAME:
        ffn_cost = 6 * t * d * X4_SWIGLU_FFN
    else:
        raise ValueError(name)
    total = 0
    for window in base.C0_WINDOWS:
        total += 4 * (_attn_cost(t, d, window) + ffn_cost)
    return int(total + 2 * t * d * base.VOCAB_SIZE)


def estimate_flops(name: str, tokens: int, sequence_length: int = base.TRAIN_SEQUENCE_LENGTH) -> dict[str, float]:
    forward_per_token = estimate_forward_flops(name, sequence_length) / sequence_length
    return {
        "inference_forward_flops_per_token": forward_per_token,
        "train_flops_estimate": forward_per_token * tokens * 3.0,
        "convention": "matmul-heavy analytic estimate; multiply-add=2 FLOPs; training≈3x forward",
    }


def validation_log_token_auc(rows: list[dict[str, object]]) -> dict[str, float]:
    if len(rows) < 2:
        return {"auc": float("nan"), "mean_nll": float("nan")}
    x = np.log(np.array([float(row["consumed_tokens"]) for row in rows]))
    y = np.array([float(row["validation_nll"]) for row in rows])
    auc = float(np.trapezoid(y, x))
    width = float(x[-1] - x[0])
    return {"auc": auc, "mean_nll": auc / width}
