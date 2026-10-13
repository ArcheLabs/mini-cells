from __future__ import annotations

import native_clm_runtime as base
from native_clm_candidates import VERIFIED_DATASET_REVISION
from native_clm_phase_b import M4_NAME, build_phase_b, estimate_flops as estimate_phase_b_flops, phase_b_config
from native_clm_phase_c import X3_NAME, X4_NAME, build_phase_c, estimate_flops as estimate_phase_c_flops, phase_c_config

base.DATASET_REVISION = VERIFIED_DATASET_REVISION
_BASE_MODEL_CONFIG = base.model_config
_BASE_ESTIMATE_FLOPS = base.estimate_flops

REPLICATION_SEEDS = (91002, 91003)
CONFIRMATION_SEEDS = (91101, 91102, 91103)
REPLICATION_MODELS = (base.T1_NAME, M4_NAME, X3_NAME, X4_NAME)
MODEL_IDS = {
    base.T1_NAME: "T1",
    M4_NAME: "M4",
    X3_NAME: "X3",
    X4_NAME: "X4",
}
PARAMETER_TOLERANCE = 0.01


def build_replication_model(name: str, vocab_size: int = base.VOCAB_SIZE):
    if name == base.T1_NAME:
        return base.ModernTransformerLM(vocab_size)
    if name == M4_NAME:
        return build_phase_b(name, vocab_size)
    if name in (X3_NAME, X4_NAME):
        return build_phase_c(name, vocab_size)
    raise ValueError(name)


def replication_config(name: str, vocab_size: int = base.VOCAB_SIZE) -> dict[str, object]:
    if name == base.T1_NAME:
        return _BASE_MODEL_CONFIG(name, vocab_size)
    if name == M4_NAME:
        return {
            **phase_b_config(name),
            "vocab_size": vocab_size,
            "context_length": base.CONTEXT_LENGTH,
        }
    if name in (X3_NAME, X4_NAME):
        return {
            **phase_c_config(name),
            "vocab_size": vocab_size,
            "context_length": base.CONTEXT_LENGTH,
        }
    raise ValueError(name)


def estimate_replication_flops(
    name: str,
    *,
    sequence_length: int,
    tokens: int,
) -> dict[str, float]:
    if name == base.T1_NAME:
        return _BASE_ESTIMATE_FLOPS(name, sequence_length=sequence_length, tokens=tokens)
    if name == M4_NAME:
        return estimate_phase_b_flops(name, tokens, sequence_length)
    if name in (X3_NAME, X4_NAME):
        return estimate_phase_c_flops(name, tokens, sequence_length)
    raise ValueError(name)


def parameter_summary() -> dict[str, object]:
    t1 = base.count_parameters(build_replication_model(base.T1_NAME))
    result: dict[str, object] = {
        "T1_anchor": t1,
        "tolerance": PARAMETER_TOLERANCE,
        "seeds": list(REPLICATION_SEEDS),
    }
    for name in REPLICATION_MODELS:
        params = base.count_parameters(build_replication_model(name))
        result[name] = {
            "id": MODEL_IDS[name],
            "parameters": params,
            "over_t1": params / t1,
            "relative_error": abs(params / t1 - 1.0),
        }
    return result


def validate_replication_request(seeds: tuple[int, ...] | list[int]) -> None:
    normalized = tuple(int(seed) for seed in seeds)
    if normalized != REPLICATION_SEEDS:
        raise RuntimeError(
            f"replication seeds are frozen to {REPLICATION_SEEDS}; got {normalized}"
        )
    if any(seed in CONFIRMATION_SEEDS for seed in normalized):
        raise RuntimeError("confirmation seeds are reserved and must remain untouched")
