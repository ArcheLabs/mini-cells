"""Count historical FP16 loop attempts separately from successful AdamW calls."""

import math


def completed_budget(checkpoint):
    import torch

    if checkpoint.get("step") != 10000 or checkpoint.get("consumed_tokens") != 10000000:
        raise RuntimeError("TRAINING_LOOP_BUDGET_INCOMPLETE")
    actual = checkpoint.get("actual_optimizer_updates")
    if type(actual) is not int or not 0 < actual <= 10000:
        raise RuntimeError("INVALID_ACTUAL_OPTIMIZER_UPDATE_COUNT")
    optimizer = checkpoint.get("optimizer_state_dict", {})
    states = optimizer.get("state", {})
    expected = {p for group in optimizer.get("param_groups", []) for p in group["params"]}
    if not states or set(states) != expected:
        raise RuntimeError("ADAM_STATE_INCOMPLETE")
    for state in states.values():
        value = state.get("step")
        if value is None:
            raise RuntimeError("ADAM_STEP_COUNTER_MISSING")
        number = float(value.item() if hasattr(value, "item") else value)
        if not math.isfinite(number) or number != actual:
            raise RuntimeError("ADAM_STEP_COUNTER_DISAGREES_WITH_OBSERVER")
        for field in ("exp_avg", "exp_avg_sq"):
            value = state.get(field)
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item():
                raise RuntimeError("ADAM_MOMENT_STATE_INVALID")
    scaler = checkpoint.get("scaler_state_dict", {})
    scale = scaler.get("scale")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("SCALER_STATE_INVALID")
    return {
        "training_steps": 10000,
        "successful_optimizer_updates": actual,
        "skipped_optimizer_updates": 10000 - actual,
        "training_loop_steps": 10000,
        "optimizer_step_attempts": 10000,
        "actual_optimizer_updates": actual,
        "scaler_skipped_updates": 10000 - actual,
        "skipped_update_fraction": (10000 - actual) / 10000,
        "training_tokens": 10000000,
        "budget_basis": "HISTORICAL_LOOP_ATTEMPTS_AND_CONSUMED_TOKENS",
        "optimizer_counter_corroboration": "ALL_ADAM_PARAMETER_STEP_COUNTERS_MATCH",
        "skipped_step_locations": "UNKNOWN_NOT_RECORDED_BY_ORIGINAL_OBSERVER",
        "found_inf_evidence": "NOT_LOGGED_IN_EXISTING_CHECKPOINT",
        "amp_skip_observation": "SKIPS_PRESENT_REPORTED" if actual < 10000 else "NO_SKIPS",
        "compensation_updates": 0,
    }
