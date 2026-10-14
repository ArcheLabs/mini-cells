"""Versioned engineering settings; scientific model/objective/budget are untouched."""

VERSION = "pd001-light-v1"
DEFAULTS = {
    "training_profile": "light",
    "console_interval": 100,
    "diagnostics_interval": 100,
    "train_probe_interval": 100,
    "checkpoint_interval": 100,
    "record_full_flops_each_step": False,
    "full_path_capture_each_step": False,
    "benchmark_warmup_steps": 5,
    "benchmark_measure_steps": 20,
    "max_session_seconds": None,
    "pause_after_step": None,
}


def settings(c):
    result = {**DEFAULTS, **c.get("performance", {})}
    if result["training_profile"] not in ("light", "instrumented"):
        raise ValueError("unknown training performance profile")
    for key in ("console_interval", "diagnostics_interval", "train_probe_interval", "checkpoint_interval"):
        if not isinstance(result[key], int) or result[key] < 1:
            raise ValueError("positive performance interval required: " + key)
    for key in ("max_session_seconds", "pause_after_step"):
        if result[key] is not None and result[key] <= 0:
            raise ValueError("optional pause limit must be positive")
    if result["training_profile"] == "light" and (
        result["record_full_flops_each_step"] or result["full_path_capture_each_step"]
    ):
        raise ValueError("resident instrumentation requires instrumented profile")
    if result["training_profile"] == "instrumented" and c["training"]["steps"] > 100:
        raise RuntimeError(
            "instrumented training is restricted to at most 100 synthetic/approved diagnostic steps"
        )
    if result["training_profile"] == "instrumented":
        result["record_full_flops_each_step"] = True
        result["full_path_capture_each_step"] = True
    return result


def retrain_review(c, source):
    """A performance repair is not permission to repeat the 5000-step experiment."""
    review = c.get("performance_review", {})
    if (
        review.get("user_approved") is not True
        or review.get("source_sha") != source
        or not all(
            review.get(k) is True
            for k in ("correctness_pass", "same_gpu_benchmark_pass", "kaggle_persistence_pass")
        )
    ):
        raise RuntimeError(
            "PD-001 performance review pending: run the short performance notebook; full retraining is blocked"
        )
    from exp006.reuse import canonical_sha256

    expected = canonical_sha256({k: v for k, v in c.items() if k not in ("performance_review", "execution")})
    if review.get("config_sha256") != expected:
        raise RuntimeError("performance review config identity mismatch")
    import re

    for key in ("benchmark_sha256", "persistence_sha256", "approval_sha256"):
        if not re.fullmatch("[0-9a-f]{64}", review.get(key, "")):
            raise RuntimeError("performance review evidence missing: " + key)
