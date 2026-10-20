"""Authorization is independent of engineering success; no Confirmation execution mode."""

from .common import canonical, ContinuationError


def require_gpu_smoke(c, authorization):
    if (
        not authorization
        or authorization.get("scope") != "GPU_SMOKE_ONLY"
        or authorization.get("config_sha256") != canonical(c)
        or authorization.get("approved") is not True
    ):
        raise ContinuationError("GPU_NOT_AUTHORIZED_CPU_ENGINEERING_ONLY")
    steps = c["balance"]["pilot_steps_per_candidate"] * len(c["balance"]["candidates"]) + c[
        "pilot"
    ]["steps_per_arm"] * len(c["arms"])
    if (
        steps != c["pilot"]["total_optimizer_updates"]
        or c["pilot"]["training_tokens"]
        != steps * c["training"]["sequence_length"] * c["training"]["effective_batch"]
    ):
        raise ContinuationError("PILOT_DECLARED_BUDGET_MISMATCH")
    if c["balance"]["minimum_cell_share"] is None:
        raise ContinuationError("BALANCE_SELECTION_THRESHOLD_NOT_FROZEN")
    return True


def require_development(c, evidence):
    required = (
        "minimum_practical_delta_nll",
        "noninferiority_margin",
        "minimum_cell_exposure",
        "mechanism_thresholds",
    )
    if (
        not c["release"]["formal_development_authorized"]
        or not c["candidate_training"]["frozen"]
        or c["balance"]["coefficient"] is None
        or any(c["statistics"][k] is None for k in required)
    ):
        raise ContinuationError("FORMAL_DEVELOPMENT_NOT_RELEASED")
    if (
        not evidence
        or evidence.get("config_sha256") != canonical(c)
        or not evidence.get("cross_session_verified")
        or not evidence.get("gpu_pilot_completed")
    ):
        raise ContinuationError("GPU_AND_PERSISTENCE_EVIDENCE_REQUIRED")
    return True


def choose_balance(c, measurements):
    threshold = c["balance"]["minimum_cell_share"]
    if threshold is None:
        raise ContinuationError("BALANCE_SELECTION_THRESHOLD_NOT_FROZEN")
    candidates = c["balance"]["candidates"]
    if set(measurements) != set(candidates):
        raise ValueError("all predeclared candidates required")
    import numpy as np

    for candidate in sorted(candidates):
        row = measurements[candidate]
        shares = np.asarray(row["cell_shares"], float)
        if row["role"] != "calibration" or shares.shape != (
            c["balance"]["measurement_windows"],
            4,
            c["routing"]["cells"],
        ):
            raise ValueError("fixed Calibration windows required")
        if not np.isfinite(shares).all() or not np.allclose(shares.sum(-1), 1):
            raise ValueError("invalid selected-count shares")
        if (shares >= threshold).all():
            return {
                "status": "CANDIDATE_SELECTED_NOT_FORMALLY_RELEASED",
                "coefficient": candidate,
                "uses_development": False,
            }
    return {"status": "NO_GO", "coefficient": None, "uses_development": False}
