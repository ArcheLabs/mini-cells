"""Research identity is new; the unchanged R3 engine uses its original arm names."""

from copy import deepcopy
import os
from pathlib import Path
from minicells_runtime.store import ContinuationError, canonical, digest, read
from clm001_r3 import common as original

BASE = Path(__file__).resolve().parents[2]
EXPERIMENT = "CLM-001-R3b"
ARMS = ("Q-R3b", "QC-R3b", "QS-R3b")
ENGINE = dict(zip(ARMS, original.ARMS))
LIGHT = list(range(0, 501, 50))
HEAVY = [0, 100, 200, 300, 400, 450, 500]


def load(path=None):
    value = read(path or BASE / "config/proposed.json")
    validate(value)
    return value


def validate(value):
    if value.get("experiment_id") != EXPERIMENT or tuple(value.get("arms", [])) != ARMS:
        raise ContinuationError("R3B_IDENTITY_REQUIRED")
    c = value["engine"]
    original.validate(c)
    d = value["diagnostics"]
    if not c.get("synthetic"):
        reference = original.load()
        for key in (
            "model",
            "training",
            "policy",
            "oracle",
            "shadow",
            "data",
            "alignment",
            "competition",
            "evaluation",
        ):
            if c[key] != reference[key]:
                raise ContinuationError("FROZEN_R3_SCIENCE_CHANGED", key)
        if d["steps"] != LIGHT or d["functional_steps"] != HEAVY:
            raise ContinuationError("R3B_PLAN_CHANGED")
    if not d["steps"] or d["steps"][0] != 0 or d["steps"][-1] != c["training"]["steps"]:
        raise ContinuationError("R3B_PLAN_INCOMPLETE")
    if sorted(set(d["steps"])) != d["steps"] or set(d["functional_steps"]) - set(d["steps"]):
        raise ContinuationError("R3B_PLAN_INVALID")
    if value.get("authorization", {}).get("confirmation"):
        raise ContinuationError("CONFIRMATION_SEALED")


def runtime(candidate, source, *, beta, pilot_hash, run_id, assets_hash):
    validate(candidate)
    c = deepcopy(candidate["engine"])
    c["competition"]["beta"] = beta
    c["authorization"]["development_training"] = True
    c["r3b"] = {
        "experiment_id": EXPERIMENT,
        "arms": list(ARMS),
        "run_id": run_id,
        "diagnostics": deepcopy(candidate["diagnostics"]),
        "candidate_sha256": canonical(candidate),
        "protocol_sha256": digest(BASE / "SPEC.md"),
        "analysis_protocol": "r3b-longitudinal-v1",
        "assets_manifest_sha256": assets_hash,
    }
    from .diagnostic_schedule import load_plan

    c["r3b"]["diagnostic_execution"] = load_plan()
    c["execution"] = {
        "source_sha": source,
        "cpu_pass": True,
        "gradient_pilot_pass": True,
        "local_alignment_pass": True,
        "pilot_hash": pilot_hash,
        "configuration_sha256": canonical(c),
    }
    return c


def identity(c, arm, source, data):
    old = __import__("clm001_r3.checkpoint", fromlist=["identity"]).identity(
        c, ENGINE[arm], source, data
    )
    return {
        **old,
        "experiment_id": c["r3b"]["experiment_id"],
        "arm": arm,
        "implementation_arm": ENGINE[arm],
        "run_id": c["r3b"]["run_id"],
        "protocol_sha256": c["r3b"]["protocol_sha256"],
        "observation_protocol_sha256": canonical(c["r3b"]["diagnostics"]),
    }


def require_training_release(candidate, source, acceptance, authorization, pilot):
    # All inputs are committed report paths or pinned HF-retrieved documents,
    # never booleans fabricated by the Notebook or environment overrides.
    if candidate["engine"].get("synthetic"):
        return
    if authorization.get("release_mode") == "USER_APPROVED_ESTIMATED_BUDGET_DEVELOPMENT":
        from .development_release import validate as validate_release

        validate_release(candidate, source, authorization, pilot)
        return
    if (
        pilot.get("status") != "GPU_PILOT_PASS"
        or pilot.get("source_sha") != source
        or pilot.get("candidate_sha256") != canonical(candidate)
    ):
        raise ContinuationError("GATE_D_NOT_VERIFIED")
    if not all(
        pilot.get(k) is True
        for k in (
            "gradient_and_alignment_pass",
            "two_t4_hardware",
            "resource_target_met",
            "snapshot_and_event_budget_reviewed",
        )
    ) or pilot.get("diagnostic_plan_sha256") != canonical(candidate["diagnostics"]):
        raise ContinuationError("GATE_D_PRODUCTION_BUDGET_NOT_REVIEWED")
    if (
        authorization.get("formal_500_step_authorized") is not True
        or authorization.get("source_sha") != source
        or authorization.get("candidate_sha256") != canonical(candidate)
        or authorization.get("pilot_sha256") != canonical(pilot)
    ):
        raise ContinuationError("GATE_E_FORMAL_TRAINING_NOT_AUTHORIZED")
    if authorization.get("assets_manifest_sha256") != pilot.get(
        "assets_manifest_sha256"
    ) or authorization.get("beta") != pilot.get("beta"):
        raise ContinuationError("GATE_E_RUNTIME_NOT_FROZEN")


def safe_error(error):
    # Read exception attributes without invoking urllib HTTPError's delegated
    # file-object properties; those may themselves fail on an interrupted body.
    attributes = vars(error)
    raw_code = attributes.get("code", type(error).__name__)
    code = raw_code if isinstance(raw_code, str) else type(error).__name__ + "_" + str(raw_code)
    response = attributes.get("response")
    status = (
        getattr(response, "status_code", None)
        if response is not None
        else raw_code
        if isinstance(raw_code, int)
        else None
    )
    import re

    message = str(error)
    if re.search(r"(?i)secret|password|authorization|bearer|token[ :=]", message):
        message = "[SENSITIVE_MESSAGE_REDACTED]"
    message = re.sub(r"https?://[^\s]+", "[URL_REDACTED]", message)
    message = re.sub(r"(?:hf_|ghp_|github_pat_)[A-Za-z0-9_]+", "[TOKEN_REDACTED]", message)
    for name, secret in os.environ.items():
        if (
            any(word in name.upper() for word in ("TOKEN", "SECRET", "PASSWORD"))
            and len(secret) >= 8
        ):
            message = message.replace(secret, "[SECRET_REDACTED]")
    return {
        "failure_code": code,
        "type": type(error).__name__,
        "http_status": status,
        "message": message[:1200],
    }
