"""Explicit human-approved development release; no fabricated performance PASS."""

from pathlib import Path
import time
from minicells_runtime.store import read, canonical, digest, safe, ContinuationError
from .common import BASE

MODE = "USER_APPROVED_ESTIMATED_BUDGET_DEVELOPMENT"


def validate(candidate, source, approval, evidence):
    from clm001_r3.common import ROOT, source_sha
    from .diagnostic_schedule import load_plan

    if (
        approval.get("release_mode") != MODE
        or approval.get("formal_500_step_authorized") is not True
        or approval.get("estimated_budget_accepted") is not True
        or approval.get("two_hour_guarantee") is not False
    ):
        raise ContinuationError("DEVELOPMENT_RELEASE_NOT_AUTHORIZED")
    if source != source_sha() or len(source) != 40:
        raise ContinuationError("AUTHORIZED_SOURCE_MISMATCH")
    attestation = read(BASE / "reports/authorized-development/source-attestation.json")
    if canonical(attestation) != approval.get("source_attestation_sha256"):
        raise ContinuationError("AUTHORIZED_CODE_ATTESTATION_MISMATCH")
    for relative, wanted in attestation["source_files"].items():
        if digest(safe(ROOT, relative)) != wanted:
            raise ContinuationError("AUTHORIZED_CODE_CHANGED", relative)
    if (
        approval.get("candidate_sha256") != canonical(candidate)
        or approval.get("execution_plan_sha256") != canonical(load_plan())
        or approval.get("pilot_sha256") != canonical(evidence)
        or evidence.get("status") != "TRANSFERRED_ENGINEERING_EVIDENCE"
        or evidence.get("performance_gate_status") != "NOT_VERIFIED"
    ):
        raise ContinuationError("AUTHORIZED_RELEASE_IDENTITY_MISMATCH")
    for relative, wanted in approval["evidence_files"].items():
        if digest(safe(BASE, relative)) != wanted:
            raise ContinuationError("AUTHORIZED_EVIDENCE_CHANGED", relative)
    pilot = read(BASE / evidence["original_pilot_file"])
    delta = read(BASE / evidence["delta_file"])
    if (
        pilot["gradient_and_alignment_pass"] is not True
        or pilot["two_t4_hardware"] is not True
        or pilot["source_sha"] != evidence["original_pilot_source_sha"]
        or delta["status"] != "TRAINING_DELTA_PASS"
        or delta["historical_parent_unchanged"] is not True
        or delta["traces"][0]["state_fingerprint"] != delta["traces"][1]["state_fingerprint"]
        or delta["verification_source_sha"] != attestation["delta_reference_source_sha"]
    ):
        raise ContinuationError("ENGINEERING_EVIDENCE_NOT_VERIFIED")
    if (
        approval["assets_manifest_sha256"] != pilot["assets_manifest_sha256"]
        or evidence["assets_manifest_sha256"] != pilot["assets_manifest_sha256"]
        or approval["beta"] != pilot["beta"]
        or evidence["beta"] != pilot["beta"]
    ):
        raise ContinuationError("AUTHORIZED_RUNTIME_NOT_FROZEN")


def choose_auto_mode(store, source, run_id, approval):
    if (
        approval.get("release_mode") != MODE
        or approval.get("formal_500_step_authorized") is not True
    ):
        return "auto", None
    try:
        ref = store.locate(source, run_id)
        return "resume", ref
    except ContinuationError as error:
        if error.code != "HF_RUN_NOT_FOUND":
            raise
    return "train", None


def retain(path):
    """Keep local leftovers intact; the exact HF revision stays authoritative."""
    path = Path(path)
    if path.exists():
        target = path.with_name(path.name + "-retained-" + str(time.time_ns()))
        path.rename(target)
        print({"retained_local_output": str(target)}, flush=True)
