"""Evidence closure limited to the three named experiments, with explicit byte-availability limits."""

from minicells_fast002.audit import reconcile

from .common import BASE, ROOT, atomic, digest, read, sha

V2 = "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3f-v2"
V2_REPORT = (
    V2 + "/reports/published/3a0e651fff7770f8576e19ac76244262587941fe9d7ed6a3263d7246cfbe9dbb"
)
FAST002_PREFIX = "research/projects/cellular-language-models/experiments/clm-002-fast-002/reports/published/40ab59e0c7a4395ba019ecf28da6469a/"


def audit(c, reports, output):
    dependencies = read(BASE / "registration/SOURCE_DEPENDENCIES.json")
    for path, expected in dependencies.items():
        if digest(ROOT / path) != expected:
            raise RuntimeError("R1_FROZEN_REUSED_SOURCE_CHANGED:" + path)
    first = reconcile(c, reports, output)
    if (
        digest(BASE / "registration/historical/FAST002_CALIBRATION.json")
        != c["history_fast002"]["calibration_sha256"]
    ):
        raise RuntimeError("R1_HISTORICAL_FAST002_CALIBRATION_CHANGED")
    cal = read(BASE / "registration/historical/FAST002_CALIBRATION.json")
    if (
        len(cal["rows"]) != 20
        or cal["candidate_optimizer_calls"] != 160
        or cal["main_updates"] != 0
    ):
        raise RuntimeError("R1_FAST002_HISTORICAL_EVIDENCE_MISMATCH")
    manifest = read(ROOT / V2_REPORT / "RUN_MANIFEST.json")
    for name in ("LOCAL_SCALE_SCAN.json", "METRICS.json"):
        expected = next(
            row["sha256"] for row in manifest["artifact_files"] if row["path"] == "reports/" + name
        )
        if digest(ROOT / V2_REPORT / name) != expected:
            raise RuntimeError("R1_R3FV2_HISTORICAL_REPORT_HASH_MISMATCH")
    if sha(read(ROOT / V2 / "config/frozen.json")) != manifest["config_sha256"]:
        raise RuntimeError("R1_HISTORICAL_R3FV2_CONFIG_CHANGED")
    scans = read(ROOT / V2_REPORT / "LOCAL_SCALE_SCAN.json")
    metrics = read(ROOT / V2_REPORT / "METRICS.json")
    anchors = {
        arm: [row for row in scans[arm]["attempts"] if row["scale"] == 2.0**-12]
        for arm in ("A-C", "B-C")
    }
    if not all(len(rows) == 1 and rows[0]["passed"] for rows in anchors.values()):
        raise RuntimeError("R1_HISTORICAL_4096_ANCHOR_NOT_VERIFIED")
    definition = "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3f-connection-admission/implementation/minicells_r3f/evaluation.py"
    definition_proof = read(BASE / "registration/historical/R3FV2_DEFINITION_PROOF.json")
    if (
        definition_proof["source_sha"] != metrics["identity"]["source_sha"]
        or digest(ROOT / definition) != definition_proof["sha256"]
    ):
        raise RuntimeError("R1_R3FV2_TRUST_DEFINITION_CHANGED")
    result = {
        "status": "PUBLISHED_EVIDENCE_VERIFIED_REAL_PARENT_BYTES_PENDING",
        "fast001_totals": first["totals"],
        "stage_a": first["stage_a"],
        "fast002": {
            "contexts": 20,
            "candidate_optimizer_calls": 160,
            "main_updates": 0,
            "admitted": sum(row["gate_passed"] for row in cal["rows"]),
        },
        "r3fv2": {
            "identity": metrics["identity"],
            "anchor_4096_verified": True,
            "metrics": metrics["arms"],
            "metric_definition_sha256": digest(ROOT / definition),
            "private_historical_weight_bytes": "NOT_RECOVERED_NOT_USED_AS_R1_PARENT",
        },
        "step50_availability": "NOT_IN_PUBLISHED_REPORTS; ORIGINAL_EXECUTOR_OVERWRITES_LATEST_EACH_STEP",
        "no_backward_recovery_from_step500": True,
        "real_parent_bytes_loaded": False,
        "historical_registered_config_and_results_unchanged": True,
        "scientific_optimizer_updates": 0,
        "gpu_used": False,
    }
    atomic(output / "HISTORICAL_EVIDENCE_AUDIT.json", result)
    return result
