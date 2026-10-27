"""Use the existing GitHub transport with a metadata-only R1 report allowlist."""

import copy
import csv
import hashlib
import io
import json
from pathlib import Path

from minicells_fast002 import publication as previous
from minicells_r3k.reuse import bind

from .common import BASE, ID, ROOT, atomic, digest, read, sha

REPORTS = BASE.relative_to(ROOT) / "reports/published"
NAMES = {
    "HISTORICAL_EVIDENCE_AUDIT.json",
    "HISTORICAL_AUDIT.json",
    "DATASET_MANIFEST.json",
    "PARENT_STATE_AUDIT.json",
    "REPLAY_AUDIT.json",
    "ADMISSION_DECISION.json",
    "COST_LEDGER.json",
    "GATE_DECISION.json",
    "FINAL_REPORT.md",
    "CONNECTION_SCALE_SCAN.csv",
    "FAILURE.json",
    "CI_ACCEPTANCE.json",
    "PROTOCOL_AMENDMENT.md",
    "REGISTRATION.json",
    "FINAL_ANALYSIS.json",
}


CONTEXT_NAMES = {
    "REPLAY_AUDIT.json",
    "CANDIDATE.json",
    "ADMISSION_DECISION.json",
    "COST_LEDGER.json",
    "COMPLETE.json",
    "SCAN_PROGRESS.json",
    "CONNECTION_SCALE_SCAN.csv",
}


def allowed(name):
    parts = Path(name).parts
    if len(parts) == 2 and parts[0] == "jobs" and parts[1].endswith(".json"):
        return parts[1][:-5].replace("-", "").isalnum()
    if len(parts) == 3 and parts[0] == "proposals":
        return parts[1].replace("-", "").isalnum() and parts[2] in {
            "CANDIDATE.json",
            "ADMISSION_DECISION.json",
        }
    return name in NAMES or (
        len(parts) == 3
        and parts[0] == "contexts"
        and parts[1].replace("-", "").isalnum()
        and parts[2] in CONTEXT_NAMES
    )


def payload(root):
    root = Path(root)
    manifest = read(root / "RUN_MANIFEST.json")
    if manifest.get("experiment_id") != ID or manifest.get("scope") not in {
        "STATEFUL_CALIBRATION_ENGINEERING",
        "BOUNDED_STATEFUL_GATE1_DIAGNOSTIC",
        "FORMAL_STAGE_B_CONTROLLED_JOINT_VALUE",
    }:
        raise previous.old.transport.PublicationError("R1_REPORT_SCOPE_MISMATCH")
    files = {}
    total = 0
    for name, expected in manifest["artifact_files"].items():
        path = root / name
        if not allowed(name) or any(p.is_symlink() for p in (path, *path.parents)):
            raise previous.old.transport.PublicationError("R1_UNSAFE_ARTIFACT")
        raw = path.read_bytes()
        total += len(raw)
        if len(raw) > previous.old.transport.MAX_FILE_BYTES or total > (
            48_000_000
            if manifest["scope"] == "FORMAL_STAGE_B_CONTROLLED_JOINT_VALUE"
            else 24_000_000
        ):
            raise previous.old.transport.PublicationError("R1_REPORT_TOO_LARGE")
        if expected != {"bytes": len(raw), "sha256": digest(path)}:
            raise previous.old.transport.PublicationError("R1_ARTIFACT_BYTES_CHANGED")
        if name.endswith(".json"):
            json.loads(raw)
        elif name.endswith(".csv"):
            list(csv.DictReader(io.StringIO(raw.decode())))
        else:
            raw.decode("utf-8")
        files[name] = raw
    snapshot = copy.deepcopy(manifest)
    for key in list(snapshot):
        if key.startswith("publication") or key == "remote_commit_sha":
            snapshot.pop(key)
    snapshot["publication_status"] = "GIT_PAYLOAD_SNAPSHOT"
    files["RUN_MANIFEST.json"] = (json.dumps(snapshot, sort_keys=True, indent=2) + "\n").encode()
    index = {
        name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in files.items()
    }
    return manifest, files, index, sha(index)


def checkout(root, env):
    return bind(previous.old.transport._checkout, REPORTS=REPORTS)(root, env)


def publish(root):
    return bind(
        previous.old.transport.publish,
        payload=payload,
        _checkout=checkout,
        REPORTS=REPORTS,
        EXPERIMENT_ID=ID,
        BRANCH_PREFIX="results/fast002-r1-",
    )(root)


def existing(root):
    return bind(previous.old.transport.publish_existing, publish=publish)(root)


def finalize(root, manifest, enabled):
    manifest["artifact_files"] = {
        p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in root.rglob("*")
        if p.is_file() and allowed(p.relative_to(root).as_posix())
    }
    atomic(root / "RUN_MANIFEST.json", manifest)
    return publish(root) if enabled else {"status": "LOCAL_RETAINED_NOT_PUBLISHED"}
