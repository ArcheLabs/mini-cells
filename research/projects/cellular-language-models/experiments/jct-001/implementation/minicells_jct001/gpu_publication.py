"""Scope-specific publication using the existing audited report-only publisher."""

import copy
import json
from pathlib import Path
from minicells_r3k.reuse import bind
from . import publication as parent
from .common import BASE, ROOT, read, digest, sha

REPORTS = BASE.relative_to(ROOT) / "reports/synthetic-pilot"
ALLOWED = parent.ALLOWED + ("PILOT_CONFIG.json", "PARTIAL_RESULTS.json")


def payload(root):
    root = Path(root)
    manifest = read(root / "RUN_MANIFEST.json")
    if (
        manifest.get("experiment_id") != "JCT-001-S1"
        or manifest.get("scope") != "SYNTHETIC_KNOWN_PATH_PILOT"
        or manifest.get("confirmation_materialized") is not False
        or len(manifest.get("source_sha", "")) != 40
        or len(manifest.get("run_id", "")) != 32
    ):
        raise parent.PublicationError("JCT_PILOT_PUBLICATION_IDENTITY_INVALID")
    files = {}
    total = 0
    complete = manifest["run_status"] == "SYNTHETIC_PILOT_COMPLETE_EXPLORATORY_REVIEW_REQUIRED"
    for name in ALLOWED:
        path = root / name
        if not path.exists():
            if (complete and name != "PARTIAL_RESULTS.json") or name == "RUN_MANIFEST.json":
                raise parent.PublicationError("JCT_PILOT_REPORT_MISSING_" + name)
            continue
        if path.is_symlink() or not path.is_file():
            raise parent.PublicationError("JCT_PILOT_REGULAR_FILES_REQUIRED")
        data = path.read_bytes()
        total += len(data)
        if len(data) > parent.MAX_FILE_BYTES or total > parent.MAX_TOTAL_BYTES:
            raise parent.PublicationError("JCT_PILOT_REPORT_BUDGET_EXCEEDED")
        if name != "RUN_MANIFEST.json":
            if manifest.get("artifact_files", {}).get(name) != dict(
                bytes=len(data), sha256=digest(path)
            ):
                raise parent.PublicationError("JCT_PILOT_REPORT_HASH_MISMATCH")
            if name.endswith(".json") and read(path).get("identity") != manifest["identity"]:
                raise parent.PublicationError("JCT_PILOT_REPORT_IDENTITY_MISMATCH")
        files[name] = data
    snapshot = copy.deepcopy(manifest)
    for key in tuple(snapshot):
        if key.startswith("publication_") or key in ("remote_commit_sha", "publication"):
            snapshot.pop(key)
    snapshot["publication_status"] = "GIT_PAYLOAD_SNAPSHOT"
    snapshot["publication_receipt_location"] = (
        "Kaggle Output PUBLICATION.json; remote commit metadata"
    )
    files["RUN_MANIFEST.json"] = (
        json.dumps(snapshot, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    index = {
        name: dict(bytes=len(data), sha256=__import__("hashlib").sha256(data).hexdigest())
        for name, data in files.items()
    }
    return manifest, files, index, sha(index)


def _checkout(root, env):
    return bind(parent._checkout, REPORTS=REPORTS)(root, env)


def preflight(root):
    return bind(parent.preflight, _checkout=_checkout)(root)


def publish(root):
    return bind(
        parent.publish,
        _checkout=_checkout,
        payload=payload,
        REPORTS=REPORTS,
        EXPERIMENT_ID="JCT-001-S1",
        BRANCH_PREFIX="results/jct001-synthetic-pilot-",
    )(root)


def publish_existing(output):
    return bind(parent.publish_existing, publish=publish)(output)
