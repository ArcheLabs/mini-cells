"""Reuse existing Secret/auth, report staging, idempotent push and remote SHA verification."""

import copy
import hashlib
import json
from pathlib import Path

from minicells_jct001 import publication as transport
from minicells_r3k.reuse import bind

from .common import BASE, ID, ROOT, digest, read, sha

REPORTS = BASE.relative_to(ROOT) / "reports/published"
# 50 raw 500-step curves plus 17,024 hash-only dataset rows; no weights/tokens.
MAX_TOTAL_BYTES = 24_000_000


def allowed(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    return name in {
        "DATASET_MANIFEST.json",
        "PARENT_FORK_ACCEPTANCE.json",
        "STAGE_A_DECISION.json",
        "FINAL_ANALYSIS.json",
        "BUDGET.json",
        "FAILURE.json",
        "REGISTRATION.json",
        "CI_ACCEPTANCE.json",
    } or (
        len(path.parts) == 2
        and path.parts[0] in {"data-manifests", "jobs"}
        and path.suffix == ".json"
    )


def payload(root):
    root = Path(root)
    manifest = read(root / "RUN_MANIFEST.json")
    if (
        manifest.get("experiment_id") != ID
        or manifest.get("scope") != "CONTROLLED_SYNTHETIC_MECHANISM_STUDY"
    ):
        raise transport.PublicationError("FAST_PUBLICATION_SCOPE_INVALID")
    files, total = {}, 0
    for name, expected in manifest["artifact_files"].items():
        path = root / name
        if (
            not allowed(name)
            or not path.is_file()
            or any(p.is_symlink() for p in (path, *path.parents))
        ):
            raise transport.PublicationError("FAST_UNSAFE_REPORT_PATH")
        data = path.read_bytes()
        total += len(data)
        if len(data) > transport.MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
            raise transport.PublicationError("FAST_REPORT_PAYLOAD_LIMIT_EXCEEDED")
        if expected != {"bytes": len(data), "sha256": digest(path)}:
            raise transport.PublicationError("FAST_REPORT_HASH_CHANGED")
        json.loads(data)
        files[name] = data
    snapshot = copy.deepcopy(manifest)
    for key in list(snapshot):
        if key.startswith("publication") or key == "remote_commit_sha":
            snapshot.pop(key)
    snapshot["publication_status"] = "GIT_PAYLOAD_SNAPSHOT"
    files["RUN_MANIFEST.json"] = (
        json.dumps(snapshot, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    index = {
        name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in files.items()
    }
    return manifest, files, index, sha(index)


def _checkout(root, env):
    return bind(transport._checkout, REPORTS=REPORTS)(root, env)


def publish(root):
    return bind(
        transport.publish,
        payload=payload,
        _checkout=_checkout,
        REPORTS=REPORTS,
        EXPERIMENT_ID=ID,
        BRANCH_PREFIX="results/clm002-fast001-",
    )(root)


def publish_existing(root):
    return bind(transport.publish_existing, publish=publish)(root)


def finalize(root, manifest, *, publish_github):
    root = Path(root)
    manifest["artifact_files"] = {
        p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(root.rglob("*.json"))
        if allowed(p.relative_to(root).as_posix())
    }
    from .common import atomic

    atomic(root / "RUN_MANIFEST.json", manifest)
    if publish_github:
        return publish(root)
    return {"status": "LOCAL_RETAINED_NOT_PUBLISHED"}
