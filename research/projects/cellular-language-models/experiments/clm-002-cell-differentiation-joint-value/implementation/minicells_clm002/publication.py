"""CLM-002 report-only scope over existing Git/auth/remote-verification components."""

import copy
import hashlib
import json
from pathlib import Path

from minicells_jct001 import publication as parent
from minicells_r3k.reuse import bind

from .common import BASE, ID, ROOT, digest, read, sha

PublicationError = parent.PublicationError

REPORTS = BASE.relative_to(ROOT) / "reports/published/gate0"
TOP = {
    "ARCHITECTURE_AUDIT.json",
    "DATA_BOUNDARY_PROOF.json",
    "CONFIG.json",
    "SEED_REGISTRY.json",
    "CPU_ENGINEERING.json",
    "MEASUREMENT_UNIT_PROOF.json",
    "COST_UNIT.json",
    "GATE_STATUS.json",
    "DECISION.json",
    "PROTOCOL.zh-CN.md",
    "STAGE_A_DEVELOPMENT_REPORT.md",
    "STAGE_A_CONFIRMATION_REPORT.md",
    "STAGE_B_FACTORIAL_REPORT.md",
    "SEED_LEVEL_RESULTS.csv",
    "CELL_TASK_CONTRIBUTION.csv",
    "ROUTE_STABILITY.csv",
    "COST_COMPARISON.csv",
}
ARM = {
    "RUN_MANIFEST.json",
    "CONFIG.json",
    "METRICS.json",
    "CELL_SPECIALIZATION.json",
    "ROUTING_SUMMARY.json",
    "TRAINING_COST.json",
    "CHECKPOINT_MANIFEST.json",
}


def permitted(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    return name in TOP or name == "PARENT_FORK_AUDIT.json" or (
        len(path.parts) == 7 and path.parts[0] == "artifacts" and path.name in ARM
    )


def payload(root):
    root = Path(root)
    m = read(root / "RUN_MANIFEST.json")
    if (
        m.get("experiment_id") != ID
        or m.get("scope") not in ("GATE0_SYNTHETIC_CPU_ONLY", "GATE0_CPU_PARENT_FORK_ONLY")
        or m.get("scientific_optimizer_updates") != 0
        or m.get("confirmation_materialized") is not False
    ):
        raise parent.PublicationError("CLM002_PUBLICATION_SCOPE_INVALID")
    files = {}
    total = 0
    for name, item in m["artifact_files"].items():
        if not permitted(name):
            raise parent.PublicationError("CLM002_NON_REPORT_FILE_REJECTED")
        path = root / name
        if not path.is_file() or any(
            root.joinpath(*path.relative_to(root).parts[:i]).is_symlink()
            for i in range(1, len(path.relative_to(root).parts) + 1)
        ):
            raise parent.PublicationError("CLM002_REPORT_SYMLINK_OR_MISSING")
        content = path.read_bytes()
        total += len(content)
        if (
            len(content) > parent.MAX_FILE_BYTES
            or total > parent.MAX_TOTAL_BYTES
            or item != {"bytes": len(content), "sha256": digest(path)}
        ):
            raise parent.PublicationError("CLM002_REPORT_HASH_OR_SIZE_INVALID")
        if name.endswith(".json"):
            ident = read(path).get("identity", {})
            if any(ident.get(k) != v for k, v in m["identity"].items()):
                raise parent.PublicationError("CLM002_REPORT_IDENTITY_MISMATCH")
        files[name] = content
    complete = m["run_status"] == "CPU_ENGINEERING_COMPLETE_GATE0_BLOCKED"
    if m["scope"] == "GATE0_CPU_PARENT_FORK_ONLY" and set(files) != {"PARENT_FORK_AUDIT.json"}:
        raise parent.PublicationError("CLM002_PARENT_FORK_REPORT_SCOPE_INVALID")
    if complete and not TOP <= set(files):
        raise parent.PublicationError("CLM002_REQUIRED_REPORTS_MISSING")
    snapshot = copy.deepcopy(m)
    for key in tuple(snapshot):
        if key.startswith("publication_") or key in ("publication", "remote_commit_sha"):
            snapshot.pop(key)
    snapshot["publication_status"] = "GIT_PAYLOAD_SNAPSHOT"
    files["RUN_MANIFEST.json"] = (
        json.dumps(snapshot, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    index = {
        name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in files.items()
    }
    return m, files, index, sha(index)


def _checkout(root, env):
    return bind(parent._checkout, REPORTS=REPORTS)(root, env)


def _print(message, **kw):
    print(str(message).replace("[JCT GitHub]", "[CLM002 GitHub]"), **kw)


def preflight(root):
    return bind(parent.preflight, _checkout=_checkout, print=_print)(root)


def publish(root):
    return bind(
        parent.publish,
        _checkout=_checkout,
        payload=payload,
        REPORTS=REPORTS,
        EXPERIMENT_ID=ID,
        BRANCH_PREFIX="results/clm002-gate0-",
        print=_print,
    )(root)


def publish_existing(output):
    return bind(parent.publish_existing, publish=publish)(output)
