"""Use existing JCT report-only publisher and existing GITHUB_TOKEN handling."""

import copy
import hashlib
import json
import runpy
from pathlib import Path

from .common import BASE, ROOT, digest, sha

ALLOWED = (
    "RUN_MANIFEST.json",
    "baseline_assets.json",
    "PARITY.json",
    "EVENT_RESULTS.json",
    "EVENT_SUMMARY.csv",
    "INCREMENTAL_TIMING.json",
    "DECISIONS.json",
    "RESOURCE_MANIFEST.json",
    "SUMMARY.md",
    "elapsed_ms.png",
    "physical_updates.png",
)


def parent():
    deps = BASE.parent / "clm-001-qfirst-integrated/r3m-f-d1/run.py"
    runpy.run_path(str(deps), run_name="epclm_publication_dependencies")["_bootstrap"]()
    import sys

    sys.path.insert(0, str(BASE.parent / "jct-001/implementation"))
    from minicells_jct001 import publication
    from minicells_r3k.reuse import bind

    return publication, bind


def payload(root):
    root = Path(root)
    manifest = json.loads((root / "RUN_MANIFEST.json").read_text())
    if manifest["experiment_id"] != "EPCLM-001" or manifest["optimizer_updates"] != 0:
        raise ValueError("EPCLM_PUBLICATION_IDENTITY_INVALID")
    files = {}
    for name in ALLOWED:
        p = root / name
        if not p.exists():
            continue
        if p.is_symlink() or p.stat().st_size > 4_000_000:
            raise ValueError("EPCLM_PUBLICATION_FILE_REJECTED")
        if name != "RUN_MANIFEST.json" and manifest["artifact_files"].get(name) != {
            "bytes": p.stat().st_size,
            "sha256": digest(p),
        }:
            raise ValueError("EPCLM_REPORT_HASH_MISMATCH")
        files[name] = p.read_bytes()
    snapshot = copy.deepcopy(manifest)
    for key in tuple(snapshot):
        if key.startswith("publication") or key == "remote_commit_sha":
            snapshot.pop(key)
    files["RUN_MANIFEST.json"] = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode()
    if sum(map(len, files.values())) > 12_000_000:
        raise ValueError("EPCLM_PUBLICATION_BUDGET_EXCEEDED")
    index = {
        name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in files.items()
    }
    return manifest, files, index, sha(index)


def adapted():
    p, bind = parent()
    reports = BASE.relative_to(ROOT) / "reports/published"
    checkout = bind(p._checkout, REPORTS=reports)
    return p, bind, reports, checkout


def preflight(root):
    p, bind, _, checkout = adapted()
    return bind(p.preflight, _checkout=checkout)(root)


def publish(root):
    p, bind, reports, checkout = adapted()
    return bind(
        p.publish,
        ROOT=ROOT,
        REPORTS=reports,
        _checkout=checkout,
        payload=payload,
        EXPERIMENT_ID="EPCLM-001",
        BRANCH_PREFIX="results/epclm001-",
    )(root)
