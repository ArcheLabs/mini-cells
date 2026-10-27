"""Existing verified transport, isolated FAST002 report scope; weights never in Git."""

from minicells_fast001 import publication as old
from minicells_r3k.reuse import bind

from .common import BASE, ID, ROOT, atomic, digest

REPORTS = BASE.relative_to(ROOT) / "reports/published"


def allowed(name):
    return old.allowed(name) or name in {
        "HISTORICAL_AUDIT.json",
        "CALIBRATION.json",
        "CHECKPOINT_REUSE_PROOF.json",
        "FRESH_CONFIRM_PROOF.json",
        "REPAIR_IDENTITY.json",
    }


payload = bind(old.payload, ID=ID, allowed=allowed)


def checkout(root, env):
    return bind(old.transport._checkout, REPORTS=REPORTS)(root, env)


def publish(root):
    return bind(
        old.transport.publish,
        payload=payload,
        _checkout=checkout,
        REPORTS=REPORTS,
        EXPERIMENT_ID=ID,
        BRANCH_PREFIX="results/clm002-fast002-",
    )(root)


def existing(root):
    return bind(old.transport.publish_existing, publish=publish)(root)


def finalize(root, manifest, enabled):
    manifest["artifact_files"] = {
        p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(root.rglob("*.json"))
        if allowed(p.relative_to(root).as_posix())
    }
    atomic(root / "RUN_MANIFEST.json", manifest)
    return publish(root) if enabled else {"status": "LOCAL_RETAINED_NOT_PUBLISHED"}
