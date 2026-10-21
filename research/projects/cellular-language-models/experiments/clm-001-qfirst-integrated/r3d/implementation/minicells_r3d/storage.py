"""Adapt existing immutable Store/segment transport; no duplicate checkpoint format."""

from pathlib import Path
import uuid
from minicells_runtime.store import safe
from minicells_r3c_moe_bridge.checkpoint import Store, restore_segment, publish_segment
from minicells_r3c_moe_bridge.development import (
    verified_manifest,
    restore_latest,
    export_checkpoints,
)
from minicells_r3c_moe_bridge.development import job_identity as parent_identity
from .common import atomic, read, ContinuationError


def restore_parents(c, backend, root):
    root = Path(root)
    ref = c["parent"]["hf_reference"]
    manifest = verified_manifest(backend, ref)
    expected_manifest = {
        "experiment_id": c["engine"]["experiment_id"],
        "phase": "EXPLORATORY_DEVELOPMENT",
        "source_sha": c["parent"]["source_sha"],
        "config_sha256": c["parent"]["config_sha256"],
    }
    if any(manifest["identity"].get(k) != v for k, v in expected_manifest.items()):
        raise ContinuationError("PARENT_MANIFEST_IDENTITY_MISMATCH")
    selected = set()
    for seed in c["seeds"]:
        prefix = "jobs/seed-%s-arm-D/" % seed
        name = prefix + "latest.json"
        if name not in manifest["files"]:
            raise ContinuationError("PARENT_D_STEP500_MISSING", str(seed))
        row = manifest["files"][name]
        from .common import digest

        p = backend.get(row["object"], ref["hf_revision"])
        if digest(p) != row["sha256"]:
            raise ContinuationError("PARENT_LATEST_HASH_MISMATCH")
        latest = read(p)
        ident = parent_identity(c["engine"], c["parent"]["source_sha"], seed, "D")
        if (
            latest["identity"] != ident
            or latest["step"] != 500
            or latest["sha256"] != c["parent"]["checkpoint_sha256"][str(seed)]
        ):
            raise ContinuationError("PARENT_D_STEP500_IDENTITY_MISMATCH")
        receipt = Store(Path(prefix), ident).receipt_file(500).as_posix()
        selected.update([name, prefix + latest["file"], receipt])
    # Restore only the three requested D objects, never A/B/C or old snapshots.
    restore_segment(
        backend, ref, root, manifest["identity"], uuid.uuid4().hex, include_paths=selected
    )
    for seed in c["seeds"]:
        parent_state(c, root, seed)


def parent_state(c, root, seed):
    directory = Path(root) / ("jobs/seed-%s-arm-D" % seed)
    latest = read(directory / "latest.json")
    if latest["step"] != 500 or latest["sha256"] != c["parent"]["checkpoint_sha256"][str(seed)]:
        raise ContinuationError("PARENT_D_STEP500_MISMATCH")
    ident = parent_identity(c["engine"], c["parent"]["source_sha"], seed, "D")
    state = Store(directory, ident).verified(latest)
    if state["step"] != 500 or state["data_cursor"] != 256000 or state["balance_coefficient"] != 0:
        raise ContinuationError("PARENT_D_STATE_BUDGET_MISMATCH")
    return state


def save(tr, store, curve, lineage):
    tr.verify_frozen()
    row = store.save(tr, observations={"curve": curve, "lineage": lineage})
    atomic(store.root / "latest.json", row)
    return row


def export(c, private, destination, *, completed_only=False):
    # Transport code is shared, schedule belongs to R3d.
    steps = c["training"]["b_steps"]
    wrapper = {"development": {"checkpoint_steps": [0, c["training"]["a_steps"], steps]}}
    if not completed_only:
        return export_checkpoints(wrapper, private, destination)
    # Workers never mutate a completed job. Hardlinks avoid copying model bytes.
    import os
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory(dir=Path(destination).parent) as temp:
        snapshot = Path(temp)
        for directory in (Path(private) / "jobs").glob("*"):
            if not (directory / "result.json").exists():
                continue
            shutil.copytree(directory, snapshot / "jobs" / directory.name, copy_function=os.link)
        return export_checkpoints(wrapper, snapshot, destination)


def public_publish(summary, prepared):
    from minicells_r3c_moe_bridge.publication import git, helper, BASE as R3C_BASE, ROOT
    from .common import canonical, BASE

    checkout = prepared["checkout"]
    git(checkout, "fetch", "--depth=1", "--filter=blob:none", "origin", "main")
    parent = git(checkout, "rev-parse", "FETCH_HEAD").stdout.strip()
    branch = "results/r3d-" + canonical(summary)[:16] + "-" + parent[:12]
    destination = BASE.relative_to(ROOT) / "reports/published" / (canonical(summary) + ".json")
    existing = git(checkout, "ls-remote", "origin", "refs/heads/" + branch).stdout.strip()
    if existing:
        return {"status": "ALREADY_PUBLISHED", "branch": branch, "parent_main_sha": parent}
    git(
        checkout,
        "sparse-checkout",
        "set",
        "--no-cone",
        "/README.md",
        "/" + str(R3C_BASE.relative_to(ROOT) / "reports/published") + "/",
        "/" + str(BASE.relative_to(ROOT) / "reports/published") + "/",
    )
    git(checkout, "checkout", "-b", branch, parent)
    atomic(checkout / destination, summary)
    git(checkout, "add", "--", str(destination))
    git(checkout, "commit", "-m", "Publish R3d aggregate results only")
    if git(checkout, "diff", "--name-only", parent, "HEAD").stdout.splitlines() != [
        str(destination)
    ]:
        raise RuntimeError("non-result change")
    auth = helper()
    old = auth.ROOT
    try:
        auth.ROOT = checkout
        auth.push_branch(branch, "GITHUB_TOKEN")
    finally:
        auth.ROOT = old
    return {"status": "PUBLISHED", "branch": branch, "parent_main_sha": parent}


__all__ = ["Store", "restore_latest", "verified_manifest", "publish_segment", "safe"]
