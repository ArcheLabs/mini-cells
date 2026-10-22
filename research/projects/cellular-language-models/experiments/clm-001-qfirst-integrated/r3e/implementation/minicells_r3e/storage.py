"""Reuse existing immutable Store, selective restore, serialized segment commits and readback."""

from pathlib import Path
import os
import shutil
import tempfile
from minicells_r3d.storage import restore_parents as restore_parents, parent_state as parent_state
from minicells_r3c_moe_bridge.checkpoint import Store, publish_segment
from minicells_r3c_moe_bridge.development import restore_latest as restore_latest, verified_manifest
from .common import atomic, read


def save(tr, store):
    row = store.save(tr)
    atomic(store.root / "latest.json", row)
    return row


def export(private, destination):
    """Coordinator snapshots immutable receipts, not mutable worker latest/temp files."""
    private = Path(private)
    destination = Path(destination)
    for directory in sorted(
        list(private.glob("jobs/*")) + list(private.glob("jobs/*/transactions/*"))
    ):
        rows = [(read(p), p) for p in (directory / "receipts").glob("*.json")]
        if not rows:
            continue
        row, receipt = max(rows, key=lambda r: r[0]["step"])
        Store(directory, row["identity"]).verified(row)
        target = destination / directory.relative_to(private)
        for source in (receipt, directory / row["file"]):
            dst = target / source.relative_to(directory)
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, dst)
            except OSError:
                shutil.copyfile(source, dst)
        atomic(target / "latest.json", row)
        # Result is written only after a verified terminal checkpoint, and is immutable thereafter.
        result = directory / "result.json"
        if result.exists():
            shutil.copyfile(result, target / "result.json")
    return destination


def remote_snapshot(backend, private, output, ident, pointer, session, prior):
    with tempfile.TemporaryDirectory(dir=output, prefix="snapshot-") as tmp:
        export(private, Path(tmp))
        if not list(Path(tmp).glob("jobs/*/latest.json")):
            return None
        manifest = verified_manifest(backend, prior, ident) if prior else None
        ref = publish_segment(
            backend, tmp, ident, session, pointer_path=pointer, prior_manifest=manifest
        )
        atomic(Path(output) / "hf-reference.json", ref)
        return ref


def public_publish(summary, output):
    # Reuse exact result-only main-descended publishing discipline; no pre-training GitHub handshake.
    from minicells_r3c_moe_bridge.publication import git, helper, ROOT
    from .common import BASE, canonical

    prepared = Path(output) / "github-metadata"
    prepared.mkdir(parents=True, exist_ok=True)
    if not (prepared / ".git").exists():
        git(prepared, "init", "--quiet")
        git(prepared, "remote", "add", "origin", "https://github.com/ArcheLabs/mini-cells.git")
        git(prepared, "config", "remote.origin.promisor", "true")
        git(prepared, "config", "remote.origin.partialclonefilter", "blob:none")
    git(prepared, "fetch", "--depth=1", "--filter=blob:none", "origin", "main")
    parent = git(prepared, "rev-parse", "FETCH_HEAD").stdout.strip()
    key = canonical(summary)
    branch = "results/r3e-" + key[:16] + "-" + parent[:12]
    if git(prepared, "ls-remote", "origin", "refs/heads/" + branch).stdout.strip():
        return {"status": "ALREADY_PUBLISHED", "branch": branch}
    dest = BASE.relative_to(ROOT) / "reports/published" / key
    git(prepared, "sparse-checkout", "init", "--no-cone")
    git(prepared, "sparse-checkout", "set", "--no-cone", "/" + str(dest.parent) + "/")
    git(prepared, "checkout", "-b", branch, parent)
    git(prepared, "config", "user.name", "MiniCells Kaggle")
    git(prepared, "config", "user.email", "minicells@users.noreply.github.com")
    allowed = (
        "RUN_MANIFEST.json",
        "METRICS.json",
        "POLICY_STABILITY.json",
        "ADAPTATION_RESULTS.json",
        "GRAPH_SUMMARY.json",
        "TRAINING_COST.json",
        "SCIENTIFIC-FINDINGS.md",
    )
    for leaf in allowed:
        src = Path(output) / "reports" / leaf
        if src.exists():
            target = prepared / dest / leaf
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
    git(prepared, "add", "--", str(dest))
    git(prepared, "commit", "-m", "Publish R3e aggregate results only")
    changed = git(prepared, "diff", "--name-only", parent, "HEAD").stdout.splitlines()
    if not changed or any(
        not p.startswith(str(dest) + "/") or Path(p).name not in allowed for p in changed
    ):
        raise RuntimeError("RESULT_ONLY_PUBLICATION_REQUIRED")
    auth = helper()
    old_root = auth.ROOT
    old_git = auth.run_git

    def bounded(*args, **kwargs):
        kwargs.setdefault("timeout", 45)
        return old_git(*args, **kwargs)

    try:
        auth.ROOT = prepared
        auth.run_git = bounded
        auth.push_branch(branch, "GITHUB_TOKEN")
    finally:
        auth.ROOT = old_root
        auth.run_git = old_git
    return {
        "status": "PUBLISHED",
        "branch": branch,
        "parent_main_sha": parent,
        "report_directory": str(dest),
    }
