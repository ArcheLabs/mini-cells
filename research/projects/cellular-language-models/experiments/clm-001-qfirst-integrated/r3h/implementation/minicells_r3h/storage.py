"""Existing Store/serialized segment transport; retain every declared immutable checkpoint."""

from pathlib import Path
import tempfile
import shutil
import os
from minicells_r3e.storage import Store, save, restore_latest, snapshot_lock
from minicells_r3c_moe_bridge.checkpoint import publish_segment
from minicells_r3c_moe_bridge.development import verified_manifest
from .parent_import import read, atomic


def remote_snapshot(backend, private, output, ident, pointer, session, prior):
    private = Path(private)
    with tempfile.TemporaryDirectory(dir=output, prefix="snapshot-") as tmp:
        dest = Path(tmp)
        with snapshot_lock(private):
            for job in private.glob("jobs/*"):
                receipts = list((job / "receipts").glob("*.json"))
                latest = None
                for receipt in receipts:
                    row = read(receipt)
                    Store(job, row["identity"]).verified(row)
                    for src in (receipt, job / row["file"]):
                        target = dest / src.relative_to(private)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if not target.exists():
                            try:
                                os.link(src, target)
                            except OSError:
                                shutil.copyfile(src, target)
                    if latest is None or row["step"] > latest["step"]:
                        latest = row
                if latest:
                    atomic(dest / job.relative_to(private) / "latest.json", latest)
                    if (job / "result.json").exists():
                        shutil.copyfile(
                            job / "result.json", dest / job.relative_to(private) / "result.json"
                        )
        if not list(dest.glob("jobs/*/latest.json")):
            return prior
        manifest = verified_manifest(backend, prior, ident) if prior else None
        ref = publish_segment(
            backend, dest, ident, session, pointer_path=pointer, prior_manifest=manifest
        )
        atomic(Path(output) / "hf-reference.json", ref)
        return ref


__all__ = ["Store", "save", "restore_latest", "remote_snapshot"]
