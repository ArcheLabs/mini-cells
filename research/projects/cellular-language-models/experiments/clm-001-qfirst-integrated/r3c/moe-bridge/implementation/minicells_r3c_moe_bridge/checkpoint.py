"""Immutable content-addressed full joint state, plus one private HF commit per segment."""

from pathlib import Path
import os
import tempfile
import shutil
import torch
from minicells_runtime.store import atomic, canonical, digest, read, safe, ContinuationError
from .common import fingerprint


class Store:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.identity = identity

    def save(self, trainer):
        key = canonical({"identity": self.identity, "step": trainer.step})
        receipt = self.root / "receipts" / (key + ".json")
        state = trainer.state()
        semantic = fingerprint(state)
        if receipt.exists():
            old = read(receipt)
            if old["identity"] != self.identity or old["state_fingerprint"] != semantic:
                raise ContinuationError("R3C_CHECKPOINT_IMMUTABLE_CONFLICT")
            self.verified(old)
            return old
        directory = self.root / "objects"
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(
                {"identity": self.identity, "state": state, "state_fingerprint": semantic}, stream
            )
            stream.flush()
            os.fsync(stream.fileno())
        sha = digest(temporary)
        target = directory / (sha + ".pt")
        try:
            try:
                os.link(temporary, target)
            except FileExistsError:
                if digest(target) != sha:
                    raise ContinuationError("R3C_CHECKPOINT_HASH_MISMATCH")
        finally:
            temporary.unlink()
        row = {
            "schema": "minicells.r3c.checkpoint.v1",
            "identity": self.identity,
            "step": trainer.step,
            "state_fingerprint": semantic,
            "sha256": sha,
            "bytes": target.stat().st_size,
            "file": target.relative_to(self.root).as_posix(),
            "cross_session_verified": False,
        }
        atomic(receipt, row)
        self.verified(row)
        return row

    def verified(self, row):
        if row.get("schema") != "minicells.r3c.checkpoint.v1":
            raise ContinuationError("R3C_CHECKPOINT_SCHEMA_MISMATCH")
        if row["identity"] != self.identity:
            raise ContinuationError("R3C_CHECKPOINT_IDENTITY_MISMATCH")
        path = safe(self.root, row["file"])
        if digest(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ContinuationError("R3C_CHECKPOINT_HASH_MISMATCH")
        # Bytes/schema/identity are checked before trusted project pickle deserialization.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload["identity"] != self.identity
            or payload["state_fingerprint"] != row["state_fingerprint"]
            or fingerprint(payload["state"]) != row["state_fingerprint"]
        ):
            raise ContinuationError("R3C_CHECKPOINT_STATE_MISMATCH")
        return payload["state"]

    def restore(self, trainer, row):
        trainer.restore(self.verified(row))


def publish_segment(backend, root, identity, session_id):
    if not backend.info().private:
        raise ContinuationError("HF_PRIVATE_REPO_REQUIRED")
    backend.check_write()
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContinuationError("UNSAFE_PATH")
        if path.is_file() and path.suffix in (".pt", ".json") and not path.name.endswith(".tmp"):
            sha = digest(path)
            files[path.relative_to(root).as_posix()] = {
                "sha256": sha,
                "bytes": path.stat().st_size,
                "object": "objects/sha256/" + sha,
            }
    if not files or not session_id:
        raise ContinuationError("R3C_EMPTY_SEGMENT_OR_SESSION")
    manifest = {
        "schema": "minicells.r3c.segment.v1",
        "identity": identity,
        "writer_session_id": session_id,
        "files": files,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "segment.json"
        atomic(path, manifest)
        key = "r3c-moe-bridge/segments/" + canonical(manifest) + ".json"
        additions = {row["object"]: root / name for name, row in files.items()}
        additions[key] = path
        revision = backend.commit(
            additions, backend.info().sha, "R3c bounded segment checkpoint bundle"
        )
        if digest(backend.get(key, revision)) != digest(path):
            raise ContinuationError("HF_READBACK_MISMATCH")
    return {
        "hf_repo": backend.repo,
        "hf_revision": revision,
        "manifest_path": key,
        "manifest_sha256": digest(backend.get(key, revision)),
        "writer_session_id": session_id,
        "cross_session_verified": False,
        "scope": "IMMEDIATE_PRIVATE_READBACK_ONLY",
    }


def restore_segment(backend, ref, root, identity, session_id):
    if ref["hf_repo"] != backend.repo:
        raise ContinuationError("R3C_REPOSITORY_IDENTITY_MISMATCH")
    path = backend.get(ref["manifest_path"], ref["hf_revision"])
    if digest(path) != ref["manifest_sha256"]:
        raise ContinuationError("HF_READBACK_MISMATCH")
    manifest = read(path)
    if manifest.get("schema") != "minicells.r3c.segment.v1":
        raise ContinuationError("R3C_SEGMENT_SCHEMA_MISMATCH")
    if manifest["identity"] != identity or manifest["writer_session_id"] == session_id:
        raise ContinuationError("R3C_INDEPENDENT_SESSION_REQUIRED")
    if not backend.info(ref["hf_revision"]).private:
        raise ContinuationError("HF_PRIVATE_REPO_REQUIRED")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, row in manifest["files"].items():
        from minicells_artifacts.hf_storage import relative

        relative(name)
        source = backend.get(row["object"], ref["hf_revision"])
        if digest(source) != row["sha256"] or source.stat().st_size != row["bytes"]:
            raise ContinuationError("HF_READBACK_MISMATCH")
        target = root / name
        if any(p.is_symlink() for p in (target, *target.parents)):
            raise ContinuationError("UNSAFE_PATH")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != row["sha256"]:
            raise ContinuationError("R3C_RESTORE_COLLISION")
        if not target.exists():
            shutil.copyfile(source, target)
    proof = {
        "status": "READBACK_VERIFIED",
        "cross_session_verified": True,
        "hf_revision": ref["hf_revision"],
        "manifest_sha256": ref["manifest_sha256"],
        "restored_session_id": session_id,
        "identity": identity,
    }
    atomic(root / "readback.json", proof)
    return proof
