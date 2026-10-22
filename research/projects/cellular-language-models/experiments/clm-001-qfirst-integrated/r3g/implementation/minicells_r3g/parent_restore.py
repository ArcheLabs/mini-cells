"""Pin the actual resolved HF revision; selectively restore verified immutable Store rows."""

from pathlib import Path
import shutil
from minicells_artifacts.hf_storage import HubBackend
from minicells_r3c_moe_bridge.development import verified_manifest
from minicells_r3c_moe_bridge.checkpoint import Store
from .common import atomic, canonical, digest, parent_identity, read


class Inputs:
    def __init__(self, c, output, local=None, backend=None):
        self.c = c
        self.expected = parent_identity(c)
        self.output = Path(output)
        self.local = None
        self.files = []
        self.path_cache = {}
        if local is not None and (Path(local) / "identity.json").exists():
            local = Path(local).resolve()
            if read(local / "identity.json") != self.expected:
                raise RuntimeError("PARENT_OUTPUT_IDENTITY_MISMATCH")
            self.local = local / "private-checkpoints"
            self.backend = None
            self.reference = {
                "origin": "VERIFIED_LOCAL_KAGGLE_OUTPUT",
                "path": str(local),
                "identity": self.expected,
            }
        else:
            self.backend = backend or HubBackend(c["parent"]["hf_repo"])
            revision = self.backend.info().sha
            if not self.backend.info(revision).private:
                raise RuntimeError("PRIVATE_PARENT_REPO_REQUIRED")
            pointer = self.backend.optional(c["parent"]["pointer"], revision)
            if not pointer or pointer.get("identity") != self.expected:
                raise RuntimeError("PARENT_POINTER_IDENTITY_MISMATCH")
            self.reference = {
                **pointer,
                "hf_repo": self.backend.repo,
                "hf_revision": revision,
                "revision_scope": "RESOLVED_AND_PINNED_FOR_THIS_AUDIT_NOT_ASSUMED_HISTORICAL_HEAD",
            }
            self.manifest = verified_manifest(self.backend, self.reference, self.expected)
        atomic(self.output / "INPUT_REFERENCE.json", self.reference)

    def file(self, name):
        if name in self.path_cache:
            if not self.path_cache[name].is_file():
                raise FileNotFoundError(name)
            return self.path_cache[name]
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError("UNSAFE_INPUT_PATH")
        if self.local is not None:
            path = self.local / rel
            if not path.exists():
                raise FileNotFoundError(name)
            if path.is_symlink():
                raise RuntimeError("UNSAFE_INPUT_PATH")
            before = digest(path)
        else:
            item = self.manifest["files"].get(name)
            if item is None:
                raise FileNotFoundError(name)
            source = self.backend.get(item["object"], self.reference["hf_revision"])
            if source.stat().st_size != item["bytes"] or digest(source) != item["sha256"]:
                raise RuntimeError("PARENT_FILE_HASH_MISMATCH")
            path = self.output / "input-cache" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and digest(path) != item["sha256"]:
                raise RuntimeError("INPUT_CACHE_CONFLICT")
            if not path.exists():
                shutil.copyfile(source, path)
            before = item["sha256"]
        row = {"path": name, "bytes": path.stat().st_size, "sha256": before}
        if row not in self.files:
            self.files.append(row)
        self.path_cache[name] = path
        return path

    def json(self, name):
        return read(self.file(name))

    def checkpoint(self, root, sequence=None):
        if sequence is None:
            row = self.json(root + "/latest.json")
        else:
            latest = self.json(root + "/latest.json")
            name = Store(Path(root), latest["identity"]).receipt_file(sequence).as_posix()
            row = self.json(name)
        ident = row["identity"]
        if any(ident.get(k) != v for k, v in self.expected.items()):
            raise RuntimeError("PARENT_STORE_IDENTITY_MISMATCH")
        arm = root.split("/")[1] if root.startswith("jobs/") else root.split("/")[1]
        if ident.get("arm") != arm or ident.get("seed") != self.c["seed"]:
            raise RuntimeError("PARENT_ARM_OR_SEED_MISMATCH")
        if sequence is not None and row["step"] != sequence:
            raise RuntimeError("PARENT_RECEIPT_SEQUENCE_MISMATCH")
        relative = row["file"]
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeError("UNSAFE_INPUT_PATH")
        statefile = self.file(root + "/" + relative)
        actual_root = statefile
        for _ in Path(relative).parts:
            actual_root = actual_root.parent
        state = Store(actual_root, ident).verified(row)
        return state, {
            "root": root,
            "receipt": row,
            "receipt_identity": ident,
            "state_fingerprint": canonical(
                {"file": row["sha256"], "state": row["state_fingerprint"]}
            ),
        }

    def unchanged(self):
        for row in self.files:
            path = (self.local if self.local is not None else self.output / "input-cache") / row[
                "path"
            ]
            if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise RuntimeError("AUDIT_MUTATED_INPUT_FILE")
        return True
