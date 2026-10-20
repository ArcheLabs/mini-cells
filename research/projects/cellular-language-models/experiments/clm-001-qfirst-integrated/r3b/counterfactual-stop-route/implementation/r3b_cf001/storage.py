"""Immutable chunk receipts, cumulative private packets and saved-version readback."""

from pathlib import Path
import json
import os
import tempfile
from minicells_runtime.store import atomic, canonical, digest, read, ContinuationError
from kaggle_posthoc.kaggle_source import relative


class Chunks:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.index = {}
        for p in sorted((self.root / "receipts").glob("*.json")):
            row = read(p)
            key = canonical(row["identity"])
            if p.stem != key or row["identity"]["run"] != identity:
                raise ContinuationError("CF_CHUNK_IDENTITY_CONFLICT")
            self._read(row)
            self.index[key] = row
        # Crash between object and receipt: recover only fully hashed matching objects.
        for p in sorted((self.root / "objects").glob("*.json")):
            value = read(p)
            if digest(p) != p.stem:
                raise ContinuationError("CF_CHUNK_HASH_MISMATCH")
            if value.get("identity", {}).get("run") != identity:
                raise ContinuationError("CF_CHUNK_IDENTITY_CONFLICT")
            if canonical(value["payload"]) != value["payload_sha256"]:
                raise ContinuationError("CF_CHUNK_CONTENT_MISMATCH")
            self.put(value["identity"]["task"], value["payload"])
        for p in sorted((self.root / "readback").glob("*.json")):
            witness = read(p)
            key = witness["chunk_key"]
            if (
                p.stem != canonical(witness)
                or key not in self.index
                or self.index[key]["sha256"] != witness["sha256"]
            ):
                raise ContinuationError("CF_READBACK_WITNESS_MISMATCH")
            self.index[key].update(
                cross_session_persistence_verified=True, readback_origin=witness["origin"]
            )

    def _read(self, row):
        file = self.root / relative(row["file"])
        if (
            any(p.is_symlink() for p in (file, *file.parents))
            or self.root.resolve() not in file.resolve().parents
            or digest(file) != row["sha256"]
        ):
            raise ContinuationError("CF_CHUNK_HASH_MISMATCH")
        value = read(file)
        if (
            value["identity"] != row["identity"]
            or canonical(value["payload"]) != value["payload_sha256"]
        ):
            raise ContinuationError("CF_CHUNK_CONTENT_MISMATCH")
        return value["payload"]

    def key(self, task):
        return {"run": self.identity, "task": task}

    def get(self, task):
        row = self.index.get(canonical(self.key(task)))
        return self._read(row) if row else None

    def put(self, task, payload):
        identity = self.key(task)
        key = canonical(identity)
        if key in self.index:
            old = self._read(self.index[key])
            if canonical(old) != canonical(payload):
                raise ContinuationError("CF_CHUNK_IMMUTABLE_CONFLICT")
            return old
        value = {"identity": identity, "payload": payload, "payload_sha256": canonical(payload)}
        data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        import hashlib

        sha = hashlib.sha256(data).hexdigest()
        name = "objects/" + sha + ".json"
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as h:
            temp = Path(h.name)
            h.write(data)
            h.flush()
            os.fsync(h.fileno())
        try:
            try:
                os.link(temp, path)
            except FileExistsError:
                if digest(path) != sha:
                    raise ContinuationError("CF_CHUNK_IMMUTABLE_CONFLICT")
        finally:
            temp.unlink()
        row = {
            "identity": identity,
            "sha256": sha,
            "file": name,
            "result_data_verified": True,
            "cross_session_persistence_verified": False,
        }
        atomic(self.root / "receipts" / (key + ".json"), row)
        self.index[key] = row
        self._read(row)
        atomic(self.root / "index.json", {"chunks": self.index})
        return payload

    def import_saved(self, root, origin):
        root = Path(root)
        count = 0
        for p in sorted(root.rglob("receipts/*.json")):
            row = read(p)
            if row.get("identity", {}).get("run") != self.identity:
                continue
            other = Chunks.__new__(Chunks)
            other.root = p.parent.parent
            value = other._read(row)
            self.put(row["identity"]["task"], value)
            key = canonical(row["identity"])
            witness = {"chunk_key": key, "sha256": self.index[key]["sha256"], "origin": origin}
            atomic(self.root / "readback" / (canonical(witness) + ".json"), witness)
            self.index[key].update(cross_session_persistence_verified=True, readback_origin=origin)
            count += 1
        atomic(self.root / "index.json", {"chunks": self.index})
        return count

    def snapshot(self, output):
        output = Path(output)
        files = {
            p.relative_to(output).as_posix(): digest(p)
            for p in sorted(self.root.rglob("*.json"))
            if p.is_file()
        }
        value = {
            "schema": "minicells.cf001.snapshot.v1",
            "identity": self.identity,
            "files": files,
            "output_saved_version": "OUTPUT_VERSION_PENDING_SAVE",
            "training_updates": 0,
        }
        atomic(output / "cf-snapshot.json", value)
        return value
