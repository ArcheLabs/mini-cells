"""Byte-verified continuation bundles. Hashes attest integrity, not untrusted provenance.

Only load checkpoints from this project's trusted private Kaggle Outputs. Verification
is deliberately stdlib-only and must finish before any torch deserialization.
"""

from pathlib import Path, PurePosixPath
from abc import ABC, abstractmethod
import hashlib
import json
import os
import shutil

SCHEMA = 1
DEFAULT_LIMIT = 16 * 1024**3


class ContinuationError(RuntimeError):
    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(code + (": " + str(detail) if detail else ""))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ContinuationError("MANIFEST_INVALID", str(error)) from error


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def safe(root, relative):
    root = Path(root)
    if not isinstance(relative, str):
        raise ContinuationError("UNSAFE_PATH", relative)
    p = PurePosixPath(relative)
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or p.is_absolute()
        or any(v in (".", "..") for v in relative.split("/"))
    ):
        raise ContinuationError("UNSAFE_PATH", relative)
    # Check ancestors, including the mount root; never resolve a symlink then accept it.
    candidate = root / relative
    for part in (candidate,) + tuple(candidate.parents):
        if part.is_symlink():
            raise ContinuationError("UNSAFE_PATH", relative)
    if not candidate.is_file():
        raise ContinuationError("CHECKPOINT_FILE_MISSING", relative)
    return candidate


def inventory(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if "continuation-records" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise ContinuationError("UNSAFE_PATH", path.relative_to(root))
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if name != "continuation.json" and not path.name.startswith("continuation-manifest-"):
                files[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return files


def capacity(files, limit):
    total = sum(row["bytes"] for row in files.values())
    largest = sorted(files.items(), key=lambda pair: pair[1]["bytes"], reverse=True)[:10]
    result = {"bytes": total, "limit_bytes": limit, "largest": largest}
    if total > limit:
        raise ContinuationError("OUTPUT_SIZE_BUDGET_EXCEEDED", json.dumps(result))
    return result


class CheckpointStore(ABC):
    @abstractmethod
    def verify(self, expected_source=None):
        """Verify the entire package before exposing checkpoint paths."""

    @abstractmethod
    def select(self, arm, step=None, revision=None):
        """Select an exact committed checkpoint; never glob a first match."""


class MountedInputStore(CheckpointStore):
    """Read-only access. Never writes even when validation fails."""

    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_dir():
            raise ContinuationError("RESUME_INPUT_MISSING", root)

    def verify(self, expected_source=None, _metadata=None):
        meta = _metadata if _metadata is not None else read(safe(self.root, "continuation.json"))
        if meta.get("schema_version") != SCHEMA or meta.get("status") not in ("PAUSED", "COMPLETE"):
            raise ContinuationError("CONTINUATION_NOT_COMMITTED")
        if expected_source is not None and meta.get("source_sha") != expected_source:
            raise ContinuationError("SOURCE_SHA_MISMATCH")
        mpath = safe(self.root, meta.get("files_manifest", ""))
        if digest(mpath) != meta.get("files_manifest_sha256"):
            raise ContinuationError("MANIFEST_HASH_MISMATCH")
        files = read(mpath)["files"]
        for name, row in files.items():
            path = safe(self.root, row.get("storage_path", name))
            if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise ContinuationError("CHECKPOINT_HASH_MISMATCH", name)

        def checked(name):
            if name not in files:
                raise ContinuationError("MANIFEST_COVERAGE_MISSING", name)
            return safe(self.root, files[name].get("storage_path", name))

        for name, field in (
            ("candidate-config.json", "candidate_config_sha256"),
            ("runtime-config.json", "runtime_config_sha256"),
            ("protocol.md", "protocol_sha256"),
            ("assets/manifest.json", "assets_manifest_sha256"),
            ("assets/tokenizer.json", "tokenizer_sha256"),
        ):
            if digest(checked(name)) != meta.get(field):
                raise ContinuationError("IDENTITY_MISMATCH", field)
        lock = read(checked("experiment.lock.json"))
        for field in (
            "run_id",
            "experiment_id",
            "source_sha",
            "protocol_sha256",
            "candidate_config_sha256",
            "runtime_config_sha256",
            "assets_manifest_sha256",
            "tokenizer_sha256",
        ):
            if lock.get(field) != meta.get(field):
                raise ContinuationError("IDENTITY_MISMATCH", field)
        runtime = read(checked("runtime-config.json"))
        if runtime.get("execution", {}).get("source_sha") != meta["source_sha"]:
            raise ContinuationError("SOURCE_SHA_MISMATCH", "runtime")
        for arm, progress in meta["arms"].items():
            identity = read(checked("arms/" + arm + "/identity.json"))
            for record in progress["checkpoints"]:
                cp = checked(record["checkpoint"])
                side = read(checked(record["sidecar"]))
                if (
                    digest(cp) != record["checkpoint_sha256"]
                    or side["sha256"] != record["checkpoint_sha256"]
                    or side["identity"] != identity
                ):
                    raise ContinuationError("CHECKPOINT_HASH_MISMATCH", record["checkpoint"])
            latest = read(checked("arms/" + arm + "/latest.json"))
            if (
                latest["file"] != Path(progress["checkpoint"]).name
                or latest["sha256"] != progress["checkpoint_sha256"]
                or latest["identity"] != identity
            ):
                raise ContinuationError("EVENT_STATE_INCONSISTENT", arm)
            probe = read(checked("arms/" + arm + "/frozen-probe-bank.json"))
            pp = checked("arms/" + arm + "/frozen-probe-bank.pt")
            if (
                probe.get("identity") != identity
                or digest(pp) != progress["probe_bank_sha256"]
                or probe["sha256"] != progress["probe_bank_sha256"]
            ):
                raise ContinuationError("FROZEN_PROBE_MISSING", arm)
        self.meta = meta
        return meta

    def select(self, arm, step=None, revision=None):
        meta = self.verify()
        if arm not in meta["arms"]:
            raise ContinuationError("ARM_NOT_FOUND", arm)
        records = meta["arms"][arm]["checkpoints"]
        if step is None:
            path = meta["arms"][arm]["checkpoint"]
            return next(v for v in records if v["checkpoint"] == path)
        candidates = [
            v
            for v in records
            if v["step"] == step and (revision is None or v["revision"] == revision)
        ]
        if not candidates:
            raise ContinuationError("CHECKPOINT_STEP_NOT_FOUND", (arm, step, revision))
        # Pick latest committed event revision at that step, explicitly from the index.
        return max(candidates, key=lambda v: v["revision"])

    def materialize(self, destination, expected_source=None):
        meta = self.verify(expected_source)
        destination = Path(destination)
        if destination.exists():
            raise ContinuationError("OUTPUT_ALREADY_EXISTS", destination)
        # Copy only bytes covered by the final commit, never mutable/uncommitted extras.
        temporary = destination.with_name(destination.name + ".importing")
        if temporary.exists():
            raise ContinuationError("OUTPUT_ALREADY_EXISTS", temporary)
        temporary.mkdir(parents=True)
        try:
            names = read(self.root / meta["files_manifest"])["files"]
            for name in list(names) + [meta["files_manifest"], "continuation.json"]:
                target = temporary / name
                target.parent.mkdir(parents=True, exist_ok=True)
                physical = names.get(name, {}).get("storage_path", name)
                shutil.copyfile(safe(self.root, physical), target)
                if physical != name:
                    stored = temporary / physical
                    stored.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(safe(self.root, physical), stored)
            MountedInputStore(temporary).verify(expected_source)
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary)
            raise
        return meta


class LocalOutputStore(MountedInputStore):
    """Publish a final continuation pointer only after every safe-boundary file exists."""

    def commit(self, meta, *, limit_bytes=DEFAULT_LIMIT, crash=None):
        files = inventory(self.root)
        report = capacity(files, limit_bytes)
        # Version only mutable selector/status metadata. Weight/data assets remain
        # single-copy immutable files. Old commit pointers survive a crash while
        # latest.json advances to the next event revision.
        for name, row in files.items():
            if Path(name).name in (
                "latest.json",
                "run-status.json",
                "segment-summary.json",
                "publication-state.json",
                "resource-estimate.json",
                "parent-input.json",
                "failure.json",
            ):
                physical = "continuation-records/" + row["sha256"] + ".json"
                target = self.root / physical
                if not target.exists():
                    target.parent.mkdir(exist_ok=True)
                    shutil.copyfile(safe(self.root, name), target)
                row["storage_path"] = physical
        document = {"schema_version": SCHEMA, "files": files}
        manifest_name = "continuation-manifest-" + canonical(document) + ".json"
        meta = dict(meta, schema_version=SCHEMA, files_manifest=manifest_name)
        atomic(self.root / manifest_name, document)
        if crash:
            crash("manifest")
        meta["files_manifest_sha256"] = digest(self.root / manifest_name)
        self.verify(meta["source_sha"], _metadata=meta)
        actual = {
            p.relative_to(self.root).as_posix(): {"bytes": p.stat().st_size}
            for p in self.root.rglob("*")
            if p.is_file() and not p.is_symlink()
        }
        actual["continuation.json"] = {"bytes": len(json.dumps(meta, indent=2).encode()) + 4096}
        report = capacity(actual, limit_bytes)
        meta["output_capacity"] = report
        if crash:
            crash("continuation")
        atomic(self.root / "continuation.json", meta)
        return dict(meta, capacity=report)
