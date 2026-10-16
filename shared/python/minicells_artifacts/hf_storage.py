"""Private HF revision storage; local checkpoint/event semantics stay in the adapter.

Reuse Native CLM M2/M3 HfApi + auth_check and M1 pinned hf_hub_download.
Content-addressed objects avoid duplicate assets and checkpoint overwrites.
Only explicit trusted repositories are accepted; no pickle loading happens here.
"""

import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import time
import uuid
from minicells_runtime.store import (
    ContinuationError,
    MountedInputStore,
    atomic,
    canonical,
    digest,
    read,
    safe,
)


def relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContinuationError("UNSAFE_PATH")
    if PurePosixPath(value).is_absolute() or any(x in ("", ".", "..") for x in value.split("/")):
        raise ContinuationError("UNSAFE_PATH")
    return value


def error_code(error):
    # Never serialize exception text: it may contain signed URLs or credentials.
    return {
        "type": type(error).__name__,
        "http_status": getattr(getattr(error, "response", None), "status_code", None),
    }


class HubBackend:
    def __init__(self, repo_id, token=None):
        from huggingface_hub import HfApi

        self.token = token or os.environ.get("HF_TOKEN", "").strip()
        if not self.token:
            raise ContinuationError("HF_TOKEN_MISSING")
        self.api = HfApi(token=self.token)
        self.repo = repo_id if "/" in repo_id else self.api.whoami()["name"] + "/" + repo_id

    def info(self, revision=None):
        return self.api.repo_info(self.repo, repo_type="model", revision=revision)

    def get(self, path, revision):
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=self.repo,
                repo_type="model",
                filename=relative(path),
                revision=revision,
                token=self.token,
            )
        )

    def optional(self, path, revision):
        from huggingface_hub.utils import EntryNotFoundError

        try:
            return read(self.get(path, revision))
        except EntryNotFoundError:
            return None

    def commit(self, files, parent, message):
        from huggingface_hub import CommitOperationAdd

        return self.api.create_commit(
            repo_id=self.repo,
            repo_type="model",
            parent_commit=parent,
            operations=[
                CommitOperationAdd(path_in_repo=relative(k), path_or_fileobj=str(v))
                for k, v in files.items()
            ],
            commit_message=message,
        ).oid

    def check_write(self):
        from huggingface_hub import auth_check

        # Older Kaggle SDKs lack write=; the actual create_commit/readback below
        # remains the write-permission proof in both SDK generations.
        import inspect

        kwargs = {"write": True} if "write" in inspect.signature(auth_check).parameters else {}
        auth_check(self.repo, repo_type="model", token=self.token, **kwargs)


def is_conflict(error):
    return getattr(getattr(error, "response", None), "status_code", None) in (409, 412)


class HFCheckpointStore:
    def __init__(self, backend, retries=3):
        self.backend = backend
        self.repo = backend.repo
        self.retries = retries

    def preflight(self):
        try:
            if not self.backend.info().private:
                raise ContinuationError("HF_PRIVATE_REPO_REQUIRED")
            self.backend.check_write()
            # Actual tiny write + pinned read; no training or large transfer.
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "probe.json"
                nonce = uuid.uuid4().hex
                atomic(path, {"nonce": nonce, "purpose": "storage-preflight"})
                revision = self.backend.commit(
                    {"engineering-preflight/" + nonce + ".json": path},
                    self.backend.info().sha,
                    "engineering storage preflight",
                )
                if digest(
                    self.backend.get("engineering-preflight/" + nonce + ".json", revision)
                ) != digest(path):
                    raise ContinuationError("HF_READBACK_MISMATCH")
            return {"status": "PREFLIGHT_PASSED", "hf_repo": self.repo, "hf_revision": revision}
        except Exception as error:
            if isinstance(error, ContinuationError):
                raise
            raise ContinuationError("STORAGE_BLOCKED", json.dumps(error_code(error))) from None

    def prefix(self, meta):
        # Hash identity instead of allowing experiment labels to form filesystem paths.
        return "runs/" + canonical(
            {
                k: meta[k]
                for k in (
                    "experiment_id",
                    "run_id",
                    "source_sha",
                    "runtime_config_sha256",
                    "protocol_sha256",
                    "assets_manifest_sha256",
                    "tokenizer_sha256",
                )
            }
        )

    def locate(self, source_sha, run_id):
        path = "run-index/" + canonical({"source_sha": source_sha, "run_id": run_id}) + ".json"
        revision = self.backend.info().sha
        pointer = self.backend.optional(path, revision)
        if pointer is None:
            raise ContinuationError("HF_RUN_NOT_FOUND")
        return self.lookup(pointer["run_key"], revision)

    def lookup(self, run_key, revision=None):
        relative(run_key)
        resolved = revision or self.backend.info().sha
        pointer = self.backend.optional(run_key + "/latest.json", resolved)
        if pointer is None:
            raise ContinuationError("HF_RUN_NOT_FOUND")
        return dict(pointer, hf_repo=self.repo, hf_revision=resolved, hf_repo_type="model")

    def publish_committed_checkpoint(self, root, *, readback=False):
        root = Path(root)
        meta = MountedInputStore(root).verify()
        prefix = self.prefix(meta)
        files = read(root / meta["files_manifest"])["files"]
        # Freeze source bytes before any network I/O. The adapter calls only at a
        # safe boundary; versioned selectors are used instead of mutable latest.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            objects = {}
            mapping = {}
            committed_paths = {
                record["checkpoint"]
                for progress in meta["arms"].values()
                for record in progress["checkpoints"]
            }
            for name, row in files.items():
                if name.endswith((".orphan", ".tmp")) or (
                    Path(name).name.startswith("checkpoint-step-")
                    and name.endswith(".pt")
                    and name not in committed_paths
                ):
                    raise ContinuationError("HF_UNCOMMITTED_OBJECT_REJECTED", name)
                physical = row.get("storage_path", name)
                local = safe(root, physical)
                key = "objects/sha256/" + row["sha256"]
                target = tmp / row["sha256"]
                if not target.exists():
                    shutil.copyfile(local, target)
                if digest(target) != row["sha256"]:
                    raise ContinuationError("CHECKPOINT_CHANGED_DURING_UPLOAD")
                objects[key] = target
                mapping[physical] = dict(row, object=key)
            for name in (meta["files_manifest"], "continuation.json"):
                target = tmp / (canonical(name) + ".json")
                shutil.copyfile(root / name, target)
                sha = digest(target)
                key = "objects/sha256/" + sha
                objects[key] = target
                mapping[name] = {"sha256": sha, "bytes": target.stat().st_size, "object": key}
            snapshot = {
                "schema": "minicells.hf-bundle.v1",
                "identity": {
                    k: meta[k]
                    for k in (
                        "run_id",
                        "experiment_id",
                        "source_sha",
                        "runtime_config_sha256",
                        "protocol_sha256",
                        "assets_manifest_sha256",
                        "tokenizer_sha256",
                    )
                },
                "files": mapping,
                "arms": meta["arms"],
                "complete": meta["complete"],
            }
            snap_path = prefix + "/bundles/" + canonical(snapshot) + ".json"
            document = tmp / "bundle.json"
            atomic(document, snapshot)
            pointer = {
                "bundle_path": snap_path,
                "bundle_sha256": digest(document),
                "run_key": prefix,
                "arms": {
                    a: {
                        "step": p["step"],
                        "revision": p["revision"],
                        "sha256": p["checkpoint_sha256"],
                        "event_cursor": p["event_cursor"],
                    }
                    for a, p in meta["arms"].items()
                },
            }
            ptr = tmp / "pointer.json"
            atomic(ptr, pointer)
            for attempt in range(self.retries):
                try:
                    head = self.backend.info().sha
                    previous = self.backend.optional(prefix + "/latest.json", head)
                    if previous:
                        for arm, old in previous["arms"].items():
                            new = pointer["arms"].get(arm)
                            if new is None or (new["step"], new["revision"]) < (
                                old["step"],
                                old["revision"],
                            ):
                                raise ContinuationError("HF_LATEST_ROLLBACK_REJECTED", arm)
                            if (new["step"], new["revision"]) == (
                                old["step"],
                                old["revision"],
                            ) and new["sha256"] != old["sha256"]:
                                raise ContinuationError("HF_CHECKPOINT_ID_COLLISION", arm)
                        if previous["bundle_sha256"] == pointer["bundle_sha256"]:
                            if readback:
                                for key, local in objects.items():
                                    if digest(self.backend.get(key, head)) != digest(local):
                                        raise ContinuationError("HF_READBACK_MISMATCH")
                            return self._index(
                                meta, snapshot, dict(pointer, hf_revision=head), mapping
                            )
                    # Existing content-addressed objects must retain identical bytes.
                    existing = set(
                        self.backend.info(head).siblings
                        and [x.rfilename for x in self.backend.info(head).siblings]
                        or []
                    )
                    additions = {k: v for k, v in objects.items() if k not in existing}
                    locator = (
                        "run-index/"
                        + canonical({"source_sha": meta["source_sha"], "run_id": meta["run_id"]})
                        + ".json"
                    )
                    previous_locator = self.backend.optional(locator, head)
                    if previous_locator and previous_locator["run_key"] != prefix:
                        raise ContinuationError("HF_RUN_IDENTITY_COLLISION")
                    additions[locator] = ptr
                    additions[snap_path] = document
                    additions[prefix + "/latest.json"] = ptr
                    for arm in pointer["arms"]:
                        additions[prefix + "/arms/" + relative(arm) + "/latest.json"] = ptr
                    revision = self.backend.commit(
                        additions, head, "durable checkpoint " + meta["run_id"]
                    )
                    # Prove all paths exist in that exact commit, not mutable main.
                    present = {x.rfilename for x in self.backend.info(revision).siblings}
                    if set(objects) - present or snap_path not in present:
                        raise ContinuationError("HF_REMOTE_FILES_MISSING")
                    if digest(self.backend.get(snap_path, revision)) != digest(document):
                        raise ContinuationError("HF_READBACK_MISMATCH")
                    if readback:
                        for key, local in objects.items():
                            if digest(self.backend.get(key, revision)) != digest(local):
                                raise ContinuationError("HF_READBACK_MISMATCH")
                    return self._index(meta, snapshot, dict(pointer, hf_revision=revision), mapping)
                except Exception as error:
                    if is_conflict(error) and attempt + 1 < self.retries:
                        continue
                    if isinstance(error, ContinuationError):
                        raise
                    if attempt + 1 < self.retries:
                        time.sleep(0.1 * 2**attempt)
                        continue
                    raise ContinuationError(
                        "STORAGE_BLOCKED", json.dumps(error_code(error))
                    ) from None

    def _index(self, meta, snapshot, ref, mapping):
        records = []
        for arm, progress in meta["arms"].items():
            for cp in progress["checkpoints"]:
                row = mapping[cp["checkpoint"]]
                records.append(
                    {
                        "arm": arm,
                        "step": cp["step"],
                        "event_revision": cp["revision"],
                        "hf_repo": self.repo,
                        "hf_repo_type": "model",
                        "hf_revision": ref["hf_revision"],
                        "hf_path": row["object"],
                        "sha256": row["sha256"],
                        "bytes": row["bytes"],
                        "event_cursor_sha256": canonical(cp.get("event_cursor", {})),
                    }
                )
        return {
            "schema": "minicells.checkpoint-index.v1",
            **snapshot["identity"],
            "status": "DURABLE_COMPLETE" if meta["complete"] else "DURABLE_PARTIAL",
            "checkpoint_ref": {**ref, "hf_repo": self.repo, "hf_repo_type": "model"},
            "checkpoints": records,
            "scientific_decision": False,
            "remote_readback": "RESUME_ALWAYS_FULL_SHA256",
        }

    def fetch_verified_checkpoint(self, ref, destination, expected_source=None):
        if ref.get("hf_repo") != self.repo or ref.get("hf_repo_type") != "model":
            raise ContinuationError("HF_REPO_IDENTITY_MISMATCH")
        revision = ref.get("hf_revision", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ContinuationError("HF_PINNED_REVISION_REQUIRED")
        destination = Path(destination)
        if destination.exists():
            raise ContinuationError("OUTPUT_ALREADY_EXISTS")
        try:
            if not self.backend.info().private:
                raise ContinuationError("HF_PRIVATE_REPO_REQUIRED")
            bundle = self.backend.get(relative(ref["bundle_path"]), revision)
            if digest(bundle) != ref["bundle_sha256"]:
                raise ContinuationError("HF_BUNDLE_HASH_MISMATCH")
            manifest = read(bundle)
            if manifest.get("schema") != "minicells.hf-bundle.v1":
                raise ContinuationError("HF_BUNDLE_SCHEMA_INVALID")
            if expected_source and manifest["identity"]["source_sha"] != expected_source:
                raise ContinuationError("SOURCE_SHA_MISMATCH")
            temporary = destination.with_name(destination.name + ".fetching")
            if temporary.exists():
                raise ContinuationError("OUTPUT_ALREADY_EXISTS")
            temporary.mkdir(parents=True)
            try:
                for name, row in manifest["files"].items():
                    target = temporary / relative(name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = self.backend.get(relative(row["object"]), revision)
                    if source.stat().st_size != row["bytes"] or digest(source) != row["sha256"]:
                        raise ContinuationError("CHECKPOINT_HASH_MISMATCH", name)
                    shutil.copyfile(source, target)
                local = MountedInputStore(temporary).verify(expected_source)
                for k, value in manifest["identity"].items():
                    if local.get(k) != value:
                        raise ContinuationError("IDENTITY_MISMATCH", k)
                os.replace(temporary, destination)
            except Exception:
                shutil.rmtree(temporary)
                raise
            return local
        except ContinuationError:
            raise
        except Exception as error:
            raise ContinuationError("STORAGE_BLOCKED", json.dumps(error_code(error))) from None
