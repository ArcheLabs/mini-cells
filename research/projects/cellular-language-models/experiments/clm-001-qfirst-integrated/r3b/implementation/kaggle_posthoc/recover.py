"""Recover original observations; never infer them from a COMPLETE counter."""

from pathlib import Path
import json
from minicells_runtime.store import canonical, digest, read, ContinuationError
from .local_store import proof


class ReadOnlyHFRecovery:
    """A single pinned inventory read; no AnalysisStore.completed orphan repair commits."""

    def __init__(self, backend, scope):
        self.backend = backend
        info = backend.info()
        if not info.private:
            raise ContinuationError("PRIVATE_REPOSITORY_REQUIRED")
        self.revision = info.sha
        self.prefix = "analyses/" + canonical(scope)
        self.index = backend.optional(self.prefix + "/index.json", self.revision) or {"tasks": {}}
        self.commits = 0

    def get(self, identity):
        key = canonical(identity)
        entry = self.index["tasks"].get(key)
        path = self.prefix + "/private/" + key + "/result.json"
        revision = self.revision
        if entry:
            if (
                entry.get("identity") != identity
                or entry.get("task_status") != "COMPLETE"
                or entry.get("hf_path") != path
                or entry.get("hf_repo") != self.backend.repo
            ):
                raise ContinuationError("HF_RECOVERY_IDENTITY_MISMATCH")
            revision = entry["hf_revision"]
            if len(revision) != 40:
                raise ContinuationError("PINNED_HF_REFERENCE_REQUIRED")
            file = self.backend.get(path, revision)
            if digest(file) != entry["sha256"]:
                raise ContinuationError("RESULT_HASH_MISMATCH")
            result = read(file)
        else:
            result = self.backend.optional(path, revision)
            if result is None:
                return None
        proof(result, identity)
        return result, {
            "transport": "HF_READ_ONLY",
            "hf_revision": revision,
            "hf_path": path,
            "orphan_recovered": not bool(entry),
            "source_sha256": entry["sha256"] if entry else canonical(result),
        }


def scan_raw_results(root, identities, store, provenance):
    """Only task-shaped JSON is admitted. Aggregates/HTML/counters never count."""
    planned = {canonical(t): t for t in identities}
    reused, rejected = set(), []
    for path in sorted(Path(root).rglob("*.json")):
        if path.is_symlink() or path.stat().st_size > 512 * 1024 * 1024:
            continue
        try:
            if path.parent.name == "sha256" and len(path.stem) == 64 and digest(path) != path.stem:
                raise ContinuationError("RESULT_HASH_MISMATCH")
            result = read(path)
            if (
                not isinstance(result, dict)
                or result.get("task_status") != "COMPLETE"
                or "result" not in result
            ):
                continue
            identity = result.get("identity")
            if not isinstance(identity, dict):
                continue
            key = canonical(identity)
            if key not in planned:
                rejected.append(
                    {"reason": "OLD_ENGINE_OR_TASK_IDENTITY_MISMATCH", "file_sha256": digest(path)}
                )
                continue
            proof(result, planned[key])
            store.commit(
                result,
                {**provenance, "source_file_sha256": digest(path)},
                persistence="KAGGLE_VERSION_PERSISTED",
            )
            reused.add(key)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        except Exception as error:
            rejected.append(
                {
                    "reason": getattr(error, "code", type(error).__name__),
                    "file_sha256": digest(path),
                }
            )
    return reused, rejected
