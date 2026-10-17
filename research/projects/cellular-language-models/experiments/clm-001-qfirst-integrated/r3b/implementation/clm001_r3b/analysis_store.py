"""Private immutable analysis results with an independent, CAS-updated task index."""

from pathlib import Path
import tempfile
from minicells_runtime.store import atomic, canonical, digest, read, ContinuationError
from minicells_artifacts.hf_storage import is_conflict, relative


class AnalysisStore:
    def __init__(self, backend, retries=3):
        self.backend, self.retries = backend, retries

    def prefix(self, identity):
        return "analyses/" + canonical(
            {
                "analysis_source_sha": identity["analysis_source_sha"],
                "analysis_protocol_sha256": identity["analysis_protocol_sha256"],
                "training_run_id": identity["training_run_id"],
                "training_source_sha": identity["training_source_sha"],
            }
        )

    def index(self, identity, revision=None):
        info = self.backend.info(revision)
        if not info.private:
            raise ContinuationError("PRIVATE_REPOSITORY_REQUIRED")
        path = self.prefix(identity) + "/index.json"
        value = self.backend.optional(path, info.sha)
        return info.sha, value or {"schema": "minicells.r3b.analysis-index.v1", "tasks": {}}

    def completed(self, identity):
        _, index = self.index(identity)
        key = canonical(identity)
        entry = index["tasks"].get(key)
        if not entry:
            return None
        if entry["identity"] != identity or entry["task_status"] != "COMPLETE":
            raise ContinuationError("ANALYSIS_TASK_IDENTITY_MISMATCH")
        if (
            entry["hf_repo"] != self.backend.repo
            or entry["hf_path"] != self.prefix(identity) + "/private/" + key + "/result.json"
            or len(entry["hf_revision"]) != 40
        ):
            raise ContinuationError("ANALYSIS_INDEX_POINTER_INVALID")
        private = self.backend.get(relative(entry["hf_path"]), entry["hf_revision"])
        if digest(private) != entry["sha256"]:
            raise ContinuationError("ANALYSIS_RESULT_HASH_MISMATCH")
        result = read(private)
        if result["identity"] != identity:
            raise ContinuationError("ANALYSIS_TASK_IDENTITY_MISMATCH")
        return result

    def commit(self, result):
        identity = result["identity"]
        key = canonical(identity)
        prefix = self.prefix(identity)
        path = prefix + "/private/" + key + "/result.json"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            atomic(tmp / "result.json", result)
            sha = digest(tmp / "result.json")
            # Result commit precedes the authoritative COMPLETE index. A crash
            # cannot expose a COMPLETE task whose bytes do not exist remotely.
            for _ in range(self.retries):
                head, index = self.index(identity)
                old = index["tasks"].get(key)
                if old:
                    if old["sha256"] != sha:
                        raise ContinuationError("ANALYSIS_TASK_RESULT_CONFLICT")
                    self.completed(identity)
                    return old
                try:
                    result_revision = self.backend.commit(
                        {path: tmp / "result.json"}, head, "private R3b analysis result"
                    )
                    break
                except Exception as error:
                    if not is_conflict(error):
                        raise
            else:
                raise ContinuationError("ANALYSIS_CAS_CONFLICT")
            if digest(self.backend.get(path, result_revision)) != sha:
                raise ContinuationError("ANALYSIS_RESULT_HASH_MISMATCH")
            entry = {
                "task_status": "COMPLETE",
                "identity": identity,
                "sha256": sha,
                "hf_repo": self.backend.repo,
                "hf_revision": result_revision,
                "hf_path": path,
            }
            for _ in range(self.retries):
                head, index = self.index(identity)
                old = index["tasks"].get(key)
                if old:
                    if old["sha256"] != sha:
                        raise ContinuationError("ANALYSIS_TASK_RESULT_CONFLICT")
                    return old
                index["tasks"][key] = entry
                atomic(tmp / "index.json", index)
                try:
                    revision = self.backend.commit(
                        {prefix + "/index.json": tmp / "index.json"}, head, "R3b analysis index CAS"
                    )
                    saved = self.backend.optional(prefix + "/index.json", revision)
                    if saved != index:
                        raise ContinuationError("ANALYSIS_INDEX_READBACK_FAILED")
                    return entry
                except Exception as error:
                    if not is_conflict(error):
                        raise
            raise ContinuationError("ANALYSIS_CAS_CONFLICT")

    def collect(self, identities):
        return [self.completed(identity) for identity in identities]
