"""Private immutable analysis results with an independent, CAS-updated task index."""

from pathlib import Path
import tempfile
import time
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
        head, index = self.index(identity)
        key = canonical(identity)
        entry = index["tasks"].get(key)
        if not entry:
            # A session can die after immutable result readback but before index CAS.
            # Adopt the exact saved bytes instead of recomputing an expensive task.
            orphan = self.backend.optional(
                self.prefix(identity) + "/private/" + key + "/result.json", head
            )
            if orphan is None:
                return None
            if orphan.get("identity") != identity:
                raise ContinuationError("ANALYSIS_TASK_IDENTITY_MISMATCH")
            self.commit(orphan)
            return self.completed(identity)
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
        if (
            result.get("optimizer_updates") != 0
            or result.get("training_state_unchanged") is not True
        ):
            raise ContinuationError("ANALYSIS_READ_ONLY_PROOF_REQUIRED")
        return result

    def commit(self, result):
        if (
            result.get("optimizer_updates") != 0
            or result.get("training_state_unchanged") is not True
            or result.get("task_status") != "COMPLETE"
        ):
            raise ContinuationError("ANALYSIS_READ_ONLY_PROOF_REQUIRED")
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
                    result_commit_start = time.perf_counter()
                    result_revision = self.backend.commit(
                        {path: tmp / "result.json"}, head, "private R3b analysis result"
                    )
                    result_commit_seconds = time.perf_counter() - result_commit_start
                    break
                except Exception as error:
                    if not is_conflict(error):
                        raise
            else:
                raise ContinuationError("ANALYSIS_CAS_CONFLICT")
            result_readback_start = time.perf_counter()
            if digest(self.backend.get(path, result_revision)) != sha:
                raise ContinuationError("ANALYSIS_RESULT_HASH_MISMATCH")
            result_readback_seconds = time.perf_counter() - result_readback_start
            entry = {
                "transport_timings": {
                    "result_successful_commit_seconds": result_commit_seconds,
                    "result_readback_seconds": result_readback_seconds,
                },
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
                    index_commit_start = time.perf_counter()
                    revision = self.backend.commit(
                        {prefix + "/index.json": tmp / "index.json"}, head, "R3b analysis index CAS"
                    )
                    index_commit_seconds = time.perf_counter() - index_commit_start
                    index_readback_start = time.perf_counter()
                    saved = self.backend.optional(prefix + "/index.json", revision)
                    if saved != index:
                        raise ContinuationError("ANALYSIS_INDEX_READBACK_FAILED")
                    return {
                        **entry,
                        "transport_timings": {
                            **entry["transport_timings"],
                            "index_successful_commit_seconds": index_commit_seconds,
                            "index_readback_seconds": time.perf_counter() - index_readback_start,
                        },
                    }
                except Exception as error:
                    if not is_conflict(error):
                        raise
            raise ContinuationError("ANALYSIS_CAS_CONFLICT")

    def collect(self, identities):
        return [self.completed(identity) for identity in identities]

    def save_coverage(self, scope, rows, session):
        """Independent authoritative coverage; never writes training latest.json."""
        from .full_analysis import counts

        counts(rows)
        path = self.prefix(scope) + "/coverage.json"
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "coverage.json"
            for _ in range(self.retries):
                head, _ = self.index(scope)
                previous = self.backend.optional(path, head) or {"sessions": []}
                sessions = {r["session_id"]: r for r in previous["sessions"]}
                sessions[session["session_id"]] = session
                value = {
                    "schema": "minicells.r3b.coverage.v1",
                    "scope": scope,
                    "tasks": rows,
                    "counts": counts(rows),
                    "sessions": list(sessions.values()),
                }
                atomic(file, value)
                try:
                    revision = self.backend.commit({path: file}, head, "R3b coverage CAS")
                    if self.backend.optional(path, revision) != value:
                        raise ContinuationError("ANALYSIS_COVERAGE_READBACK_FAILED")
                    return revision, value
                except Exception as error:
                    if not is_conflict(error):
                        raise
            raise ContinuationError("ANALYSIS_CAS_CONFLICT")

    def coverage(self, scope):
        revision, _ = self.index(scope)
        return self.backend.optional(self.prefix(scope) + "/coverage.json", revision)
