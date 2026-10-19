"""Cumulative private results, atomic local commits, and explicit saved-version provenance."""

from pathlib import Path
import json
import os
import tempfile
from minicells_runtime.store import atomic, canonical, digest, read, ContinuationError


def proof(result, identity):
    if (
        result.get("identity") != identity
        or result.get("task_status") != "COMPLETE"
        or result.get("optimizer_updates") != 0
        or result.get("training_state_unchanged") is not True
    ):
        raise ContinuationError("RESULT_IDENTITY_OR_READ_ONLY_PROOF_INVALID")
    if not isinstance(result.get("sample_count"), int) or result["sample_count"] < 0:
        raise ContinuationError("RESULT_COVERAGE_INVALID")
    payload = result.get("result", {})
    stage = identity.get("stage")
    try:
        if stage == "quality":
            observed = payload["quality"]
            if set(observed) != set(identity["policies"]):
                raise ValueError("policy coverage")
            for row in observed.values():
                if len(row["nll"]) != len(row["groups"]) or len(row["cost"]) != len(row["nll"]):
                    raise ValueError("sample coverage")
            samples = sum(len(row["nll"]) for row in observed.values())
        elif stage == "functional":
            samples = len(payload["functional"]["state_records"])
            for key in ("records", "summary", "crossover"):
                payload["functional"][key]
        elif stage == "action":
            samples = len(payload["router"])
            payload["action_type_calibration"]
        else:
            raise ValueError("stage")
        if samples != result["sample_count"]:
            raise ValueError("sample count")
    except (KeyError, TypeError, ValueError):
        raise ContinuationError("RESULT_RAW_OBSERVATION_COVERAGE_INVALID") from None


def safe_file(root, name):
    root = Path(root).resolve()
    name = Path(name)
    if name.is_absolute() or ".." in name.parts:
        raise ContinuationError("UNSAFE_RESULT_PATH")
    target = root / name
    if any(p.is_symlink() for p in (target, *target.parents) if p != root.parent):
        raise ContinuationError("UNSAFE_RESULT_PATH")
    if root not in target.resolve().parents:
        raise ContinuationError("UNSAFE_RESULT_PATH")
    return target


class LocalResultStore:
    def __init__(self, root, context=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        context_path = self.root / "store-context.json"
        if context is not None and not context_path.exists():
            atomic(context_path, context)
        self.context = (
            read(context_path)
            if context_path.exists()
            else {"transport": "LOCAL_UNCOMMITTED_RECOVERY"}
        )
        self.index_path = self.root / "analysis-index.json"
        self.index = (
            read(self.index_path)
            if self.index_path.exists()
            else {"schema": "minicells.r3b.local-results.v1", "tasks": {}}
        )
        self.rebuild()

    def rebuild(self):
        # Each immutable receipt is the commit point; an index write may have crashed.
        for path in sorted((self.root / "receipts").glob("*.json")):
            if path.is_symlink():
                raise ContinuationError("UNSAFE_RESULT_PATH")
            entry = read(path)
            key = canonical(entry["identity"])
            if path.stem != key:
                raise ContinuationError("RESULT_RECEIPT_IDENTITY_MISMATCH")
            old = self.index["tasks"].get(key)
            if old and old["sha256"] != entry["sha256"]:
                raise ContinuationError("LOCAL_RESULT_CONFLICT")
            self._read(entry)
            self.index["tasks"][key] = entry
        for path in sorted((self.root / "objects/sha256").glob("*.json")):
            if path.is_symlink() or digest(path) != path.stem:
                raise ContinuationError("RESULT_HASH_MISMATCH")
            result = read(path)
            proof(result, result["identity"])
            key = canonical(result["identity"])
            if key not in self.index["tasks"]:
                self.commit(result, self.context, persistence="COMPUTED_LOCAL")
        for path in sorted((self.root / "persistence").glob("*.json")):
            witness = read(path)
            key = canonical(witness["identity"])
            entry = self.index["tasks"].get(key)
            if not entry or entry["sha256"] != witness["sha256"]:
                raise ContinuationError("PERSISTENCE_RECEIPT_MISMATCH")
            entry.update(
                cross_session_persistence_verified=True,
                persistence=witness["persistence"],
                storage_provenance=witness["storage_provenance"],
            )
        atomic(self.index_path, self.index)

    def _read(self, entry):
        path = safe_file(self.root, entry["file"])
        if digest(path) != entry["sha256"]:
            raise ContinuationError("RESULT_HASH_MISMATCH")
        result = read(path)
        proof(result, entry["identity"])
        return result

    def completed(self, identity):
        entry = self.index["tasks"].get(canonical(identity))
        return self._read(entry) if entry else None

    def commit(self, result, provenance, *, persistence="COMPUTED_LOCAL"):
        identity = result["identity"]
        proof(result, identity)
        encoded = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        import hashlib

        sha = hashlib.sha256(encoded).hexdigest()
        key = canonical(identity)
        old = self.index["tasks"].get(key)
        if old:
            self._read(old)
            # Semantic equality permits importing an old JSON serialization without
            # overwriting its original bytes. New differing timestamps are conflicts.
            if canonical(self._read(old)) != canonical(result):
                raise ContinuationError("LOCAL_RESULT_CONFLICT")
            if not old["cross_session_persistence_verified"] and persistence in (
                "KAGGLE_VERSION_PERSISTED",
                "HF_VERIFIED",
            ):
                witness = {
                    "identity": identity,
                    "sha256": old["sha256"],
                    "persistence": persistence,
                    "storage_provenance": provenance,
                }
                atomic(self.root / "persistence" / (canonical(witness) + ".json"), witness)
                old.update(
                    cross_session_persistence_verified=True,
                    persistence=persistence,
                    storage_provenance=provenance,
                )
                atomic(self.index_path, self.index)
            return old
        file = "objects/sha256/" + sha + ".json"
        path = safe_file(self.root, file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temp = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            try:
                os.link(temp, path)  # publish immutable bytes without replacing a name
            except FileExistsError:
                if digest(path) != sha:
                    raise ContinuationError("LOCAL_RESULT_CONFLICT")
        finally:
            temp.unlink()
        entry = {
            "identity": identity,
            "sha256": sha,
            "file": file,
            "task_status": "COMPLETE",
            "result_data_verified": True,
            "cross_session_persistence_verified": persistence
            in ("KAGGLE_VERSION_PERSISTED", "HF_VERIFIED"),
            "persistence": persistence,
            "storage_provenance": provenance,
        }
        self._read(entry)
        receipt = self.root / "receipts" / (key + ".json")
        if receipt.exists() and read(receipt) != entry:
            raise ContinuationError("LOCAL_RESULT_CONFLICT")
        atomic(receipt, entry)
        self.index["tasks"][key] = entry
        atomic(self.index_path, self.index)
        return entry

    def import_snapshot(self, root, provenance):
        other = LocalResultStore.__new__(LocalResultStore)
        other.root = Path(root)
        index = read(safe_file(root, "analysis-index.json"))
        imported, conflicts = 0, []
        for entry in index.get("tasks", {}).values():
            try:
                value = other._read(entry)
                self.commit(value, provenance, persistence="KAGGLE_VERSION_PERSISTED")
                imported += 1
            except Exception as error:
                conflicts.append(
                    {
                        "task_key": canonical(entry["identity"]),
                        "reason": getattr(error, "code", type(error).__name__),
                    }
                )
        return imported, conflicts

    def save_coverage(self, rows, session):
        from clm001_r3b.full_analysis import counts

        atomic(
            self.root / "coverage-ledger.json",
            {
                "tasks": rows,
                "counts": counts(rows),
                "session": session,
                "output_version": "OUTPUT_VERSION_PENDING_SAVE",
                "training_updates": 0,
            },
        )
