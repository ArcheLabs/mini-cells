"""Read-only audit of the 59 old observations; never calls the old 102-task executor."""

from pathlib import Path
from minicells_runtime.store import canonical, digest, read, ContinuationError
from kaggle_posthoc.kaggle_source import fetch
from kaggle_posthoc.local_store import proof
from .common import RESULT_RELATIVE, ENGINE_SHA


def audit_old(api, inventories, cache, code_root, *, on_progress=None):
    published = read(Path(code_root) / RESULT_RELATIVE / "coverage-ledger.json")
    rows = published["tasks"]
    if len(rows) != 102 or sum(r["persistence"] == "COMPUTED_LOCAL" for r in rows) != 59:
        raise ContinuationError("CF_OLD_PUBLIC_LEDGER_CHANGED")
    targets = {r["task_key"]: r for r in rows if r["persistence"] == "COMPUTED_LOCAL"}
    verified = {}
    raw_paths = {}

    def progress(current=None):
        missing = sorted(set(targets) - set(verified))
        report = {
            "status": "OLD_ANALYSIS_PERSISTENCE_VERIFIED" if not missing else "NOT_VERIFIED",
            "expected_local_results": 59,
            "verified_saved_results": len(verified),
            "previous_hf_verified_results": 43,
            "tasks": list(verified.values()),
            "missing_task_keys": missing,
            "old_diagnostics_recomputed": 0,
            "training_updates": 0,
            "current_file": current,
        }
        if on_progress is not None:
            on_progress(report)
        return report

    progress()
    for inv in sorted(inventories, key=lambda i: i["version_number"], reverse=True):
        names = {r["name"] for r in inv["files"]}
        selected = {}
        for key, row in targets.items():
            if key in verified:
                continue
            matches = [
                n
                for n in names
                if n.endswith("/private-results/objects/sha256/" + row["result_sha256"] + ".json")
            ]
            if matches:
                selected[key] = sorted(matches)[0]
        if not selected:
            continue
        destination = Path(cache) / ("old-saved-" + str(inv["version_number"]))
        print(
            {
                "stage": "OLD59_SAVED_READBACK",
                "version": inv["version_number"],
                "files": len(selected),
            },
            flush=True,
        )
        for key, name in selected.items():
            progress({"task_key": key, "version": inv["version_number"], "status": "READING"})
            print(
                {
                    "stage": "OLD59_FILE_START",
                    "version": inv["version_number"],
                    "verified": len(verified),
                    "expected": 59,
                    "task_key": key,
                },
                flush=True,
            )
            # The shared fetch still performs size/resume/SHA handling. Verify
            # and checkpoint each file before proceeding to the next one.
            fetch(api, inv, [name], destination)
            path = destination / name
            value = read(path)
            row = targets[key]
            if digest(path) != row["result_sha256"] or canonical(value["identity"]) != key:
                raise ContinuationError("CF_OLD_RESULT_HASH_OR_TASK_MISMATCH")
            proof(value, value["identity"])
            if value["identity"]["analysis_source_sha"] != ENGINE_SHA:
                raise ContinuationError("CF_OLD_ENGINE_MISMATCH")
            verified[key] = {
                "task_key": key,
                "sha256": row["result_sha256"],
                "kaggle_kernel": "bingjiangli/minicells",
                "version": inv["version_number"],
                "result_data_verified": True,
                "cross_session_persistence_verified": True,
            }
            raw_paths[key] = path
            progress({"task_key": key, "version": inv["version_number"], "status": "VERIFIED"})
            print(
                {"stage": "OLD59_FILE_VERIFIED", "verified": len(verified), "expected": 59},
                flush=True,
            )
    report = progress()
    reference = None
    for key, row in targets.items():
        if (
            row["arm"] == "QC-R3b"
            and row["step"] == 500
            and row["stage"] == "quality"
            and "greedy" in row["policies"]
            and key in raw_paths
        ):
            reference = read(raw_paths[key])
    return report, reference
