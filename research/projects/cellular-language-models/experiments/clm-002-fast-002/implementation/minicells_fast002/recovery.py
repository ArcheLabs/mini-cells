"""Runtime import of versioned Kaggle Output through the existing CF streaming adapter."""

import os
import re
import subprocess
import sys
from pathlib import Path

from . import checkpoints
from .common import BASE, atomic, digest, read

KERNEL = "bingjiangli/minicells"
CLI_COMMIT = "6894864ff4095cf98e9e36d3a694ff5fc9ba3818"
LOOKBACK = 12


def backend():
    # Prefer existing credentials. A Notebook session alone does not authenticate
    # reads of private saved versions through the Kaggle API.
    if not os.environ.get("KAGGLE_API_TOKEN"):
        try:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret("KAGGLE_API_TOKEN").strip()
            if token:
                os.environ["KAGGLE_API_TOKEN"] = token
        except Exception:  # noqa: BLE001 -- optional hosted Secret reader; preserve existing credentials
            print("[FAST002 Recovery] Checking existing Kaggle credentials", flush=True)
    probe = """import sys
try:
 from kaggle.api.kaggle_api_extended import KaggleApi
except ModuleNotFoundError:
 sys.exit(3)
except Exception:
 sys.exit(4)
sys.exit(0 if callable(getattr(KaggleApi, '_kernel_version_label', None)) else 3)
"""
    tested = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, timeout=20, check=False
    )
    if tested.returncode == 3:
        print("[FAST002 Recovery] Prepare pinned versioned Kaggle client", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--upgrade",
                "git+https://github.com/Kaggle/kaggle-api.git@" + CLI_COMMIT,
            ],
            check=True,
            timeout=90,
        )
    elif tested.returncode:
        raise RuntimeError("FAST002_KAGGLE_AUTH_REQUIRED_ENABLE_EXISTING_KAGGLE_API_TOKEN")
    lineage = BASE.parent / "clm-001-qfirst-integrated"
    sys.path.insert(0, str(lineage / "r3b/implementation"))
    sys.path.insert(0, str(lineage / "r3b/counterfactual-stop-route/implementation"))
    from kaggle_posthoc.kaggle_source import fetch
    from r3b_cf001.transport import RecoverableKaggleOutputAPI

    class OutputAPI(RecoverableKaggleOutputAPI):
        # Retain CF's streaming, SDK socket deadlines, progress, and official
        # body resume. Avoid multiplying its retries with the legacy outer loop.
        def _call(self, operation, label, *, pace=True):
            metrics = self._metrics()["api_attempts"]
            metrics[label] = metrics.get(label, 0) + 1
            with self._watch(label, None):
                return operation()

        def _transport_call(self, operation, label, file_id=None):
            return operation()

    api = OutputAPI()
    if not callable(getattr(api.api, "_kernel_version_label", None)):
        raise TypeError("FAST002_PINNED_KAGGLE_CLIENT_MISSING_VERSION_SUPPORT")
    return api, fetch


def expected(c, reports):
    return {
        f"{seed}-{arm}": read(reports / "jobs" / f"{seed}-{arm}.json")
        for seed in c["seeds"]
        for arm in c["arms"]
        if arm.startswith("A-") or arm[:3] in ("B00", "B10")
    }


def relative_files(c, jobs):
    prefix = "clm002-fast001-output/" + c["historical"]["run_id"] + "/"
    paths = {
        key: prefix
        + "private/"
        + ("A" if row["arm"].startswith("A-") else "B")
        + "/"
        + key
        + "/latest.pt"
        for key, row in jobs.items()
    }
    return prefix, paths


def find_version(api, c, required, version=None):
    current = api.metadata() if version is None else None
    if version is None:
        value = current.get("version_number")
        match = re.fullmatch(r"v?(\d+)", str(value or current.get("version_label") or ""))
        if not match:
            raise RuntimeError("FAST002_KAGGLE_LATEST_VERSION_NOT_EXPOSED_EXACT_VERSION_NEEDED")
        latest = int(match[1])
        versions = range(latest, max(0, latest - LOOKBACK), -1)
    else:
        versions = [version]
    searched = []
    for number in versions:
        names, token, seen = set(), None, set()
        print("[FAST002 Recovery] Inspect saved version", number, flush=True)
        try:
            while True:
                page = api.page(number, token)
                names.update(item["file_name"] for item in page["files"])
                # Do not enumerate irrelevant remaining code/assets once the
                # exact known output paths have all been found.
                if required.issubset(names):
                    meta = api.metadata(number)
                    supplied = meta.get("version_number")
                    if supplied is not None and int(supplied) != number:
                        raise RuntimeError("FAST002_KAGGLE_VERSION_RESPONSE_MISMATCH")
                    return number, names, meta, searched
                token = page.get("next_page_token")
                if not token:
                    break
                if token in seen:
                    raise RuntimeError("FAST002_KAGGLE_PAGINATION_LOOP")
                seen.add(token)
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status not in (404, 410):
                raise
        searched.append(
            {
                "version": number,
                "required_files_present": len(required & names),
                "required_files": len(required),
            }
        )
    raise RuntimeError(
        "FAST002_SAVED_OUTPUT_NOT_FOUND_IN_RECENT_VERSIONS_NO_RETRAIN:" + str(searched)
    )


def resolve(c, reports, roots, cache, *, version=None):
    cache = Path(cache)
    jobs = expected(c, reports)
    try:
        paths = checkpoints.locate(c, reports, [*roots, cache])
        return paths, {
            "status": "VERIFIED_LOCAL_OR_RECOVERY_CACHE",
            "network_required": False,
            "historical_run_id": c["historical"]["run_id"],
            "checkpoint_count": len(paths),
        }
    except RuntimeError as error:
        if not str(error).startswith(
            "FAST002_ATTACH_ORIGINAL_FAST001_SAVE_VERSION_OUTPUT_MISSING:"
        ):
            raise
    api, fetch = backend()
    prefix, selected = relative_files(c, jobs)
    manifest_name = prefix + "RUN_MANIFEST.json"
    number, _names, meta, searched = find_version(
        api, c, set(selected.values()) | {manifest_name}, version
    )
    destination = cache / ("version-" + str(number))
    inventory = {
        "version_number": number,
        "files": [
            {"name": name, "bytes": None}
            for name in sorted(set(selected.values()) | {manifest_name})
        ],
    }
    fetch(api, inventory, [manifest_name], destination)
    manifest = read(destination / manifest_name)
    expected_identity = read(reports / "RUN_MANIFEST.json")["identity"]
    if manifest.get("identity") != expected_identity:
        raise RuntimeError("FAST002_SAVED_OUTPUT_RUN_PROVENANCE_MISMATCH_NO_WEIGHT_DOWNLOAD")
    transferred, reused, files = 0, 0, {}
    for index, (key, name) in enumerate(sorted(selected.items()), 1):
        path = destination / name
        pinned = jobs[key]["checkpoint_sha256"]
        if path.is_file() and digest(path) == pinned:
            reused += path.stat().st_size
        else:
            if path.is_file():
                # Keep partial files for the existing official Range/resume path.
                # Quarantine only complete/oversized scratch files with wrong hashes.
                size = api.size(number, name)
                for item in inventory["files"]:
                    if item["name"] == name:
                        item["bytes"] = size
                if path.stat().st_size >= size:
                    path.rename(path.with_suffix(".hash-mismatch-" + digest(path)[:12]))
            print(
                "[FAST002 Recovery] Download checkpoint", index, "/", len(selected), key, flush=True
            )
            transfer = fetch(api, inventory, [name], destination)
            transferred += transfer["downloaded_bytes"]
            if digest(path) != pinned:
                raise RuntimeError("FAST002_RECOVERED_CHECKPOINT_SHA256_MISMATCH:" + key)
        files[key] = {"sha256": pinned, "bytes": path.stat().st_size}
    paths = checkpoints.locate(c, reports, [destination])
    proof = {
        "status": "EXACT_KAGGLE_OUTPUT_CHECKPOINTS_RECOVERED",
        "kernel": KERNEL,
        "version_number": number,
        "metadata": meta,
        "searched_newer_versions": searched,
        "historical_identity": expected_identity,
        "historical_run_id": c["historical"]["run_id"],
        "files": files,
        "checkpoint_count": len(paths),
        "downloaded_checkpoint_bytes": transferred,
        "reused_checkpoint_bytes": reused,
        "byte_scope": "COMPLETED_REQUESTED_FILE_PAYLOAD; RESUME_WIRE_BYTES_NOT_MEASURED",
        "signed_urls_persisted": False,
        "listing_scope": "PAGINATED_ONCE_PER_CANDIDATE_VERSION_STOP_WHEN_REQUIRED_PATHS_FOUND",
        "transport": "EXISTING_CF001_RECOVERABLE_KAGGLE_OUTPUT_API",
        "network_metrics": api._metrics(),
        "model_loaded_during_recovery": False,
        "scientific_optimizer_updates": 0,
    }
    atomic(cache / "RECOVERY_SOURCE.json", proof)
    return paths, proof
