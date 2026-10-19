"""Authenticated, versioned Kaggle Output enumeration/download; never an HF input fallback."""

from pathlib import Path, PurePosixPath
import re
import shutil
import time
from datetime import datetime, timezone
from minicells_runtime.store import canonical, digest, read, ContinuationError

KERNEL = "bingjiangli/minicells"
CLI_COMMIT = "6894864ff4095cf98e9e36d3a694ff5fc9ba3818"


def value(obj, *names):
    for name in names:
        found = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if found is not None:
            return found
    return None


def relative(name):
    if not isinstance(name, str) or "\\" in name or any(ord(c) < 32 for c in name):
        raise ContinuationError("UNSAFE_KAGGLE_OUTPUT_PATH")
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not p.parts or str(p) != name:
        raise ContinuationError("UNSAFE_KAGGLE_OUTPUT_PATH")
    return name


class KaggleOutputAPI:
    def __init__(self):
        from kaggle.api.kaggle_api_extended import KaggleApi

        self.api = KaggleApi()
        self.api.authenticate()
        self._output_urls = {}

    def _call(self, operation, label, *, pace=True):
        """Retry a single request, retaining completed enumeration pages."""
        sleep = getattr(self, "_sleep", time.sleep)
        for attempt in range(5):
            if pace:
                previous = getattr(self, "_last_request_at", None)
                delay = 1.0 - (time.monotonic() - previous) if previous is not None else 0
                if delay > 0:
                    sleep(delay)
                self._last_request_at = time.monotonic()
            try:
                return operation()
            except Exception as error:
                response = getattr(error, "response", None)
                status = getattr(response, "status_code", None)
                if status not in (429, 500, 502, 503, 504) or attempt == 4:
                    raise
                # Reuse the pinned CLI's RFC Retry-After parser.
                delay = self.api._get_retry_after_delay(response)
                delay = max(1, delay) if delay is not None else min(120, 15 * 2**attempt)
                print(
                    {
                        "stage": "KAGGLE_REQUEST_RETRY",
                        "request": label,
                        "http_status": status,
                        "attempt": attempt + 1,
                        "wait_seconds": delay,
                    },
                    flush=True,
                )
                while delay > 0:
                    chunk = min(30, delay)
                    sleep(chunk)
                    delay -= chunk

    def metadata(self, version=None):
        from kaggle.api.kaggle_api_extended import ApiGetKernelRequest

        request = ApiGetKernelRequest()
        request.user_name, request.kernel_slug = KERNEL.split("/")
        if version is not None:
            request.version_label = self.api._kernel_version_label(str(version))

        def get():
            with self.api.build_kaggle_client() as client:
                return client.kernels.kernels_api_client.get_kernel(request)

        response = self._call(get, "metadata")
        m = response.metadata
        # Never store source code, signed file URLs or the unrestricted API payload.
        return {
            "is_private": value(m, "is_private"),
            "ref": value(m, "ref"),
            "version_number": value(m, "version_number", "current_version_number"),
            "version_label": value(m, "version_label"),
            "script_version_id": value(m, "script_version_id"),
            "created_at": str(
                value(m, "last_run_time", "create_time", "lastRunTime") or "NOT_EXPOSED"
            ),
        }

    def page(self, version, token=None):
        # Use the same authoritative relative file_name as kernels_output.
        # ListKernelFiles exposes display names, which are not safe path keys.
        from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest

        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug = KERNEL.split("/")
        request.version_label = self.api._kernel_version_label(str(version))
        self.api._set_paging(request, 200, token)

        def get():
            with self.api.build_kaggle_client() as client:
                return client.kernels.kernels_api_client.list_kernel_session_output(request)

        response = self._call(get, "output-page")
        files = []
        for item in response.files or []:
            name = relative(item.file_name)
            self._output_urls[(version, name)] = item.url
            files.append({"file_name": name})
        return {"files": files, "next_page_token": response.next_page_token}

    def _file_response(self, version, name):
        import requests

        # The authenticated versioned Kaggle response supplies this URL.
        # Use it exactly as the official downloader does, without a guessed
        # storage-provider/domain whitelist.
        url = self._output_urls[(version, relative(name))]
        return requests.get(
            url, stream=True, timeout=(15, 60), headers={"Accept-Encoding": "identity"}
        )

    def size(self, version, name):
        # Signed URLs authorize GET, not necessarily HEAD. Never persist URLs.
        def get():
            with self._file_response(version, name) as response:
                response.raise_for_status()
                size = response.headers.get("Content-Length")
                if size is None or not size.isdigit():
                    raise ContinuationError("KAGGLE_FILE_SIZE_UNKNOWN")
                return int(size)

        return self._call(get, "file-size", pace=False)

    def download(self, version, names, destination):
        # Reuse the pinned CLI streaming/resume implementation, but NOT its
        # kernels_output wrapper: that wrapper relists every page for each
        # filtered download. Reuse the version-pinned signed file URLs instead.
        for name in names:
            relative(name)
            path = Path(destination) / name
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ContinuationError("UNSAFE_KAGGLE_OUTPUT_PATH")

            def get(name=name, path=path):
                with self._file_response(version, name) as response:
                    response.raise_for_status()
                    self.api.download_file(response, str(path), None, quiet=True, resume=True)

            self._call(get, "selected-file-download", pace=False)


def inventory(api, version):
    if type(version) is not int or version < 1:
        raise ContinuationError("KAGGLE_EXACT_VERSION_REQUIRED")
    started = time.perf_counter()
    meta = api.metadata(version)
    supplied = meta.get("version_number")
    if supplied is not None and int(supplied) != version:
        raise ContinuationError("KAGGLE_VERSION_RESPONSE_MISMATCH")
    files, token, seen = {}, None, set()
    duplicate_entries, pages = 0, 0
    while True:
        response = api.page(version, token)
        pages += 1
        for item in value(response, "files") or []:
            name = relative(value(item, "file_name", "name"))
            size = value(item, "total_bytes", "size", "content_length")
            if size is not None and int(size) < 0:
                raise ContinuationError("KAGGLE_FILE_SIZE_UNKNOWN")
            row = {"name": name, "bytes": int(size) if size is not None else None}
            if name in files:
                if files[name] != row:
                    raise ContinuationError("KAGGLE_OUTPUT_PATH_SIZE_CONFLICT")
                # Repeated listing entries are not a second physical file.
                # Equal size is not a content proof: frozen SHA verification
                # remains mandatory after download and before checkpoint load.
                duplicate_entries += 1
            else:
                files[name] = row
        token = value(response, "next_page_token", "nextPageToken")
        if not token:
            break
        if token in seen:
            raise ContinuationError("KAGGLE_PAGINATION_LOOP")
        seen.add(token)
    rows = [files[name] for name in sorted(files)]
    print(
        {
            "stage": "KAGGLE_VERSION_LISTED",
            "version": version,
            "unique_files": len(rows),
            "pages": pages,
            "repeated_entries": duplicate_entries,
        },
        flush=True,
    )
    return {
        "kernel": KERNEL,
        "version_number": version,
        "metadata": meta,
        "files": rows,
        "file_count": len(rows),
        "listing_pages": pages,
        "repeated_listing_entries": duplicate_entries,
        "total_bytes": sum(r["bytes"] for r in rows)
        if all(r["bytes"] is not None for r in rows)
        else None,
        "size_status": "KNOWN"
        if all(r["bytes"] is not None for r in rows)
        else "RESOLVED_BEFORE_SELECTED_DOWNLOAD",
        "listing_api": "ListKernelSessionOutput/file_name",
        "inventory_sha256": canonical([r["name"] for r in rows]),
        "inventory_hash_scope": "sorted authoritative relative paths; selected sizes resolved before download",
        "enumeration_seconds": time.perf_counter() - started,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "run_status": "NOT_EXPOSED_BY_THIS_FILE_LIST_RESPONSE",
        "cli_source_commit": CLI_COMMIT,
        "script_version_mapping": "VERIFIED" if meta.get("script_version_id") else "NOT_EXPOSED",
    }


def fetch(api, inv, names, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ContinuationError("UNSAFE_KAGGLE_OUTPUT_PATH")
    rows_by_name = {r["name"]: r for r in inv["files"]}
    # Resolve sizes only for selected files; do not issue one HTTP request per
    # irrelevant historical artifact merely to enumerate versions.
    size_started = time.perf_counter()
    for name in names:
        relative(name)
        if name not in rows_by_name:
            raise ContinuationError("KAGGLE_REQUIRED_FILE_MISSING")
        if rows_by_name[name]["bytes"] is None:
            rows_by_name[name]["bytes"] = api.size(inv["version_number"], name)
    size_seconds = time.perf_counter() - size_started
    sizes = {n: r["bytes"] for n, r in rows_by_name.items()}
    required = []
    for name in names:
        relative(name)
        if name not in sizes:
            raise ContinuationError("KAGGLE_REQUIRED_FILE_MISSING")
        path = destination / name
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ContinuationError("UNSAFE_KAGGLE_OUTPUT_PATH")
        if path.exists():
            if path.stat().st_size > sizes[name]:
                raise ContinuationError("KAGGLE_LOCAL_FILE_SIZE_CONFLICT")
            if path.stat().st_size < sizes[name]:
                # Official pinned CLI resumes a partial download; final size and
                # the frozen manifest hash still have to pass afterwards.
                required.append(name)
        else:
            required.append(name)
    if shutil.disk_usage(destination).free < sum(sizes[n] for n in required) + 128 * 1024 * 1024:
        raise ContinuationError("KAGGLE_DOWNLOAD_DISK_CAPACITY")
    start = time.perf_counter()
    api.download(inv["version_number"], required, destination)
    hashes = {}
    for name in names:
        path = destination / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != sizes[name]:
            raise ContinuationError("KAGGLE_DOWNLOAD_INCOMPLETE")
        hashes[name] = digest(path)
    return {
        "version": inv["version_number"],
        "download_seconds": time.perf_counter() - start,
        "selected_file_header_seconds": size_seconds,
        "downloaded_bytes": sum(sizes[n] for n in required),
        "byte_scope": "completed requested file payload; resumed bytes and CLI log excluded; wire bytes NOT_MEASURED",
        "reused_bytes": sum(sizes[n] for n in names if n not in required),
        "file_sha256": hashes,
    }


def locate_training(api, inventories, cache, engine):
    """Match exact frozen inventory first; only then load trusted PyTorch state."""
    search_started = time.perf_counter()
    expected = read(engine.REPORTS / "checkpoint-index.json")
    expected_set = {
        (r["arm"], r["step"], r["event_revision"], r["sha256"]) for r in expected["checkpoints"]
    }
    if len(expected_set) != 57:
        raise ContinuationError("FROZEN_CHECKPOINT_COUNT_MISMATCH")
    # Metadata-first selection: do not download identical multi-GB inputs repeatedly.
    for inv in sorted(inventories, key=lambda i: i["version_number"]):
        candidates = [
            r["name"]
            for r in inv["files"]
            if r["name"].endswith("/continuation.json") or r["name"] == "continuation.json"
        ]
        for name in candidates:
            download = Path(cache) / ("version-" + str(inv["version_number"]))
            fetch(api, inv, [name], download)
            meta = read(download / name)
            if meta.get("source_sha") != engine.SOURCE or meta.get("run_id") != engine.INTERNAL_RUN:
                continue
            actual = {
                (a, r["step"], r["revision"], r["checkpoint_sha256"])
                for a, p in meta.get("arms", {}).items()
                for r in p.get("checkpoints", [])
            }
            if actual != expected_set or not meta.get("complete"):
                continue
            prefix = str(PurePosixPath(name).parent)
            prefix = "" if prefix == "." else prefix + "/"
            names = {r["name"] for r in inv["files"]}
            required_meta = [prefix + meta["files_manifest"]]
            fetch(api, inv, required_meta, download)
            manifest_path = download / required_meta[0]
            if digest(manifest_path) != meta["files_manifest_sha256"]:
                raise ContinuationError("MANIFEST_HASH_MISMATCH")
            files = read(manifest_path)["files"]
            required = [prefix + relative(row.get("storage_path", n)) for n, row in files.items()]
            required += [name, required_meta[0]]
            if not set(required).issubset(names):
                inv["training_match"] = "INCOMPLETE_FILES"
                inv["missing_training_files"] = sorted(set(required) - names)
                continue
            transfer = fetch(api, inv, required, download)
            root = download / prefix
            validation_started = time.perf_counter()
            # The manifest is hashed and every file checked before any torch.load.
            for n, row in files.items():
                path = root / relative(row.get("storage_path", n))
                if digest(path) != row["sha256"]:
                    raise ContinuationError("CHECKPOINT_HASH_MISMATCH")
            meta = engine.verify(root)
            engine.validate_input(expected, meta, expected["checkpoint_ref"])
            transfer["integrity_and_runner_validation_seconds"] = (
                time.perf_counter() - validation_started
            )
            transfer["selection_download_validation_seconds"] = time.perf_counter() - search_started
            transfer["metadata_download_bytes"] = {"value": None, "classification": "NOT_MEASURED"}
            inv["training_match"] = "VERIFIED_57_OF_57"
            return inv, root, meta, transfer
    raise ContinuationError("KAGGLE_TRAINING_OUTPUT_VERSION_UNRESOLVED")


def discover(api, candidate=65, *, minimum_version=1, on_inventory=None):
    """Every candidate is enumerated through an explicit v<N> selector.

    A current metadata version hint expands the search; otherwise record the
    bounded probe range honestly. Missing numbers do not end discovery.
    """
    current = api.metadata()
    if current.get("is_private") is not True:
        raise ContinuationError("KAGGLE_PRIVATE_OUTPUT_REQUIRED")
    hint = current.get("version_number")
    if hint is None:
        label = current.get("version_label") or ""
        matched = re.fullmatch(r"v?(\d+)", str(label))
        hint = int(matched[1]) if matched else None
    upper = max(candidate, int(hint)) if hint else candidate + 32
    found, rejected = [], []
    for version in [candidate, *range(upper, minimum_version - 1, -1)]:
        if any(i["version_number"] == version for i in found) or any(
            i["version_number"] == version for i in rejected
        ):
            continue
        try:
            item = inventory(api, version)
            found.append(item)
            if on_inventory:
                on_inventory(item)
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if status not in (404, 410):
                raise
            rejected.append(
                {"version_number": version, "status": "NOT_AVAILABLE", "http_status": status}
            )
    return found, {
        "candidate": candidate,
        "search_upper_bound": upper,
        "latest_number_exposed": hint,
        "exhaustive_to_exposed_latest": bool(hint) and minimum_version == 1,
        "search_lower_bound": minimum_version,
        "older_versions": "DEFERRED_UNLESS_INPUT_MISSING" if minimum_version > 1 else "SCANNED",
        "unavailable_versions": rejected,
        "current_visibility_verified_private": True,
        "user_reported_script_version_id": 359735394,
        "script_version_mapping": "UNRESOLVED_UNLESS_EXPOSED_BY_VERSION_METADATA",
    }
