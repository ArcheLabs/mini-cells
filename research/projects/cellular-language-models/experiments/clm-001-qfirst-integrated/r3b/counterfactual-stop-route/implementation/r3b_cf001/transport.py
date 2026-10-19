"""Retry transport failures before streaming; retain the official body/resume handling."""

from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
import time
import requests
from kaggle_posthoc.kaggle_source import KaggleOutputAPI, relative
from minicells_runtime.store import canonical, ContinuationError


class IdleTimeoutAdapter(requests.adapters.HTTPAdapter):
    """Instance-local socket timeout for the official SDK, including its POSTs."""

    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (10, 30)
        return super().send(request, **kwargs)


class RecoverableKaggleOutputAPI(KaggleOutputAPI):
    single_pass_downloads = True
    socket_timeout = (10, 30)

    def __init__(self):
        super().__init__()
        self._client_builder = self.api.build_kaggle_client
        self.api.build_kaggle_client = self._client_context

    @contextmanager
    def _client_context(self):
        if not hasattr(self, "_sdk_client"):
            context = self._client_builder()
            client = context.__enter__()
            try:
                client.http_client()._session.mount("https://", IdleTimeoutAdapter())
            except Exception:
                context.__exit__(None, None, None)
                raise ContinuationError("CF_KAGGLE_CLIENT_SHAPE_CHANGED") from None
            self._sdk_context, self._sdk_client = context, client
        yield self._sdk_client

    def _metrics(self):
        if not hasattr(self, "network_metrics"):
            self.network_metrics = {
                "storage_get_attempts": 0,
                "size_response_reuses": 0,
                "official_download_calls": 0,
                "completed_file_payload_bytes": 0,
                "max_completed_file_bytes": 0,
                "transport_retries": 0,
                "api_attempts": {},
                "socket_timeout_connect_read_seconds": list(self.socket_timeout),
                "wire_bytes": "NOT_MEASURED",
                "request_count_scope": "adapter GET/RPC calls; redirects and official body retry requests are not counted",
                "official_internal_retry_requests": "NOT_OBSERVED_BY_ADAPTER",
            }
        return self.network_metrics

    def _session_get(self, url, **kwargs):
        if not hasattr(self, "_session"):
            self._session = requests.Session()
        return self._session.get(url, **kwargs)

    @contextmanager
    def _watch(self, label, file_id, path=None):
        stopped = Event()
        start = time.monotonic()

        def beat():
            while not stopped.wait(getattr(self, "_heartbeat_seconds", 10)):
                print(
                    {
                        "stage": "CF_NETWORK_HEARTBEAT",
                        "request": label,
                        "file_id": file_id,
                        "elapsed_seconds": round(time.monotonic() - start, 1),
                        "local_file_bytes": path.stat().st_size
                        if path and path.is_file()
                        else None,
                    },
                    flush=True,
                )

        watcher = Thread(target=beat, daemon=True)
        watcher.start()
        try:
            yield
        finally:
            stopped.set()
            watcher.join()

    def _transport_call(self, operation, label, file_id=None):
        # The official downloader already retries/resumes body failures. This
        # layer covers connection/header failures before that downloader starts.
        for attempt in range(3):
            try:
                return operation()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as error:
                # Invalid TLS certificates and HTTP authentication failures do
                # not become transient successes through repeated requests.
                if isinstance(error, requests.exceptions.SSLError) or attempt == 2:
                    raise
                delay = 2**attempt
                self._metrics()["transport_retries"] += 1
                print(
                    {
                        "stage": "CF_KAGGLE_TRANSPORT_RETRY",
                        "request": label,
                        "file_id": file_id,
                        "error_type": type(error).__name__,
                        "attempt": attempt + 1,
                        "wait_seconds": delay,
                    },
                    flush=True,
                )
                getattr(self, "_sleep", time.sleep)(delay)

    def _call(self, operation, label, *, pace=True):
        parent = super()._call

        def counted():
            counts = self._metrics()["api_attempts"]
            counts[label] = counts.get(label, 0) + 1
            return operation()

        if label in ("selected-file-download", "file-size", "connectivity-prefix"):
            # _file_response handles header retries; do not multiply the
            # pinned CLI's existing five body retries by an outer retry loop.
            return parent(operation, label, pace=pace)
        with self._watch(label, None):
            return self._transport_call(lambda: parent(counted, label, pace=pace), label)

    def _fresh_response(self, version, name, headers=None):
        file_id = canonical([version, relative(name)])[:16]

        def request():
            self._metrics()["storage_get_attempts"] += 1
            with self._watch("response-headers", file_id):
                return self._session_get(
                    self._output_urls[(version, name)],
                    stream=True,
                    timeout=self.socket_timeout,
                    headers={"Accept-Encoding": "identity", **(headers or {})},
                )

        return self._transport_call(
            request,
            "file-response-headers",
            file_id,
        )

    def _clear_pending(self):
        pending = getattr(self, "_pending_response", None)
        if pending is not None:
            pending[1].close()
            self._pending_response = None

    def size(self, version, name):
        key = (version, relative(name))
        pending = getattr(self, "_pending_response", None)
        if pending is not None and pending[0] == key:
            return int(pending[1].headers["Content-Length"])
        self._clear_pending()

        def get():
            response = self._fresh_response(version, name)
            try:
                response.raise_for_status()
                size = response.headers.get("Content-Length")
                if size is None or not size.isdigit():
                    raise ContinuationError("KAGGLE_FILE_SIZE_UNKNOWN")
                self._pending_response = (key, response)
                return int(size)
            except Exception:
                response.close()
                raise

        return self._call(get, "file-size", pace=False)

    def _file_response(self, version, name):
        key = (version, relative(name))
        pending = getattr(self, "_pending_response", None)
        if pending is not None and pending[0] == key:
            self._pending_response = None
            self._metrics()["size_response_reuses"] += 1
            return pending[1]
        self._clear_pending()
        return self._fresh_response(version, name)

    def download(self, version, names, destination):
        for name in dict.fromkeys(names):
            path = Path(destination) / relative(name)
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ContinuationError("UNSAFE_KAGGLE_OUTPUT_PATH")
            file_id = canonical([version, name])[:16]

            def get(path=path, name=name):
                with self._file_response(version, name) as response:
                    response.raise_for_status()
                    print(
                        {
                            "stage": "CF_DOWNLOAD_START",
                            "version": version,
                            "file_id": file_id,
                            "expected_bytes": response.headers.get("Content-Length"),
                        },
                        flush=True,
                    )
                    with self._watch("file-body", file_id, path):
                        self._metrics()["official_download_calls"] += 1
                        # One in-process retry retains the official Range/validator
                        # implementation. No outer loop multiplies body retries.
                        self.api.download_file(
                            response,
                            str(path),
                            None,
                            quiet=True,
                            resume=True,
                            timeout=self.socket_timeout,
                            max_retries=1,
                        )
                    size = path.stat().st_size
                    stats = self._metrics()
                    stats["completed_file_payload_bytes"] += size
                    stats["max_completed_file_bytes"] = max(stats["max_completed_file_bytes"], size)
                    print(
                        {"stage": "CF_DOWNLOAD_COMPLETE", "file_id": file_id, "file_bytes": size},
                        flush=True,
                    )

            self._call(get, "selected-file-download", pace=False)

    def metadata(self, version=None):
        cache = getattr(self, "_metadata_cache", {})
        if version not in cache:
            cache[version] = super().metadata(version)
        self._metadata_cache = cache
        return cache[version]

    def page(self, version, token=None):
        cache = getattr(self, "_page_cache", {})
        key = (version, token)
        if key not in cache:
            cache[key] = super().page(version, token)
        self._page_cache = cache
        return cache[key]

    def connectivity_probe(self, version):
        if self.metadata().get("is_private") is not True:
            raise ContinuationError("KAGGLE_PRIVATE_OUTPUT_REQUIRED")
        first = self.page(version)
        if not first["files"]:
            raise ContinuationError("CF_PROBE_OUTPUT_EMPTY")
        name = first["files"][0]["file_name"]
        start = time.perf_counter()

        def get():
            with self._fresh_response(version, name, {"Range": "bytes=0-1023"}) as response:
                response.raise_for_status()
                # Bound the consumed prefix even if a server ignores Range.
                chunk = next(response.iter_content(chunk_size=1024), b"")
                if not chunk and response.headers.get("Content-Length") != "0":
                    raise ContinuationError("CF_PROBE_OUTPUT_EMPTY")
                return len(chunk)

        with self._watch("connectivity-prefix", canonical([version, name])[:16]):
            count = self._call(get, "connectivity-prefix", pace=False)
        result = {
            "status": "PASS",
            "consumed_prefix_bytes": count,
            "wall_seconds": time.perf_counter() - start,
            "full_input_integrity": "NOT_YET_VERIFIED",
        }
        print({"stage": "CF_STORAGE_PROBE", **result}, flush=True)
        return result

    def close(self):
        self._clear_pending()
        if hasattr(self, "_session"):
            self._session.close()
        if hasattr(self, "_sdk_context"):
            self._sdk_context.__exit__(None, None, None)
            del self._sdk_context, self._sdk_client
