"""Retry transport failures before streaming; retain the official body/resume handling."""

import time
import requests
from kaggle_posthoc.kaggle_source import KaggleOutputAPI, relative
from minicells_runtime.store import canonical


class RecoverableKaggleOutputAPI(KaggleOutputAPI):
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
        if label in ("selected-file-download", "file-size"):
            # _file_response handles header retries; do not multiply the
            # pinned CLI's existing five body retries by an outer retry loop.
            return parent(operation, label, pace=pace)
        return self._transport_call(lambda: parent(operation, label, pace=pace), label)

    def _file_response(self, version, name):
        parent = super()._file_response
        return self._transport_call(
            lambda: parent(version, name),
            "file-response-headers",
            canonical([version, relative(name)])[:16],
        )
