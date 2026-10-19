"""Synthetic failures only: no credentials, real network or GPU."""

from types import SimpleNamespace
import pytest
import requests
from r3b_cf001.transport import RecoverableKaggleOutputAPI


class Response:
    headers = {"Content-Length": "7"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False

    def close(self):
        self.closed = True

    def iter_content(self, chunk_size=1024):
        yield b"x" * min(chunk_size, 7)

    def raise_for_status(self):
        pass


@pytest.fixture
def api():
    value = RecoverableKaggleOutputAPI.__new__(RecoverableKaggleOutputAPI)
    value._output_urls = {(69, "old/result.json"): "https://example.invalid/?signed=PRIVATE"}
    value.waits = []
    value._sleep = value.waits.append
    value._session = SimpleNamespace(
        get=lambda *a, **kw: requests.get(*a, **kw), close=lambda: None
    )
    yield value
    value.close()


@pytest.mark.parametrize(
    "failure",
    [
        requests.exceptions.ReadTimeout,
        requests.exceptions.ConnectTimeout,
        requests.exceptions.ConnectionError,
    ],
)
def test_header_transport_recovers_without_changing_signed_url(api, monkeypatch, capsys, failure):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        if len(calls) == 1:
            raise failure("signed=PRIVATE must never enter retry logs")
        return Response()

    monkeypatch.setattr(requests, "get", get)
    assert api.size(69, "old/result.json") == 7
    assert len(calls) == 2 and api.waits == [1]
    assert calls[0] == calls[1]
    assert "PRIVATE" not in capsys.readouterr().out


def test_header_retries_are_bounded_and_do_not_report_success(api, monkeypatch):
    calls = []

    def get(*args, **kwargs):
        calls.append(1)
        raise requests.exceptions.ReadTimeout("synthetic")

    monkeypatch.setattr(requests, "get", get)
    with pytest.raises(requests.exceptions.ReadTimeout):
        api.size(69, "old/result.json")
    assert len(calls) == 3 and api.waits == [1, 2]


def test_tls_configuration_error_not_retried(api, monkeypatch):
    def get(*args, **kwargs):
        raise requests.exceptions.SSLError("invalid certificate")

    monkeypatch.setattr(requests, "get", get)
    with pytest.raises(requests.exceptions.SSLError):
        api.size(69, "old/result.json")
    assert not api.waits


def test_api_metadata_timeout_retried_but_http_auth_denial_not_retried(api):
    calls = []

    def metadata():
        calls.append(1)
        if len(calls) == 1:
            raise requests.exceptions.ReadTimeout("synthetic")
        return 42

    assert api._call(metadata, "metadata", pace=False) == 42
    assert api.waits == [1]
    calls.clear()

    def denied():
        calls.append(1)
        error = requests.exceptions.HTTPError("forbidden")
        error.response = SimpleNamespace(status_code=403)
        raise error

    with pytest.raises(requests.exceptions.HTTPError):
        api._call(denied, "metadata", pace=False)
    assert len(calls) == 1


def test_existing_retry_after_policy_is_preserved(api):
    api.api = SimpleNamespace(_get_retry_after_delay=lambda response: 7)
    calls = []

    def limited():
        calls.append(1)
        if len(calls) == 1:
            error = requests.exceptions.HTTPError("rate limited")
            error.response = SimpleNamespace(status_code=429)
            raise error
        return "ok"

    assert api._call(limited, "output-page", pace=False) == "ok"
    assert api.waits == [7]


def test_body_retry_not_multiplied_and_official_resume_kept(api, monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "get", lambda *a, **kw: Response())
    calls = []

    def exhausted(response, path, client, **kwargs):
        calls.append(kwargs)
        raise requests.exceptions.ReadTimeout("official downloader exhausted its retries")

    api.api = SimpleNamespace(download_file=exhausted)
    with pytest.raises(requests.exceptions.ReadTimeout):
        api.download(69, ["old/result.json"], tmp_path)
    assert calls == [{"quiet": True, "resume": True, "timeout": (10, 30), "max_retries": 1}]
    assert not api.waits


def test_59_files_use_59_gets_instead_of_118_and_no_unselected_files(api, monkeypatch, tmp_path):
    from pathlib import Path
    from kaggle_posthoc.kaggle_source import KaggleOutputAPI, fetch

    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    def write(response, name, client, **kwargs):
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 7)

    monkeypatch.setattr(requests, "get", get)
    api.api = SimpleNamespace(download_file=write)
    names = ["results/" + str(i) + ".json" for i in range(59)]
    urls = {(69, n): "https://example.invalid/" + n for n in names}
    old = KaggleOutputAPI.__new__(KaggleOutputAPI)
    old._output_urls, old.api = urls, api.api

    def inv():
        return {
            "version_number": 69,
            "files": [{"name": n, "bytes": None} for n in names + ["unselected.pt"]],
        }

    first = fetch(old, inv(), names, tmp_path / "old")
    assert len(calls) == 118
    calls.clear()
    api._output_urls = urls
    second = fetch(api, inv(), names + names[:1], tmp_path / "new")
    assert len(calls) == 59
    assert first["file_sha256"] == second["file_sha256"]
    assert first["downloaded_bytes"] == second["downloaded_bytes"] == 59 * 7
    assert api.network_metrics["size_response_reuses"] == 59
    assert api.network_metrics["max_completed_file_bytes"] == 7


def test_small_probe_is_before_bulk_listing_and_does_not_consume_full_file(api, monkeypatch):
    monkeypatch.setattr(api, "metadata", lambda: {"is_private": True})
    monkeypatch.setattr(api, "page", lambda version: {"files": [{"file_name": "old/result.json"}]})
    calls = []

    class LargeResponse(Response):
        headers = {"Content-Length": "1000000000"}

        def iter_content(self, chunk_size):
            yield b"x" * chunk_size
            raise AssertionError("probe tried to read more than one prefix chunk")

    response = LargeResponse()
    monkeypatch.setattr(requests, "get", lambda url, **kw: calls.append(kw) or response)
    result = api.connectivity_probe(69)
    assert result["consumed_prefix_bytes"] == 1024 and response.closed
    assert calls[0]["headers"]["Range"] == "bytes=0-1023"
    assert calls[0]["timeout"] == (10, 30)
    assert api.network_metrics["storage_get_attempts"] == 1


def test_timeout_policy_is_idle_only_and_no_nested_probe_retry(api, monkeypatch):
    monkeypatch.setattr(api, "metadata", lambda: {"is_private": True})
    monkeypatch.setattr(api, "page", lambda version: {"files": [{"file_name": "old/result.json"}]})
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise requests.exceptions.ReadTimeout("synthetic")

    monkeypatch.setattr(requests, "get", fail)
    with pytest.raises(requests.exceptions.ReadTimeout):
        api.connectivity_probe(69)
    assert len(calls) == 3 and api.waits == [1, 2]
    # This tuple limits connection/no-data waits, not total file or session duration.
    assert all(c["timeout"] == (10, 30) for c in calls)


def test_sdk_session_reused_and_timeout_adapter_is_instance_local(api):
    from contextlib import contextmanager

    mounts, entered, closed = [], [], []
    session = SimpleNamespace(mount=lambda scheme, adapter: mounts.append((scheme, adapter)))
    client = SimpleNamespace(http_client=lambda: SimpleNamespace(_session=session))

    @contextmanager
    def builder():
        entered.append(1)
        try:
            yield client
        finally:
            closed.append(1)

    api._client_builder = builder
    with api._client_context() as a:
        assert a is client
    with api._client_context() as b:
        assert b is client
    assert entered == [1] and not closed
    assert mounts[0][0] == "https://"
    api.close()
    assert closed == [1]


def test_sdk_default_timeout_is_applied_without_global_patching(monkeypatch):
    from r3b_cf001.transport import IdleTimeoutAdapter

    seen = []
    monkeypatch.setattr(
        requests.adapters.HTTPAdapter, "send", lambda self, request, **kw: seen.append(kw) or "ok"
    )
    assert IdleTimeoutAdapter().send(object(), timeout=None) == "ok"
    assert seen[-1]["timeout"] == (10, 30)
    IdleTimeoutAdapter().send(object(), timeout=(2, 3))
    assert seen[-1]["timeout"] == (2, 3)


def test_storage_connection_pool_reused_and_pending_response_bounded(api, monkeypatch):
    api._clear_pending()
    del api._session
    sessions, responses = [], []

    class Session:
        def __init__(self):
            sessions.append(self)

        def get(self, *a, **kw):
            response = Response()
            responses.append(response)
            return response

        def close(self):
            pass

    monkeypatch.setattr(requests, "Session", Session)
    api.size(69, "old/result.json")
    api._output_urls[(69, "second.json")] = "https://example.invalid/second"
    api.size(69, "second.json")
    assert len(sessions) == 1 and responses[0].closed
    assert api.size(69, "second.json") == 7 and len(responses) == 2


def test_transfer_heartbeat_reports_file_bytes_without_signed_url(api, tmp_path, capsys):
    import time

    file = tmp_path / "partial"
    file.write_bytes(b"123")
    api._heartbeat_seconds = 0.01
    with api._watch("file-body", "safe-file-id", file):
        time.sleep(0.04)
    text = capsys.readouterr().out
    assert "CF_NETWORK_HEARTBEAT" in text and "local_file_bytes" in text
    assert "PRIVATE" not in text


def test_probe_metadata_and_first_page_are_cached_for_discovery(api, monkeypatch):
    from kaggle_posthoc.kaggle_source import KaggleOutputAPI

    calls = []

    def metadata(self, version=None):
        calls.append(("metadata", version))
        return {"is_private": True, "version_number": version or 71}

    def page(self, version, token=None):
        calls.append(("page", version, token))
        return {"files": [], "next_page_token": ""}

    monkeypatch.setattr(KaggleOutputAPI, "metadata", metadata)
    monkeypatch.setattr(KaggleOutputAPI, "page", page)
    assert api.metadata() == api.metadata()
    assert api.page(65) == api.page(65)
    assert calls == [("metadata", None), ("page", 65, None)]
