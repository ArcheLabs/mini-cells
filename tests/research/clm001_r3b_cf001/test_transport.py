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
        return False

    def raise_for_status(self):
        pass


@pytest.fixture
def api():
    value = RecoverableKaggleOutputAPI.__new__(RecoverableKaggleOutputAPI)
    value._output_urls = {(69, "old/result.json"): "https://example.invalid/?signed=PRIVATE"}
    value.waits = []
    value._sleep = value.waits.append
    return value


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
    assert calls == [{"quiet": True, "resume": True}]
    assert not api.waits
