import hashlib
import json

import pytest
from minicells_fast001.common import atomic
from minicells_fast002 import recovery


def setup(tmp_path):
    c = {"seeds": [1], "arms": ["A-S", "B00-S", "B10-S"], "historical": {"run_id": "a" * 32}}
    reports = tmp_path / "reports"
    identity = {"source_sha": "b" * 40, "run_id": "a" * 32, "config_sha256": "c" * 64}
    atomic(reports / "RUN_MANIFEST.json", {"identity": identity})
    blobs = {}
    for arm in c["arms"]:
        content = ("original immutable " + arm).encode()
        atomic(
            reports / "jobs" / f"1-{arm}.json",
            {"seed": 1, "arm": arm, "checkpoint_sha256": hashlib.sha256(content).hexdigest()},
        )
        blobs[f"1-{arm}"] = content
    jobs = recovery.expected(c, reports)
    prefix, paths = recovery.relative_files(c, jobs)
    files = {paths[key]: content for key, content in blobs.items()}
    files[prefix + "RUN_MANIFEST.json"] = json.dumps({"identity": identity}).encode()
    return c, reports, files


class API:
    def __init__(self, files):
        self.files = files
        self.pages = []

    def metadata(self, version=None):
        return {"version_number": version or 10, "script_version_id": "metadata-only"}

    def page(self, version, token=None):
        self.pages.append((version, token))
        if version == 10:
            return {"files": [{"file_name": "unrelated.txt"}], "next_page_token": None}
        names = sorted(self.files)
        selected = names[:2] if token is None else names[2:]
        return {
            "files": [{"file_name": n} for n in selected],
            "next_page_token": "page2" if token is None else None,
        }

    def size(self, version, name):
        return len(self.files[name])

    def _metrics(self):
        return {"request_scope": "TEST_FAKE_NO_NETWORK"}


def test_runtime_recovery_single_listing_exact_hash_and_cache(tmp_path, monkeypatch):
    c, reports, files = setup(tmp_path)
    api = API(files)
    calls = []

    def fetch(api, inventory, names, destination):
        calls.extend(names)
        for name in names:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(files[name])
        return {"downloaded_bytes": sum(len(files[n]) for n in names)}

    monkeypatch.setattr(recovery, "backend", lambda: (api, fetch))
    paths, proof = recovery.resolve(c, reports, [], tmp_path / "cache")
    assert len(paths) == 3 and proof["version_number"] == 9
    assert api.pages == [(10, None), (9, None), (9, "page2")]
    assert (
        len(calls) == 4
    )  # one small manifest + three required weights; no arrays/code/log download
    assert not proof["signed_urls_persisted"] and not proof["model_loaded_during_recovery"]
    monkeypatch.setattr(
        recovery,
        "backend",
        lambda: pytest.fail("valid cache must not authenticate or access network"),
    )
    restored, reused = recovery.resolve(c, reports, [], tmp_path / "cache")
    assert restored == paths and reused["network_required"] is False


def test_wrong_saved_run_stops_before_checkpoint_download(tmp_path, monkeypatch):
    c, reports, files = setup(tmp_path)
    manifest = next(n for n in files if n.endswith("RUN_MANIFEST.json"))
    files[manifest] = json.dumps({"identity": {"run_id": "wrong"}}).encode()
    calls = []

    def fetch(api, inventory, names, destination):
        calls.extend(names)
        for name in names:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(files[name])
        return {"downloaded_bytes": 0}

    monkeypatch.setattr(recovery, "backend", lambda: (API(files), fetch))
    with pytest.raises(RuntimeError, match="PROVENANCE_MISMATCH_NO_WEIGHT_DOWNLOAD"):
        recovery.resolve(c, reports, [], tmp_path / "cache", version=9)
    assert calls == [manifest]


def test_mismatched_checkpoint_is_not_loaded_or_accepted(tmp_path, monkeypatch):
    c, reports, files = setup(tmp_path)
    target = next(n for n in files if n.endswith("latest.pt"))
    files[target] = b"wrong checkpoint bytes"

    def fetch(api, inventory, names, destination):
        for name in names:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(files[name])
        return {"downloaded_bytes": 0}

    monkeypatch.setattr(recovery, "backend", lambda: (API(files), fetch))
    with pytest.raises(RuntimeError, match="CHECKPOINT_SHA256_MISMATCH"):
        recovery.resolve(c, reports, [], tmp_path / "cache", version=9)


def test_partial_download_is_preserved_for_official_resume(tmp_path, monkeypatch):
    c, reports, files = setup(tmp_path)
    target = next(n for n in files if n.endswith("latest.pt"))
    path = tmp_path / "cache" / "version-9" / target
    path.parent.mkdir(parents=True)
    path.write_bytes(files[target][:3])

    def fetch(api, inventory, names, destination):
        for name in names:
            output = destination / name
            if name == target:
                assert output.read_bytes() == files[target][:3]
                assert next(row for row in inventory["files"] if row["name"] == name)[
                    "bytes"
                ] == len(files[name])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(files[name])
        return {"downloaded_bytes": sum(len(files[n]) for n in names)}

    monkeypatch.setattr(recovery, "backend", lambda: (API(files), fetch))
    paths, proof = recovery.resolve(c, reports, [], tmp_path / "cache", version=9)
    assert len(paths) == proof["checkpoint_count"] == 3
    assert path.read_bytes() == files[target]
    assert not list(path.parent.glob("*.hash-mismatch-*"))
