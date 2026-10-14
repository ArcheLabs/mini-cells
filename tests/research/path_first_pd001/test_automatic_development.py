"""Automatic flow tests use synthetic bundles only: no downloads or GPU work."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from exp006.reuse import file_sha256, write_json
from pd001.configs import load, authorize
from pd001 import development
from experiments.common.development import seed_for, record


def fake_bundle(c, destination, source, sha):
    destination = Path(destination)
    destination.mkdir(parents=True)
    np.save(destination / "train.npy", np.arange(128, dtype=np.int32) % 23)
    np.save(destination / "development.npy", np.arange(128, dtype=np.int32) % 23)
    (destination / "tokenizer.json").write_text('{"fixture":true}')
    write_json(
        destination / "windows.json",
        {
            "train": {"starts": [0, 4, 8], "groups": [0, 1, 2], "story_hashes": ["train-1", "train-2"]},
            "development": {"starts": [0, 4], "groups": [0, 1], "story_hashes": ["dev-1", "dev-2"]},
        },
    )
    manifest = {
        "experiment_id": "006-path-first-clm",
        "source_parquet_sha256": sha,
        "sequence_length": c["training"]["sequence_length"],
        "vocab_size": c["model"]["vocab_size"],
        "files": {
            name: file_sha256(destination / name)
            for name in ("train.npy", "development.npy", "tokenizer.json", "windows.json")
        },
        "confirmation_materialized": False,
        "story_selection_audit": {"raw_scan_limit_exclusive": 60000, "raw_rows_scanned": 7},
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


@pytest.fixture
def prepared(proposal, tmp_path, monkeypatch):
    _, p = load()
    raw = tmp_path / "source-fixture"
    raw.write_text("not real corpus")
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=lambda **kw: str(raw))
    )
    monkeypatch.setattr(development, "prepare", fake_bundle)
    monkeypatch.setattr(development, "occupied_seeds", lambda root: {17, 6060611})
    output = tmp_path / "run"
    result = development.configure(proposal, p, output, "a" * 40)
    return result, output


def test_auto_prepares_without_manual_registration(prepared, proposal, monkeypatch):
    (c, p, path, assets, seed), output = prepared
    assert c["data"]["manifest_sha256"] == file_sha256(assets / "manifest.json")
    assert c["data"]["tokenizer_sha256"] == file_sha256(assets / "tokenizer.json")
    assert seed not in (17, 6060611)
    assert not c["seeds"]["confirmation"] and not c["authorization"]["confirmation"]
    assert p["status"] == "AUTOMATIC_DEVELOPMENT" and p["scientific_decision"] is False
    assert path.is_file() and (output / "execution-identity.json").is_file()
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: SimpleNamespace(returncode=0))
    authorize(c, p, "P50-A32", seed, "development", "train", "a" * 40, "cpu")
    # Defaults remain immutable; only execution metadata is populated.
    assert proposal["data"]["manifest_sha256"] is None
    assert proposal["authorization"]["development"] is False


def test_auto_never_unlocks_confirmation(prepared):
    (c, p, _, _, seed), _ = prepared
    for mode in ("train", "evaluate", "confirmation", "publish"):
        with pytest.raises(RuntimeError, match="never grants confirmation"):
            authorize(c, p, "P50-A32", seed, "confirmation", mode, "a" * 40)


def test_auto_assets_checked_and_repeatable(prepared, proposal):
    (c, p, path, assets, seed), output = prepared
    _, base = load()
    second = development.configure(proposal, base, output, "a" * 40)
    assert second[4] == seed and second[2] == path
    (assets / "train.npy").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="corrupt development asset"):
        development.configure(proposal, base, output, "a" * 40)


def test_seed_and_identity_are_internal_and_immutable(tmp_path):
    chosen = seed_for("fixture", "a" * 40, set())
    assert seed_for("fixture", "a" * 40, {chosen}) == chosen + 1
    record(tmp_path, "identity.json", {"source": "a"})
    record(tmp_path, "identity.json", {"source": "a"})
    with pytest.raises(ValueError, match="identity changed"):
        record(tmp_path, "identity.json", {"source": "b"})


def test_auto_rejects_missing_identity_and_foreign_assets(prepared):
    (c, p, _, assets, seed), _ = prepared
    invalid = copy.deepcopy(c)
    invalid["data"]["manifest_sha256"] = None
    with pytest.raises(RuntimeError, match="automatic asset identity missing"):
        authorize(invalid, p, "M0", seed, "development", "train", "a" * 40)
    m = json.loads((assets / "manifest.json").read_text())
    m["confirmation_materialized"] = True
    write_json(assets / "manifest.json", m)
    from pd001.assets import load_bundle

    c["data"]["manifest_sha256"] = file_sha256(assets / "manifest.json")
    import unittest.mock

    with unittest.mock.patch("pd001.assets.authorize"):
        with pytest.raises(ValueError, match="confirmation material"):
            load_bundle(assets, c, p, "M0", seed, "development", "a" * 40)


def test_publication_preflight_uses_current_main_only(tmp_path, monkeypatch):
    import pd001.publication as publication

    commands = []
    pushes = []

    def fake_git(cmd, **kw):
        commands.append(cmd)
        code = 2 if "ls-remote" in cmd else 0
        return SimpleNamespace(
            returncode=code, stdout="b" * 40 + "\n" if "rev-parse" in cmd else "", stderr=""
        )

    monkeypatch.setattr(publication.subprocess, "run", fake_git)
    monkeypatch.setitem(
        sys.modules,
        "publish_core_validation_007",
        SimpleNamespace(preflight_push=lambda branch, secret: pushes.append((branch, secret))),
    )
    publication.prepare_checkout(tmp_path / "pub", "a" * 40)
    assert any(cmd[-1] == "main" and "fetch" in cmd for cmd in commands)
    assert not any(cmd[-1] == "a" * 40 and "checkout" in cmd for cmd in commands)
    assert pushes == [("results/pd001-" + ("a" * 12), "GITHUB_TOKEN")]


def test_chunked_metadata_is_bounded(tmp_path):
    from pd001.training import write_curve

    write_curve(tmp_path, [{"step": i} for i in range(250)])
    index = json.loads((tmp_path / "learning-curve.json").read_text())
    assert index["steps"] == 250 and len(index["chunks"]) == 3
    assert all(len(json.loads((tmp_path / name).read_text())) <= 100 for name in index["chunks"])


def test_one_click_runs_training_profiles_and_publication_in_order(proposal, tmp_path, monkeypatch):
    import pd001.runner as runner
    import pd001.orchestration as orchestration
    import pd001.publication as publication
    import pd001.report as reporting

    _, p = load()
    calls = []
    c = copy.deepcopy(proposal)
    config = tmp_path / "runtime.json"
    config.write_text("{}")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "manifest.json").write_text("{}")
    (assets / "windows.json").write_text("{}")

    def prepare(*a, **kw):
        calls.append("assets")
        return c, p, config, assets, 100000017

    def matrix(*a, **kw):
        calls.append("train_all_workers")
        runs = []
        for arm in ("M0", "P50-A32"):
            folder = tmp_path / "runs" / arm
            folder.mkdir(parents=True)
            runs.append({"arm": arm, "device": "cpu", "run": str(folder)})
        return {"runs": runs, "workers_finished": True, "scientific_decision": False}

    def profile(c, p, args):
        assert "train_all_workers" in calls
        calls.append("profile_" + args.arm)
        return {"fixture": True}

    def push(c, p, arm, *a, **kw):
        assert "profile_M0" in calls and "profile_P50-A32" in calls
        calls.append("publish_" + arm)

    monkeypatch.setenv("GITHUB_TOKEN", "fixture-not-a-real-token")
    monkeypatch.setenv("HF_TOKEN", "fixture-not-a-real-token")
    monkeypatch.setattr(
        runner, "preflight", lambda *a: (calls.append("preflight") or {"status": "ENGINEERING_READY"})
    )
    monkeypatch.setattr(development, "configure", prepare)
    monkeypatch.setattr(publication, "preflight_development", lambda *a: calls.append("return_check"))
    monkeypatch.setattr(publication, "publish", push)
    monkeypatch.setattr(orchestration, "matrix", matrix)
    monkeypatch.setattr(runner, "scientific", profile)
    monkeypatch.setattr(
        reporting, "report", lambda *a: {"status": "DEVELOPMENT_DIAGNOSTIC", "scientific_decision": False}
    )
    args = SimpleNamespace(
        source_commit="a" * 40, output=str(tmp_path), seed=None, devices=["cpu"], arms=["M0", "P50-A32"]
    )
    result = runner.auto_run(c, p, args)
    assert result["status"] == "DEVELOPMENT_COMPLETED"
    assert calls == [
        "preflight",
        "assets",
        "return_check",
        "train_all_workers",
        "profile_M0",
        "profile_P50-A32",
        "publish_M0",
        "publish_P50-A32",
    ]
    assert not result["confirmation_access"]
