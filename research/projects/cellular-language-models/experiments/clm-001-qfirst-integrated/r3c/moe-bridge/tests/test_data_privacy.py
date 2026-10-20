import json
import pytest
from minicells_r3c_moe_bridge import data
from minicells_r3c_moe_bridge.common import configuration, atomic, digest
from minicells_r3c_moe_bridge.publication import public_summary
from test_state_and_storage import FakeHub


def test_real_data_reuses_frozen_validator(tmp_path, monkeypatch):
    c = configuration()
    atomic(tmp_path / "manifest.json", {"fixture": True})
    (tmp_path / "tokenizer.json").write_text("{}")
    c["input"]["assets_manifest_sha256"] = digest(tmp_path / "manifest.json")
    c["input"]["tokenizer_sha256"] = digest(tmp_path / "tokenizer.json")
    sentinel = object()
    seen = []
    monkeypatch.setattr(
        data, "original_load", lambda config, folder: seen.append(config) or sentinel
    )
    assert data.load_assets(c, tmp_path) is sentinel and seen[0] == data.legacy_configuration()
    c["input"]["confirmation_materialized"] = True
    with pytest.raises(RuntimeError, match="SEALED"):
        data.load_assets(c, tmp_path)


def test_asset_subset_never_fetches_checkpoint(tmp_path, monkeypatch):
    c = configuration()
    hub = FakeHub(tmp_path / "hub")
    hub.root.mkdir()
    source = hub.root / "asset"
    source.write_bytes(b"asset")
    blob = hub.root / "bundle"
    atomic(
        blob,
        {
            "identity": {
                "assets_manifest_sha256": c["input"]["assets_manifest_sha256"],
                "tokenizer_sha256": c["input"]["tokenizer_sha256"],
            },
            "files": {
                "assets/train.npy": {"object": "asset", "sha256": digest(source)},
                "runs/private.pt": {"object": "must-not-fetch", "sha256": "0" * 64},
            },
        },
    )
    monkeypatch.setattr(data, "load_assets", lambda c, p: "VERIFIED")
    assert (
        data.private_asset_subset(
            hub,
            {"hf_revision": "pinned", "bundle_path": "bundle", "bundle_sha256": digest(blob)},
            tmp_path / "output",
            c,
        )
        == "VERIFIED"
    )
    assert all(name != "must-not-fetch" for name, rev in hub.requests)
    assert (tmp_path / "output/train.npy").read_bytes() == b"asset"


def test_synthetic_public_reports_only_aggregate(config, tmp_path):
    from minicells_r3c_moe_bridge.runner import engineering

    report = engineering(tmp_path / "run")
    report["raw_tokens"] = "DO_NOT_PUBLISH"
    report["checkpoint_payload"] = "PRIVATE"
    summary = public_summary(report)
    serialized = json.dumps(summary)
    assert "DO_NOT_PUBLISH" not in serialized and "PRIVATE" not in serialized
    assert not summary["full_engineering_pass"] and not summary["scientific_decision"]
    assert (tmp_path / "run/reports/parity-ledger.json").exists()


def test_engineering_repeat_is_idempotent(tmp_path):
    from minicells_r3c_moe_bridge.runner import engineering

    first = engineering(tmp_path / "repeat")
    second = engineering(tmp_path / "repeat")
    assert [r["checkpoint_sha256"] for r in first["arms"]] == [
        r["checkpoint_sha256"] for r in second["arms"]
    ]
    assert all(r["exact_full_state"] for r in second["resume"])
