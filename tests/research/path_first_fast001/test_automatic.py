import copy
import pytest
from exp006.reuse import canonical_sha256
from fast001.io import write_json
from fast001 import automatic as auto


def wire(monkeypatch, events):
    monkeypatch.setattr(auto, "preflight", lambda *a: events.append("preflight"))
    monkeypatch.setattr(auto, "gpu_check", lambda *a: events.append("hardware"))

    def prepare(c, path):
        events.append("data")
        path.mkdir()

    monkeypatch.setattr(auto, "prepare", prepare)
    monkeypatch.setattr(auto, "load_assets", lambda *a: events.append("data-check"))
    monkeypatch.setattr(
        auto,
        "benchmark",
        lambda *a, **kw: events.append("pilot") or {"projected_total_seconds": 9999},
    )

    def freeze(c, pilot, path, **kw):
        assert kw == {"approve": True, "accept_slow": True}
        events.append("freeze")
        frozen = copy.deepcopy(c)
        frozen["approval"]["base_config_sha256"] = canonical_sha256(c)
        write_json(path, frozen)
        return frozen

    monkeypatch.setattr(auto, "freeze", freeze)

    def matrix(*a, **kw):
        assert kw["arms"] == ["CLM-S16", "CLM-S8", "M0-TINY"] and kw["resume"] is True
        events.append("train")
        return {"status": "DEVELOPMENT_COMPLETED"}

    monkeypatch.setattr(auto, "matrix", matrix)
    for name in ("evaluate", "profile", "suite"):
        monkeypatch.setattr(auto, name, lambda *a, _name=name: events.append(_name))
    monkeypatch.setattr(
        auto, "report", lambda *a: events.append("report") or {"status": "DEVELOPMENT_COMPLETED"}
    )


def test_single_launch_checks_then_trains_without_prior_pilot(proposed, tmp_path, monkeypatch):
    events = []
    wire(monkeypatch, events)
    result = auto.run(proposed, tmp_path / "work", ["cuda:0", "cuda:1"])
    assert events[:7] == ["preflight", "hardware", "data", "data-check", "pilot", "freeze", "train"]
    assert events[-1] == "report" and result["status"] == "DEVELOPMENT_COMPLETED"


def test_failed_preflight_cannot_start_pilot_or_training(proposed, tmp_path, monkeypatch):
    events = []
    wire(monkeypatch, events)

    def failed(*a):
        raise ValueError("functional check failed")

    monkeypatch.setattr(auto, "preflight", failed)
    with pytest.raises(ValueError):
        auto.run(proposed, tmp_path / "work", ["cuda:0"])
    assert events == []


def test_existing_matching_config_skips_new_pilot_and_resumes(proposed, tmp_path, monkeypatch):
    events = []
    wire(monkeypatch, events)
    output = tmp_path / "work"
    (output / "assets").mkdir(parents=True)
    frozen = copy.deepcopy(proposed)
    frozen["approval"]["base_config_sha256"] = canonical_sha256(proposed)
    write_json(output / "execution-config.json", frozen)
    monkeypatch.setattr(auto, "authorize", lambda *a: events.append("identity"))
    auto.run(proposed, output, ["cuda:0"])
    assert "pilot" not in events and "freeze" not in events and "train" in events
