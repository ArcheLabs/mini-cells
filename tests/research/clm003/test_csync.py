"""C-sync changed-code tests; synthetic CPU only, no Confirmation access."""

import copy
import json
import os
import sys
from pathlib import Path

import csync as sync
import pytest
import screening as sc
import torch
from minicells_fast001.data import Stream

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fast002_r1"))
from test_stateful import fixture


def config():
    return json.loads((sc.BASE / "CSYNC_MANIFEST.json").read_text())


def test_rng_and_label_free_contract():
    state = {"weight": torch.zeros(6, 8), "bias": torch.zeros(6)}
    before = torch.random.get_rng_state().clone()
    a = sync.direction(state, 901)
    b = sync.direction(state, 901)
    c = sync.direction(state, 902)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], c[k]) for k in a)
    assert torch.equal(before, torch.random.get_rng_state())
    assert sum(x.square().sum() for x in a.values()).item() == pytest.approx(1)
    with pytest.raises(RuntimeError, match="UNLABELLED"):
        sync.validate_probe(
            {"role": "routing_probe", "inputs": [], "sha256": "x", "labels": []}, "x"
        )
    with pytest.raises(RuntimeError, match="IDENTITY"):
        sync.validate_probe({"role": "routing_probe", "inputs": [], "sha256": "x"}, "x")


def test_all_three_metrics_are_required():
    m = config()["matching"]
    target = {"action_change": 0.1, "path_change": 0.2, "kl": 0.001}
    assert sync.matches(target, target, m)
    for key in target:
        other = dict(target)
        other[key] = 1
        assert not sync.matches(other, target, m)
    target["kl"] = float("nan")
    assert not sync.matches(target, target, m)


def test_native_short_paired_trial(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "fsync", lambda fd: None)
    c, parent, folder = fixture(tmp_path)
    base = copy.deepcopy(sc.manifest())
    c["stage_b"] = copy.deepcopy(base["runtime_template"]["stage_b"])
    base.update(steps=2, update_steps=[2], runtime_template=c)
    x = torch.tensor([[7, 8, 9, 10]])
    base["runtime_template"]["admission"].update(
        manifest_sha256="fixture", ordered_input_hashes=[sc.rv.sha(x.squeeze(0).tolist())]
    )
    bundle = {"role": "admission", "manifest_sha256": "fixture", "inputs": [x]}
    probe = {"role": "routing_probe", "inputs": [torch.tensor([[11, 12, 13, 14]])]}
    probe["sha256"] = sync.probe_identity(probe["inputs"])
    m = config()
    m["probe"]["identity"] = probe["sha256"]
    m["update_steps"] = [2]
    # Bounded engineering fixture budget; production K/scales remain unchanged in manifest.
    m["random"].update(K=2, scales=[0.25, 0.01, 0.0001])
    a = sc.engine(base, parent, 7001, "A", tmp_path / "trial", bundle, unit=True)
    initial = copy.deepcopy(a.model.connection.state_dict())
    stream = Stream(a.fast, folder, 7001, "H", "B")
    for _ in range(2):
        a.update(stream, a.stage_directory / "latest.pt", [])
    before = copy.deepcopy(a.model)
    before.connection.load_state_dict(initial)
    event = sync.accepted_event(
        before,
        a.model,
        seed=7001,
        step=2,
        accepted=sc.rv.fingerprint(initial) != sc.rv.fingerprint(a.model.connection.state_dict()),
        manifest=m,
        probe=probe,
    )
    e = sc.engine(base, parent, 7001, "B", tmp_path / "trial", bundle, unit=True)
    e.screen_arm = "C-sync"
    stream = Stream(e.fast, folder, 7001, "H", "B")
    logs = []

    def hook(engine):
        if engine.step == 2:
            logs.append(sync.synchronize(engine, event, probe, m, log_path=tmp_path / "EVENT.json"))

    for _ in range(2):
        e.update(stream, e.stage_directory / "latest.pt", [], failure_hook=hook)
    assert [r["input_sha256"] for r in a.curve] == [r["input_sha256"] for r in e.curve]
    r = logs[0]
    assert r["cost"]["candidate_evaluations"] <= 6
    assert r["cost"]["optimizer_calls"] == 0
    assert r["status"] in {"MATCHED", "UNMATCHABLE", "A_REJECTED_NO_C_UPDATE"}
    if r["status"] != "MATCHED":
        assert r["before"] == r["after"]
    # Replay of an uncertain event is blocked, never silently resampled.
    with pytest.raises(RuntimeError, match="ALREADY_ATTEMPTED"):
        sync.synchronize(e, event, probe, m, log_path=tmp_path / "EVENT.json")
    refused = dict(event, accepted=False)
    skipped = sync.synchronize(e, refused, probe, m, log_path=tmp_path / "REJECTED.json")
    assert skipped["cost"]["candidate_evaluations"] == 0
    assert skipped["before"] == skipped["after"]
    # Force impossible geometry; fail closed at K * scales, no tolerance relaxation.
    impossible = dict(
        event, accepted=True, geometry={"action_change": 2, "path_change": 2, "kl": 2}
    )
    failed = sync.synchronize(e, impossible, probe, m, log_path=tmp_path / "UNMATCHABLE.json")
    assert failed["status"] == "UNMATCHABLE"
    assert failed["cost"]["candidate_evaluations"] == 6
    assert failed["before"] == failed["after"]
    with pytest.raises(RuntimeError, match="CONTEXT"):
        sync.synchronize(e, dict(event, step=1), probe, m, log_path=tmp_path / "BAD.json")
    # Unit-only sentinel exercises successful commit; never counted as native feasibility.
    sentinel = {
        "action_change": 0.01,
        "path_change": 0.02,
        "kl": 0.00001,
        "parameter_delta_norm": 0.25,
    }
    monkeypatch.setattr(sync, "trust", lambda *a, **kw: dict(sentinel))
    monkeypatch.setattr(sync, "decide", lambda *a, **kw: {"passed": True})
    accepted = dict(event, accepted=True, geometry=sentinel)
    matched = sync.synchronize(e, accepted, probe, m, log_path=tmp_path / "MOCK_MATCHED.json")
    assert matched["status"] == "MATCHED"
    assert matched["before"] != matched["after"]
    assert matched["cost"]["candidate_evaluations"] == 1
    assert not sync.feasibility([failed], m)["mechanism_interpretable"]
    assert not sync.feasibility([], m)["mechanism_interpretable"]
    out = os.environ.get("CSYNC_FIXTURE_REPORT")
    if out:
        dest = Path(out)
        dest.mkdir(parents=True, exist_ok=True)
        for name in ["EVENT.json", "REJECTED.json", "UNMATCHABLE.json"]:
            (dest / name).write_bytes((tmp_path / name).read_bytes())
        sc.rv.atomic(
            dest / "FIXTURE_COST.json",
            {
                "scope": "SYNTHETIC_CPU_NOT_SCIENCE",
                "A": a.journal.summary(),
                "C": e.journal.summary(),
                "C_events": [r["cost"], failed["cost"]],
            },
        )
    for engine in [a, e]:
        for handle in engine._work_handles:
            handle.remove()
