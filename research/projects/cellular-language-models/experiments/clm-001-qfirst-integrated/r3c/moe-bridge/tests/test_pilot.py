from copy import deepcopy
import numpy as np
import pytest
import torch
from minicells_r3c_moe_bridge import pilot, publication
from minicells_r3c_moe_bridge.data import synthetic_batches
from minicells_r3c_moe_bridge.training import Trainer
from minicells_r3c_moe_bridge.checkpoint import publish_segment, restore_segment
from minicells_r3c_moe_bridge.release import require_development
from test_state_and_storage import FakeHub


@pytest.fixture
def short(config, monkeypatch):
    c = deepcopy(config)
    c["input"]["asset_bundle_ref"] = pilot.pilot_configuration()["input"]["asset_bundle_ref"]
    c["balance"].update(minimum_cell_share=0, pilot_steps_per_candidate=1)
    c["pilot"].update(steps_per_arm=1, total_optimizer_updates=8, training_tokens=64)
    monkeypatch.setattr(
        pilot, "train_batches", lambda data, c, rng, device: synthetic_batches(c, rng, device)
    )

    def windows(data, c, role, device):
        assert role == "calibration"
        for i in range(2):
            x, y = synthetic_batches(c, np.random.default_rng(19 + i), device)[0]
            yield str(i), x, y

    monkeypatch.setattr(pilot, "fixed_windows", windows)
    return c


def test_released_budget_and_formal_still_denied():
    c = pilot.pilot_configuration()
    assert c["pilot"]["total_optimizer_updates"] == 40 and c["pilot"]["training_tokens"] == 20480
    assert c["balance"]["minimum_cell_share"] == 1 / 128
    with pytest.raises(RuntimeError, match="NOT_RELEASED"):
        require_development(c, None)


def test_all_jobs_automatic_and_repeated_run_skips_updates(short, tmp_path, monkeypatch):
    calls = []
    old = Trainer.update

    def update(self, *a, **k):
        calls.append((self.model.arm, self.step))
        return old(self, *a, **k)

    monkeypatch.setattr(Trainer, "update", update)
    first = pilot.execute_pilot(short, {}, tmp_path, "source", "cpu")
    assert first["logical_optimizer_updates"] == 8 and len(first["arms"]) == 5 and len(calls) == 8
    again = pilot.execute_pilot(short, {}, tmp_path, "source", "cpu")
    assert len(calls) == 8 and first == again
    summary = publication.pilot_public_summary(first)
    assert summary["formal_500_step_authorized"] is False
    assert all(
        r["cost"]["cell_rows"] == 32 and r["cost"]["context_rows"] == 4 for r in summary["arms"]
    )


def test_mid_job_resume_does_not_repeat_committed_update(short, tmp_path, monkeypatch):
    old = pilot.timed_update
    calls = []

    def fail(trainer, batches, device):
        if trainer.step == 1:
            raise RuntimeError("interrupted before next optimizer update")
        calls.append(trainer.step)
        return old(trainer, batches, device)

    monkeypatch.setattr(pilot, "timed_update", fail)
    with pytest.raises(RuntimeError):
        pilot.execute_job(short, {}, tmp_path, "source", "test", "C", 0.01, 2, "cpu")
    monkeypatch.setattr(
        pilot, "timed_update", lambda tr, b, d: (calls.append(tr.step), old(tr, b, d))[1]
    )
    result = pilot.execute_job(short, {}, tmp_path, "source", "test", "C", 0.01, 2, "cpu")
    assert calls == [0, 1] and result["steps"] == 2


def test_export_only_latest_and_one_hf_commit_with_pointer(short, tmp_path):
    c = deepcopy(short)
    c["balance"]["pilot_steps_per_candidate"] = 2
    pilot.execute_pilot(c, {}, tmp_path / "run", "source", "cpu")
    export = pilot.export_latest(tmp_path / "run/private-checkpoints", tmp_path / "export")
    assert len(list(export.rglob("*.pt"))) == 8
    assert len(list((tmp_path / "run/private-checkpoints").rglob("*.pt"))) == 11
    assert len(list(export.rglob("*step-*.json"))) == 0
    hub = FakeHub(tmp_path / "hub")
    identity = {"source": "test"}
    ref = publish_segment(hub, export, identity, "writer", pointer_path="pilot/latest.json")
    assert hub.commits == 1 and (hub.root / "pilot/latest.json").exists()
    assert pilot.export_matches(hub, ref, export)
    restore_segment(hub, ref, tmp_path / "restore", identity, "reader")
    assert len(list((tmp_path / "restore").rglob("*.pt"))) == 8


def test_secret_and_preflight_failure_stop_before_assets(short, tmp_path, monkeypatch):
    c = pilot.pilot_configuration()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    events = []
    monkeypatch.setattr(pilot, "pilot_configuration", lambda: c)
    monkeypatch.setattr(pilot, "load_secrets", lambda: events.append("secrets"))

    class Hub(FakeHub):
        def __init__(self, repo):
            super().__init__(tmp_path / "hub")

        def check_write(self):
            events.append("hf-check")

    monkeypatch.setattr(pilot, "HubBackend", Hub)

    def blocked(*a):
        events.append("github-check")
        raise RuntimeError("blocked")

    monkeypatch.setattr(publication, "prepare_pilot_publication", blocked)
    monkeypatch.setattr(
        pilot, "private_asset_subset", lambda *a: pytest.fail("assets accessed after failure")
    )
    with pytest.raises(RuntimeError, match="blocked"):
        pilot.automatic(tmp_path / "run", None)
    assert events == ["secrets", "hf-check", "github-check"]


def test_no_go_is_terminal_and_does_not_start_arms(short, tmp_path, monkeypatch):
    monkeypatch.setattr(
        pilot, "choose_balance", lambda c, m: {"status": "NO_GO", "coefficient": None}
    )
    report = pilot.execute_pilot(short, {}, tmp_path, "source", "cpu")
    assert report["status"] == "NO_GO_BALANCE_EXPOSURE" and report["arms"] == []
    assert report["logical_optimizer_updates"] == 3


def test_pointer_restore_and_resume_uses_saved_job_states(short, tmp_path):
    original = pilot.execute_pilot(short, {}, tmp_path / "a", "source", "cpu")
    export = pilot.export_latest(tmp_path / "a/private-checkpoints", tmp_path / "export")
    hub = FakeHub(tmp_path / "hub")
    ident = {"run": "test"}
    ref = publish_segment(hub, export, ident, "a", pointer_path="latest.json")
    restore_segment(hub, ref, tmp_path / "b/private-checkpoints", ident, "b")
    restored = pilot.execute_pilot(short, {}, tmp_path / "b", "source", "cpu")
    assert original == restored


def test_failure_report_keeps_saved_state_and_no_scientific_pass(short, tmp_path, monkeypatch):
    pilot.execute_job(
        short, {}, tmp_path / "private-checkpoints", "source", "balance-0", "C", 0.0, 1, "cpu"
    )
    report = pilot.partial_report(
        short, tmp_path, "source", RuntimeError("do not expose secret text")
    )
    assert report["logical_optimizer_updates"] == 1 and not report["gpu_pilot_completed"]
    assert "secret" not in str(report.get("failure"))
    summary = publication.pilot_public_summary(report)
    assert not summary["scientific_decision"] and not summary["actual_kaggle_persistence_verified"]


def test_one_launch_return_and_fresh_session_reuses_hf(short, tmp_path, monkeypatch):
    from minicells_r3c_moe_bridge.common import read

    events = []
    hub = FakeHub(tmp_path / "hub")
    hub.optional = (
        lambda name, revision: read(hub.root / name) if (hub.root / name).exists() else None
    )
    monkeypatch.setattr(pilot, "pilot_configuration", lambda: short)
    monkeypatch.setattr(pilot, "source_sha", lambda: "f" * 40)
    monkeypatch.setattr(pilot, "load_secrets", lambda: events.append("secrets"))
    monkeypatch.setattr(pilot, "HubBackend", lambda repo: hub)
    monkeypatch.setattr(pilot, "private_asset_subset", lambda *args: events.append("assets") or {})
    monkeypatch.setattr(
        publication,
        "prepare_pilot_publication",
        lambda *args: events.append("github-preflight") or {},
    )
    monkeypatch.setattr(
        publication,
        "publish_pilot_result",
        lambda *args: events.append("publish") or {"branch": "results/test"},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *a: None)
    original = pilot.execute_pilot
    monkeypatch.setattr(
        pilot, "execute_pilot", lambda c, d, out, src, dev: original(c, d, out, src, "cpu")
    )
    first = pilot.automatic(tmp_path / "session-a", tmp_path / "assets")
    assert first["logical_optimizer_updates"] == 8 and hub.commits == 1
    # Fresh filesystem restores saved model+optimizer/RNG, and never repeats a completed update.
    monkeypatch.setattr(Trainer, "update", lambda *a, **k: pytest.fail("retrained a completed job"))
    second = pilot.automatic(tmp_path / "session-b", tmp_path / "assets")
    assert second["logical_optimizer_updates"] == 8 and hub.commits == 1
    third = pilot.automatic(tmp_path / "session-b", tmp_path / "assets")
    assert third["logical_optimizer_updates"] == 8 and hub.commits == 1
    assert events[:3] == ["secrets", "github-preflight", "assets"] and events.count("publish") == 3
