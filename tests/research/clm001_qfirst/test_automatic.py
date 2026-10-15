import json
import pytest
from clm001 import automatic
from clm001.common import load, authorize, source_sha, canonical_sha256


def configured():
    return automatic.runtime_config(
        load(), {"asset_sha256": "asset", "tokenizer_sha256": "tokenizer"}, source_sha()
    )


def pilot(c):
    return {
        "source_commit": source_sha(),
        "config_sha256": canonical_sha256(c),
        "asset_sha256": "asset",
        "tokenizer_sha256": "tokenizer",
        "arms": {
            arm: {
                "local_restore_pass": True,
                "oracle": {"states": 1},
                "update_rows": [{"live_ce": 3.0, "synchronized_total_seconds": 1.0}],
                "projected_500_update_seconds": 500.0,
            }
            for arm in c["arms"]
        },
    }


def test_automatic_requires_real_passes_without_manual_flags():
    c = configured()
    assert not c["gates"]["gpu_pilot_approved"] and not c["gates"]["portable_pass"]
    authorize(c, "gpu-pilot")
    with pytest.raises(RuntimeError):
        authorize(c, "train-dev")
    c["execution"]["checks"].update(gpu_functional=True, local_restore=True)
    authorize(c, "train-dev")
    assert c["policy"]["lambda_cost"] == 0.01 and not c["gates"]["thresholds_frozen"]
    c["training"]["steps"] = 50
    c["training"]["target_tokens"] = 25600
    with pytest.raises(RuntimeError):
        authorize(c, "train-dev")


@pytest.mark.parametrize("bad", ["source", "restart", "cost", "oracle", "nonfinite"])
def test_automatic_preflight_fails_closed(bad):
    c = configured()
    r = pilot(c)
    a = r["arms"]["QF-FULL"]
    if bad == "source":
        r["source_commit"] = "wrong"
    elif bad == "restart":
        a["local_restore_pass"] = False
    elif bad == "cost":
        a["projected_500_update_seconds"] = 7201
    elif bad == "oracle":
        a["oracle"]["states"] = 0
    else:
        a["update_rows"][0]["live_ce"] = float("nan")
    with pytest.raises(RuntimeError):
        automatic.validate_pilot(r, c)


def install_fake_flow(monkeypatch, tmp_path, fail_cpu=False):
    from clm001 import cli, pilot as pilot_module

    calls = []

    def cpu(c, output):
        calls.append("cpu")
        if fail_cpu:
            raise RuntimeError("failed CPU check")

    monkeypatch.setattr(cli, "cpu_gate", cpu)
    monkeypatch.setattr(automatic.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(automatic.torch.cuda, "device_count", lambda: 2)

    def publish(*args, **kwargs):
        calls.append("return-preflight" if kwargs.get("preflight_only") else "return")
        return {"status": "OK"}

    monkeypatch.setattr(automatic, "publish", publish)
    monkeypatch.setattr(automatic, "prepare", lambda c, path: calls.append("assets"))
    monkeypatch.setattr(
        automatic,
        "load_assets",
        lambda *a: {"asset_sha256": "asset", "tokenizer_sha256": "tokenizer"},
    )

    def gpu(c, *args):
        calls.append("gpu-preflight")
        return pilot(c)

    monkeypatch.setattr(pilot_module, "run", gpu)

    def workers(c, config, output, assets):
        calls.append("development")
        authorize(c, "train-dev")
        for arm in c["arms"]:
            (output / arm).mkdir()
            (output / arm / "metrics.json").write_text("{}")
            (output / arm / "shadow-journal.json").write_text("{}")

    monkeypatch.setattr(automatic, "workers", workers)
    monkeypatch.setattr(
        automatic,
        "summarize",
        lambda *a, **kw: {"status": "DEVELOPMENT_COMPLETED", "scientific_decision": False},
    )
    return calls


def test_one_invocation_continues_and_returns(monkeypatch, tmp_path):
    calls = install_fake_flow(monkeypatch, tmp_path)
    result = automatic.run(load(), tmp_path / "out", tmp_path / "assets")
    assert calls == ["cpu", "return-preflight", "assets", "gpu-preflight", "development", "return"]
    assert result["status"] == "COMPLETED" and not result["real_kaggle_persistence_verified"]
    c = json.loads((tmp_path / "out/runtime-config.json").read_text())
    assert not c["gates"]["thresholds_frozen"] and c["execution"]["checks"]["local_restore"]


def test_failed_cpu_never_reaches_gpu_or_training(monkeypatch, tmp_path):
    calls = install_fake_flow(monkeypatch, tmp_path, True)
    with pytest.raises(RuntimeError):
        automatic.run(load(), tmp_path / "out", tmp_path / "assets")
    assert calls == ["cpu"]


def test_unfrozen_admission_measures_but_never_promotes(config):
    from clm001.model import Model
    from clm001.training import Trainer
    from clm001.shadows import Nursery
    from clm001.admission import assess
    from test_core import batch

    config["execution"] = {"mode": "automatic-development"}
    config["shadow"]["min_cost_aware_improvement"] = None
    config["shadow"]["nll_regression_tolerance"] = None
    m = Model(config, "QF-FULL")
    t = Trainer(m)
    n = Nursery()
    candidate = n.propose(m, "cell", t.opt)
    n.probation(candidate, m, [batch()], 1)
    before = {"paired_identity": "test", "nll": [5.0, 5.0], "cost": [1.0, 1.0]}
    after = {"paired_identity": "test", "nll": [1.0, 1.0], "cost": [0.1, 0.1]}
    assert assess(candidate, m, before, after, [0, 1], config) is False
    assert (
        candidate.state == "REJECTED" and candidate.metrics["admission_thresholds_frozen"] is False
    )


@pytest.mark.parametrize("failure_stage", ["return", "assets", "gpu"])
def test_preflight_failure_never_starts_development(monkeypatch, tmp_path, failure_stage):
    from clm001 import pilot as pilot_module

    calls = install_fake_flow(monkeypatch, tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("injected " + failure_stage + " failure")

    if failure_stage == "return":
        monkeypatch.setattr(automatic, "publish", fail)
    elif failure_stage == "assets":
        monkeypatch.setattr(automatic, "prepare", fail)
    else:
        monkeypatch.setattr(pilot_module, "run", fail)
    with pytest.raises(RuntimeError):
        automatic.run(load(), tmp_path / "out", tmp_path / "assets")
    assert "development" not in calls
