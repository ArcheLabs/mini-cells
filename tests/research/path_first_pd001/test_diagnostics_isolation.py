import copy
import numpy as np
import pytest
import torch
from pd001 import training
from performance_helpers import fixture, final_loader, tensors_equal


@pytest.mark.parametrize("arm", ["P50-F16", "P50-A32-J1", "M0"])
def test_periodic_diagnostics_and_probes_do_not_change_rng_or_weights(
    proposal, tmp_path, monkeypatch, arm
):
    c, p, m, assets = fixture(proposal, arm)
    monkeypatch.setattr(training, "arm_config", lambda *a: m)
    monkeypatch.setattr(training, "load_bundle", final_loader)
    for interval in (1, 100):
        config = copy.deepcopy(c)
        config["performance"].update(diagnostics_interval=interval, train_probe_interval=interval)
        training._train_registered(
            config,
            p,
            arm,
            17,
            "development",
            "a" * 40,
            assets,
            "synthetic",
            tmp_path / str(interval),
            "cpu",
            False,
        )
    a = torch.load(tmp_path / "1/checkpoint.pt", weights_only=False)
    b = torch.load(tmp_path / "100/checkpoint.pt", weights_only=False)
    tensors_equal(a["model"], b["model"])
    tensors_equal(a["optimizer"], b["optimizer"])
    tensors_equal(a["extra"]["path_rngs"], b["extra"]["path_rngs"])
    assert np.array_equal(a["extra"]["numpy_global_rng"][1], b["extra"]["numpy_global_rng"][1])
    for row in b["extra"]["history"]:
        assert row["train_probe_greedy_nll"] is None and row["probe_sampled"] is False
        assert not row["diagnostics_sampled"] and row["forward_gemm_flops"] is None


def test_light_never_installs_resident_instruments(proposal, tmp_path, monkeypatch):
    from pd001 import engine

    c, p, m, assets = fixture(proposal)
    monkeypatch.setattr(training, "arm_config", lambda *a: m)
    monkeypatch.setattr(training, "load_bundle", final_loader)

    def forbidden(*a, **kw):
        raise AssertionError("resident diagnostic in light step")

    monkeypatch.setattr(training, "Counter", forbidden)
    monkeypatch.setattr(engine, "TrainingGemmCounter", forbidden)
    monkeypatch.setattr(engine, "gradients", forbidden)
    training._train_registered(
        c,
        p,
        "P50-F16",
        17,
        "development",
        "a" * 40,
        assets,
        "synthetic",
        tmp_path / "run",
        "cpu",
        False,
    )


def test_baseline_stability_keeps_original_first_and_final_optimizer_window(
    proposal, tmp_path, monkeypatch
):
    from pd001.analysis import baseline_review

    c, p, m, assets = fixture(proposal, "M0", steps=5)
    p = copy.deepcopy(p)
    p["thresholds"]["baseline_stability"] = {
        "tail_steps": 2,
        "minimum_nll_drop": -100,
        "tail_relative_std_max": 100,
        "tail_absolute_slope_max": 100,
    }
    monkeypatch.setattr(training, "arm_config", lambda *a: m)
    monkeypatch.setattr(training, "load_bundle", final_loader)
    result = training._train_registered(
        c, p, "M0", 17, "development", "a" * 40, assets, "synthetic", tmp_path / "run", "cpu", False
    )
    history = torch.load(tmp_path / "run/checkpoint.pt", weights_only=False)["extra"]["history"]
    assert [r["step"] for r in history if r["train_probe_greedy_nll"] is not None] == [1, 4, 5]
    assert result["baseline_stable"]
    assert not result["authorization"]["registered"]
    dense = [{"train_probe_greedy_nll": v} for v in [3.0, 2.8, 2.7, 2.6, 2.5]]
    sparse = copy.deepcopy(dense)
    sparse[1]["train_probe_greedy_nll"] = None
    sparse[2]["train_probe_greedy_nll"] = None
    assert baseline_review(dense, p["thresholds"]["baseline_stability"]) == baseline_review(
        sparse, p["thresholds"]["baseline_stability"]
    )
    sparse[-1]["train_probe_greedy_nll"] = None
    assert baseline_review(sparse, p["thresholds"]["baseline_stability"]) is False
