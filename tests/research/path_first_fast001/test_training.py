import copy
import json
import numpy as np
import pytest
import torch
from fast001.models import make_model
from fast001.config import engine_config, authorize, freeze
from fast001.data import load_assets
from fast001.train import train
from fast001.io import write_json
from fast001.resume import export, restore
from pd001.engine import step


def test_light_instrumented_loss_gradient_rng_equivalence(tiny, assets):
    data = load_assets(tiny, assets)
    for arm in ("M0-TINY", "CLM-S8", "CLM-S16"):
        outputs = []
        for instrument in (False, True):
            model = make_model(tiny, arm)
            opt = torch.optim.AdamW(model.parameters(), lr=0.0003)
            path = [torch.Generator().manual_seed(11)]
            row = step(
                model,
                opt,
                engine_config(tiny),
                1,
                np.array(data["windows"]["train"]["starts"][:2]),
                data["train"],
                path,
                "cpu",
                capture=instrument,
                diagnostics=instrument,
                dispatch=instrument,
                routing=instrument,
            )
            outputs.append((copy.deepcopy(model.state_dict()), path[0].get_state(), row))
        assert all(torch.equal(p, outputs[1][0][n]) for n, p in outputs[0][0].items())
        assert torch.equal(outputs[0][1], outputs[1][1])
        assert outputs[0][2]["ce"] == outputs[1][2]["ce"]
        assert outputs[0][2]["observed_training_gemm_flops"] is None
        assert outputs[1][2]["observed_training_gemm_flops"] > 0


def assert_nested(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_nested(left, right)
    else:
        assert a == b


def test_atomic_pause_portable_restore_and_checkpoint_rng(tiny, assets, tmp_path):
    whole = tmp_path / "whole"
    partial = tmp_path / "partial"
    final = train(tiny, "CLM-S8", assets, whole)
    pause = train(tiny, "CLM-S8", assets, partial, pause_after_step=2)
    assert pause["status"] == "PAUSED_RESUMABLE"
    work = tmp_path / "work"
    work.mkdir()
    (work / "runs").mkdir()
    import shutil

    shutil.copytree(partial, work / "runs/CLM-S8")
    shutil.copytree(assets, work / "assets")
    write_json(work / "execution-config.json", tiny)
    bundle = tmp_path / "portable"
    export(work, bundle)
    restored = tmp_path / "new-session"
    c = restore(bundle, restored)
    # A genuinely new Python process resumes a user-attached portable fixture.
    import subprocess
    import sys
    import os
    from fast001.config import BASE

    subprocess.run(
        [
            sys.executable,
            str(BASE / "run_fast001.py"),
            "--mode",
            "train",
            "--arm",
            "CLM-S8",
            "--config",
            str(restored / "execution-config.json"),
            "--assets",
            str(restored / "assets"),
            "--output",
            str(restored / "runs/CLM-S8"),
            "--device",
            "cpu",
            "--resume",
        ],
        check=True,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    resumed = json.loads((restored / "runs/CLM-S8/metrics.json").read_text())
    assert resumed["status"] == "DEVELOPMENT_COMPLETED"
    a = torch.load(whole / "checkpoint.pt", weights_only=False)
    b = torch.load(restored / "runs/CLM-S8/checkpoint.pt", weights_only=False)
    for key in ("identity", "step", "model", "optimizer", "numpy_rng", "torch_rng", "python_rng"):
        assert_nested(a[key], b[key])
    assert_nested(a["extra"]["path_rngs"], b["extra"]["path_rngs"])
    assert [r["data_starts"] for r in a["extra"]["history"]] == [
        r["data_starts"] for r in b["extra"]["history"]
    ]
    assert final["quality"] == resumed["quality"]
    # Completed arms are reused without another optimizer step.
    checkpoint = restored / "runs/CLM-S8/checkpoint.pt"
    before = checkpoint.read_bytes()
    train(c, "CLM-S8", restored / "assets", restored / "runs/CLM-S8", resume=True)
    assert checkpoint.read_bytes() == before
    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        train(c, "CLM-S8", restored / "assets", restored / "runs/CLM-S8", resume=True)


def test_data_sampler_same_across_clm_depths(tiny, assets, tmp_path):
    for arm in ("CLM-S8", "CLM-S16"):
        train(tiny, arm, assets, tmp_path / arm)
    a = json.loads((tmp_path / "CLM-S8/history.json").read_text())
    b = json.loads((tmp_path / "CLM-S16/history.json").read_text())
    assert [r["data_starts"] for r in a] == [r["data_starts"] for r in b]


def test_development_training_and_gpu_fixture_fail_closed(proposed, tiny, tmp_path):
    with pytest.raises(RuntimeError, match="User review"):
        authorize(proposed, "cuda:0", "train")
    with pytest.raises(RuntimeError, match="CPU-only"):
        authorize(tiny, "cuda:0", "train")
    with pytest.raises(RuntimeError, match="explicit"):
        freeze(proposed, {}, tmp_path / "frozen.json")
    with pytest.raises(RuntimeError, match="actual"):
        freeze(proposed, {}, tmp_path / "frozen.json", approve=True)
    bad = copy.deepcopy(tiny)
    bad["approval"]["confirmation"] = True
    with pytest.raises(ValueError, match="sealed"):
        authorize(bad, "cpu", "train")


def test_nan_fails_before_optimizer_update(tiny, assets):
    model = make_model(tiny, "CLM-S8")
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    with torch.no_grad():
        model.embedding.weight.fill_(float("nan"))
    data = load_assets(tiny, assets)
    with pytest.raises((FloatingPointError, RuntimeError)):
        step(
            model,
            opt,
            engine_config(tiny),
            1,
            np.array([0, 4]),
            data["train"],
            [torch.Generator().manual_seed(1)],
            "cpu",
        )
    assert len(opt.state) == 0


def test_interrupted_checkpoint_receipt_transaction_recovers_prior_pair(tiny, assets, tmp_path):
    whole = tmp_path / "whole"
    partial = tmp_path / "partial"
    train(tiny, "CLM-S8", assets, whole)
    train(tiny, "CLM-S8", assets, partial, pause_after_step=3)
    assert (partial / "checkpoint.previous.pt").exists()
    # Simulate process interruption after replacing checkpoint, before committing receipt.
    (partial / "checkpoint.pt").write_bytes(b"incomplete latest transaction")
    resumed = train(tiny, "CLM-S8", assets, partial, resume=True)
    assert resumed["status"] == "DEVELOPMENT_COMPLETED"
    a = torch.load(whole / "checkpoint.pt", weights_only=False)
    b = torch.load(partial / "checkpoint.pt", weights_only=False)
    assert_nested(a["model"], b["model"])
    assert_nested(a["optimizer"], b["optimizer"])
