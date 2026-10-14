import copy
import subprocess
import sys
import os
import torch
import pytest
from pathlib import Path
from pd001 import training
from pd001.persistence import export_bundle, import_bundle
from pd001.configs import BASE, ROOT
from exp006.reuse import canonical_sha256, write_json
from performance_helpers import fixture, final_loader, tensors_equal


def test_atomic_pause_portable_restart_and_completed_skip(proposal, tmp_path, monkeypatch):
    c, p, m, assets = fixture(proposal, steps=4)
    c["performance"]["pause_after_step"] = 2
    monkeypatch.setattr(training, "arm_config", lambda *a: m)
    monkeypatch.setattr(training, "load_bundle", final_loader)
    work = tmp_path / "work"
    work.mkdir()
    write_json(work / "execution-config.json", c)
    write_json(
        work / "execution-identity.json",
        {"source_sha": "a" * 40, "config_sha256": canonical_sha256(c)},
    )
    args = (c, p, "P50-F16", 17, "development", "a" * 40, assets, "synthetic")
    result = training._train_registered(*args, work / "runs/P50-F16", "cpu", False)
    assert result["status"] == "PAUSED_RESUMABLE" and result["step"] == 2
    assert not (work / "runs/P50-F16/metrics.json").exists()
    export_bundle(work, tmp_path / "portable", "a" * 40)
    with pytest.raises(ValueError, match="source mismatch"):
        import_bundle(tmp_path / "portable", tmp_path / "wrong", "b" * 40)
    restored = import_bundle(tmp_path / "portable", tmp_path / "restart", "a" * 40)
    assert restored == c
    # REAL new Python process simulates losing the source process and /working.
    runner = tmp_path / "restart.py"
    runner.write_text(
        """import json,sys\nfrom pathlib import Path\nfrom pd001 import training\nfrom performance_helpers import fixture,final_loader\nc=json.loads(Path(sys.argv[1]).read_text())\n_,p,m,assets=fixture(c,steps=4)\ntraining.arm_config=lambda *a:m\ntraining.load_bundle=final_loader\ntraining._train_registered(c,p,'P50-F16',17,'development','a'*40,assets,'synthetic',Path(sys.argv[2]),'cpu',True)\n"""
    )
    paths = [
        BASE / "implementation",
        BASE.parent / "implementation",
        ROOT / "src",
        ROOT / "shared/python",
        Path(__file__).parent,
    ]
    subprocess.run(
        [
            sys.executable,
            str(runner),
            str(tmp_path / "restart/execution-config.json"),
            str(tmp_path / "restart/runs/P50-F16"),
        ],
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(map(str, paths)),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    control = copy.deepcopy(c)
    control["performance"]["pause_after_step"] = None
    training._train_registered(
        control,
        p,
        "P50-F16",
        17,
        "development",
        "a" * 40,
        assets,
        "synthetic",
        tmp_path / "straight",
        "cpu",
        False,
    )
    a = torch.load(tmp_path / "straight/checkpoint.pt", weights_only=False)
    b = torch.load(tmp_path / "restart/runs/P50-F16/checkpoint.pt", weights_only=False)
    for key in ("model", "optimizer", "numpy_rng", "torch_rng", "python_rng"):
        tensors_equal(a[key], b[key])
    tensors_equal(a["extra"]["path_rngs"], b["extra"]["path_rngs"])
    # Completed result is returned after hash+identity verification, without any step.
    monkeypatch.setattr(
        training,
        "optimization_step",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("retrained completed arm")),
    )
    complete = training._train_registered(*args, tmp_path / "restart/runs/P50-F16", "cpu", True)
    assert complete["status"] == "DEVELOPMENT_COMPLETED"
    # Content corruption is detected before checkpoint deserialization/training.
    with (tmp_path / "portable/runs/P50-F16/checkpoint.pt").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        import_bundle(tmp_path / "portable", tmp_path / "corrupt", "a" * 40)


def test_pause_does_not_trigger_profile_or_publication(proposal, tmp_path, monkeypatch):
    import pd001.runner as runner
    from types import SimpleNamespace
    from pd001.configs import load

    monkeypatch.setattr("pd001.performance.retrain_review", lambda *a: None)
    monkeypatch.setattr(runner, "preflight", lambda *a: {"status": "ENGINEERING_READY"})
    monkeypatch.setenv("HF_TOKEN", "fixture")
    monkeypatch.setenv("GITHUB_TOKEN", "fixture")
    monkeypatch.setattr(
        "pd001.development.configure",
        lambda *a: (proposal, load()[1], tmp_path / "cfg", tmp_path / "assets", 17),
    )
    monkeypatch.setattr("pd001.publication.preflight_development", lambda *a: None)
    monkeypatch.setattr(
        "pd001.orchestration.matrix",
        lambda *a: {"status": "PAUSED_RESUMABLE", "workers_finished": False, "runs": []},
    )
    monkeypatch.setattr(
        runner, "scientific", lambda *a: (_ for _ in ()).throw(AssertionError("profile on pause"))
    )
    monkeypatch.setattr(
        "pd001.publication.publish",
        lambda *a: (_ for _ in ()).throw(AssertionError("publish on pause")),
    )
    monkeypatch.setattr("pd001.persistence.export_bundle", lambda *a: {})
    result = runner.auto_run(
        proposal,
        load()[1],
        SimpleNamespace(
            source_commit="a" * 40, output=tmp_path, seed=17, devices=["cpu"], arms=["M0"]
        ),
    )
    assert result["status"] == "PAUSED_RESUMABLE"
