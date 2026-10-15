from types import SimpleNamespace
import pytest
import torch
from clm001_r3 import (
    automatic,
    cli,
    data,
    publication,
    pilot,
    model,
    training,
    checkpoint,
    audit,
    specialization,
    profiling,
    evaluation,
)
from clm001_r3.common import load, authorize


@pytest.mark.parametrize("alignment", ["LOCAL_ALIGNMENT_PASS", "INCONCLUSIVE"])
def test_one_launch_gates_beta_then_training_and_resume(monkeypatch, tmp_path, alignment):
    c = load()
    output = tmp_path / "output"
    assets = tmp_path / "assets"
    assets.mkdir()
    events = []
    monkeypatch.setattr(cli, "cpu_gate", lambda *args: events.append("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)

    def publish(*args, **kwargs):
        events.append("return-check" if kwargs.get("preflight_only") else "publish")
        return {"status": "TEST_PUBLICATION"}

    monkeypatch.setattr(publication, "publish", publish)
    monkeypatch.setattr(
        data,
        "load_assets",
        lambda *args: {
            "asset_sha256": "test-assets",
            "tokenizer_sha256": c["data"]["required_r1_tokenizer_sha256"],
        },
    )

    def gradient_pilot(config, assets, output, arm, device):
        events.append("pilot-" + arm)
        return {
            "status": "GPU_PILOT_PASS",
            "p90_step_seconds": 0.01,
            "shadow_rehearsal_seconds": 0.1,
            "gradient_rows": [{"q_grad_norm": 2.0, "lcg_grad_norm": 1.0}],
        }

    monkeypatch.setattr(pilot, "run", gradient_pilot)

    class FakeModel:
        def __init__(self, *args):
            pass

        def to(self, *args):
            return self

    monkeypatch.setattr(model, "Model", FakeModel)
    monkeypatch.setattr(training, "Trainer", lambda model: SimpleNamespace(model=model))
    monkeypatch.setattr(checkpoint, "restore", lambda *args: None)
    monkeypatch.setattr(audit, "capture", lambda *args: ([], []))
    monkeypatch.setattr(
        audit, "measure", lambda *args, **kwargs: {"summary": {"status": alignment}}
    )
    monkeypatch.setattr(specialization, "snapshot", lambda *args, **kwargs: {})
    monkeypatch.setattr(evaluation, "evaluate", lambda *args, **kwargs: {})

    def workers(active, *args):
        authorize(active, "train-dev", "cuda:0")
        events.append("train")
        assert active["competition"]["beta"] == 1.0 and not active["authorization"]["confirmation"]
        return [{"arm": arm, "exit_code": 0} for arm in c["arms"]]

    monkeypatch.setattr(automatic, "workers", workers)
    monkeypatch.setattr(profiling, "completed", lambda *args: events.append("profile"))
    monkeypatch.setattr(automatic, "aggregate", lambda *args: {"status": "TEST_ANALYSIS"})
    automatic.run(c, output, assets)
    if alignment == "LOCAL_ALIGNMENT_PASS":
        assert (
            events.index("pilot-QS-R3")
            < events.index("train")
            < events.index("profile")
            < events.index("publish")
        )
        automatic.run(c, output, assets)
        assert (
            sum(e.startswith("pilot-") for e in events) == 3
        )  # Cached verified gates, no repeated GPU Pilot.
        assert (
            events.count("train") == 2
        )  # Worker resumes its checkpoint/completed result, not fresh training.
    else:
        assert "train" not in events and "profile" not in events
        assert events[-1] == "publish"
