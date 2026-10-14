import copy
import torch
import pytest
from pd001.models import PDCLM
from pd001.losses import joint_loss
from pd001 import training
from performance_helpers import original, fixture, final_loader, tensors_equal, BASELINE


@pytest.mark.parametrize("mechanism", ["J0", "J1"])
def test_original_replay_forward_connection_and_all_gradients(proposal, mechanism):
    c, _, config, _ = fixture(proposal, "P50-A32-J1")
    config["adaptive"] = True
    old = original("models").PDCLM(config, "P50-A32-J1", 17).train()
    new = PDCLM(config, "P50-A32-J1", 17).train()
    new.load_state_dict(old.state_dict())
    x = torch.tensor([[1, 2]])
    replay = torch.tensor([[[0, 0, 2, 1, 3], [1, 1, 2, 0, 3]]])
    signals = []

    def run(model, capture):
        scores = []
        handle = model.controller.register_forward_hook(
            lambda m, i, o: scores.append(o[0].detach().clone())
        )
        outputs = [
            model(x, replay=replay, capture=capture) for _ in range(1 if mechanism == "J0" else 2)
        ]
        loss, log = joint_loss(outputs, torch.tensor([[2, 3]]), c["training"], mechanism)
        loss.backward()
        handle.remove()
        signals.append(scores)
        return outputs, loss, log

    a, al, alog = run(old, True)
    b, bl, blog = run(new, False)
    assert torch.equal(al, bl)
    for key in alog:
        assert torch.equal(alog[key], blog[key])
    for x, y in zip(a, b):
        for field in (
            "logits",
            "actions",
            "log_prob",
            "values",
            "cost_units",
            "lengths",
            "forced_stop",
        ):
            assert torch.equal(getattr(x, field), getattr(y, field))
        assert x.state_records and not y.state_records
    tensors_equal(signals[0], signals[1])
    for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad), name
    for prefix in ("cells.", "controller.scorer.", "controller.source.", "controller.critic."):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for n, p in new.named_parameters()
            if n.startswith(prefix)
        )
    assert new.controller.scorer[-1].weight.grad[-1].abs().sum() > 0  # STOP score


@pytest.mark.parametrize("arm", ["M0", "P50-F16", "P50-A32-J1"])
def test_real_original_runner_vs_light_multiple_steps(proposal, tmp_path, monkeypatch, arm):
    c, p, m, assets = fixture(proposal, arm)
    old = original("training")
    old.Counter = original("accounting").Counter
    old.gradients = original("diagnostics").gradients
    for module in (old, training):
        monkeypatch.setattr(module, "arm_config", lambda *a: m)
        monkeypatch.setattr(module, "load_bundle", final_loader)
    old._train_registered(
        c, p, arm, 17, "development", BASELINE, assets, "synthetic", tmp_path / "old", "cpu", False
    )
    training._train_registered(
        c, p, arm, 17, "development", "a" * 40, assets, "synthetic", tmp_path / "new", "cpu", False
    )
    a = torch.load(tmp_path / "old/checkpoint.pt", weights_only=False)
    b = torch.load(tmp_path / "new/checkpoint.pt", weights_only=False)
    for key in ("model", "optimizer", "step", "numpy_rng", "torch_rng", "python_rng", "cuda_rng"):
        if key in a:
            tensors_equal(a[key], b[key])
    tensors_equal(a["extra"]["path_rngs"], b["extra"]["path_rngs"])
    assert a["identity"]["source_sha"] != b["identity"]["source_sha"]
    assert b["identity"]["implementation_version"] == "pd001-light-v1"


def test_scientific_config_and_model_source_unchanged(proposal):
    import json
    import subprocess
    from pd001.configs import ROOT, BASE

    for name in ("models", "router", "losses"):
        current = (BASE / "implementation/pd001" / f"{name}.py").read_text()
        old = subprocess.check_output(
            [
                "git",
                "show",
                BASELINE
                + ":"
                + str((BASE / "implementation/pd001" / f"{name}.py").relative_to(ROOT)),
            ],
            cwd=ROOT,
            text=True,
        )
        assert current == old
    old = json.loads(
        subprocess.check_output(
            [
                "git",
                "show",
                BASELINE + ":" + str((BASE / "config/proposed.json").relative_to(ROOT)),
            ],
            cwd=ROOT,
            text=True,
        )
    )
    new = copy.deepcopy(proposal)
    new.pop("performance")
    assert old == new
