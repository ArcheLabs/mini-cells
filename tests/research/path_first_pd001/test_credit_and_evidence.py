import copy
import itertools
import json
import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from pd001.losses import joint_loss
from pd001.configs import load, authorize, BASE
from pd001.analysis import paired_interval, noninferiority, seed_summary
from pd001.checkpoint import save, restore
from pd001.models import PDCLM
from pd001.training import train
from pd001.assets import load_bundle


def enumerated(theta, actions):
    """Learnable residual Cell changes the state used by the NEXT policy, including STOP."""
    h = theta.new_tensor(0.3)
    logps = []
    mask = []
    cost = 0.0
    for action in actions:
        logits = torch.stack([theta[1] * h + theta[2], -theta[1] * h])
        logps.append(logits.log_softmax(0)[action])
        mask.append(True)
        if action == 1:
            break
        h = h + torch.tanh(theta[0] * h + 0.2)
        cost += 1.0
    while len(logps) < 2:
        logps.append(theta.new_zeros(()))
        mask.append(False)
    q = torch.stack(logps).sum().exp()
    out = SimpleNamespace(
        logits=torch.stack([h, -h]).reshape(1, 1, 2),
        log_prob=torch.stack(logps).reshape(1, 1, 2),
        values=theta.new_zeros((1, 1, 2)),
        entropy=theta.new_zeros((1, 1, 2)),
        decision_mask=torch.tensor(mask).reshape(1, 1, 2),
        cost_units=theta.new_full((1, 1), cost),
    )
    return q, out


@pytest.mark.parametrize("mechanism", ["J0", "J1"])
def test_exact_enumerated_joint_cell_policy_gradient(mechanism):
    theta = torch.tensor([0.4, 0.7, -0.2], dtype=torch.float64, requires_grad=True)
    paths = [(1,), (0, 1), (0, 0)]
    rollouts = [enumerated(theta, path) for path in paths]
    assert torch.allclose(sum(p for p, _ in rollouts), theta.new_tensor(1.0))
    target = torch.tensor([[1]])
    settings = {"critic_coefficient": 0.0, "entropy_coefficient": 0.0, "moe_balance_coefficient": 0.0}
    exact = sum(
        p * (F.cross_entropy(o.logits.reshape(1, 2), target.flatten()) + 0.13 * o.cost_units.mean())
        for p, o in rollouts
    )
    exact_grad = torch.autograd.grad(exact, theta, retain_graph=True)[0]
    estimate = torch.zeros_like(theta)
    if mechanism == "J0":
        for probability, out in rollouts:
            objective, _ = joint_loss([out], target, settings, "J0", 0.13)
            estimate += probability.detach() * torch.autograd.grad(objective, theta, retain_graph=True)[0]
    else:
        for (pa, a), (pb, b) in itertools.product(rollouts, repeat=2):
            objective, _ = joint_loss([a, b], target, settings, "J1", 0.13)
            estimate += (pa * pb).detach() * torch.autograd.grad(objective, theta, retain_graph=True)[0]
    assert torch.allclose(exact_grad, estimate, atol=1e-10, rtol=1e-9)
    assert exact_grad[0].abs() > 0  # genuine state-computed policy dependence, not preset numbers


@pytest.mark.parametrize("mechanism", ["J0", "J1"])
def test_sample_average_joint_gradient_matches_enumeration(mechanism):
    theta = torch.tensor([0.4, 0.7, -0.2], dtype=torch.float64, requires_grad=True)
    paths = [(1,), (0, 1), (0, 0)]
    rollouts = [enumerated(theta, path) for path in paths]
    targets = torch.tensor([[1]])
    settings = {"critic_coefficient": 0.0, "entropy_coefficient": 0.0, "moe_balance_coefficient": 0.0}
    exact = sum(
        p * (F.cross_entropy(o.logits.reshape(1, 2), targets.flatten()) + 0.13 * o.cost_units.mean())
        for p, o in rollouts
    )
    expected = torch.autograd.grad(exact, theta, retain_graph=True)[0].numpy()
    gradients = []
    probabilities = []
    entries = (
        [(p, [o]) for p, o in rollouts]
        if mechanism == "J0"
        else [(pa * pb, [a, b]) for (pa, a), (pb, b) in itertools.product(rollouts, repeat=2)]
    )
    for probability, outs in entries:
        objective, _ = joint_loss(outs, targets, settings, mechanism, 0.13)
        gradients.append(torch.autograd.grad(objective, theta, retain_graph=True)[0].numpy())
        probabilities.append(float(probability.detach()))
    rng = np.random.default_rng(17)
    samples = np.asarray(gradients)[
        rng.choice(len(entries), size=60000, p=np.asarray(probabilities) / sum(probabilities))
    ]
    stderr = samples.std(0) / np.sqrt(len(samples))
    assert np.all(np.abs(samples.mean(0) - expected) < 6 * stderr + 1e-6)


def test_j1_other_return_is_detached_and_critic_has_no_label_input(tiny, proposal):
    m = PDCLM(tiny, "P50-A32-J1").train()
    from pd001.losses import sample_paths

    outs = sample_paths(
        m, torch.tensor([[1]]), [torch.Generator().manual_seed(21), torch.Generator().manual_seed(22)], "J1"
    )
    inputs = []
    handle = m.controller.critic.register_forward_pre_hook(
        lambda mod, args: inputs.append(args[0].detach().clone())
    )
    m(torch.tensor([[1]]), generator=torch.Generator().manual_seed(23), policy="sampled")
    handle.remove()
    assert inputs and inputs[0].shape[-1] == tiny["width"] + tiny["step_width"]
    # No cross-trajectory derivative through the LOO baseline in the actor term.
    losses = [F.cross_entropy(o.logits.flatten(0, 1), torch.tensor([2])) for o in outs]
    actor = (losses[0] - losses[1]).detach() * outs[0].log_prob.sum()
    cross = torch.autograd.grad(actor, outs[1].logits, allow_unused=True)
    assert cross == (None,)


def test_checkpoint_resume_joint_optimizer_rng(tiny, proposal, tmp_path):
    from pd001.losses import sample_paths

    c = copy.deepcopy(tiny)
    c["budget"] = 4
    c["adaptive"] = False
    settings = proposal["training"]

    def build():
        model = PDCLM(c, "P50-A32-J1", seed=17).train()
        opt = torch.optim.AdamW(model.parameters(), lr=0.002)
        data = np.random.default_rng(17)
        paths = [torch.Generator().manual_seed(101), torch.Generator().manual_seed(102)]
        return model, opt, data, paths

    def step(model, opt, data, paths):
        x = torch.tensor(data.integers(1, 15, size=(1, 2)))
        opt.zero_grad(set_to_none=True)
        outs = sample_paths(model, x, paths, "J1")
        loss, _ = joint_loss(outs, (x + 1) % 23, settings, "J1")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return outs

    a, opt, data, paths = build()
    step(a, opt, data, paths)
    identity = {"experiment_id": "006-PD1", "seed": 17}
    save(tmp_path / "checkpoint.pt", a, opt, 1, identity, data, paths, {"fixture": True})
    random_before = random.getstate()
    np_before = np.random.get_state()
    expected = step(a, opt, data, paths)
    b, opt2, data2, paths2 = build()
    first, extra = restore(tmp_path / "checkpoint.pt", b, opt2, identity, data2, paths2)
    assert first == 1 and extra["fixture"] is True
    assert random.getstate() == random_before
    assert np.array_equal(np.random.get_state()[1], np_before[1])
    observed = step(b, opt2, data2, paths2)
    assert all(torch.equal(x.actions, y.actions) for x, y in zip(expected, observed))
    assert all(torch.equal(a.state_dict()[name], value) for name, value in b.state_dict().items())
    with pytest.raises(ValueError, match="identity mismatch"):
        restore(tmp_path / "checkpoint.pt", b, opt2, {"seed": 18}, data2, paths2)


@pytest.mark.parametrize("mode", ["train", "evaluate", "profile", "publish", "confirmation"])
def test_all_scientific_modes_fail_closed(proposal, mode):
    _, p = load()
    with pytest.raises(RuntimeError, match="PROPOSED_NOT_REGISTERED"):
        authorize(proposal, p, "P50-A32", 17, "development", mode, "a" * 40)


def test_gate_precedes_assets_and_files(proposal, tmp_path, monkeypatch):
    _, p = load()

    def forbidden(*args, **kwargs):
        raise AssertionError("asset load must not occur")

    monkeypatch.setattr("pd001.training.load_bundle", forbidden)
    with pytest.raises(RuntimeError, match="PROPOSED_NOT_REGISTERED"):
        train(proposal, p, "P50-A32", 17, "development", "a" * 40, tmp_path / "missing", tmp_path / "run")
    assert not (tmp_path / "run").exists()
    with pytest.raises(RuntimeError, match="PROPOSED_NOT_REGISTERED"):
        load_bundle(
            tmp_path / "missing", proposal, p, "P50-A32", 17, "confirmation", "a" * 40, evaluation=True
        )


def test_story_bootstrap_clusters_and_noninferiority():
    base = {
        "identity": {
            "asset_sha": "a",
            "source_sha": "s",
            "config_sha": "c",
            "protocol_sha": "p",
            "tokenizer_sha": "t",
            "evaluation_sha": "e",
            "role": "confirmation",
            "seed": 17,
            "target_tokens": 100,
            "training_budget_sha": "b",
        },
        "baseline_stable": True,
        "quality": {
            "greedy": {"window_ids": [0, 1, 2], "groups": [10, 10, 20], "window_nll": [2.0, 2.0, 2.0]}
        },
    }
    candidate = copy.deepcopy(base)
    candidate["quality"]["greedy"]["window_nll"] = [2.001] * 3
    result = noninferiority(base, candidate, replicates=200)
    assert result["within_seed_noninferiority"] and result["stories"] == 2
    assert result["upper95_one_sided"] < np.log(1.02)
    base["baseline_stable"] = False
    assert noninferiority(base, candidate)["status"] == "BASELINE_UNDERTRAINED"
    base["baseline_stable"] = True
    candidate["identity"]["tokenizer_sha"] = "wrong"
    with pytest.raises(ValueError, match="tokenizer_sha"):
        noninferiority(base, candidate)
    assert seed_summary([])["status"] == "INCOMPLETE"
    cluster = paired_interval([0, 0, 0], [1, 1, -1], [1, 1, 2], replicates=200)
    assert cluster["stories"] == 2


def test_proposal_not_registered_and_history_untouched():
    _, p = load()
    assert p["status"] == "PROPOSED_NOT_REGISTERED"
    assert p["gpu_runs"] == 0 and not p["confirmation_access"]
    assert json.loads((BASE.parent / "protocol.json").read_text())["status"] == "PROPOSED_NOT_REGISTERED"


def test_cpu_fixture_training_final_loader_and_resume(tiny, proposal, tmp_path, monkeypatch):
    """Nonresearch synthetic fixture exercises the REAL internal runner, not a language run."""
    import pd001.training as training
    from exp006.reuse import file_sha256

    c = copy.deepcopy(proposal)
    c["training"].update(
        steps=2, effective_batch=2, microbatch=1, sequence_length=3, tokens=12, checkpoint_every=1
    )
    c["evaluation"].update(windows=2, bootstrap_replicates=30)
    fixture = copy.deepcopy(tiny)
    fixture.update(budget=4, adaptive=False)
    monkeypatch.setattr(training, "arm_config", lambda config, arm: fixture)
    tokens = np.arange(23, dtype=np.int32)
    assets = {
        "train": tokens,
        "train_starts": [0, 3, 6, 9],
        "manifest_sha256": "f" * 64,
        "tokenizer_sha256": "e" * 64,
    }
    loaded = []

    def final_loader(*args, **kwargs):
        frozen = kwargs["frozen_checkpoint"]
        loaded.append(frozen)
        assert frozen["complete"] and frozen["step"] == 2
        return {**assets, "tokens": tokens, "starts": [0, 3], "groups": [0, 1], "evaluation_sha": "d" * 64}

    monkeypatch.setattr(training, "load_bundle", final_loader)
    _, protocol = load()
    straight = training._train_registered(
        c,
        protocol,
        "P50-A32-J1",
        17,
        "confirmation",
        "a" * 40,
        assets,
        "not-read",
        tmp_path / "straight",
        "cpu",
        False,
        confirmation_assets="sealed-not-read",
    )
    assert len(loaded) == 1 and straight["identity"]["experiment_id"] == "006-PD1"
    assert loaded[0]["sha"] == file_sha256(tmp_path / "straight/checkpoint.pt")
    original_save = training.save

    def interrupt(*args, **kwargs):
        original_save(*args, **kwargs)
        if args[3] == 1:
            raise RuntimeError("fixture interruption")

    monkeypatch.setattr(training, "save", interrupt)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        training._train_registered(
            c,
            protocol,
            "P50-A32-J1",
            17,
            "confirmation",
            "a" * 40,
            assets,
            "not-read",
            tmp_path / "resumed",
            "cpu",
            False,
            confirmation_assets="sealed-not-read",
        )
    assert json.loads((tmp_path / "resumed/failure.json").read_text())["status"] == "INCOMPLETE"
    assert len(loaded) == 1  # no intermediate holdout load
    monkeypatch.setattr(training, "save", original_save)
    recovered = training._train_registered(
        c,
        protocol,
        "P50-A32-J1",
        17,
        "confirmation",
        "a" * 40,
        assets,
        "not-read",
        tmp_path / "resumed",
        "cpu",
        True,
        confirmation_assets="sealed-not-read",
    )
    a = torch.load(tmp_path / "straight/checkpoint.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed/checkpoint.pt", weights_only=False)
    assert all(torch.equal(p, b["model"][name]) for name, p in a["model"].items())
    assert straight["quality"]["greedy"]["window_nll"] == recovered["quality"]["greedy"]["window_nll"]
    assert (
        straight["stability"]["observed_training_gemm_flops"]
        == recovered["stability"]["observed_training_gemm_flops"]
    )
    assert recovered["cost"]["total_flops_status"] == "OFFLINE_ONLY"
    assert recovered["cost"]["forward_gemm_flops"] is None


def test_report_requires_real_matched_moe(proposal, tmp_path):
    from pd001.report import report

    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"experiment_id": "006-PD1", "metrics": []}))
    result = report(p, proposal)
    assert result["status"] == "INCOMPLETE" and not result["scientific_decision"]


def test_registered_matrix_and_publication_stay_gated(proposal, tmp_path):
    from pd001.orchestration import matrix
    from pd001.publication import publish

    _, p = load()
    with pytest.raises(RuntimeError, match="PROPOSED_NOT_REGISTERED"):
        matrix(
            proposal,
            p,
            ["M0", "P50-A32"],
            17,
            "development",
            "a" * 40,
            "missing",
            tmp_path / "work",
            "missing",
            ["cpu"],
        )
    with pytest.raises(RuntimeError, match="PROPOSED_NOT_REGISTERED"):
        publish(proposal, p, "P50-A32", 17, "development", "a" * 40, "missing", tmp_path / "pub")
    assert not (tmp_path / "work").exists() and not (tmp_path / "pub").exists()


def test_automatic_preflight_then_approval_then_execution():
    from pd001.runner import automatic

    calls = []

    def pre():
        calls.append("preflight")
        return {"status": "ENGINEERING_READY"}

    def approve():
        calls.append("approval")

    def execute():
        calls.append("execute")
        return {"status": "finished"}

    assert automatic(pre, approve, execute)["status"] == "finished"
    assert calls == ["preflight", "approval", "execute"]


@pytest.mark.parametrize("failure", ["preflight", "approval", "execute"])
def test_automatic_failure_propagates_and_stops(failure):
    from pd001.runner import automatic

    calls = []

    def stage(name):
        def invoke():
            calls.append(name)
            if name == failure:
                raise RuntimeError("fixture failure")
            return {"status": "ENGINEERING_READY"}

        return invoke

    with pytest.raises(RuntimeError, match="fixture failure"):
        automatic(stage("preflight"), stage("approval"), stage("execute"))
    assert (
        calls
        == ["preflight", "approval", "execute"][: ["preflight", "approval", "execute"].index(failure) + 1]
    )
