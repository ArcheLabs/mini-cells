import copy
import itertools
import torch
from fast001.config import ARMS, counts, model_config, shape_hash
from fast001.models import make_model
from pd001.losses import joint_loss, terminal_ce
from pd001.accounting import Counter


def test_actual_parameter_balance_and_identical_initialization(proposed):
    rows = counts(proposed)
    assert set(rows) == set(ARMS)
    assert 0.45 <= rows["CLM-S8"]["ratio_to_moe"] <= 0.55
    a, b = [make_model(proposed, arm) for arm in ARMS[1:]]
    assert shape_hash(a) == shape_hash(b)
    assert all(torch.equal(a.state_dict()[n], p) for n, p in b.state_dict().items())
    assert model_config(proposed, "CLM-S8") == {**model_config(proposed, "CLM-S16"), "budget": 8}
    assert a.controller.actions == proposed["model"]["cell_count"] + 1
    assert a.controller.scorer[-1].out_features == a.controller.actions


def test_sparse_selected_calls_self_loops_context_and_gradients(tiny):
    model = make_model(tiny, "CLM-S8").train()
    replay = torch.tensor([[[0, 0, 2, 1, 0, 2, 1, 1]]]).expand(1, 4, -1)
    x = torch.tensor([[1, 2, 3, 4]])
    y = (x + 1) % 64
    with Counter(model) as counter:
        out = model(x, replay=replay, capture=True)
    objective, _ = joint_loss([out], y, tiny["training"])
    objective.backward()
    assert counter.last["verified"] and counter.last["expected"]["cells.0.up"] == 12
    assert counter.last["expected"]["context.query"] == 8
    assert out.lengths.unique().tolist() == [8] and not out.stop_prob.any()
    for module in (model.cells[0], model.controller.scorer, model.context, model.embedding):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    # An unselected Cell cannot execute, even when its weights exist.
    calls = []
    h = model.cells[1].register_forward_hook(lambda *a: calls.append(1))
    model(x, replay=torch.zeros_like(replay))
    h.remove()
    assert not calls


def test_real_moe_top_two_six_layers(tiny):
    model = make_model(tiny, "M0-TINY").train()
    x = torch.tensor([[1, 2, 3, 4]])
    out = model(x)
    loss, _ = joint_loss([out], x, tiny["training"])
    loss.backward()
    assert len(model.blocks) == 6
    assert out.actions.shape[-2:] == (6, 2)
    assert all(block.ffn.top_k == 2 and len(block.ffn.cells) == 4 for block in model.blocks)


def test_causal_future_and_replay_cached_decode(tiny):
    x = torch.tensor([[1, 2, 3, 4]])
    for arm in ARMS:
        model = make_model(tiny, arm).eval()
        replay = (
            None
            if arm == "M0-TINY"
            else torch.tensor([0, 2] * 8)[: model.config["budget"]][None, None, :].expand(1, 4, -1)
        )
        kwargs = {} if replay is None else {"replay": replay}
        full = model(x, **kwargs)
        changed = x.clone()
        changed[0, -1] = 9
        assert torch.allclose(
            full.logits[:, :-1], model(changed, **kwargs).logits[:, :-1], atol=1e-6
        )
        prefix = model.prefill(x[:, :-1], **({} if replay is None else {"replay": replay[:, :-1]}))
        last = model.decode_step(
            x[:, -1], prefix.memory, **({} if replay is None else {"replay": replay[:, -1:]})
        )
        assert torch.allclose(last.logits, full.logits[:, -1:], atol=1e-5, rtol=1e-4)


def test_actual_cell_router_joint_gradient_exact_enumeration(tiny):
    # Full enumeration includes trainable Cells affecting subsequent route probabilities.
    config = copy.deepcopy(tiny)
    config["arms"]["CLM-S8"]["max_actions"] = 2
    model = make_model(config, "CLM-S8").double().train()
    x = torch.tensor([[1]])
    target = torch.tensor([[2]])
    params = [
        model.cells[0].up.weight,
        model.controller.scorer[0].weight,
        model.controller.source.weight,
    ]
    expected = [torch.zeros_like(p) for p in params]
    estimated = [torch.zeros_like(p) for p in params]
    for actions in itertools.product(range(3), repeat=2):
        out = model(x, replay=torch.tensor(actions).reshape(1, 1, 2))
        probability = out.log_prob.sum().exp()
        loss = terminal_ce(out, target).mean()
        grads = torch.autograd.grad(
            probability * loss, params, retain_graph=True, allow_unused=True
        )
        settings = {**config["training"], "critic_coefficient": 0.0, "entropy_coefficient": 0.0}
        proxy, _ = joint_loss([out], target, settings)
        estimate = torch.autograd.grad(proxy, params, allow_unused=True)
        for i, (g, e) in enumerate(zip(grads, estimate)):
            if g is not None:
                expected[i] += g.detach()
            if e is not None:
                estimated[i] += probability.detach() * e.detach()
    for a, b in zip(expected, estimated):
        assert torch.allclose(a, b, atol=1e-9, rtol=1e-6)


def test_critic_does_not_train_state_or_step_embedding(tiny):
    model = make_model(tiny, "CLM-S8").train()
    state = torch.randn(2, 8, requires_grad=True)
    _, value = model.controller(state, torch.tensor([4, 4]), 0)
    value.square().sum().backward()
    assert state.grad is None and model.controller.step.weight.grad is None
    assert any(p.grad is not None for p in model.controller.critic.parameters())


def test_j0_actor_is_token_normalized_not_divided_by_path_length(tiny):
    from types import SimpleNamespace

    values = []
    for depth in (8, 16):
        out = SimpleNamespace(
            logits=torch.zeros(1, 1, 64, requires_grad=True),
            log_prob=torch.full((1, 1, depth), -0.5, requires_grad=True),
            decision_mask=torch.ones(1, 1, depth, dtype=torch.bool),
            values=torch.zeros(1, 1, depth),
            entropy=torch.zeros(1, 1, depth),
            cost_units=torch.zeros(1, 1),
        )
        _, log = joint_loss([out], torch.tensor([[1]]), tiny["training"])
        values.append(float(log["actor_loss"]))
    assert abs(values[1] - 2 * values[0]) < 1e-6
