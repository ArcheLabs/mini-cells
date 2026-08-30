from __future__ import annotations

import io

import torch

from minicells.language_models import TextNCALM
from minicells.overcomplete_cellular_textnca import CLMv2Config
from minicells.textnca_to_clm_v2 import convert_textnca_to_clm_v2


def _models():
    torch.manual_seed(19)
    source = TextNCALM(
        vocab_size=43, max_context=12, dim=16, heads=4, ffn_dim=32,
        windows=(2, 4, 8), iterations=(2, 1, 1), stage_supervision=True,
    )
    return source, convert_textnca_to_clm_v2(source)


def test_alpha_one_has_exact_scaffold_and_recurrent_parity() -> None:
    source, model = _models()
    inputs = torch.randint(0, 43, (2, 9))
    expected_states = []
    actual_states = []
    expected_hooks = [stage.gru.register_forward_hook(
        lambda _module, _inputs, output: expected_states.append(output.detach().clone())
    ) for stage in source.stages]
    actual_hooks = [stage.gru.register_forward_hook(
        lambda _module, _inputs, output: actual_states.append(output.detach().clone())
    ) for stage in model.stages]
    expected = source(inputs)
    actual = model(inputs)
    for hook in [*expected_hooks, *actual_hooks]:
        hook.remove()
    torch.testing.assert_close(actual.logits, expected.logits)
    for left, right in zip(actual_states, expected_states):
        torch.testing.assert_close(left, right)


def test_random_program_branch_cannot_influence_alpha_one_main_state() -> None:
    source, model = _models()
    with torch.no_grad():
        for parameter in model.sparse_parameters():
            parameter.mul_(100)
    inputs = torch.randint(0, 43, (2, 9))
    torch.testing.assert_close(model(inputs).logits, source(inputs).logits)
    with_local, stats = model(inputs, return_local_imitation=True)
    torch.testing.assert_close(with_local.logits, source(inputs).logits)
    assert float(stats.local_relative_mse) > 0


def test_alpha_zero_never_calls_dense_scaffold_at_inference() -> None:
    _, model = _models()
    model.set_scaffold_alpha(0)
    calls = 0

    def count(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    hooks = [stage.dense_scaffold.register_forward_hook(count) for stage in model.stages]
    model(torch.randint(0, 43, (2, 9)))
    for hook in hooks:
        hook.remove()
    assert calls == 0


def test_experts_are_independent_and_topk_is_exact() -> None:
    _, model = _models()
    bank = model.stages[0].program_bank
    assert bank.programs[0].in_proj.weight.data_ptr() != bank.programs[1].in_proj.weight.data_ptr()
    inputs = torch.randn(2, 7, 16)
    gates, _, _ = bank.route(inputs)
    torch.testing.assert_close(gates.detach().sum(-1), torch.full((2, 7), 6.0))


def test_receptor_is_strictly_pointwise_local() -> None:
    _, model = _models()
    receptor = model.stages[0].program_bank.receptor
    inputs = torch.randn(2, 7, 16)
    baseline = receptor(inputs)
    changed = inputs.clone()
    changed[1, 3] += 10
    actual = receptor(changed)
    keep = torch.ones(2, 7, dtype=torch.bool)
    keep[1, 3] = False
    torch.testing.assert_close(actual[keep], baseline[keep])


def test_local_imitation_gradients_are_isolated_to_new_branch() -> None:
    _, model = _models()
    model.freeze_inherited_backbone()
    model.set_scaffold_alpha(1)
    _, stats = model(torch.randint(0, 43, (2, 9)), return_local_imitation=True)
    loss = stats.local_relative_mse + 0.01 * stats.balance_loss + 1e-4 * stats.router_z_loss
    loss.backward()
    sparse_gradients = [parameter.grad for parameter in model.sparse_parameters()]
    assert any(gradient is not None and float(gradient.norm()) > 0 for gradient in sparse_gradients)
    for stage in model.stages:
        assert all(parameter.grad is None for parameter in stage.dense_scaffold.parameters())
        assert stage.program_bank.receptor.out_proj.weight.grad is not None
    assert model.token_embedding.weight.grad is None


def test_balance_loss_detects_collapsed_usage() -> None:
    _, model = _models()
    model.freeze_inherited_backbone()
    model.set_scaffold_alpha(1)
    with torch.no_grad():
        for stage in model.stages:
            stage.program_bank.receptor.out_proj.weight.zero_()
            stage.program_bank.receptor.out_proj.bias.copy_(torch.arange(12.0))
    _, stats = model(torch.randint(0, 43, (2, 9)), return_local_imitation=True)
    assert float(stats.balance_loss) > 0


def test_active_hidden_equivalents() -> None:
    _, model = _models()
    expected = {6: 512, 5: 448, 4: 384, 3: 320}
    for top_k, hidden in expected.items():
        model.set_program_top_k(top_k)
        model.set_scaffold_alpha(0)
        _, stats = model(torch.randint(0, 43, (1, 5)), return_stats=True)
        assert stats.active_hidden_equivalent == hidden
        assert stats.genome_hidden_equivalent == 896


def test_masked_dense_and_sparse_dispatch_match() -> None:
    _, model = _models()
    model.set_scaffold_alpha(0)
    inputs = torch.randint(0, 43, (2, 9))
    model.set_execution_backend("masked_dense")
    dense = model(inputs).logits
    model.set_execution_backend("sparse_dispatch")
    sparse = model(inputs).logits
    torch.testing.assert_close(sparse, dense, rtol=5e-5, atol=1e-6)


def test_save_load_preserves_alpha_k_and_program_bank() -> None:
    source, model = _models()
    model.set_scaffold_alpha(0.25)
    model.set_program_top_k(4)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = convert_textnca_to_clm_v2(source)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert restored.config.scaffold_alpha == 0.25
    assert restored.config.top_k == 4
    torch.testing.assert_close(
        restored.stages[0].program_bank.programs[0].in_proj.weight,
        model.stages[0].program_bank.programs[0].in_proj.weight,
    )
