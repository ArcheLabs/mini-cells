from __future__ import annotations

import io

import torch
from torch.nn import functional as F

from minicells.language_models import TextNCALM
from minicells.textnca_to_clm import convert_textnca_to_sparse_cellular, verify_dense_equivalence


def _model() -> TextNCALM:
    torch.manual_seed(7)
    return TextNCALM(
        vocab_size=41, max_context=16, dim=16, heads=4, ffn_dim=19,
        windows=(3, 5, 8), iterations=(2, 1, 1), stage_supervision=True,
    )


def test_dense_conversion_has_forward_and_stage_parity() -> None:
    teacher = _model()
    student = convert_textnca_to_sparse_cellular(teacher, num_programs=8)
    inputs = torch.randint(0, 41, (2, 11))
    verify_dense_equivalence(teacher, student, inputs)
    assert student.conversion_metadata["partition_sizes"] == [3, 3, 3, 2, 2, 2, 2, 2]


def test_dense_conversion_has_gradient_parity() -> None:
    teacher = _model()
    student = convert_textnca_to_sparse_cellular(teacher, num_programs=8)
    inputs = torch.randint(0, 41, (2, 9))
    targets = torch.randint(0, 41, (2, 9))
    F.cross_entropy(teacher(inputs).logits.flatten(0, 1), targets.flatten()).backward()
    F.cross_entropy(student(inputs).logits.flatten(0, 1), targets.flatten()).backward()
    pairs = (
        (teacher.token_embedding.weight, student.token_embedding.weight),
        (teacher.stages[0].ffn[0].weight, student.stages[0].update.in_proj.weight),
        (teacher.stages[0].ffn[2].weight, student.stages[0].update.out_proj.weight),
        (teacher.stages[0].gru.weight_ih, student.stages[0].gru.weight_ih),
    )
    for expected, actual in pairs:
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)


def test_masked_dense_and_sparse_dispatch_match_fixed_hard_routing() -> None:
    model = convert_textnca_to_sparse_cellular(_model(), num_programs=8)
    model.set_routing_mode("hard")
    model.set_program_top_k(4)
    inputs = torch.randint(0, 41, (2, 10))
    model.set_execution_backend("masked_dense")
    dense, dense_stats = model(inputs, return_stats=True)
    model.set_execution_backend("sparse_dispatch")
    sparse, sparse_stats = model(inputs, return_stats=True)
    torch.testing.assert_close(sparse.logits, dense.logits, rtol=2e-5, atol=2e-6)
    assert sparse_stats.receptor_flops > 0
    assert sparse_stats.program_usage.shape == (8,)
    assert dense_stats.program_coactivation.shape == (8, 8)


def test_sleeping_cells_preserve_state_and_program_mask_removes_contribution() -> None:
    model = convert_textnca_to_sparse_cellular(_model(), num_programs=8)
    stage = model.stages[0]
    inputs = torch.randn(2, 4, 16)
    gates = torch.ones(2, 4, 8)
    full = stage.update.routed(inputs, gates, "masked_dense")
    gates[..., 3] = 0
    masked = stage.update.routed(inputs, gates, "masked_dense")
    expected = stage.update._program(inputs, 3)
    torch.testing.assert_close(full - masked, expected)
    model.set_routing_mode("hard")
    with torch.no_grad():
        for receptor in (item.receptor for item in model.stages):
            receptor.out_proj.bias[0] = -100
    initial = torch.randn(1, 5, 16)
    next_state, _ = stage(initial)
    torch.testing.assert_close(next_state, initial)


def test_receptor_is_strictly_pointwise_local() -> None:
    model = convert_textnca_to_sparse_cellular(_model(), num_programs=8)
    receptor = model.stages[0].receptor
    perception = torch.randn(2, 6, 16)
    baseline = receptor(perception, None)
    changed = perception.clone()
    changed[0, 2] += 3.0
    routed = receptor(changed, None)
    unchanged = torch.ones(2, 6, dtype=torch.bool)
    unchanged[0, 2] = False
    torch.testing.assert_close(routed[unchanged], baseline[unchanged])


def test_program_sparsity_phase_keeps_cells_exactly_active() -> None:
    model = convert_textnca_to_sparse_cellular(_model(), num_programs=8)
    model.set_routing_mode("soft_program")
    model(torch.randint(0, 41, (2, 7)))
    program_ratio, cell_ratio = model.routing_ratios()
    torch.testing.assert_close(cell_ratio, torch.ones_like(cell_ratio))
    assert 0 < float(program_ratio) < 1


def test_state_dict_preserves_routing_configuration_and_usage() -> None:
    model = convert_textnca_to_sparse_cellular(_model(), num_programs=8)
    model.set_routing_mode("hard_program")
    model.set_program_top_k(4)
    model(torch.randint(0, 41, (1, 5)))
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = convert_textnca_to_sparse_cellular(_model(), num_programs=8)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert restored.routing_config.routing_mode == "hard_program"
    assert restored.routing_config.program_top_k == 4
    torch.testing.assert_close(restored.program_usage_ema, model.program_usage_ema)
