from __future__ import annotations

import torch
from torch.nn import functional as F

from minicells.language_2d import LatentTissueNCALM
from minicells.language_models import TextNCALM, count_parameters


def _build_2d(*, tissue_height: int = 4) -> LatentTissueNCALM:
    torch.manual_seed(9)
    return LatentTissueNCALM(
        vocab_size=64,
        tissue_height=tissue_height,
        max_context=8,
        dim=32,
        heads=4,
        ffn_dim=64,
        windows=(2, 4, 8),
        iterations=(1, 1, 1),
        carry_bias=2.0,
    )


def _build_1d() -> TextNCALM:
    torch.manual_seed(9)
    return TextNCALM(
        vocab_size=64,
        max_context=8,
        dim=32,
        heads=4,
        ffn_dim=64,
        windows=(2, 4, 8),
        iterations=(1, 1, 1),
        rms_norm=False,
        carry_bias=2.0,
        tie_embeddings=True,
        stage_supervision=False,
    )


def test_2d_forward_shape_and_parameter_overhead_are_bounded() -> None:
    model = _build_2d()
    inputs = torch.randint(0, 64, (2, 8))
    output = model(inputs)
    assert output.logits.shape == (2, 8, 64)

    baseline = _build_1d()
    overhead = count_parameters(model) / count_parameters(baseline)
    assert overhead < 1.08


def test_only_token_row_receives_token_embedding_at_initialization() -> None:
    model = _build_2d()
    left = torch.tensor([[1, 2, 3, 4]])
    right = torch.tensor([[1, 2, 9, 4]])
    left_state = model._initial_state(left)
    right_state = model._initial_state(right)
    assert not torch.equal(left_state[:, :, 0, :], right_state[:, :, 0, :])
    assert torch.equal(left_state[:, :, 1:, :], right_state[:, :, 1:, :])


def test_2d_model_is_causal_along_text_axis() -> None:
    model = _build_2d().eval()
    left = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    right = left.clone()
    right[0, -1] = 11
    with torch.no_grad():
        left_logits = model(left).logits
        right_logits = model(right).logits
    torch.testing.assert_close(left_logits[:, :-1], right_logits[:, :-1], rtol=0.0, atol=1e-6)


def test_latent_rows_receive_gradient_through_vertical_communication() -> None:
    model = _build_2d()
    inputs = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))
    logits = model(inputs).logits
    loss = F.cross_entropy(logits.reshape(-1, 64), targets.reshape(-1))
    loss.backward()

    assert model.row_embedding.weight.grad is not None
    assert float(model.row_embedding.weight.grad[1:].abs().sum()) > 0
    vertical_grad = model.stages[0].vertical.conv.weight.grad
    assert vertical_grad is not None
    assert float(vertical_grad.abs().sum()) > 0


def test_diagnostics_and_latent_ablation_are_well_formed() -> None:
    model = _build_2d(tissue_height=4).eval()
    inputs = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        diagnostics = model.diagnose(inputs)
        ablated = model.forward_with_ablation(inputs, 2)
    assert len(diagnostics.row_cosine_to_token) == 4
    assert len(diagnostics.row_update_rms) == 3
    assert all(len(stage_values) == 4 for stage_values in diagnostics.row_update_rms)
    assert ablated.logits.shape == (2, 8, 64)
