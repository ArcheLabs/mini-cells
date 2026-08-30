import torch
from minicells.clm_growth import ProgressiveGrowthCLM
from minicells.developmental_tissue import (
    StressObservation,
    TissueConfig,
    TissueFFN,
    convert_model_experts_to_tissues,
    count_module_parameters,
)
from minicells.language_models import TextNCALM, count_parameters
from minicells.upcycled_cellular_textnca import UpcyclingConfig, convert_textnca_to_upcycled
from torch import nn


def _dense_ffn() -> nn.Sequential:
    torch.manual_seed(9)
    return nn.Sequential(nn.Linear(8, 13), nn.GELU(), nn.Linear(13, 8))


def _upcycled_model():
    torch.manual_seed(13)
    source = TextNCALM(
        vocab_size=23,
        max_context=8,
        dim=8,
        heads=2,
        ffn_dim=12,
        windows=(2, 3, 4),
        iterations=(1, 1, 1),
        carry_bias=2.0,
    )
    return convert_textnca_to_upcycled(
        source,
        config=UpcyclingConfig(num_experts=4, top_k=1),
    )


def test_hidden_partition_is_exact_and_parameter_neutral() -> None:
    dense = _dense_ffn()
    inputs = torch.randn(3, 5, 8)
    expected = dense(inputs)
    dense_parameters = count_module_parameters(dense)

    for cells in (1, 2, 3, 4, 7, 13):
        tissue = TissueFFN.from_dense_ffn(
            dense,
            config=TissueConfig(cells_per_tissue=cells),
        )
        assert tissue.cell_count == cells
        assert sum(tissue.cell_hidden_widths) == 13
        assert count_module_parameters(tissue) == dense_parameters
        torch.testing.assert_close(tissue(inputs), expected, rtol=1e-5, atol=1e-6)


def test_model_conversion_changes_granularity_not_initial_function_or_parameters() -> None:
    model = _upcycled_model()
    inputs = torch.randint(0, 23, (2, 6))
    expected = model(inputs).logits
    expected_parameters = count_parameters(model)

    for cells in (1, 2, 3, 4, 6, 12):
        tissue_model = convert_model_experts_to_tissues(
            model,
            config=TissueConfig(cells_per_tissue=cells),
        )
        assert count_parameters(tissue_model) == expected_parameters
        assert all(
            isinstance(expert, TissueFFN) and expert.cell_count == cells
            for stage in tissue_model.stages
            for expert in stage.program_bank.experts
        )
        torch.testing.assert_close(
            tissue_model(inputs).logits,
            expected,
            rtol=2e-5,
            atol=2e-6,
        )


def test_progressive_clm_conversion_preserves_hierarchical_routing_function() -> None:
    model = ProgressiveGrowthCLM(_upcycled_model())
    inputs = torch.randint(0, 23, (2, 6))
    expected = model(inputs).logits.detach()
    expected_parameters = count_parameters(model)

    tissue_model = convert_model_experts_to_tissues(
        model,
        config=TissueConfig(cells_per_tissue=3),
    )

    assert count_parameters(tissue_model) == expected_parameters
    assert all(
        isinstance(expert, TissueFFN) and expert.cell_count == 3
        for stage in tissue_model.stages
        for expert in stage.program_bank.experts.values()
    )
    torch.testing.assert_close(tissue_model(inputs).logits, expected, rtol=2e-5, atol=2e-6)


def test_microcell_division_preserves_full_model_function() -> None:
    model = convert_model_experts_to_tissues(
        _upcycled_model(),
        config=TissueConfig(cells_per_tissue=3, juvenile_plasticity=4.0),
    )
    inputs = torch.randint(0, 23, (2, 6))
    expected = model(inputs).logits.detach().clone()
    tissue = model.stages[1].program_bank.experts[0]
    previous_count = tissue.cell_count
    previous_parameters = count_module_parameters(tissue)

    event = tissue.divide_cell(1)

    assert tissue.cell_count == previous_count + 1
    assert count_module_parameters(tissue) > previous_parameters
    assert event["function_preserving_rule"] == (
        "clone_parent_and_halve_both_outgoing_projections"
    )
    assert float(tissue.cells[2].plasticity.item()) == 4.0
    torch.testing.assert_close(model(inputs).logits, expected, rtol=2e-5, atol=2e-6)


def test_neighbor_capacity_relaxes_local_stress_and_requires_persistence() -> None:
    config = TissueConfig(
        cells_per_tissue=3,
        stress_ema_decay=0.0,
        mitosis_threshold=0.8,
        minimum_overload_steps=2,
    )
    tissue = TissueFFN.from_dense_ffn(_dense_ffn(), config=config)
    overloaded = StressObservation(
        usage=1.0,
        residual_loss=1.0,
        novelty=1.0,
        gradient_conflict=1.0,
        neighbor_capacity=0.0,
    )
    supported = StressObservation(
        usage=1.0,
        residual_loss=1.0,
        novelty=1.0,
        gradient_conflict=1.0,
        neighbor_capacity=1.0,
    )

    assert tissue.instantaneous_stress(supported) < tissue.instantaneous_stress(overloaded)
    tissue.observe_stress(1, overloaded)
    assert not tissue.should_divide(1)
    tissue.observe_stress(1, overloaded)
    assert tissue.should_divide(1)
    tissue.observe_stress(1, supported)
    assert not tissue.should_divide(1)


def test_child_is_juvenile_then_plasticity_decays_toward_mature_value() -> None:
    config = TissueConfig(
        cells_per_tissue=2,
        mature_plasticity=1.0,
        juvenile_plasticity=4.0,
        plasticity_half_life_tokens=100,
    )
    tissue = TissueFFN.from_dense_ffn(_dense_ffn(), config=config)
    tissue.divide_cell(0)
    child = tissue.cells[1]
    assert float(child.plasticity.item()) == 4.0

    tissue.advance_age(100)

    assert abs(float(child.plasticity.item()) - 2.5) < 1e-6
    groups = tissue.optimizer_param_groups(1e-3)
    child_group = next(group for group in groups if group["cell_index"] == 1)
    mature_group = next(group for group in groups if group["cell_index"] == 2)
    assert child_group["lr"] > mature_group["lr"]


def test_division_adds_child_as_local_neighbor_and_inherits_parent_neighborhood() -> None:
    tissue = TissueFFN.from_dense_ffn(
        _dense_ffn(),
        config=TissueConfig(cells_per_tissue=4),
    )
    assert tissue.neighbors(1) == (0, 2)

    tissue.divide_cell(1)

    assert 2 in tissue.neighbors(1)
    assert 1 in tissue.neighbors(2)
    assert 0 in tissue.neighbors(2)
    assert 3 in tissue.neighbors(2)
