from types import SimpleNamespace

import numpy as np
import torch

from minicells.language_growing_organism import build_cellular_model
from minicells.language_localized_learning import LocalizedLearningState
from minicells.language_tissue_specificity import (
    EXAMPLE_TOP1_MIN,
    MATCHING_VALUE_MIN,
    SPECIFICITY_NORM_MIN,
    allocate_fixed_tissue,
    family_pass,
    summarize_specificity,
)


def test_specificity_summary_detects_diagonal_advantage() -> None:
    names = ["A", "B", "C"]
    summary = summarize_specificity(names, np.array([2.0, 1.0, 0.5]), "A")
    assert summary.matching_value == 2.0
    assert summary.mean_wrong_value == 0.75
    assert summary.strict_margin == 1.0
    assert summary.normalized_specificity > 0.6
    assert summary.matching_rank == 1


def test_flat_generic_utility_does_not_pass_identity_gate() -> None:
    names = ["A", "B", "C"]
    summary = summarize_specificity(names, np.array([1.00, 1.01, 0.99]), "A")
    assert summary.normalized_specificity < SPECIFICITY_NORM_MIN
    assert summary.matching_rank != 1
    assert not family_pass(summary.matching_value, summary.normalized_specificity, 1, EXAMPLE_TOP1_MIN - 0.01)


def test_family_pass_requires_benefit_and_all_identity_components() -> None:
    assert family_pass(MATCHING_VALUE_MIN + 0.1, 0.11, 2, 0.51)
    assert not family_pass(MATCHING_VALUE_MIN - 0.01, 0.2, 3, 0.8)
    assert not family_pass(1.0, 0.09, 3, 0.8)
    assert not family_pass(1.0, 0.2, 1, 0.8)
    assert not family_pass(1.0, 0.2, 3, 0.49)


def _probe_for(model):
    pressure = torch.zeros(model.max_cells)
    pressure[1] = 10.0
    split_direction = torch.zeros(model.max_cells, model.dim)
    split_direction[:, 0] = 1.0
    return SimpleNamespace(pressure=pressure, split_direction=split_direction)


def test_fixed_tissue_allocation_preserves_old_memory_and_changes_only_geometry() -> None:
    torch.manual_seed(20020)
    one = build_cellular_model(128, "G")
    three = build_cellular_model(128, "G")
    three.load_state_dict(one.state_dict())
    one_state = LocalizedLearningState.capture(one)
    three_state = LocalizedLearningState.capture(three)
    old_one = one.cell_memory.detach().clone()
    old_three = three.cell_memory.detach().clone()

    parent_one, newborn_one = allocate_fixed_tissue(one, one_state, _probe_for(one), tissue_size=1)
    parent_three, newborn_three = allocate_fixed_tissue(three, three_state, _probe_for(three), tissue_size=3)

    assert parent_one == parent_three == 1
    assert len(newborn_one) == 1
    assert len(newborn_three) == 3
    assert newborn_one[0] == newborn_three[0]
    base = one_state.base_alive
    assert torch.equal(one.cell_memory.detach()[base], old_one[base])
    assert torch.equal(three.cell_memory.detach()[base], old_three[base])
    assert torch.equal(one.cell_memory.detach()[newborn_one[0]], three.cell_memory.detach()[newborn_three[0]])
    assert three.parent[newborn_three[0]].item() == parent_three
    assert three.parent[newborn_three[1]].item() == newborn_three[0]
    assert three.parent[newborn_three[2]].item() == newborn_three[1]
