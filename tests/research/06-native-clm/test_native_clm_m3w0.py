from __future__ import annotations

import math

import torch

from minicells.native_clm_m3l2 import OnlineAddressNativeCLM
from minicells.native_clm_m3w0 import (
    M3W0Thresholds,
    analyze_factorial,
    classify_results,
    restore_operator_groups,
    root_ancestor,
)
from minicells.native_clm_v0 import NativeCLM, NativeCLMConfig


def _config() -> NativeCLMConfig:
    return NativeCLMConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        n_layers=2,
        n_heads=2,
        d_ff=32,
        initial_cells=2,
        active_cells=1,
        cellular_layer_index=0,
        certificate_max_rank=4,
        tie_embeddings=True,
    )


def _matrix(a: float, b: float, c: float, d: float) -> dict:
    return {
        "A": {"loss": a},
        "B": {"loss": b},
        "C": {"loss": c},
        "D": {"loss": d},
    }


def test_root_ancestor_and_group_restoration() -> None:
    torch.manual_seed(7)
    config = _config()
    m1 = NativeCLM(config)
    model = OnlineAddressNativeCLM(config)
    child = model.spawn_cell(parent_id=0, route_key=torch.ones(config.d_model), inherit_scale=1.0)
    grandchild = model.spawn_cell(
        parent_id=child,
        route_key=torch.ones(config.d_model),
        inherit_scale=1.0,
    )
    assert root_ancestor(model, grandchild) == 0

    with torch.no_grad():
        for cell in model.cellular.cells:
            cell.weight.add_(1.0)
    root_before = model.cellular.cells[0].weight.detach().clone()
    restore_operator_groups(model, m1, restore_roots=False, restore_descendants=True)
    assert torch.equal(model.cellular.cells[0].weight, root_before)
    assert torch.equal(model.cellular.cells[child].weight, m1.cellular.cells[0].weight)
    assert torch.equal(model.cellular.cells[grandchild].weight, m1.cellular.cells[0].weight)


def test_factorial_shapley_identity() -> None:
    thresholds = M3W0Thresholds()
    result = analyze_factorial(
        seed=1,
        m1_matrix=_matrix(1.0, 4.0, 5.0, 3.0),
        final_matrix=_matrix(1.5, 2.0, 2.5, 1.5),
        root_restore_matrix=_matrix(1.1, 2.4, 3.0, 1.8),
        descendant_root_restore_matrix=_matrix(1.4, 3.0, 3.5, 2.1),
        all_lineage_restore_matrix=_matrix(1.0, 4.0, 5.0, 3.0),
        thresholds=thresholds,
    )
    assert result["identity_ok"]
    total = result["A_root_shapley"] + result["A_descendant_shapley"]
    assert math.isclose(total, 0.5, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(
        result["A_root_fraction"] + result["A_descendant_fraction"],
        1.0,
        rel_tol=0,
        abs_tol=1e-12,
    )


def test_root_dominant_children_carry_plasticity_classification() -> None:
    thresholds = M3W0Thresholds()
    results = []
    for seed in (1, 2, 3):
        result = analyze_factorial(
            seed=seed,
            m1_matrix=_matrix(1.0, 4.0, 5.0, 3.0),
            final_matrix=_matrix(1.5, 2.0, 2.5, 1.5),
            root_restore_matrix=_matrix(1.1, 2.2, 2.8, 1.7),
            descendant_root_restore_matrix=_matrix(1.45, 3.8, 4.7, 2.8),
            all_lineage_restore_matrix=_matrix(1.0, 4.0, 5.0, 3.0),
            thresholds=thresholds,
        )
        assert result["A_root_fraction"] >= 0.60
        assert min(result["root_restore_new_domain_gain_retention"].values()) >= 0.70
        results.append(result)
    assert classify_results(results, thresholds) == "ROOT_WRITE_DOMINANT_CHILDREN_CARRY_PLASTICITY"


def test_root_dominant_transfer_gap_classification() -> None:
    thresholds = M3W0Thresholds()
    results = []
    for seed in (1, 2, 3):
        result = analyze_factorial(
            seed=seed,
            m1_matrix=_matrix(1.0, 4.0, 5.0, 3.0),
            final_matrix=_matrix(1.5, 2.0, 2.5, 1.5),
            root_restore_matrix=_matrix(1.1, 3.5, 4.4, 2.6),
            descendant_root_restore_matrix=_matrix(1.45, 2.1, 2.7, 1.6),
            all_lineage_restore_matrix=_matrix(1.0, 4.0, 5.0, 3.0),
            thresholds=thresholds,
        )
        assert result["A_root_fraction"] >= 0.60
        results.append(result)
    assert classify_results(results, thresholds) == "ROOT_WRITE_DOMINANT_TRANSFER_GAP"


def test_identity_failure_is_inconclusive() -> None:
    thresholds = M3W0Thresholds()
    result = analyze_factorial(
        seed=1,
        m1_matrix=_matrix(1.0, 4.0, 5.0, 3.0),
        final_matrix=_matrix(1.5, 2.0, 2.5, 1.5),
        root_restore_matrix=_matrix(1.1, 2.2, 2.8, 1.7),
        descendant_root_restore_matrix=_matrix(1.4, 3.0, 3.5, 2.1),
        all_lineage_restore_matrix=_matrix(1.01, 4.0, 5.0, 3.0),
        thresholds=thresholds,
    )
    assert not result["identity_ok"]
    assert classify_results([result], thresholds) == "INCONCLUSIVE_IDENTITY"
