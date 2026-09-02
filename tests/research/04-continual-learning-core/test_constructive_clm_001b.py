from __future__ import annotations

import pytest

from minicells.constructive_clm_001b import build_world, run_seed


@pytest.mark.research
def test_001b_world_has_no_singleton_training_exposure() -> None:
    world = build_world(201)
    assert len(world.train_pairs) == 12
    assert len(world.heldout_pair_types) == 3
    assert all(len(transaction.factors) == 2 for transaction in world.train_transactions)
    assert all(len(transaction.factors) == 2 for transaction in world.heldout_pairs)
    assert all(len(transaction.factors) == 3 for transaction in world.heldout_triples)
    assert set(world.train_pairs).isdisjoint(set(world.heldout_pair_types))


@pytest.mark.research
@pytest.mark.parametrize("seed", [201, 202])
def test_001b_development_regression(seed: int) -> None:
    result = run_seed(seed)
    assert result["pass"] is True
    assert result["no_singleton_training"] is True
    assert result["prototype_count"] == 12
    assert result["active_cells"] == 6
    assert result["fit"]["valid"] is True
    assert result["fit"]["largest_clique_size"] == 4
    assert result["fit"]["star_count"] == 6
    assert result["alignment"]["covered_factors"] == 6
    assert result["heldout_pair"]["route_recall"] >= 0.95
    assert result["heldout_triple"]["route_recall"] >= 0.95
    assert result["heldout_pair"]["mse"] < result["transaction_memory_pair_mse"]
    assert result["heldout_pair"]["mse"] < result["shuffled_effect_pair_mse"]
