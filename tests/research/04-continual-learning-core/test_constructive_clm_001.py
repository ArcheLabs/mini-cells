from minicells.constructive_clm_001 import (
    GrowingCellMemory,
    Transaction,
    build_world,
    run_seed,
)


def test_structured_world_forms_bounded_reusable_coordinates():
    result = run_seed(1001)
    assert result["pass"]
    assert result["active_cells"] <= result["world"]["factor_count"] + 2
    assert result["late_spawns"] <= 1
    assert result["heldout_pair_route_recall"] >= 0.85
    assert result["heldout_pair_mse"] < result["shuffled_address_pair_mse"]


def test_hidden_factor_labels_are_not_used_by_learner():
    world = build_world(1002)
    learner_a = GrowingCellMemory()
    learner_b = GrowingCellMemory()

    for step, tx in enumerate(world.transactions[:10]):
        learner_a.observe(tx, step)
        relabeled = Transaction(
            tuple(99 for _ in tx.factors),
            tx.x.clone(),
            tx.y.clone(),
            tx.phase,
        )
        learner_b.observe(relabeled, step)

    assert learner_a.active_cells == learner_b.active_cells
    for cell_a, cell_b in zip(learner_a.cells, learner_b.cells):
        assert cell_a.key.equal(cell_b.key)
        assert cell_a.value.equal(cell_b.value)
