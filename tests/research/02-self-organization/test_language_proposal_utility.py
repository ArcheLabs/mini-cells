from __future__ import annotations

import copy

import torch

from minicells.language_growing_organism import build_cellular_model
from minicells.language_localized_learning import (
    LocalizedLearningState,
    conservative_fork,
    restore_structure,
    set_newborn_tissue_active,
)
from minicells.language_proposal_utility import (
    BOUNDARY_FEATURES,
    forward_with_fixed_recruitment,
    measure_proposal_batch,
)
from minicells.language_utility_skill_data import (
    MODEL_LENGTH,
    SKILL_FAMILIES,
    apply_family,
    generate_utility_skill_corpus,
)


def _model_with_newborn(vocab_size: int = 128):
    torch.manual_seed(19019)
    model = build_cellular_model(vocab_size, "G")
    state = LocalizedLearningState.capture(model)
    direction = torch.zeros(model.dim)
    direction[0] = 1.0
    child = conservative_fork(model, 1, step=0, direction=direction)
    assert child is not None
    return model, state


def test_utility_skill_families_are_deterministic_and_distinct() -> None:
    first = generate_utility_skill_corpus(60, seed=1234)
    second = generate_utility_skill_corpus(60, seed=1234)
    assert torch.equal(first.sequences, second.sequences)
    assert first.sequences.shape[1] == MODEL_LENGTH + 1
    assert first.loss_mask.shape == (MODEL_LENGTH,)
    assert set(first.family_names) == set(SKILL_FAMILIES)

    values = (1, 2, 3, 4, 5, 6)
    assert apply_family("REVERSE_INC", values) == (7, 6, 5, 4, 3, 2)
    assert apply_family("MOD_ADD", values) == (1, 3, 6, 0, 5, 1)
    assert apply_family("PARITY", values) == (1, 1, 0, 0, 1, 1)
    assert apply_family("DELAY_COPY", values) == (1, 2, 3, 1, 2, 3)
    assert apply_family("LOCAL_RULE", values) == (1, 3, 3, 5, 5, 7)
    assert apply_family("LOOKUP", (2, 7, 5, 9, 5, 0)) == (9, 9, 9, 9, 9, 9)


def test_recruitment_extremes_recover_base_and_fully_active_dynamics() -> None:
    model, state = _model_with_newborn()
    inputs = torch.randint(0, 100, (2, 12))

    closed = forward_with_fixed_recruitment(model, inputs, state, 0.0).output.logits
    saved_alive, saved_adjacency = set_newborn_tissue_active(model, state, False)
    base = model.forward_variable(inputs).output.logits
    restore_structure(model, saved_alive, saved_adjacency)
    assert torch.allclose(closed, base, atol=1e-6, rtol=1e-6)

    opened = forward_with_fixed_recruitment(model, inputs, state, 1.0).output.logits
    fully_active = model.forward_variable(inputs).output.logits
    assert torch.allclose(opened, fully_active, atol=1e-6, rtol=1e-6)


def test_gradient_utility_matches_small_finite_difference() -> None:
    model, state = _model_with_newborn()
    corpus = generate_utility_skill_corpus(8, seed=99, families=("MOD_ADD",))
    inputs = corpus.inputs[:4]
    targets = corpus.targets[:4]
    measured = measure_proposal_batch(
        model,
        state,
        inputs,
        targets,
        corpus.loss_mask,
        epsilon=1e-3,
    )
    gradient = measured["oracle_gradient"]
    finite = measured["oracle_fd"]
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(finite).all()
    assert torch.allclose(gradient, finite, atol=2e-2, rtol=0.15)


def test_label_free_features_do_not_depend_on_targets() -> None:
    model, state = _model_with_newborn()
    corpus = generate_utility_skill_corpus(8, seed=101, families=("LOCAL_RULE",))
    inputs = corpus.inputs[:4]
    targets = corpus.targets[:4]
    changed_targets = targets.roll(shifts=1, dims=-1)

    first = measure_proposal_batch(model, state, inputs, targets, corpus.loss_mask, epsilon=0.01)
    second = measure_proposal_batch(model, state, inputs, changed_targets, corpus.loss_mask, epsilon=0.01)
    for feature in BOUNDARY_FEATURES:
        assert torch.equal(first[feature], second[feature]), feature
    assert not torch.equal(first["oracle_gradient"], second["oracle_gradient"])


def test_closed_utility_is_independent_of_newborn_phenotype() -> None:
    model, state = _model_with_newborn()
    inputs = torch.randint(0, 100, (2, 10))
    before = forward_with_fixed_recruitment(model, inputs, state, 0.0).output.logits
    child = state.newborn_cells(model)[0]
    with torch.no_grad():
        model.cell_memory[child].normal_(mean=10.0, std=3.0)
        model.connect(0, child)
        model.connect(child, 3)
    after = forward_with_fixed_recruitment(model, inputs, state, 0.0).output.logits
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-6)
