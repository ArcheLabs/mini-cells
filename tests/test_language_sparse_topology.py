from __future__ import annotations

import numpy as np
import pytest
import torch

from minicells.language_models import count_parameters
from minicells.language_skill_data import (
    ALL_TASKS,
    BASE_TASKS,
    COMPOSITION_MAP,
    DIGIT_BASE,
    DIGIT_COUNT,
    EOS,
    MODEL_LENGTH,
    SEP,
    apply_task,
    encode_example,
    generate_skill_corpus,
)
from minicells.language_sparse_topology import (
    ACTIVE_LATENT,
    TISSUE_HEIGHT,
    build_sparse_topology_model,
)
from minicells.language_topology_metrics import (
    composition_reuse_scores,
    permutation_mi_null,
    weighted_mutual_information,
)


def tiny_model(variant: str):
    torch.manual_seed(123)
    return build_sparse_topology_model(
        32,
        variant=variant,
        tissue_height=TISSUE_HEIGHT,
        active_latent=ACTIVE_LATENT,
        max_context=MODEL_LENGTH,
        dim=16,
        heads=2,
        ffn_dim=32,
        windows=(4, 8, 16),
        iterations=(2, 2, 2),
    )


def test_skill_operations_and_compositions_are_deterministic() -> None:
    digits = (1, 2, 3, 4, 5, 6)
    assert apply_task("REVERSE", digits) == (6, 5, 4, 3, 2, 1)
    assert apply_task("ROTATE", digits) == (2, 3, 4, 5, 6, 1)
    assert apply_task("INC", digits) == (2, 3, 4, 5, 6, 7)
    assert apply_task("SWAP", digits) == (2, 1, 4, 3, 6, 5)
    left, right = COMPOSITION_MAP["REVERSE_ROTATE"]
    assert apply_task("REVERSE_ROTATE", digits) == apply_task(right, apply_task(left, digits))


def test_encoded_example_is_autoregressive_and_fixed_length() -> None:
    encoded = encode_example("REVERSE", (1, 2, 3, 4, 5, 6))
    assert len(encoded) == MODEL_LENGTH + 1
    assert encoded[2 : 2 + DIGIT_COUNT] == tuple(DIGIT_BASE + value for value in (1, 2, 3, 4, 5, 6))
    assert SEP in encoded
    assert encoded[-1] == EOS


def test_synthetic_corpus_is_reproducible_and_balanced() -> None:
    first = generate_skill_corpus(1000, seed=15)
    second = generate_skill_corpus(1000, seed=15)
    assert torch.equal(first.sequences, second.sequences)
    assert torch.equal(first.task_ids, second.task_ids)
    counts = torch.bincount(first.task_ids, minlength=len(ALL_TASKS))
    assert counts.min().item() == counts.max().item() == 100
    assert int(first.loss_mask.sum()) == DIGIT_COUNT + 1


def test_variants_have_equal_parameter_count_and_no_trainable_row_embedding() -> None:
    models = [tiny_model(code) for code in "ABC"]
    assert len({count_parameters(model) for model in models}) == 1
    for model in models:
        names = {name for name, _ in model.named_parameters()}
        assert not any("row_embedding" in name for name in names)
        buffers = {name for name, _ in model.named_buffers()}
        assert "row_encoding" in buffers


def test_sparse_activity_has_row_zero_plus_exactly_two_latent_rows() -> None:
    model = tiny_model("B")
    input_ids = torch.randint(0, 32, (2, MODEL_LENGTH))
    result = model.forward_variable(input_ids, stage_depths=(1, 1, 1), collect_topology=True)
    diagnostics = result.diagnostics
    assert diagnostics is not None
    activity = diagnostics.activity
    assert torch.all(activity[..., 0] == 1)
    assert torch.allclose(activity.sum(dim=-1), torch.full_like(activity[..., 0], 1 + ACTIVE_LATENT))
    assert diagnostics.logical_active_fraction.item() == pytest.approx((1 + ACTIVE_LATENT) / TISSUE_HEIGHT)


def test_dynamic_edges_are_nonlocal_and_only_target_active_receivers() -> None:
    model = tiny_model("C")
    input_ids = torch.randint(0, 32, (2, MODEL_LENGTH))
    result = model.forward_variable(input_ids, stage_depths=(1, 1, 1), collect_topology=True)
    diagnostics = result.diagnostics
    assert diagnostics is not None
    edges = diagnostics.edges
    nonzero = torch.nonzero(edges > 0, as_tuple=False)
    assert len(nonzero) > 0
    for item in nonzero:
        source = int(item[-2])
        receiver = int(item[-1])
        assert abs(source - receiver) > 1
    incoming = edges.sum(dim=-2)
    assert torch.allclose(incoming, diagnostics.activity)


def test_ablation_removes_row_from_sparse_activity() -> None:
    model = tiny_model("C")
    input_ids = torch.randint(0, 32, (2, MODEL_LENGTH))
    result = model.forward_variable(
        input_ids,
        stage_depths=(1, 1, 1),
        collect_topology=True,
        ablate_row=3,
    )
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert torch.all(diagnostics.activity[..., 3] == 0)


def test_weighted_mi_and_permutation_null_detect_task_region_specialization() -> None:
    task_ids = np.repeat(np.arange(4), 50)
    features = np.zeros((200, 4), dtype=np.float64)
    features[np.arange(200), task_ids] = 1.0
    observed = weighted_mutual_information(task_ids, features)
    stats = permutation_mi_null(task_ids, features, seed=7, permutations=200)
    assert observed > 1.0
    assert stats["observed"] > stats["null_p99"]
    assert stats["empirical_p"] < 0.02


def test_composition_reuse_prefers_true_component_pair() -> None:
    activity = {
        "REVERSE": (1.0, 0.0, 0.0, 0.0),
        "ROTATE": (0.0, 1.0, 0.0, 0.0),
        "INC": (0.0, 0.0, 1.0, 0.0),
        "SWAP": (0.0, 0.0, 0.0, 1.0),
    }
    for composite, (left, right) in COMPOSITION_MAP.items():
        activity[composite] = tuple(
            np.asarray(activity[left], dtype=np.float64) + np.asarray(activity[right], dtype=np.float64)
        )
    rows = composition_reuse_scores(activity, COMPOSITION_MAP, BASE_TASKS)
    assert len(rows) == len(COMPOSITION_MAP)
    assert all(float(row["reuse_margin_vs_best_wrong"]) > 0 for row in rows)
