from __future__ import annotations

import itertools
import math

import torch

from research.minicells.language_depth_ablation import (
    VARIANTS,
    factorial_contrast,
    geometric_ratio_from_log_contrast,
    resolve_stage_depths,
    step_embedding_rms,
    variant_by_code,
)
from research.minicells.language_scaling import build_minicells_v2
from research.minicells.language_stabilization import scale_step_embeddings, stabilizing_forward


def test_full_factorial_contains_all_eight_combinations() -> None:
    observed = {
        (variant.random_depth, variant.low_step_init, variant.uses_stability_loss)
        for variant in VARIANTS
    }
    assert observed == set(itertools.product((False, True), repeat=3))
    assert {variant.code for variant in VARIANTS} == set("ABCDEFGH")


def test_A_and_D_reproduce_experiment_011_recipes() -> None:
    a = variant_by_code("A")
    d = variant_by_code("D")
    assert not a.random_depth
    assert a.step_embedding_init_scale == 1.0
    assert a.stability_weight == 0.0
    assert d.random_depth
    assert d.step_embedding_init_scale == 0.25
    assert d.stability_weight == 0.10


def test_fixed_cells_ignore_random_schedule_and_random_cells_use_it() -> None:
    scheduled = (2, 3, 4)
    assert resolve_stage_depths(variant_by_code("A"), scheduled) == (4, 4, 4)
    assert resolve_stage_depths(variant_by_code("B"), scheduled) == scheduled


def test_factorial_main_effect_recovers_known_log_ratio() -> None:
    # Make only random depth contribute a multiplicative 0.8 effect.
    values = {}
    for variant in VARIANTS:
        value = math.log(0.8) if variant.random_depth else 0.0
        values[variant.code] = value
    contrast = factorial_contrast(values, ("random_depth",))
    assert math.isclose(geometric_ratio_from_log_contrast(contrast), 0.8, rel_tol=1e-12)
    assert math.isclose(factorial_contrast(values, ("low_step_init",)), 0.0, abs_tol=1e-12)
    assert math.isclose(factorial_contrast(values, ("stability_loss",)), 0.0, abs_tol=1e-12)


def test_step_embedding_scaling_is_initialization_only_and_trainable() -> None:
    torch.manual_seed(123)
    model = build_minicells_v2(64)
    before = step_embedding_rms(model)
    scale_step_embeddings(model, 0.25)
    after = step_embedding_rms(model)
    assert math.isclose(after / before, 0.25, rel_tol=1e-5)
    assert all(stage.step_embedding.requires_grad for stage in model.stages)


def test_full_depth_stabilizing_forward_matches_standard_forward() -> None:
    torch.manual_seed(321)
    model = build_minicells_v2(64).eval()
    ids = torch.randint(0, 64, (2, 32))
    standard = model(ids).logits
    recurrent = stabilizing_forward(model, ids, stage_depths=(4, 4, 4)).output.logits
    torch.testing.assert_close(standard, recurrent, rtol=1e-5, atol=1e-6)
