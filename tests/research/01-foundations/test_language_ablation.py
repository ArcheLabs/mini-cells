from __future__ import annotations

import pandas as pd
import pytest
import torch

from minicells.language_ablation import FACTORIAL_SPECS, factorial_effects, validate_factorial_specs
from minicells.language_models import TextNCALM


def test_factorial_design_contains_all_eight_combinations() -> None:
    validate_factorial_specs()
    combinations = {
        (spec.rms_norm, spec.carry_bias, spec.auxiliary_loss)
        for spec in FACTORIAL_SPECS
    }
    assert len(FACTORIAL_SPECS) == 8
    assert len(combinations) == 8
    assert FACTORIAL_SPECS[0].name == "ln-c0-a0"
    assert FACTORIAL_SPECS[-1].name == "rms-c2-aux"


def test_factorial_effects_recover_known_main_and_interaction_terms() -> None:
    rows = []
    for spec in FACTORIAL_SPECS:
        codes = spec.factor_codes()
        response = (
            10.0
            - 0.5 * codes["R"]
            - 0.2 * codes["C"]
            + 0.1 * codes["A"]
            + 0.3 * codes["R"] * codes["C"]
        )
        rows.append(
            {
                "rms_norm": spec.rms_norm,
                "carry_bias": spec.carry_bias,
                "auxiliary_loss": spec.auxiliary_loss,
                "validation_nll": response,
            }
        )
    effects = factorial_effects(pd.DataFrame(rows)).set_index("term")
    assert effects.loc["R", "effect_nll"] == pytest.approx(-1.0)
    assert effects.loc["C", "effect_nll"] == pytest.approx(-0.4)
    assert effects.loc["A", "effect_nll"] == pytest.approx(0.2)
    assert effects.loc["RC", "effect_nll"] == pytest.approx(0.6)
    assert effects.loc["RA", "effect_nll"] == pytest.approx(0.0)
    assert effects.loc["CA", "effect_nll"] == pytest.approx(0.0)
    assert effects.loc["RCA", "effect_nll"] == pytest.approx(0.0)


def test_factor_switches_preserve_matching_random_initialization() -> None:
    kwargs = {
        "vocab_size": 64,
        "max_context": 16,
        "dim": 16,
        "heads": 4,
        "ffn_dim": 32,
        "windows": (4, 8, 16),
        "iterations": (1, 1, 1),
    }
    torch.manual_seed(55005)
    baseline = TextNCALM(
        **kwargs,
        rms_norm=False,
        carry_bias=0.0,
        stage_supervision=False,
    )
    torch.manual_seed(55005)
    full = TextNCALM(
        **kwargs,
        rms_norm=True,
        carry_bias=2.0,
        stage_supervision=True,
    )

    assert torch.equal(baseline.token_embedding.weight, full.token_embedding.weight)
    assert torch.equal(baseline.position_embedding.weight, full.position_embedding.weight)
    assert torch.equal(baseline.stages[0].attention.qkv.weight, full.stages[0].attention.qkv.weight)
    assert torch.equal(baseline.stages[0].gru.weight_ih, full.stages[0].gru.weight_ih)
    assert torch.equal(baseline.stages[0].gru.weight_hh, full.stages[0].gru.weight_hh)

    hidden = baseline.stages[0].gru.hidden_size
    baseline_update = baseline.stages[0].gru.bias_ih[hidden : 2 * hidden]
    full_update = full.stages[0].gru.bias_ih[hidden : 2 * hidden]
    assert not torch.equal(baseline_update, full_update)


def test_auxiliary_switch_does_not_change_parameter_count_for_same_norm() -> None:
    torch.manual_seed(1)
    no_aux = TextNCALM(vocab_size=128, rms_norm=True, carry_bias=0.0, stage_supervision=False)
    torch.manual_seed(1)
    aux = TextNCALM(vocab_size=128, rms_norm=True, carry_bias=0.0, stage_supervision=True)
    assert sum(p.numel() for p in no_aux.parameters()) == sum(p.numel() for p in aux.parameters())
