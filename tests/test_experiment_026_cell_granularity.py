from __future__ import annotations

import math

import torch

from minicells.clm_growth import ProgressiveGrowthCLM
from minicells.developmental_tissue import TissueConfig, TissueFFN, convert_model_experts_to_tissues
from minicells.granularity_30m import (
    DIAGNOSTIC_PERCEPTION_CAP,
    DOMAINS,
    DiagnosticBatch,
    _profile_specialization,
    collect_cell_diagnostics,
    continuation_lr,
    domain_for_step,
    model_structure,
    schedule_manifest,
    summarize_diagnostics,
)
from minicells.language_models import TextNCALM, count_parameters


def _tiny_progressive(granularity: int) -> ProgressiveGrowthCLM:
    torch.manual_seed(26026)
    source = TextNCALM(
        vocab_size=31,
        max_context=8,
        dim=8,
        heads=2,
        ffn_dim=16,
        windows=(2, 3, 4),
        iterations=(1, 1, 1),
        carry_bias=2.0,
    )
    model = ProgressiveGrowthCLM(source)
    converted = convert_model_experts_to_tissues(
        model,
        config=TissueConfig(cells_per_tissue=granularity),
        inplace=True,
    )
    assert isinstance(converted, ProgressiveGrowthCLM)
    return converted


def _batches(vocab_size: int = 31) -> dict[str, DiagnosticBatch]:
    result = {}
    for index, domain in enumerate(DOMAINS):
        generator = torch.Generator().manual_seed(30000 + index)
        inputs = torch.randint(0, vocab_size, (2, 6), generator=generator)
        targets = torch.randint(0, vocab_size, (2, 6), generator=generator)
        result[domain] = DiagnosticBatch(domain, inputs, targets)
    return result


def test_domain_schedule_is_exactly_balanced_per_block() -> None:
    for block in range(20):
        values = [domain_for_step(block * 4 + offset) for offset in range(4)]
        assert sorted(values) == sorted(DOMAINS)
    manifest = schedule_manifest(4_000)
    assert manifest["domain_tokens"] == {domain: 1_000 for domain in DOMAINS}


def test_continuation_lr_has_warmup_and_positive_floor() -> None:
    assert continuation_lr(0, total_steps=2_000) == 0.0
    assert 0.0 < continuation_lr(1, total_steps=2_000) < continuation_lr(500, total_steps=2_000)
    assert math.isclose(continuation_lr(500, total_steps=2_000), 1e-4)
    assert math.isclose(continuation_lr(2_000, total_steps=2_000), 1e-5, rel_tol=1e-6)


def test_specialization_entropy_behaves_as_registered() -> None:
    assert abs(_profile_specialization([1.0, 1.0, 1.0, 1.0])) < 1e-12
    assert _profile_specialization([1.0, 0.0, 0.0, 0.0]) > 0.999
    assert 0.0 < _profile_specialization([4.0, 2.0, 1.0, 1.0]) < 1.0


def test_all_granularities_are_parameter_neutral_and_keep_12_tissues() -> None:
    base = _tiny_progressive(1)
    expected = count_parameters(base)
    for granularity in (1, 2, 4, 8):
        model = _tiny_progressive(granularity)
        structure = model_structure(model)
        assert structure["program_tissues"] == 12
        assert structure["micro_cells"] == 12 * granularity
        assert count_parameters(model) == expected
        assert all(
            isinstance(expert, TissueFFN) and expert.cell_count == granularity
            for stage in model.stages
            for expert in stage.program_bank.experts.values()
        )


def test_diagnostics_emit_cell_and_tissue_rows_without_mutating_structure() -> None:
    model = _tiny_progressive(4)
    before = model_structure(model)
    batches = _batches()
    cells, tissues, profiles = collect_cell_diagnostics(
        model,
        batches=batches,
        domain_nlls={domain: 1.0 + 0.1 * index for index, domain in enumerate(DOMAINS)},
        baseline_profiles=None,
    )
    summary = summarize_diagnostics(cells, tissues)
    after = model_structure(model)

    assert before == after
    assert len(tissues) == 12
    assert len(cells) == 48
    assert len(profiles) == 48
    assert all(set(profile) == set(DOMAINS) for profile in profiles.values())
    assert all(0.0 <= float(row["specialization"]) <= 1.0 for row in cells)
    assert all(0.0 <= float(row["gradient_conflict"]) <= 1.0 for row in cells)
    assert all(0.0 <= float(row["diagnostic_stress"]) <= 1.0 for row in cells)
    assert 0.0 <= summary["median_cell_specialization"] <= 1.0
    assert DIAGNOSTIC_PERCEPTION_CAP >= 1


def test_age_zero_profiles_have_unit_stability_when_reused_as_baseline() -> None:
    model = _tiny_progressive(2)
    batches = _batches()
    nlls = {domain: 1.0 for domain in DOMAINS}
    _, _, baseline = collect_cell_diagnostics(
        model,
        batches=batches,
        domain_nlls=nlls,
        baseline_profiles=None,
    )
    cells, tissues, _ = collect_cell_diagnostics(
        model,
        batches=batches,
        domain_nlls=nlls,
        baseline_profiles=baseline,
    )
    assert min(float(row["profile_stability_vs_age0"]) for row in cells) > 0.9999
    assert min(float(row["mean_profile_stability_vs_age0"]) for row in tissues) > 0.9999
