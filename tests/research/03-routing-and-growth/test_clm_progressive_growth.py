import json

import torch

from minicells.clm_growth import ProgressiveGrowthCLM, next_growth_event, replicate_seed, stop_target
from minicells.growth_pressure import (PressureCandidate, calibrate_model_pressure, cosine_kmeans_2,
                                       gradient_disagreement, pressure_score, select_pressure_parent,
                                       select_random_parent)
from minicells.growth_reporting import aggregate_formal_results, write_ppl_history
from minicells.growth_validation import (ExecutionCapture, compare_captures, progressive_growth_decision,
                                         root_router_balance_loss, student_teacher_kl)
from minicells.language_data import make_training_schedule
from minicells.language_models import TextNCALM
from minicells.language_scaling import _promote_stream_requirements
from minicells.upcycled_cellular_textnca import UpcyclingConfig, convert_textnca_to_upcycled


def _model() -> ProgressiveGrowthCLM:
    torch.manual_seed(11)
    source = TextNCALM(vocab_size=23, max_context=8, dim=8, heads=2, ffn_dim=12,
                       windows=(2, 3, 4), iterations=(1, 1, 1), carry_bias=2.0)
    source = convert_textnca_to_upcycled(source, config=UpcyclingConfig(num_experts=4, top_k=1))
    return ProgressiveGrowthCLM(source)


def test_zero_birth_matches_clm01() -> None:
    torch.manual_seed(4)
    source = TextNCALM(vocab_size=23, max_context=8, dim=8, heads=2, ffn_dim=12,
                       windows=(2, 3, 4), iterations=(1, 1, 1), carry_bias=2.0)
    upcycled = convert_textnca_to_upcycled(source, config=UpcyclingConfig(num_experts=4, top_k=1))
    growth = ProgressiveGrowthCLM(upcycled)
    inputs = torch.randint(0, 23, (2, 6))
    torch.testing.assert_close(growth(inputs).logits, upcycled(inputs).logits, rtol=1e-5, atol=1e-6)


def test_two_births_preserve_function_and_are_recursive() -> None:
    model = _model()
    inputs = torch.randint(0, 23, (2, 6))
    perceptions = torch.randn(512, 8)
    first = model.birth(stage=1, parent_id="s1-e2", routed_perceptions=perceptions,
                        token=500_000, validation_inputs=inputs)
    second = model.birth(stage=1, parent_id=first["child"], routed_perceptions=perceptions,
                         token=1_000_000, validation_inputs=inputs)
    assert first["parity"]["status"] == "CLM_GROWTH_EQUIVALENCE"
    assert second["parity"]["status"] == "CLM_GROWTH_EQUIVALENCE"
    assert model.expert_counts_by_stage() == [4, 6, 4]
    assert model.growth_history[-1]["parent"] == first["child"]


def test_masked_dense_and_sparse_parity_after_growth() -> None:
    model = _model()
    model.birth(stage=0, parent_id="s0-e0", routed_perceptions=torch.randn(512, 8),
                token=500_000)
    inputs = torch.randint(0, 23, (2, 6))
    dense = model(inputs, execution_backend="masked_dense").logits
    sparse = model(inputs, execution_backend="sparse_dispatch").logits
    torch.testing.assert_close(dense, sparse, rtol=2e-5, atol=2e-6)


def test_pressure_formula_and_geometry_are_deterministic() -> None:
    gradients = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]
    assert gradient_disagreement(gradients) == 1.0
    assert pressure_score(.25, .4) == .35
    samples = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-.9, -.1]])
    torch.testing.assert_close(cosine_kmeans_2(samples), cosine_kmeans_2(samples))


def test_normal_training_does_not_collect_pressure_perceptions() -> None:
    model = _model()
    model(torch.randint(0, 23, (2, 6)))
    assert all(not any(bank.last_perceptions.values())
               for bank in (stage.program_bank for stage in model.stages))


def test_pressure_collection_is_bounded() -> None:
    model = _model()
    batches = [(torch.randint(0, 23, (8, 8)), torch.randint(0, 23, (8, 8))) for _ in range(12)]
    _, perceptions = calibrate_model_pressure(model, batches, min_samples=1)
    assert perceptions
    assert all(value.shape[0] <= 8192 for value in perceptions.values())


def test_resume_boundaries_execute_each_birth_once() -> None:
    assert next_growth_event(500_000, []) == (0, 500_000)
    assert next_growth_event(500_000, [{"birth_index": 1}]) is None
    assert next_growth_event(1_000_000, [{"birth_index": 1}]) == (1, 1_000_000)
    assert next_growth_event(1_000_000, [{"birth_index": 1}, {"birth_index": 2}]) is None


def test_stop_after_tokens() -> None:
    assert stop_target(1_500_000, 523_001, 1_000) == 524_000
    assert stop_target(1_500_000, None, 1_000) == 1_500_000


def test_corpus_requirements_preserve_experiment_005_prefixes() -> None:
    manifest = {"train_stream_tokens": 800_000, "validation_stream_tokens": 100_000}
    assert _promote_stream_requirements(
        manifest,
        train_stream_tokens=1_500_127,
        validation_stream_tokens=4_032,
    ) == (1_500_127, 100_000)


def test_replicate_schedules_are_paired_and_distinct() -> None:
    schedules = [make_training_schedule(50_000, seed=replicate_seed(i), budget_tokens=8_000,
                                        batch_size=4, sequence_length=8).starts for i in range(3)]
    for replicate in range(3):
        assert schedules[replicate] == make_training_schedule(
            50_000, seed=replicate_seed(replicate), budget_tokens=8_000,
            batch_size=4, sequence_length=8).starts
    assert len(set(schedules)) == 3


def test_kl_is_student_to_teacher() -> None:
    student = torch.tensor([[[2.0, -1.0]]], requires_grad=True)
    teacher = torch.tensor([[[-2.0, 1.0]]])
    value = student_teacher_kl(student, teacher)
    student_logp = torch.log_softmax(student, -1)
    expected = (student_logp.exp() * (student_logp - torch.log_softmax(teacher, -1))).sum(-1).mean()
    torch.testing.assert_close(value, expected)


def test_root_balance_has_router_gradient_before_and_after_growth() -> None:
    model = _model()
    inputs = torch.randint(0, 23, (2, 6))
    for grow in (False, True):
        if grow:
            model.birth(stage=0, parent_id="s0-e0", routed_perceptions=torch.randn(512, 8), token=500_000)
        model.zero_grad(set_to_none=True)
        _, stats = model(inputs, return_stats=True)
        root_router_balance_loss(stats.root_usage).backward()
        assert any(parameter.grad is not None for stage in model.stages
                   for parameter in stage.program_bank.router.root_router.parameters())


def test_birth_checks_real_ppl_ratio_and_aligned_routes() -> None:
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    states = (torch.zeros(1),)
    before = ExecutionCapture(logits, states, (torch.tensor([[[0, 1]]]),))
    after = ExecutionCapture(logits.flip(-1), states, (torch.tensor([[[1, 0]]]),))
    result = compare_captures(before, after, validation_targets=torch.tensor([[0, 1]]))
    assert result["ppl_ratio"] != 1.0
    assert not result["non_parent_root_routes_unchanged"]


def test_pressure_and_random_use_same_eligible_pool() -> None:
    pool = [PressureCandidate(0, "a", .4, .2, .48, 600, True),
            PressureCandidate(0, "b", .5, .0, .5, 600, True),
            PressureCandidate(0, "c", .9, .9, 1.71, 5, False)]
    assert select_pressure_parent(pool).expert_id == "b"
    assert select_random_parent(pool, seed=2).expert_id in {"a", "b"}


def test_growth_and_pressure_selection_decisions_are_separate() -> None:
    rows = []
    for replicate in range(3):
        rows.extend([{"arm": "fixed4", "replicate": replicate, "ppl": 10.0},
                     {"arm": "pressure_growth", "replicate": replicate, "ppl": 9.9,
                      "viable": True, "equivalent_births": 2},
                     {"arm": "random_growth", "replicate": replicate, "ppl": 10.1}])
    decision = progressive_growth_decision(rows)
    assert decision["growth_utility"]["status"] == "CLM_PROGRESSIVE_GROWTH_SIGNAL"
    assert decision["pressure_selection"]["status"] == "CLM_GROWTH_PRESSURE_SELECTION_SIGNAL"
    assert decision["formal_gpu_experiment_run"] is False


def test_formal_aggregation_uses_matched_fixed4_and_derives_viability(tmp_path) -> None:
    for replicate in range(3):
        for arm, ppl in (("fixed4", 10.0), ("pressure_growth", 9.9), ("random_growth", 10.1)):
            directory = tmp_path / f"r{replicate}-{arm}"
            directory.mkdir(parents=True)
            complete = {
                "type": "worker_complete", "arm": arm, "replicate": replicate,
                "consumed_tokens": 1_500_000, "target_tokens": 1_500_000,
            }
            (directory / "events.jsonl").write_text(json.dumps(complete) + "\n", encoding="utf-8")
            write_ppl_history(directory / "ppl-history.csv", [{
                "replicate": replicate, "arm": arm, "tokens": 1_500_000,
                "phase": "complete", "ppl": ppl, "nll": 2.0,
                "fixed4_ppl": None, "clm01_start_ppl": 11.0,
                "textnca_frozen_ppl": 12.0, "ppl_vs_fixed4": None,
                "ppl_vs_clm01": ppl / 11.0, "ppl_vs_textnca": ppl / 12.0,
                "health": "GREEN",
            }])
            if arm != "fixed4":
                history = [
                    {"birth_index": 1, "stage": 0, "parent": "s0-e0", "child": "s0-e4",
                     "parity": {"status": "CLM_GROWTH_EQUIVALENCE"}},
                    {"birth_index": 2, "stage": 1, "parent": "s1-e1", "child": "s1-e4",
                     "parity": {"status": "CLM_GROWTH_EQUIVALENCE"}},
                ]
                diagnostic = {
                    "offset_tokens": 500_000, "child_usage": .1, "relative_l2": .01,
                    "split_entropy": .5, "causal_merge_back_penalty": .001,
                }
                diagnostics = [
                    {"birth_index": 1, **diagnostic},
                    {"birth_index": 2, **diagnostic},
                ]
                (directory / "growth-history.json").write_text(
                    json.dumps(history), encoding="utf-8"
                )
                (directory / "newborn-diagnostics.json").write_text(
                    json.dumps(diagnostics), encoding="utf-8"
                )

    aggregate = aggregate_formal_results(tmp_path, formal_gpu_experiment_run=True)
    pressure = [row for row in aggregate["formal_rows"] if row["arm"] == "pressure_growth"]
    assert len(pressure) == 3
    assert all(abs(row["fixed4_ppl"] - 10.0) < 1e-12 for row in pressure)
    assert all(abs(row["ppl_vs_fixed4"] - .99) < 1e-12 for row in pressure)
    assert aggregate["decision"]["growth_equivalence"]["status"] == "CLM_GROWTH_EQUIVALENCE"
    assert aggregate["decision"]["growth_viability"]["status"] == "CLM_PROGRESSIVE_GROWTH_VIABILITY"
    assert aggregate["decision"]["growth_utility"]["status"] == "CLM_PROGRESSIVE_GROWTH_SIGNAL"
    assert aggregate["decision"]["pressure_selection"]["status"] == "CLM_GROWTH_PRESSURE_SELECTION_SIGNAL"
    assert aggregate["decision"]["formal_gpu_experiment_run"] is True
