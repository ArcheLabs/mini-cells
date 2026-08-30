from pathlib import Path

import torch

from minicells.continual_learning import PARAMETER_COUNT, delta_vector, load_q88_model, model_hash
from minicells.optimizer_search import (
    EvalStats,
    SearchConfig,
    _step,
    choose_candidate,
    evaluate_stats,
    fixed_probe,
    objective_score,
    training_batch,
)

GENESIS = Path("service/generated/genesis_model.bin")


def test_python_mirror_matches_rust_generation_zero_metrics():
    flat = load_q88_model(GENESIS)
    parent = model_hash(flat)
    batch = training_batch(parent, 0, 1)
    base = evaluate_stats(flat, batch)
    delta = delta_vector(parent, 0, PARAMETER_COUNT)
    plus = evaluate_stats((flat + 4 * delta).clamp(-2048, 2048), batch)
    minus = evaluate_stats((flat - 4 * delta).clamp(-2048, 2048), batch)
    assert (base.loss, base.correct, base.total) == (22425, 1, 80)
    assert (plus.loss, plus.correct) == (22930, 1)
    assert (minus.loss, minus.correct) == (22735, 1)


def test_fixed_probe_matches_rust_local_gate_genesis():
    probe = fixed_probe(load_q88_model(GENESIS))
    assert probe.total_loss == 607901
    assert probe.correct_tokens == 28
    assert probe.total_tokens == 2168


def test_accuracy_lex_prioritizes_correct_tokens_then_loss():
    base = EvalStats(loss=100, correct=2, total=10)
    plus = EvalStats(loss=120, correct=3, total=10)
    minus = EvalStats(loss=80, correct=2, total=10)
    assert objective_score(plus, "accuracy-lex") < objective_score(base, "accuracy-lex")
    assert choose_candidate(base, plus, minus, "accuracy-lex") == 1
    assert choose_candidate(base, plus, minus, "loss") == -1


def test_block_delta_is_deterministic_and_localized():
    parent = bytes(range(32))
    a = delta_vector(parent, 7, 128)
    b = delta_vector(parent, 7, 128)
    assert torch.equal(a, b)
    assert int((a != 0).sum().item()) <= 128
    blocks = (PARAMETER_COUNT + 127) // 128
    block = 7 % blocks
    start = block * 128
    stop = min(start + 128, PARAMETER_COUNT)
    assert int((a[:start] != 0).sum().item()) == 0
    assert int((a[stop:] != 0).sum().item()) == 0


def test_step_recheck_never_accepts_non_improving_actual_proposal():
    flat = load_q88_model(GENESIS)
    config = SearchConfig(
        block_size=PARAMETER_COUNT,
        perturbation_q=8,
        step_q=1,
        objective="loss",
        batch_groups=1,
        apply_mode="step-recheck",
    )
    for generation in range(24):
        next_flat, record = _step(flat, generation, config)
        if record.accepted:
            assert record.proposal_loss < record.base_loss
            assert record.next_model_hash != record.parent_model_hash
        else:
            assert record.next_model_hash == record.parent_model_hash
        flat = next_flat


def test_evaluated_candidate_requires_same_step_and_perturbation():
    try:
        SearchConfig(
            block_size=PARAMETER_COUNT,
            perturbation_q=4,
            step_q=1,
            apply_mode="evaluated-candidate",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected evaluated-candidate q/step mismatch to be rejected")
