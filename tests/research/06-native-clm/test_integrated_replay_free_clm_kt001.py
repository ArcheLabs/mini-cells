from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from minicells.integrated_replay_free_clm_kt001 import (
    SEED_REGISTRY_PATH,
    canonical_arm_map,
    validate_causal_matrix,
)
from minicells.integrated_replay_free_clm_kt001_aggregate import (
    aggregate_formal_seed_decisions,
    a_regression,
    phase_gains,
)
from minicells.integrated_replay_free_clm_kt001_mechanics import (
    capture_pre_step_cell_weights,
    finalize_realized_adamw_transaction_,
    force_shadow_expansion_,
)
from minicells.integrated_replay_free_clm_kt001_replay import MatchedReplayIterator
from minicells.integrated_replay_free_clm_kt001_runner import KT001RunnerConfig
from minicells.native_clm_m3 import NativeCLMM3GrowthConfig
from minicells.native_clm_m3l2 import MomentAccumulator, OnlineAddressNativeCLM
from minicells.native_clm_v0 import NativeCLM, NativeCLMConfig


def _tiny_config(*, cells: int = 1, active: int = 1) -> NativeCLMConfig:
    return NativeCLMConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=4,
        n_layers=1,
        n_heads=1,
        d_ff=8,
        initial_cells=cells,
        active_cells=active,
        cellular_layer_index=0,
        certificate_max_rank=2,
        tie_embeddings=False,
    )


def test_causal_matrix_is_frozen_and_explicit() -> None:
    validate_causal_matrix()
    arms = canonical_arm_map()
    assert tuple(arms) == (
        "unsafe",
        "write_transaction_only",
        "read_history_only",
        "full_no_replay",
        "matched_replay_oracle",
    )
    assert arms["unsafe"].mechanisms == ()
    assert arms["full_no_replay"].legacy_gradient_projection is True
    assert arms["full_no_replay"].realized_update_write_safety is True
    assert arms["full_no_replay"].historical_address_read is True
    assert arms["full_no_replay"].raw_replay is False
    assert arms["matched_replay_oracle"].raw_replay is True


def test_runner_schedule_is_fixed_before_formal() -> None:
    config = KT001RunnerConfig()
    config.validate()
    assert config.calibration_batches == 64
    assert config.forced_shadow_expansions_per_phase == 1
    assert config.bootstrap_sampling_seed == 74001


def test_final_realized_transaction_uses_r0b_projection() -> None:
    model = NativeCLM(_tiny_config())
    cell = model.cellular.cells[0]
    with torch.no_grad():
        q = torch.tensor([1.0, 2.0, 0.0, 0.0])
        q = q / torch.linalg.vector_norm(q)
        cell.certificate_basis[0].copy_(q)
        cell.certificate_rank.fill_(1)
        cell.weight.zero_()

    before = capture_pre_step_cell_weights(model)
    with torch.no_grad():
        cell.weight[0, :2] = torch.tensor([1.0, -1.0])

    result = finalize_realized_adamw_transaction_(
        model,
        before,
        arm=canonical_arm_map()["full_no_replay"],
        step=1,
    )
    rows = result["invariant_rows"]
    assert result["realized_update_projection_applied"] is True
    assert len(rows) == 1
    assert rows[0]["violation_ratio"] < 1e-6


def test_matched_replay_iterator_is_exactly_half_replay(tmp_path: Path) -> None:
    current = tmp_path / "current.txt"
    old_a = tmp_path / "a.txt"
    old_b = tmp_path / "b.txt"
    current.write_bytes(bytes(range(32)) * 20)
    old_a.write_bytes(bytes(reversed(range(32))) * 20)
    old_b.write_bytes(bytes(range(16)) * 40)

    model = SimpleNamespace(config=SimpleNamespace(max_seq_len=8))
    config = SimpleNamespace(batch_size=4, num_workers=0)
    iterator = MatchedReplayIterator(
        model=model,
        current_path=current,
        historical_paths={"A": old_a, "B": old_b},
        train_config=config,
        seed=123,
    )
    x, y = next(iterator)
    assert x.shape == y.shape == (4, 8)
    for _ in range(9):
        next(iterator)
    metadata = iterator.metadata()
    assert metadata["steps"] == 10
    assert metadata["current_examples"] == metadata["replay_examples"]
    assert metadata["replay_example_fraction"] == 0.5
    assert sum(metadata["historical_domain_steps"].values()) == 10


def test_forced_shadow_birth_uses_canonical_m3l2_path() -> None:
    model = OnlineAddressNativeCLM(_tiny_config(cells=8, active=2))
    parent = 0

    old = MomentAccumulator(model.config.d_model)
    current = MomentAccumulator(model.config.d_model)
    old_values = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [1.0, -0.1, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
        ]
    )
    current_values = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
            [-0.1, 1.0, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
        ]
    )
    old.update(old_values)
    current.update(current_values)
    model.historical_sketches[parent] = old.to_sketch(
        rank=32,
        diagonal_regularization=model.address_config.diagonal_regularization,
    )
    model.current_moments[parent] = current

    optimizer = torch.optim.AdamW(
        [cell.weight for cell in model.cellular.cells],
        lr=1e-3,
        weight_decay=0.01,
    )
    probe = torch.randint(0, model.config.vocab_size, (2, model.config.max_seq_len))
    event = force_shadow_expansion_(
        model,
        optimizer,
        growth_config=NativeCLMM3GrowthConfig(),
        global_step=0,
        probe_tokens=probe,
    )
    assert event["forced_by_protocol"] is True
    assert event["trigger_uses_evaluation_metrics"] is False
    assert event["parent_id"] == parent
    assert event["child_id"] == 8
    assert model.cell_count == 9
    assert event["birth_root_topk_match"] == 1.0
    assert event["birth_logits_max_abs_drift"] <= 1e-5


def _summary(
    a0: float,
    a1: float,
    b0: float,
    b1: float,
    c0: float,
    c1: float,
    d0: float,
    d1: float,
):
    def row(loss: float):
        return {"loss": loss, "active_fraction_vs_dense": 0.2}

    return {
        "evaluation_matrix": {
            "initial": {"A": row(a0), "B": row(b0), "C": row(c0), "D": row(d0)},
            "after_B": {"A": row(a0), "B": row(b1), "C": row(c0), "D": row(d0)},
            "after_C": {"A": row(a0), "B": row(b1), "C": row(c1), "D": row(d0)},
            "after_D": {"A": row(a1), "B": row(b1), "C": row(c1), "D": row(d1)},
        }
    }


def test_metric_semantics_match_native_continual_definitions() -> None:
    summary = _summary(1.0, 1.1, 2.0, 1.0, 4.0, 2.0, 6.0, 3.0)
    assert abs(a_regression(summary) - 0.1) < 1e-12
    assert phase_gains(summary) == {"B": 0.5, "C": 0.5, "D": 0.5}


def test_formal_aggregate_requires_unanimous_valid_outcome() -> None:
    decision = json.loads(
        Path(
            "research/experiments/04-continual-learning-core/"
            "integrated-replay-free-clm-kill-test-001/DECISION.json"
        ).read_text()
    )
    passes = [{"seed": seed, "classification": "PASS"} for seed in (1, 2, 3)]
    supported = aggregate_formal_seed_decisions(passes, decision_protocol=decision)
    assert supported["scientific_decision"] is True

    failures = [{"seed": seed, "classification": "VALID_FAIL"} for seed in (1, 2, 3)]
    rejected = aggregate_formal_seed_decisions(failures, decision_protocol=decision)
    assert rejected["scientific_decision"] is False

    mixed = [
        {"seed": 1, "classification": "PASS"},
        {"seed": 2, "classification": "VALID_FAIL"},
        {"seed": 3, "classification": "INCONCLUSIVE_ORACLE"},
    ]
    inconclusive = aggregate_formal_seed_decisions(mixed, decision_protocol=decision)
    assert inconclusive["scientific_decision"] is None


def test_formal_seed_values_do_not_leak_into_execution_entrypoints() -> None:
    registry = json.loads(Path(SEED_REGISTRY_PATH).read_text())
    formal = [str(seed) for seed in registry["formal"]]
    paths = [
        Path("scripts/research/run_integrated_replay_free_clm_kt001.py"),
        Path("scripts/research/aggregate_integrated_replay_free_clm_kt001.py"),
        Path("scripts/research/publish_integrated_replay_free_clm_kt001.py"),
    ]
    text = "\n".join(path.read_text() for path in paths)
    for seed in formal:
        assert seed not in text
