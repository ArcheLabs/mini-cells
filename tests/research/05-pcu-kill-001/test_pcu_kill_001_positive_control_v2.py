"""Regression coverage for the repaired base-model positive control and audit path."""

from __future__ import annotations

from pathlib import Path

import torch

from minicells.pcu_kill_001.cellular import GraniteArchitectureInspector
from minicells.pcu_kill_001.governance import _split_source_and_generated_status
from minicells.pcu_kill_001.pipeline_guard import persist_pre_science_evidence
from minicells.pcu_kill_001.synthetic import (
    POSITIVE_CONTROL_VERSION,
    _candidate_pool,
    _teacher_forced_candidate_scores,
    audit_dataset,
    context_oracle,
    generate_world,
)


class _RankingTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if add_special_tokens:
            return [2]
        return {"RIGHT": [5], "WRONG": [6]}[text]


class _RankingModel:
    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        logits = torch.zeros((*input_ids.shape, 8), dtype=torch.float32, device=input_ids.device)
        logits[..., 5] = 5.0
        logits[..., 6] = -5.0
        return type("Output", (), {"logits": logits})()


def test_teacher_forced_candidate_ranking_uses_causal_logits_not_generation() -> None:
    scores = _teacher_forced_candidate_scores(
        _RankingModel(),
        _RankingTokenizer(),
        "prompt",
        ("WRONG", "RIGHT"),
        device="cpu",
    )
    assert scores[1] > scores[0]


def test_candidate_pool_is_deterministic_contains_correct_and_has_no_position_bias() -> None:
    values = tuple(f"V{index:04d}" for index in range(32))
    left = _candidate_pool(values, "V0007", "sample", size=16)
    right = _candidate_pool(values, "V0007", "sample", size=16)
    assert left == right
    assert len(left) == 16
    assert len(set(left)) == 16
    assert "V0007" in left
    assert left[0] != "V0007"  # correct is not privileged by candidate ordering


def test_symbolic_context_oracle_v2_preserves_reference_capacity() -> None:
    world = generate_world(26090501, 8)
    oracle = context_oracle(world)
    assert oracle["schema"] == "minicells.pcu-kill-001.context-oracle.v2"
    assert oracle["positive_control_version"] == POSITIVE_CONTROL_VERSION
    assert oracle["retrieval_a_accuracy"] == 1.0
    assert oracle["retrieval_b_accuracy"] == 1.0
    assert oracle["composition_accuracy"] == 1.0
    assert oracle["passed"] is True


def test_generated_research_artifacts_do_not_mark_source_dirty() -> None:
    source, generated = _split_source_and_generated_status(
        "?? artifacts/research/pcu/run.json\n M src/minicells/example.py"
    )
    assert source == " M src/minicells/example.py"
    assert generated == "?? artifacts/research/pcu/run.json"


class _Metric:
    def __init__(self, passed: bool = True) -> None:
        self.passed = passed

    def to_dict(self) -> dict[str, object]:
        return {"passed": self.passed, "relative_l2": 0.0}


def test_fail_fast_audit_writes_g0_and_cache_before_oracle(tmp_path: Path) -> None:
    world = generate_world(26090501, 4)
    audit = audit_dataset(world)
    inspector = GraniteArchitectureInspector(
        target_path="model.layers.23.block_sparse_moe",
        target_layer=23,
        hidden_size=1024,
        intermediate_size=512,
        local_experts=32,
        experts_per_token=8,
        fused_projection=True,
        fused_order="gate_up",
    )
    persist_pre_science_evidence(
        phase="engineering",
        seed=26090501,
        output=tmp_path / "26090501-oracle-v2",
        device="cpu",
        tokenizer=None,
        original=None,
        cellular=None,
        manifest={"model_repo": "m", "model_revision": "r"},
        inspector=inspector,
        g0=[_Metric()],
        g0_full_moe=_Metric(),
        g0_e2e=_Metric(),
        cache_gate=_Metric(),
        world=world,
        audit=audit,
        allow_search=True,
    )
    output = tmp_path / "26090501-oracle-v2"
    assert (output / "EQUIVALENCE.json").is_file()
    assert (output / "CACHE_EQUIVALENCE.json").is_file()
    assert (output / "RUN_IDENTITY.json").is_file()
    identity = __import__("json").loads((output / "RUN_IDENTITY.json").read_text())
    assert identity["positive_control_version"] == POSITIVE_CONTROL_VERSION
