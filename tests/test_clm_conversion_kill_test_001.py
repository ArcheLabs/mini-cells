from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.research.clm_conversion_kill_test_001.dataset import (
    DATASET_VERSION,
    ENTITIES,
    PROTOCOLS,
    REGIONS,
    contextual_conflict_rows,
    formation_evaluation,
    formation_validation,
    rewrite_rows,
    training_rows,
)
from scripts.research.clm_conversion_kill_test_001.semantic_choice import (
    summarize_candidate_scores,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "research" / "validations" / "clm-conversion-kill-test-001" / "protocol.json"
)
DATASET = ROOT / "scripts" / "research" / "clm_conversion_kill_test_001" / "dataset.py"


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = b"blob " + str(len(data)).encode() + b"\0"
    return hashlib.sha1(header + data).hexdigest()


def test_controlled_dataset_has_registered_identity_and_no_question_leakage() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    train = training_rows()
    validation = formation_validation()
    evaluation = formation_evaluation()
    assert protocol["dataset"]["version"] == DATASET_VERSION
    assert len(ENTITIES) == protocol["dataset"]["entities"] == 12
    assert len(PROTOCOLS) == protocol["dataset"]["protocol_values"] == 6
    assert len(REGIONS) == protocol["dataset"]["region_values"] == 3
    assert len(train) == 36
    assert len(validation) == protocol["dataset"]["validation_rows"] == 12
    assert {name: len(rows) for name, rows in evaluation.items()} == {
        "direct": 24,
        "negation": 12,
        "relation": 12,
        "routing": 48,
    }
    train_questions = {row["question"] for row in train}
    validation_questions = {row["question"] for row in validation}
    eval_questions = {row["question"] for rows in evaluation.values() for row in rows}
    assert train_questions.isdisjoint(validation_questions)
    assert train_questions.isdisjoint(eval_questions)
    assert validation_questions.isdisjoint(eval_questions)
    allowed_answers = {*PROTOCOLS, *REGIONS}
    assert all(row["answer"] in allowed_answers for row in train)
    assert _git_blob_sha(DATASET) == protocol["dataset"]["generator_git_blob_sha"]


def test_rewrite_and_growth_use_unseen_surface_forms() -> None:
    entity = ENTITIES[0]
    old_protocol = PROTOCOLS[0]
    new_protocol = PROTOCOLS[3]
    rewrite = rewrite_rows(entity, new_protocol, prefix="unit")
    assert {row["question"] for row in rewrite["train"]}.isdisjoint(
        {row["question"] for row in rewrite["evaluation"]}
    )
    assert {row["answer"] for row in rewrite["train"] + rewrite["evaluation"]} == {
        new_protocol
    }

    growth = contextual_conflict_rows(entity, old_protocol, new_protocol)
    assert {row["answer"] for row in growth["alpha"]} == {old_protocol}
    assert {row["answer"] for row in growth["beta_train"] + growth["beta_eval"]} == {
        new_protocol
    }
    assert {row["question"] for row in growth["beta_train"]}.isdisjoint(
        {row["question"] for row in growth["beta_eval"]}
    )


def test_candidate_choice_requires_strict_reference_ranking() -> None:
    rows = [
        {"id": "a", "answer": "QX-17"},
        {"id": "b", "answer": "LM-42"},
    ]
    candidates = ("QX-17", "LM-42", "VR-08")
    summary = summarize_candidate_scores(
        rows,
        candidates,
        [
            [0.2, 0.8, 1.1],
            [0.5, 0.5, 0.9],
        ],
    )
    assert summary["strict_choice_accuracy"] == 0.5
    assert summary["rows"][0]["strict_correct"] is True
    assert summary["rows"][1]["strict_correct"] is False
    assert summary["rows"][1]["margin"] == 0.0


def test_protocol_freezes_semantic_hardening_before_formal_gpu() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["experiment"] == "CLM_CONVERSION_KILL_TEST_001"
    assert protocol["status"] == "PROTOCOL_FROZEN_GPU_PENDING"
    assert protocol["protocol_version"] == 1.2
    assert protocol["base"]["foundation_trainable"] is False
    assert protocol["substrate"]["pretrained_expert_is_cell"] is False
    assert protocol["substrate"]["pretrained_channel_group_is_cell"] is False
    assert protocol["substrate"]["layer_indices"] == [7, 15, 23]
    assert protocol["sequence_task"]["candidate_choice_append_eos"] is False
    assert (
        protocol["decision"]["formal_support"]
        == "at least 2 of 3 untouched formal seeds pass"
    )
    assert "after the first formal seed starts" in protocol["decision"]["freeze_rule"]
    checkpoint_rule = protocol["training"]["formation"]["checkpoint_rule"]
    assert "validation" in checkpoint_rule
    assert "final direct/negation/relation heldout" in checkpoint_rule
    assert "candidate-choice metrics" in checkpoint_rule

    gates = protocol["gates"]
    assert gates["minimum_direct_choice_accuracy"] == 0.75
    assert gates["minimum_negation_choice_accuracy"] == 0.75
    assert gates["minimum_relation_choice_accuracy"] == 0.75
    assert gates["minimum_semantic_write_choice_accuracy"] == 1.0
    assert gates["minimum_growth_alpha_choice_accuracy"] == 1.0
    assert gates["minimum_growth_beta_choice_accuracy"] == 1.0
    assert gates["minimum_branch_standalone_choice_accuracy"] == 1.0
    assert gates["minimum_branch_merged_choice_accuracy"] == 1.0
    assert gates["require_growth_necessity"] is True
    assert gates["maximum_spawn_mean_nll_delta"] == 1e-5
    assert gates["maximum_spawn_choice_margin_delta"] == 1e-5
