from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.research.cow_clm_001.dataset import capability_rows, knowledge_rows

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "research" / "validations" / "cow-clm-001" / "protocol.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _git_blob(path: str) -> str:
    return subprocess.check_output(["git", "hash-object", path], cwd=ROOT, text=True).strip()


def test_cow_clm_001_protocol_identity_is_frozen() -> None:
    protocol = _protocol()
    assert protocol["experiment"] == "COW_CLM_001"
    assert protocol["protocol_version"] == "1.1"
    assert protocol["status"] == "FORMAL_PROTOCOL_FROZEN_GPU_PENDING"
    assert protocol["seed"] == 26090511


def test_cow_clm_001_capacity_and_decision_are_not_posthoc_economic_gates() -> None:
    protocol = _protocol()
    assert protocol["capacity_sites"] == [1, 2, 4, 8]
    decision = protocol["decision"]
    assert decision["private_fraction_threshold"] is None
    assert "report the minimum measured fraction" in decision["private_fraction_policy"]
    assert "both knowledge and capability" in decision["pass_requires"]


def test_cow_clm_001_keeps_routing_and_lineage_out_of_scope() -> None:
    semantics = _protocol()["cell_semantics"]
    assert "oracle Cell activation" in semantics["learned_routing"]
    assert semantics["deeper_lineage"] == "out of scope until COW-CLM-002"
    assert semantics["canonical_composition"] == "inheritance/substitution, never additive sibling merge"
    assert semantics["private_delta_dtype"] == "fp32 with cast to parent dtype at execution"


def test_cow_clm_001_structural_gates_are_architectural_not_kl_thresholds() -> None:
    protocol = _protocol()
    gates = protocol["structural_gates"]
    assert gates == {
        "empty_or_zero_delta_birth": True,
        "parent_parameters_require_grad": False,
        "parent_parameter_identity_and_version_unchanged_after_training": True,
        "inactive_parent_view_restored_after_Cell_scope": True,
        "fresh_reload_required_for_positive_track": True,
    }
    assert "history_kl" not in json.dumps(protocol).lower()


def test_cow_clm_001_track_thresholds_are_frozen() -> None:
    tracks = _protocol()["tracks"]
    assert tracks["knowledge"]["facts"] == 8
    assert tracks["knowledge"]["minimum_choice_accuracy"] == 1.0
    assert tracks["capability"]["minimum_choice_accuracy"] == 0.8
    assert tracks["capability"]["heldout_exact_pairs_used_for_training"] is False


def test_knowledge_heldout_questions_are_never_training_questions() -> None:
    rows = knowledge_rows(8)
    train_questions = {row["question"] for row in rows["train"]}
    eval_questions = {row["question"] for row in rows["evaluation"]}
    assert train_questions.isdisjoint(eval_questions)
    assert len(rows["train"]) == 24
    assert len(rows["evaluation"]) == 16


def test_capability_exact_operand_pairs_are_disjoint_between_train_and_heldout() -> None:
    rows = capability_rows()
    train_ids = {row["id"] for row in rows["train"]}
    eval_ids = {row["id"] for row in rows["evaluation"]}
    assert train_ids.isdisjoint(eval_ids)
    assert len(rows["train"]) == 80
    assert len(rows["evaluation"]) == 20
    train_questions = {row["question"] for row in rows["train"]}
    eval_questions = {row["question"] for row in rows["evaluation"]}
    assert train_questions.isdisjoint(eval_questions)


def test_registered_implementation_blobs_match_frozen_protocol() -> None:
    protocol = _protocol()
    expected = protocol["implementation_git_blobs"]
    assert expected
    actual = {path: _git_blob(path) for path in expected}
    assert actual == expected


def test_nonclaims_preserve_prior_research_boundaries() -> None:
    text = "\n".join(_protocol()["nonclaims"])
    assert "existing Granite experts are Cells" in text
    assert "arbitrary sibling Cells compose" in text
    assert "expert-only COW is sufficient" in text
    assert "autonomous natural-language Cell routing" in text
    assert "does not rewrite" in text
