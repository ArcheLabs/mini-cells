from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "scripts" / "research" / "jam_knowledge_mutation_001_failure_diagnostic"
PLAN = (
    ROOT
    / "research"
    / "validations"
    / "jam-knowledge-mutation-001-failure-diagnostic"
    / "diagnostic_plan.json"
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_is_post_hoc_and_does_not_change_formal_decision() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["status"] == "POST_HOC_DIAGNOSTIC_FROZEN_GPU_PENDING"
    assert plan["upstream"]["formal_decision"] == "JAM_KNOWLEDGE_MUTATION_NOT_SUPPORTED"
    assert plan["upstream"]["formal_seeds"] == [26090711, 26090712, 26090713]
    assert plan["upstream"]["capacities"] == [1, 2, 4]
    assert plan["upstream"]["formal_misconception_reference_nll_gain_threshold"] == 0.25
    rules = plan["diagnostic_rules"]
    assert rules["retraining"] is False
    assert rules["formal_gate_changes"] is False
    assert rules["formal_decision_changes"] is False
    assert rules["new_scientific_pass_fail_gate"] is False


def test_source_validator_accepts_merged_frozen_artifacts() -> None:
    validator = _module(DIAG / "validate_sources.py", "jam001_diag_validate_test")
    result = validator.validate_sources(require_git_identity=False)
    assert result["status"] == "JAM001_FAILURE_DIAGNOSTIC_SOURCES_VALID"
    assert result["upstream_formal_decision"] == "JAM_KNOWLEDGE_MUTATION_NOT_SUPPORTED"
    assert result["formal_seeds"] == [26090711, 26090712, 26090713]
    assert result["inspected_mutations"] == 9


def test_segmented_encoding_exactly_partitions_supervised_answer_tokens() -> None:
    segmentation = _module(DIAG / "segmentation.py", "jam001_diag_segmentation_test")

    class CharTokenizer:
        is_fast = True
        eos_token_id = 2
        bos_token_id = 1
        pad_token_id = 0

        def __call__(
            self,
            text,
            *,
            add_special_tokens=False,
            return_offsets_mapping=False,
        ):
            del add_special_tokens
            text = str(text)
            output = {"input_ids": [10 + ord(char) for char in text]}
            if return_offsets_mapping:
                output["offset_mapping"] = [
                    (index, index + 1) for index in range(len(text))
                ]
            return output

    batch = segmentation.encode_segmented_rows(
        CharTokenizer(),
        [
            {
                "id": "x",
                "question": "Is X correct?",
                "answer": "The claim is incorrect. JAM fact",
            }
        ],
        prompt_template="Question: {question}\nAnswer:",
        max_length=256,
        device="cpu",
        prefix="The claim is incorrect.",
        separator=" ",
    )
    supervised = batch["labels"][:, 1:].ne(-100)
    shifted_segments = batch["segments"][:, 1:]
    prefix = supervised & shifted_segments.eq(segmentation.SEGMENT_PREFIX)
    content = supervised & shifted_segments.eq(segmentation.SEGMENT_CONTENT)
    eos = supervised & shifted_segments.eq(segmentation.SEGMENT_EOS)
    assert int(prefix.sum()) > 0
    assert int(content.sum()) > 0
    assert int(eos.sum()) == 1
    assert torch.equal(prefix | content | eos, supervised)
    assert not bool((prefix & content).any())
    assert not bool((prefix & eos).any())
    assert not bool((content & eos).any())


def test_gain_decomposition_reconstructs_full_gain() -> None:
    segmentation = _module(DIAG / "segmentation.py", "jam001_diag_decompose_test")
    base = {
        "prefix": {"mean_reference_nll": 2.0, "supervised_tokens": 2.0},
        "canonical_content": {"mean_reference_nll": 3.0, "supervised_tokens": 6.0},
        "eos": {"mean_reference_nll": 1.0, "supervised_tokens": 1.0},
        "full": {
            "mean_reference_nll": (2 * 2.0 + 6 * 3.0 + 1.0) / 9.0,
            "supervised_tokens": 9.0,
        },
    }
    mutated = {
        "prefix": {"mean_reference_nll": 1.9, "supervised_tokens": 2.0},
        "canonical_content": {"mean_reference_nll": 2.6, "supervised_tokens": 6.0},
        "eos": {"mean_reference_nll": 0.8, "supervised_tokens": 1.0},
        "full": {
            "mean_reference_nll": (2 * 1.9 + 6 * 2.6 + 0.8) / 9.0,
            "supervised_tokens": 9.0,
        },
    }
    result = segmentation.gain_decomposition(
        base,
        mutated,
        original_threshold=0.25,
    )
    assert result["decomposition_absolute_error"] < 1e-12
    assert result["canonical_content_reference_nll_gain"] > result[
        "prefix_reference_nll_gain"
    ]


def test_capacity_four_classification_uses_original_threshold_only_as_reference() -> None:
    segmentation = _module(DIAG / "segmentation.py", "jam001_diag_classify_test")
    formulation_cases = [
        {
            "decomposition": {
                "full_reference_nll_gain": full,
                "content_plus_eos_reference_nll_gain": content,
                "canonical_content_reference_nll_gain": content + 0.01,
            }
        }
        for full, content in ((0.20, 0.28), (0.21, 0.27), (0.19, 0.29))
    ]
    assert (
        segmentation.classify_capacity_four(
            formulation_cases,
            original_threshold=0.25,
        )
        == "FORMULATION_PREFIX_DILUTION_SUFFICIENT_TO_EXPLAIN_CAPACITY4_FORMAL_GATE_FAILURE"
    )

    content_cases = [
        {
            "decomposition": {
                "full_reference_nll_gain": 0.20,
                "content_plus_eos_reference_nll_gain": value,
                "canonical_content_reference_nll_gain": value - 0.01,
            }
        }
        for value in (0.22, 0.23, 0.24)
    ]
    assert (
        segmentation.classify_capacity_four(
            content_cases,
            original_threshold=0.25,
        )
        == "CANONICAL_CONTENT_GAIN_ALSO_BELOW_ORIGINAL_THRESHOLD_AT_CAPACITY4"
    )
