from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from minicells.moe_multicoordinate import (
    apply_mutation_set_,
    save_mutation_set,
    validate_coordinate_targets,
)
from minicells.moe_subexpert import MoeSubexpertError

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research" / "validations" / "jam-knowledge-mutation-001" / "protocol.json"
DATASET_MANIFEST = ROOT / "research" / "datasets" / "jam-knowledge-v0.1" / "manifest.json"
DATASET_IDENTITY = (
    ROOT / "scripts" / "research" / "jam_knowledge_mutation_001" / "dataset_identity.py"
)
SEQUENCE = ROOT / "scripts" / "research" / "jam_knowledge_mutation_001" / "sequence.py"
AGGREGATE = ROOT / "scripts" / "research" / "jam_knowledge_mutation_001" / "aggregate.py"
FROZEN_PROTOCOL_SHA256 = "e934be45009d9025adf3b48ee2551f55a7099281196265b478e503d746559a54"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_is_release_bounded_and_source_locked() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert _sha256(PROTOCOL) == FROZEN_PROTOCOL_SHA256
    assert protocol["protocol_version"] == 1.2
    assert protocol["status"] == "PROTOCOL_FROZEN_GPU_PENDING"
    assert protocol["formal_seeds"] == [26090711, 26090712, 26090713]
    assert protocol["base"]["revision"] == "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"
    assert protocol["dataset"]["repository_commit"] == "5016cb36f8eb5ca715b6fd7796384ae5b607bd12"
    assert (
        protocol["dataset"]["manifest_sha256"]
        == "d2925ef66c3a7775e5485acea0be40bdd7887e22b89e7b809cb0c07f8102be15"
    )
    assert protocol["dataset"]["concept_count"] == 180
    assert protocol["dataset"]["generated_counts"] == {
        "train": 409,
        "validation": 180,
        "factual": 180,
        "relational": 66,
        "misconceptions": 49,
        "reasoning": 50,
    }
    assert protocol["mutation"]["capacity_ladder"] == [1, 2, 4]
    assert protocol["mutation"]["maximum_coordinate_count"] == 4
    assert protocol["mutation"]["require_unique_experts"] is True
    assert protocol["mutation"]["group_size"] == 32
    assert protocol["mutation"]["expected_intermediate_size"] == 512
    assert protocol["history"]["learner_visible_selection_prompts"] == 32
    assert protocol["history"]["withheld_evaluation_prompts"] == 32
    assert protocol["evaluation"]["evaluation_never_used_for_training_or_checkpoint_selection"] is True


def test_dataset_manifest_identity_is_protocol_pinned(tmp_path: Path) -> None:
    identity = _module(DATASET_IDENTITY, "jam_dataset_identity_test")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert identity.verify_manifest_identity(protocol, DATASET_MANIFEST) == protocol["dataset"][
        "manifest_sha256"
    ]

    tampered = tmp_path / "manifest.json"
    tampered.write_bytes(DATASET_MANIFEST.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="dataset manifest identity mismatch"):
        identity.verify_manifest_identity(protocol, tampered)


def test_multicoordinate_container_applies_and_rolls_back_exactly(tmp_path: Path) -> None:
    gate = torch.nn.Parameter(torch.zeros(2, 8, 3))
    down = torch.nn.Parameter(torch.zeros(2, 3, 4))
    parameters = {"gate": gate, "down": down}

    def coordinate(expert: int, group: int, value: float):
        target = {
            "layer_index": 23,
            "expert_index": expert,
            "group_index": group,
            "group_size": 2,
            "intermediate_size": 4,
            "gate_up_name": "gate",
            "down_name": "down",
            "gate_up_canonical_name": "gate",
            "down_canonical_name": "down",
        }
        return {
            "target": target,
            "deltas": {
                "gate": torch.full((2, 3), value),
                "up": torch.full((2, 3), value * 2),
                "down": torch.full((3, 2), value * 3),
            },
        }

    manifest = save_mutation_set(
        tmp_path,
        base_manifest_identity="base",
        source_model_id="model",
        source_revision="revision",
        coordinates=[coordinate(0, 0, 1.0), coordinate(1, 1, 2.0)],
        require_unique_experts=True,
    )
    assert manifest["coordinate_count"] == 2
    before_gate = gate.detach().clone()
    before_down = down.detach().clone()
    apply_mutation_set_(parameters, tmp_path)
    assert not torch.equal(gate, before_gate)
    assert not torch.equal(down, before_down)
    apply_mutation_set_(parameters, tmp_path, scale=-1.0)
    assert torch.equal(gate, before_gate)
    assert torch.equal(down, before_down)

    with pytest.raises(MoeSubexpertError):
        validate_coordinate_targets(
            [
                {"layer_index": 23, "expert_index": 0, "group_index": 0},
                {"layer_index": 23, "expert_index": 0, "group_index": 1},
            ],
            require_unique_experts=True,
        )


def test_answer_only_encoding_masks_prompt_and_padding() -> None:
    sequence = _module(SEQUENCE, "jam_sequence_test")

    class Tokenizer:
        pad_token_id = 0
        bos_token_id = 1
        eos_token_id = 2

        def __call__(self, text, *, add_special_tokens=True):
            ids = [1] if add_special_tokens else []
            ids.extend(10 + index for index, _word in enumerate(str(text).split()))
            return {"input_ids": ids}

    batch = sequence.encode_rows(
        Tokenizer(),
        [{"id": "x", "question": "What is JAM?", "answer": "A protocol"}],
        prompt_template="Question: {question}\nAnswer:",
        max_length=32,
        device="cpu",
    )
    labels = batch["labels"][0]
    supervised = labels[labels.ne(-100)].tolist()
    assert supervised[-1] == 2
    assert len(supervised) == 3
    assert all(value == -100 for value in labels[:5].tolist())


def _write_seed_summary(
    root: Path,
    *,
    seed: int,
    status: str,
    capacity: int | None,
    protocol_sha256: str = FROZEN_PROTOCOL_SHA256,
) -> None:
    seed_root = root / f"seed-{seed}"
    seed_root.mkdir(parents=True)
    (seed_root / "seed_summary.json").write_text(
        json.dumps(
            {
                "experiment": "JAM_KNOWLEDGE_MUTATION_001",
                "protocol_sha256": protocol_sha256,
                "seed": seed,
                "status": status,
                "selected_capacity": capacity,
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_requires_two_of_three_formal_seeds(tmp_path: Path) -> None:
    aggregate_module = _module(AGGREGATE, "jam_aggregate_test")
    for seed, status, capacity in (
        (26090711, "PASS", 2),
        (26090712, "FAIL", None),
        (26090713, "PASS", 4),
    ):
        _write_seed_summary(tmp_path, seed=seed, status=status, capacity=capacity)

    decision = aggregate_module.aggregate(tmp_path)
    assert decision["protocol_sha256"] == FROZEN_PROTOCOL_SHA256
    assert decision["status"] == "JAM_KNOWLEDGE_MUTATION_SUPPORTED"
    assert decision["scientific_decision"] is True
    assert decision["passed_seeds"] == [26090711, 26090713]
    assert decision["minimum_passing_capacity_observed"] == 2


def test_aggregate_rejects_mixed_protocol_seed(tmp_path: Path) -> None:
    aggregate_module = _module(AGGREGATE, "jam_aggregate_mixed_protocol_test")
    _write_seed_summary(tmp_path, seed=26090711, status="PASS", capacity=2)
    _write_seed_summary(tmp_path, seed=26090712, status="PASS", capacity=2)
    _write_seed_summary(
        tmp_path,
        seed=26090713,
        status="PASS",
        capacity=2,
        protocol_sha256="0" * 64,
    )

    decision = aggregate_module.aggregate(tmp_path)
    assert decision["status"] == "JAM_KNOWLEDGE_MUTATION_INCOMPLETE"
    assert decision["scientific_decision"] is False
    assert decision["malformed_seeds"] == [26090713]
