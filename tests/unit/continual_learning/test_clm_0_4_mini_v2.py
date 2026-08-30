from __future__ import annotations

import json
from pathlib import Path

import pytest

from minicells.clm04mini.model import MiniCLMConfig, TinyCLMDecoder
from minicells.clm04mini.protocol import (
    ProtocolError,
    assert_seed_allowed,
    load_protocol,
)
from minicells.clm04mini.v2 import (
    DenseDecoder,
    dense_baseline_config,
    route_balanced_eval_addresses,
    v2_math_eval_examples,
    v2_math_stream,
    v2_story_eval_examples,
    v2_story_stream,
)


ROOT = Path(__file__).resolve().parents[3]
VALIDATION = (
    ROOT
    / "research"
    / "validations"
    / "clm-0.4-mini-m1-v2-language-validation"
)
PROTOCOL = VALIDATION / "protocol.json"


def test_v2_seed_boundary_and_asset_lock():
    protocol = load_protocol(PROTOCOL)
    assert_seed_allowed(protocol, mode="calibration", seed=90402)
    with pytest.raises(ProtocolError):
        assert_seed_allowed(protocol, mode="calibration", seed=90401)
    with pytest.raises(ProtocolError):
        assert_seed_allowed(protocol, mode="calibration", seed=90411)
    lock = json.loads((VALIDATION / "asset-lock.json").read_text())
    assert lock["lock_status"] == "LOCKED"
    assert lock["development_seed_observed_when_locked"] is False
    assert lock["formal_seeds_observed_when_locked"] is False
    assert lock["identity"] == {
        "dataset_revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        "routing_salt": "clm-0.4-mini-m1-v2",
        "base_tokens": 30000054,
        "tokenizer_hash": "c0fd71032df7f7f50b5c46d29032191aaa64c0c5b00aee5a3443b7430a48406b",
        "tokenizer_manifest_hash": "703b86361d32726ced0c8ed9db4a437b7a65aa99cb90e7415bc36c46e69171ae",
        "base_corpus_manifest_hash": "e34fa3386a06b0c02faed80908c6368a44044aadae1d215f4af780a5d0ea1939",
        "curriculum_manifest_hash": "1637cc834f7f9493d1187637ef2a4e38cabdb7ebb4cf48b70ba71a1da23296bb",
        "base_generator_version": "clm-0.4-mini-m1-v2-base-corpus-v1",
    }


def test_v2_controlled_streams_are_task_aligned():
    math_stream = v2_math_stream()
    story_stream = v2_story_stream()
    math = next(math_stream)
    story = next(story_stream)
    assert "Question:" in math and "Answer:" in math
    assert "Context:" in story
    assert "Question:" in story
    assert "Answer:" in story


def _cfg() -> MiniCLMConfig:
    return MiniCLMConfig(
        vocab_size=8192,
        max_seq_len=256,
        num_layers=4,
        d_model=256,
        n_heads=8,
        dense_ff_hidden=768,
        base_cells=32,
        cell_hidden=32,
        routing_salt="clm-0.4-mini-m1-v2",
    )


def test_route_balanced_eval_is_unique_and_covers_every_cell():
    cfg = _cfg()
    model = TinyCLMDecoder(cfg)
    for domain in ("math", "story"):
        addresses = route_balanced_eval_addresses(cfg, domain=domain, count=64)
        assert len(addresses) == 64
        assert len(set(addresses)) == 64
        covered = {3: set(), 4: set()}
        for address in addresses:
            for layer, values in model.base_routes(address).items():
                covered[int(layer)].update(values)
        assert covered[3] == set(range(32))
        assert covered[4] == set(range(32))


def test_eval_examples_use_held_out_balanced_addresses():
    cfg = _cfg()
    math = v2_math_eval_examples(cfg, 64)
    story = v2_story_eval_examples(cfg, 64)
    assert len({item.address_id for item in math}) == 64
    assert len({item.address_id for item in story}) == 64
    assert all(item.address_id.startswith("v2/eval/math/") for item in math)
    assert all(item.address_id.startswith("v2/eval/story/") for item in story)


def test_dense_baseline_parameter_counts_are_executable_contracts():
    protocol = load_protocol(PROTOCOL)
    equal_parameter = DenseDecoder(
        dense_baseline_config(
            protocol,
            kind="equal_parameter",
            routing_salt="test/equal-parameter",
        )
    )
    equal_compute = DenseDecoder(
        dense_baseline_config(
            protocol,
            kind="equal_active_compute",
            routing_salt="test/equal-compute",
        )
    )
    assert sum(p.numel() for p in equal_parameter.parameters()) == 5_010_464
    assert sum(p.numel() for p in equal_compute.parameters()) == 4_009_088
