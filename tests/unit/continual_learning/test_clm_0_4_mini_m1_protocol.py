from pathlib import Path

import pytest

from minicells.clm04mini.curriculum import build_curriculum, transaction_specs
from minicells.clm04mini.model import TinyCLMDecoder
from minicells.clm04mini.protocol import (
    ProtocolError,
    assert_seed_allowed,
    candidate_grid,
    formal_model_config,
    load_protocol,
)


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = (
    ROOT / "research" / "validations" / "clm-0.4-mini-language-validation" / "protocol.json"
)


def test_frozen_m1_model_maps_to_about_five_million_parameters():
    protocol = load_protocol(PROTOCOL)
    cfg = formal_model_config(protocol, routing_salt="test-salt")
    model = TinyCLMDecoder(cfg)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert cfg.vocab_size == 8192
    assert cfg.d_model == 256
    assert cfg.base_cells == 32
    assert cfg.cell_hidden == 32
    assert 4_500_000 <= parameters <= 5_500_000


def test_candidate_grids_and_seed_boundaries_are_frozen():
    protocol = load_protocol(PROTOCOL)
    assert len(candidate_grid(protocol, "direct")) == 9
    assert len(candidate_grid(protocol, "growth")) == 9
    assert candidate_grid(protocol, "direct")[0].steps == 16
    assert candidate_grid(protocol, "growth")[0].steps == 32
    assert_seed_allowed(protocol, mode="infrastructure-smoke", seed=90400)
    assert_seed_allowed(protocol, mode="calibration", seed=90401)
    for seed in (90411, 90412, 90413):
        assert_seed_allowed(protocol, mode="formal", seed=seed)
        with pytest.raises(ProtocolError):
            assert_seed_allowed(protocol, mode="infrastructure-smoke", seed=seed)
    with pytest.raises(ProtocolError):
        assert_seed_allowed(protocol, mode="calibration", seed=90411)


def test_curriculum_cardinality_revisit_and_supersede_semantics():
    manifest = build_curriculum()
    specs = transaction_specs(manifest)
    assert len(specs) == 192
    assert sum(spec.domain == "math" for spec in specs) == 96
    assert sum(spec.domain == "story" for spec in specs) == 96
    math_addresses = {spec.address_id for spec in specs if spec.domain == "math"}
    story_addresses = {spec.address_id for spec in specs if spec.domain == "story"}
    assert len(math_addresses) == 12
    assert len(story_addresses) == 24
    assert all(
        sum(spec.address_id == address for spec in specs) == 8
        for address in math_addresses
    )
    assert all(
        sum(spec.address_id == address for spec in specs) == 4
        for address in story_addresses
    )
    world_zero = [spec for spec in specs if spec.address_id == "story/world-00"]
    supersede = next(spec for spec in world_zero if spec.operation == "supersede")
    original = next(
        spec
        for spec in world_zero
        if spec.operation == "append" and spec.payload["attribute"] == "location"
    )
    assert supersede.knowledge_key == original.knowledge_key
    assert supersede.supersedes_key == original.knowledge_key
    assert supersede.payload["value"] != original.payload["value"]
