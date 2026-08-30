import json
from pathlib import Path

from minicells.clm04mini.calibration import (
    build_calibration_plan,
    minimum_base_cell_activation,
    verify_committed_plan,
)
from minicells.clm04mini.protocol import load_protocol


ROOT = Path(__file__).resolve().parents[3]
VALIDATION = ROOT / "research" / "validations" / "clm-0.4-mini-language-validation"
PROTOCOL_PATH = VALIDATION / "protocol.json"
PLAN_PATH = VALIDATION / "calibration-plan.json"
ASSETS_PATH = VALIDATION / "calibration-assets.json"


def test_calibration_plan_is_frozen_before_development_seed():
    protocol = load_protocol(PROTOCOL_PATH)
    plan = verify_committed_plan(protocol, PLAN_PATH)
    assert plan == build_calibration_plan(protocol)
    assert plan["candidate_count"] == 81
    assert plan["plan_sha256"] == "d5a3d0d18337d5fb3e9996e79300faae4dd8f07cdb8ae6f183c4a8abf4dc2704"
    assert plan["candidates"][0]["estimated_candidate_steps"] == 48
    assert plan["candidates"][9]["estimated_candidate_steps"] == 64
    assert plan["candidates"][18]["estimated_candidate_steps"] == 80
    assert plan["candidates"][-1]["estimated_candidate_steps"] == 160
    costs = [item["estimated_candidate_steps"] for item in plan["candidates"]]
    assert costs == sorted(costs)


def test_base_activation_threshold_is_half_uniform_topk_expectation():
    protocol = load_protocol(PROTOCOL_PATH)
    assert minimum_base_cell_activation(protocol, base_sequences=3200) == 100


def test_calibration_asset_identity_was_frozen_before_90401():
    payload = json.loads(ASSETS_PATH.read_text(encoding="utf-8"))
    assert payload["dataset_revision"] == "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
    assert payload["base_tokens"] == 30000057
    assert payload["development_seed_observed_when_committed"] is False
    assert payload["formal_seeds_observed_when_committed"] is False
    assert payload["tokenizer_hash"] == "c0fd71032df7f7f50b5c46d29032191aaa64c0c5b00aee5a3443b7430a48406b"
    assert payload["base_corpus_manifest_hash"] == "f73c1e9073b367cc4f8bb787331eb21f2d05331953dd99367f9e6f1192ef3e50"
    assert payload["curriculum_manifest_hash"] == "1637cc834f7f9493d1187637ef2a4e38cabdb7ebb4cf48b70ba71a1da23296bb"
