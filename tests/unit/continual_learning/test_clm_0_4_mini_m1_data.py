import json
from pathlib import Path

import pytest

from minicells.clm04mini.curriculum import TextExample, build_curriculum
from minicells.clm04mini.examples import tokenize_text_example
from minicells.clm04mini.lock import build_protocol_lock, validate_protocol_lock
from minicells.clm04mini.protocol import candidate_grid, load_protocol
from minicells.clm04mini.tokenizer import TokenizerBundle, train_tokenizer


ROOT = Path(__file__).resolve().parents[3]
VALIDATION = ROOT / "research" / "validations" / "clm-0.4-mini-language-validation"
PROTOCOL = VALIDATION / "protocol.json"
LOCK_TEMPLATE = VALIDATION / "protocol-lock.template.json"


def test_answer_only_target_mask_and_tokenizer_manifest(tmp_path):
    texts = [
        "Question: What is two plus three? Answer: five.",
        "Mira lives in Luma and works as a baker.",
    ] * 16
    manifest = train_tokenizer(texts, out_dir=tmp_path, vocab_size=512, min_frequency=1)
    tokenizer = TokenizerBundle.load(tmp_path / "tokenizer.json")
    assert manifest["tokenizer_sha256"]
    assert manifest["manifest_sha256"]
    example = tokenize_text_example(
        TextExample("x", "math/test", "Question: What is 2 plus 3? Answer:", " 5."),
        tokenizer,
        max_seq_len=48,
    )
    assert len(example.target_mask) == len(example.tokens) - 1
    assert any(example.target_mask)
    assert not all(example.target_mask)
    first_scored = example.target_mask.index(True)
    assert first_scored > 0


def test_protocol_lock_builder_accepts_only_registered_candidate_configs(tmp_path):
    protocol = load_protocol(PROTOCOL)
    template = json.loads(LOCK_TEMPLATE.read_text(encoding="utf-8"))
    direct = candidate_grid(protocol, "direct")[0]
    growth = candidate_grid(protocol, "growth")[0]
    lock = build_protocol_lock(
        protocol=protocol,
        template=template,
        protocol_path=PROTOCOL,
        direct_optimizer=direct,
        growth_optimizer=growth,
        tokenizer_manifest={"tokenizer_sha256": "a" * 64, "manifest_sha256": "b" * 64},
        base_corpus_manifest={"manifest_sha256": "c" * 64, "generator_version": "test"},
        curriculum_manifest={
            "manifest_sha256": build_curriculum()["manifest_sha256"],
            "generator_version": build_curriculum()["generator_version"],
        },
        dataset_revision="deadbeef",
        routing_salt="test-salt",
        minimum_base_cell_activation=1,
        code_commit="d" * 40,
        code_tree="e" * 40,
        environment={
            "python": "test",
            "torch": "test",
            "cuda": None,
            "gpu": None,
            "tokenizers": "test",
        },
    )
    assert lock["lock_status"] == "LOCKED"
    validate_protocol_lock(lock, protocol=protocol)
    assert lock["formal_results_observed_when_locked"] is False
