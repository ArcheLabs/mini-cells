from __future__ import annotations

import json

import pytest
import torch

from minicells.clm_conditionality_002 import (
    Conditionality002Evidence,
    aligned_route_disagreement,
    evaluate_conditionality_evidence,
    make_conditionality_002_decision,
)
from minicells.clm_release import BUNDLE_FORMAT, CLM, build_release_model, save_release_bundle


def test_aligned_route_disagreement_preserves_position_and_time() -> None:
    # At both aligned positions, four samples split 2/2 between two experts.
    routes = torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0]])
    mask = torch.nn.functional.one_hot(routes, num_classes=4).float()
    observed = aligned_route_disagreement([mask, mask.clone()])
    assert observed == pytest.approx(2.0 / 3.0)


def test_aligned_route_disagreement_is_zero_for_identical_routes() -> None:
    routes = torch.zeros(4, 3, dtype=torch.long)
    mask = torch.nn.functional.one_hot(routes, num_classes=4).float()
    assert aligned_route_disagreement([mask]) == 0.0


def test_conditionality_002_requires_quality_causality_and_aligned_variation() -> None:
    row = evaluate_conditionality_evidence(
        replicate=0,
        dense_ppl=18.2,
        dense_nll=2.90,
        dynamic={"ppl": 18.0, "nll": 2.88, "usage_entropy": 0.99},
        static={"ppl": 18.5, "nll": 2.92},
        shuffled={"ppl": 18.5, "nll": 2.92},
        aligned_disagreement=0.30,
    )
    assert row.passed
    decision = make_conditionality_002_decision([row, row, Conditionality002Evidence(
        replicate=2,
        quality_ratio_to_dense_continued=1.0,
        aligned_route_disagreement=0.0,
        static_advantage=0.01,
        shuffled_advantage=0.01,
        usage_entropy=0.99,
        passed=False,
    )])
    assert decision["status"] == "PASS"
    assert decision["diagnosis"] == "CLM_LOCAL_CONDITIONALITY_SIGNAL"


def test_clm_bundle_roundtrip(tmp_path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    torch.manual_seed(7)
    model = build_release_model()
    model.eval()
    tokenizer = tokenizers.Tokenizer(WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_path = tmp_path / "source-tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    bundle = save_release_bundle(
        model,
        tokenizer_path,
        tmp_path / "bundle",
        provenance={"test": True},
        metrics={"validation_ppl": 1.0},
    )
    config = json.loads((bundle / "config.json").read_text())
    assert config["format"] == BUNDLE_FORMAT
    loaded = CLM.from_pretrained(bundle)
    inputs = torch.randint(0, 2048, (1, 8))
    with torch.no_grad():
        expected = model(inputs).logits
        actual = loaded.model(inputs).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_generation_rejects_empty_prompt(tmp_path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    model = build_release_model()
    tokenizer = tokenizers.Tokenizer(WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    clm = CLM(model, tokenizer, torch.device("cpu"))
    with pytest.raises(ValueError):
        clm.generate("", max_new_tokens=1)
