from __future__ import annotations

from pathlib import Path

from minicells.clm04mini.model import MiniCLMConfig, TinyCLMDecoder
from minicells.clm04mini.preview import PREVIEW_MIXTURE, preview_model_config
from minicells.clm04mini.tokenizer import (
    DigitAwareTokenizerBundle,
    join_digits,
    separate_digits,
    train_digit_aware_tokenizer,
)


def test_historical_config_keeps_sparse_layers_without_shared_ffn():
    cfg = MiniCLMConfig()
    model = TinyCLMDecoder(cfg)
    assert model.blocks[2].shared_ff is None
    assert model.blocks[3].shared_ff is None
    assert model.shared_cell_ffn_parameters() == 0


def test_preview_adds_frozen_shared_capacity_without_changing_cell_addressability():
    cfg = preview_model_config()
    model = TinyCLMDecoder(cfg)
    assert cfg.shared_cell_ff_hidden == 256
    assert model.blocks[2].shared_ff is not None
    assert model.blocks[3].shared_ff is not None
    assert model.shared_cell_ffn_parameters() == 263_168
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_273_088

    address = "preview/test-address"
    cell_modules = model.modules_for_cell_ids(model.base_cell_ids(address))
    cell_parameter_ids = {
        id(parameter) for module in cell_modules for parameter in module.parameters()
    }
    shared_parameter_ids = {
        id(parameter)
        for layer in (2, 3)
        for parameter in model.blocks[layer].shared_ff.parameters()  # type: ignore[union-attr]
    }
    assert cell_parameter_ids
    assert shared_parameter_ids
    assert cell_parameter_ids.isdisjoint(shared_parameter_ids)


def test_preview_base_mix_reallocates_story_budget_to_math():
    assert PREVIEW_MIXTURE == {
        "language_carrier": 0.60,
        "controlled_base_math": 0.30,
        "controlled_base_story": 0.10,
    }
    assert sum(PREVIEW_MIXTURE.values()) == 1.0


def test_digit_aware_tokenizer_exposes_multi_digit_structure(tmp_path: Path):
    texts = [
        "Question: What is 12 plus 19? Answer: 31.",
        "Context: Ada lives in Luma. Question: Where does Ada live? Answer: Luma.",
    ] * 16
    manifest = train_digit_aware_tokenizer(
        texts,
        out_dir=tmp_path,
        vocab_size=512,
        min_frequency=1,
    )
    bundle = DigitAwareTokenizerBundle.load(tmp_path / "tokenizer.json")
    assert manifest["digit_policy"] == "split-contiguous-decimal-digits-before-bpe"
    assert bundle.encode("31", add_special_tokens=False) == bundle.encode(
        "3 1", add_special_tokens=False
    )
    assert bundle.decode(bundle.encode("31")) == "31"
    assert separate_digits("x=105, y=7") == "x=1 0 5, y=7"
    assert join_digits("x=1 0 5, y=7") == "x=105, y=7"
