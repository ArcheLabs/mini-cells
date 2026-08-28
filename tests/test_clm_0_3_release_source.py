from __future__ import annotations

from pathlib import Path

import torch

from minicells.clm_growth import ProgressiveGrowthCLM
from minicells.clm_release_benchmark import (
    AGE_ZERO_MAX_LOGITS_DIFF,
    SOURCE_006_CHECKPOINT,
    SOURCE_006_CHECKPOINT_SHA256,
    build_bridge_model,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_experiment_006_checkpoint_hash_is_release_locked() -> None:
    path = ROOT / SOURCE_006_CHECKPOINT
    assert path.is_file()
    assert sha256_file(path) == SOURCE_006_CHECKPOINT_SHA256


def test_current_clm_fixed4_conversion_preserves_frozen_source_logits_on_cpu() -> None:
    path = ROOT / SOURCE_006_CHECKPOINT
    dense = build_bridge_model(
        "textnca_continuation",
        path,
        vocab_size=2048,
        device="cpu",
    ).eval()
    clm = build_bridge_model(
        "clm_fixed4",
        path,
        vocab_size=2048,
        device="cpu",
    ).eval()
    assert isinstance(clm, ProgressiveGrowthCLM)
    generator = torch.Generator(device="cpu").manual_seed(57011)
    inputs = torch.randint(0, 2048, (2, 64), generator=generator)
    with torch.no_grad():
        dense_logits = dense(inputs).logits
        clm_logits = clm(inputs, execution_backend="masked_dense").logits
    max_diff = float((dense_logits - clm_logits).abs().max().item())
    assert max_diff <= AGE_ZERO_MAX_LOGITS_DIFF
