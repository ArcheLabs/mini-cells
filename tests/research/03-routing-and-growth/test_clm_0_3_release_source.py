from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from minicells.clm_growth import ProgressiveGrowthCLM
from minicells.clm_release_benchmark import (
    AGE_ZERO_MAX_LOGITS_DIFF,
    BRIDGE_TOKENS_PER_STEP,
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


def test_formal_worker_entry_exports_all_runtime_accounting_constants() -> None:
    """Import the actual formal worker wrapper, not only its source modules.

    ``py_compile`` cannot detect an undefined global that is only reached after
    GPU setup.  This regression test executes the entrypoint's module-loading
    path and verifies that the canonical tokens-per-step constant is available
    in the loaded worker before any formal run can start.
    """

    entry = ROOT / "scripts" / "run_clm_0_3_release_bridge_worker_entry.py"
    spec = importlib.util.spec_from_file_location("clm_release_bridge_entry_test", entry)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    assert loaded.module.BRIDGE_TOKENS_PER_STEP == BRIDGE_TOKENS_PER_STEP
    assert loaded.module.BRIDGE_TOKENS_PER_STEP == (
        loaded.module.BRIDGE_BATCH_SIZE * loaded.module.BRIDGE_SEQUENCE_LENGTH
    )
