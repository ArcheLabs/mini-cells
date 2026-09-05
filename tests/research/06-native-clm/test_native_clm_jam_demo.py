from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

from minicells.native_clm_v0 import NativeCLM, NativeCLMConfig

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "research" / "run_native_clm_jam_demo.py"
SPEC = importlib.util.spec_from_file_location("native_clm_jam_demo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def tiny_model() -> NativeCLM:
    return NativeCLM(
        NativeCLMConfig(
            vocab_size=256,
            max_seq_len=64,
            d_model=24,
            n_layers=2,
            n_heads=4,
            d_ff=48,
            initial_cells=2,
            active_cells=1,
            cellular_layer_index=0,
            certificate_max_rank=4,
        )
    )


def test_qa_encoding_masks_prompt_and_keeps_answer() -> None:
    row = {"id": "test", "question": "What is JAM?", "answer": "A protocol."}
    sample = MODULE._encode_qa(row, max_seq_len=64)
    labels = sample["labels"]
    assert sample["input_ids"].shape == labels.shape
    assert bool((labels == -100).any())
    assert bool((labels != -100).any())


def test_qa_encoding_respects_context_limit() -> None:
    row = {
        "id": "long",
        "question": "Q" * 200,
        "answer": "A" * 200,
    }
    sample = MODULE._encode_qa(row, max_seq_len=64)
    assert sample["input_ids"].numel() <= 64
    assert bool((sample["labels"] != -100).any())


def test_qa_evaluator_runs_on_native_clm() -> None:
    model = tiny_model()
    rows = [
        {"id": "a", "question": "What is JAM?", "answer": "A protocol."},
        {"id": "b", "question": "What is Refine?", "answer": "A computation phase."},
    ]
    metrics = MODULE.evaluate_qa(
        model,
        rows,
        device=torch.device("cpu"),
        precision="fp32",
        batch_size=2,
    )
    assert metrics["rows"] == 2
    assert metrics["answer_tokens"] > 0
    assert math.isfinite(float(metrics["answer_nll"]))
    assert 0.0 <= float(metrics["answer_token_accuracy"]) <= 1.0
