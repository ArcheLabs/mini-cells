from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import torch

from minicells.clm04mini.gates import evaluate_m1_gates
from minicells.clm04mini.model import MiniCLMConfig, TinyCLMDecoder
from minicells.clm04mini.examples import ScoredTokenExample
from minicells.clm04mini.performance import (
    _ORIGINAL_SCORED_LOGITS,
    _ORIGINAL_SPARSE_FORWARD,
    batched_scored_logits,
    evaluate_gate_summaries,
    grouped_sparse_forward,
    resolve_cuda_devices,
)
from minicells.clm04mini.protocol import load_protocol


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "research/validations/clm-0.4-mini-language-validation/protocol.json"


def _cfg() -> MiniCLMConfig:
    return MiniCLMConfig(
        vocab_size=64,
        max_seq_len=16,
        num_layers=4,
        d_model=32,
        n_heads=4,
        dense_ff_hidden=64,
        base_cells=8,
        cell_hidden=8,
        routing_salt="performance-test",
    )


def test_grouped_sparse_forward_matches_legacy_with_private_cell():
    torch.manual_seed(17)
    model = TinyCLMDecoder(_cfg()).eval()
    layer = model.sparse_layer(3)
    layer.spawn_private("address-1", d_model=32, hidden=8)
    x = torch.randn(7, 9, 32)
    addresses = ["address-1", "address-2", "address-1", "address-3", "address-4", "address-2", "address-5"]

    legacy = _ORIGINAL_SPARSE_FORWARD(layer, x, addresses)
    grouped = grouped_sparse_forward(layer, x, addresses)
    torch.testing.assert_close(grouped, legacy, rtol=1e-5, atol=1e-6)


def test_batched_structural_logits_match_legacy_within_registered_tolerance():
    torch.manual_seed(23)
    model = TinyCLMDecoder(_cfg()).eval()
    tokenizer = SimpleNamespace(pad_id=0)
    examples = []
    for index in range(70):
        length = 5 + index % 7
        tokens = tuple(1 + ((index + offset) % 50) for offset in range(length))
        mask = tuple([False] * (length - 3) + [True, True])
        examples.append(
            ScoredTokenExample(
                example_id=f"example-{index:03d}",
                address_id=f"address-{index % 11}",
                tokens=tokens,
                target_mask=mask,
            )
        )

    legacy = _ORIGINAL_SCORED_LOGITS(
        model, examples, tokenizer=tokenizer, device=torch.device("cpu")
    )
    batched = batched_scored_logits(
        model,
        examples,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        batch_size=16,
    )
    maximum = max(
        float((legacy[key] - batched[key]).abs().max()) for key in legacy
    )
    assert maximum <= 1e-5


def test_cached_summary_gate_algebra_matches_registered_gate_evaluator():
    protocol = load_protocol(PROTOCOL)
    always = {
        "positive_global_regression_damage": 1.0,
        "committed_new_gain": 2.0,
    }
    local_tx = {"committed_new_gain": 1.0}
    growth = {
        "positive_global_regression_damage": 0.2,
        "committed_new_gain": 1.8,
        "false_safe_rate": 0.0,
        "maximum_structural_escape_rate": 0.0,
        "effective_acceptance_rate": 0.8,
        "final_protected_retention_ratio": 0.97,
        "growth_rescue_rate": 0.75,
        "private_reuse_acceptance_rate": 0.7,
        "spawned_bundles_per_effective_commit": 0.4,
        "growth_parameter_overhead_ratio": 0.1,
        "mean_direct_dependency_coverage": 0.2,
    }
    records = [
        {"active_cells_by_layer": {"3": ["base:L3:C00", "growth:L3:a_x"], "4": ["base:L4:C00", "growth:L4:a_x"]}}
    ]

    class FakeHarness:
        def __init__(self, summary, rows=None):
            self._summary = summary
            self.records = rows or []

        def summary(self):
            return self._summary

    harnesses = {
        "local_always": FakeHarness(always),
        "local_tx": FakeHarness(local_tx),
        "local_tx_growth": FakeHarness(growth, records),
    }
    registered = evaluate_m1_gates(protocol=protocol, harnesses=harnesses)
    optimized = evaluate_gate_summaries(
        protocol=protocol,
        summaries={"local_always": always, "local_tx": local_tx, "local_tx_growth": growth},
        growth_harness=harnesses["local_tx_growth"],
    )
    assert optimized == registered


def test_cpu_device_resolution_does_not_require_cuda():
    assert resolve_cuda_devices(requested_device="cpu") == [torch.device("cpu")]
