from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from minicells.native_clm_m3l2 import (
    M3L2AddressConfig,
    MomentAccumulator,
    OnlineAddressNativeCLM,
    merge_sketch_and_moments,
    observe_online_queries,
)
from minicells.native_clm_m3l_gate import derive_sketch_gate
from minicells.native_clm_v0 import NativeCLMConfig


def _model() -> OnlineAddressNativeCLM:
    return OnlineAddressNativeCLM(
        NativeCLMConfig(
            vocab_size=32,
            max_seq_len=8,
            d_model=16,
            n_layers=2,
            n_heads=4,
            d_ff=32,
            initial_cells=2,
            active_cells=1,
            cellular_layer_index=0,
            certificate_max_rank=8,
        ),
        lineage_root_count=2,
    )


def _acc(center: float, n: int = 128) -> MomentAccumulator:
    torch.manual_seed(int((center + 3) * 100))
    values = torch.randn(n, 16) * 0.15
    values[:, 0] += center
    accumulator = MomentAccumulator(16)
    accumulator.update(values)
    return accumulator


def test_rank32_state_is_bounded_and_mergeable() -> None:
    config = M3L2AddressConfig(rank=32)
    first = _acc(-1.0).to_sketch(
        rank=config.rank,
        diagonal_regularization=config.diagonal_regularization,
    )
    assert first.rank == 16
    current = _acc(-0.5)
    merged = merge_sketch_and_moments(first, current, config=config)
    assert merged is not None
    assert merged.rank <= 32
    assert merged.storage_bytes <= config.maximum_persistent_bytes_per_cell


def test_online_query_accumulator_has_registered_per_batch_cap() -> None:
    accumulator = MomentAccumulator(16)
    values = torch.randn(1024, 16)
    accumulator.update(values, max_samples=256)
    assert accumulator.count == 256


def test_affine_child_gate_preserves_birth_function() -> None:
    torch.manual_seed(8)
    model = _model().eval()
    old = _acc(-1.5).to_sketch(rank=16, diagonal_regularization=1e-4)
    new = _acc(1.5).to_sketch(rank=16, diagonal_regularization=1e-4)
    tokens = torch.randint(0, 32, (2, 8))
    before = model(tokens, return_info=True)
    parent = int(before["cell_info"]["root_idx"][0, 0, 0])
    child = model.spawn_cell(
        parent_id=parent,
        route_key=F.normalize(new.mean, dim=0),
        inherit_scale=1.0,
    )
    model.historical_sketches[parent] = old
    model.historical_sketches[child] = new
    model.affine_gates[parent] = derive_sketch_gate(
        old,
        new,
        diagonal_regularization=1e-4,
        target_old_fpr=0.1,
    )
    after = model(tokens, return_info=True)
    assert torch.equal(before["cell_info"]["root_idx"], after["cell_info"]["root_idx"])
    assert torch.equal(before["cell_info"]["root_probs"], after["cell_info"]["root_probs"])
    assert torch.allclose(before["logits"], after["logits"], atol=1e-6, rtol=0.0)


def test_split_parent_no_longer_accumulates_historical_updates() -> None:
    torch.manual_seed(11)
    model = _model().eval()
    tokens = torch.randint(0, 32, (2, 8))
    info = model(tokens, return_info=True)["cell_info"]
    parent = int(info["top_idx"][0, 0, 0])
    child = model.spawn_cell(
        parent_id=parent,
        route_key=F.normalize(torch.randn(16), dim=0),
        inherit_scale=1.0,
    )
    assert not model.is_lineage_leaf(parent)
    assert model.is_lineage_leaf(child)

    fake_info = dict(info)
    fake_info["top_idx"] = torch.full_like(info["top_idx"], parent)
    observe_online_queries(model, fake_info)
    assert parent not in model.current_moments


def test_address_checkpoint_roundtrip_preserves_registered_config(tmp_path) -> None:
    model = _model().eval()
    old = _acc(-1.0).to_sketch(rank=16, diagonal_regularization=1e-4)
    new = _acc(1.0).to_sketch(rank=16, diagonal_regularization=1e-4)
    child = model.spawn_cell(
        parent_id=0,
        route_key=F.normalize(new.mean, dim=0),
        inherit_scale=1.0,
    )
    model.historical_sketches[0] = old
    model.historical_sketches[child] = new
    model.affine_gates[0] = derive_sketch_gate(
        old,
        new,
        diagonal_regularization=1e-4,
        target_old_fpr=0.1,
    )
    model.bootstrap_complete = True
    model.bootstrap_parameter_hash_before = "same"
    model.bootstrap_parameter_hash_after = "same"
    model.bootstrap_access_released = True
    path = tmp_path / "m3l2.pt"
    torch.save(model.checkpoint_payload(), path)
    restored, _ = OnlineAddressNativeCLM.load_checkpoint(path)
    tokens = torch.randint(0, 32, (2, 8))
    assert restored.address_state_metrics()["gate_count"] == 1
    assert restored.address_state_metrics()["sketch_count"] == 2
    assert restored.address_config == M3L2AddressConfig()
    assert restored.bootstrap_access_released is True
    assert torch.allclose(model(tokens)["logits"], restored(tokens)["logits"], atol=1e-7, rtol=0.0)


def test_address_sidecars_follow_model_apply() -> None:
    model = _model().eval()
    sketch = _acc(-1.0).to_sketch(rank=16, diagonal_regularization=1e-4)
    model.historical_sketches[0] = sketch
    model.double()
    assert model.historical_sketches[0].mean.dtype == torch.float64
    assert model.historical_sketches[0].basis.dtype == torch.float64


def test_affine_gate_separates_shifted_query_state() -> None:
    old = _acc(-2.0).to_sketch(rank=16, diagonal_regularization=1e-4)
    new = _acc(2.0).to_sketch(rank=16, diagonal_regularization=1e-4)
    gate = derive_sketch_gate(old, new, diagonal_regularization=1e-4, target_old_fpr=0.1)
    weight = gate["weight"]
    assert isinstance(weight, torch.Tensor)
    old_score = F.normalize(old.mean, dim=0).dot(weight) + float(gate["bias"])
    new_score = F.normalize(new.mean, dim=0).dot(weight) + float(gate["bias"])
    assert new_score > old_score


def test_protocol_config_is_exactly_registered_rank32_budget() -> None:
    config = M3L2AddressConfig()
    config.validate()
    assert config.rank == 32
    assert config.maximum_persistent_bytes_per_cell == 52360
    assert config.bootstrap_batches == 160
    assert config.max_queries_per_cell_per_batch == 256
    protocol = json.loads(
        Path(
            "research/validations/native-clm-v0-m3l2-online-address-state/protocol.json"
        ).read_text()
    )
    assert protocol["address_state"]["maximum_affine_gate_bytes_per_edge"] == 1552
    assert protocol["address_state"]["maximum_total_address_bytes_per_node"] == 53912


def _load_runner_module():
    path = Path("scripts/research/run_native_clm_v0_m3l2.py")
    spec = importlib.util.spec_from_file_location("m3l2_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(loss: float, cells: int) -> dict:
    return {
        "loss": loss,
        "active_fraction_vs_dense": 2 / cells,
        "cell_usage_share": [1 / cells] * cells,
    }


def _synthetic_summary(*, treatment: bool) -> dict:
    cells = 9
    initial = {
        "A": _metrics(1.0, 8),
        "B": _metrics(2.0, 8),
        "C": _metrics(3.0, 8),
        "D": _metrics(2.5, 8),
    }
    a_final = 1.10 if treatment else 1.40
    summary = {
        "arm": "online_address_state" if treatment else "lineage_cosine_control",
        "seed": 74101,
        "parent_checkpoint_sha256": "same-m1",
        "growth_config": {"frozen": True},
        "learner_replay_bytes": 0,
        "shared_and_original_router_frozen": True,
        "spawned_cells": 1,
        "final_cell_count": cells,
        "child_post_birth_route_hits": {"8": 4096},
        "growth_events": [
            {
                "birth_logits_max_abs_drift": 0.0,
                "birth_logits_mse": 0.0,
                "birth_root_topk_match": 1.0,
                "birth_root_prob_max_abs_drift": 0.0,
            }
        ],
        "lineage_chain_valid": True,
        "root_route_probes": {
            stage: {domain: f"stable-{domain}" for domain in "ABCD"}
            for stage in ("initial", "after_B", "after_C", "after_D")
        },
        "evaluation_matrix": {
            "initial": initial,
            "after_B": {
                "A": _metrics(1.02, cells),
                "B": _metrics(1.20, cells),
                "C": _metrics(3.0, cells),
                "D": _metrics(2.5, cells),
            },
            "after_C": {
                "A": _metrics(1.05, cells),
                "B": _metrics(1.22, cells),
                "C": _metrics(1.50, cells),
                "D": _metrics(2.5, cells),
            },
            "after_D": {
                "A": _metrics(a_final, cells),
                "B": _metrics(1.24, cells),
                "C": _metrics(1.53, cells),
                "D": _metrics(1.75, cells),
            },
        },
    }
    if treatment:
        summary.update(
            {
                "address_state": {
                    "maximum_rank": 32,
                    "maximum_bytes_per_cell": 52360,
                    "gate_count": 1,
                },
                "address_state_checkpoint_roundtrip": True,
                "bootstrap": {
                    "complete": True,
                    "parameter_sha256_before": "same-params",
                    "parameter_sha256_after": "same-params",
                    "A_access_after_continual_start": False,
                    "access_released_before_continual_start": True,
                },
            }
        )
    return summary


def test_formal_gate_aggregation_counts_sketch_gate_and_total_storage() -> None:
    runner = _load_runner_module()
    protocol = json.loads(
        Path(
            "research/validations/native-clm-v0-m3l2-online-address-state/protocol.json"
        ).read_text()
    )
    control = _synthetic_summary(treatment=False)
    treatment = _synthetic_summary(treatment=True)
    result = runner.compare_arms(control, treatment, thresholds=protocol["thresholds"])
    assert set(result["gates"]) == set(protocol["registered_gates"])
    assert result["gates"]["rank32_address_state_bounded"] is True
    assert result["treatment_address_state"]["maximum_affine_gate_bytes_per_edge"] == 1552
    assert result["treatment_address_state"]["maximum_total_address_bytes_per_node"] == 53912
    assert result["pass"] is True

    too_small = copy.deepcopy(protocol["thresholds"])
    too_small["maximum_total_address_bytes_per_node"] = 53911
    rejected = runner.compare_arms(control, treatment, thresholds=too_small)
    assert rejected["gates"]["rank32_address_state_bounded"] is False
    assert rejected["pass"] is False
