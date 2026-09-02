from __future__ import annotations

import torch
import torch.nn.functional as F

from minicells.native_clm_m3l2 import (
    M3L2AddressConfig,
    MomentAccumulator,
    OnlineAddressNativeCLM,
    merge_sketch_and_moments,
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
    x = torch.randn(n, 16) * 0.15
    x[:, 0] += center
    acc = MomentAccumulator(16)
    acc.update(x)
    return acc


def test_rank32_state_is_bounded_and_mergeable() -> None:
    config = M3L2AddressConfig(rank=32)
    first = _acc(-1.0).to_sketch(rank=config.rank, diagonal_regularization=config.diagonal_regularization)
    assert first.rank == 16
    current = _acc(-0.5)
    merged = merge_sketch_and_moments(first, current, config=config)
    assert merged is not None
    assert merged.rank <= 32
    assert merged.storage_bytes <= config.maximum_persistent_bytes_per_cell


def test_affine_child_gate_preserves_birth_function() -> None:
    torch.manual_seed(8)
    model = _model().eval()
    old = _acc(-1.5).to_sketch(rank=16, diagonal_regularization=1e-4)
    new = _acc(1.5).to_sketch(rank=16, diagonal_regularization=1e-4)
    tokens = torch.randint(0, 32, (2, 8))
    before = model(tokens, return_info=True)
    parent = int(before["cell_info"]["root_idx"][0, 0, 0])
    child = model.spawn_cell(parent_id=parent, route_key=F.normalize(new.mean, dim=0), inherit_scale=1.0)
    model.historical_sketches[parent] = old
    model.historical_sketches[child] = new
    model.affine_gates[parent] = derive_sketch_gate(old, new, diagonal_regularization=1e-4, target_old_fpr=0.1)
    after = model(tokens, return_info=True)
    assert torch.equal(before["cell_info"]["root_idx"], after["cell_info"]["root_idx"])
    assert torch.equal(before["cell_info"]["root_probs"], after["cell_info"]["root_probs"])
    assert torch.allclose(before["logits"], after["logits"], atol=1e-6, rtol=0.0)


def test_address_checkpoint_roundtrip(tmp_path) -> None:
    model = _model().eval()
    old = _acc(-1.0).to_sketch(rank=16, diagonal_regularization=1e-4)
    new = _acc(1.0).to_sketch(rank=16, diagonal_regularization=1e-4)
    child = model.spawn_cell(parent_id=0, route_key=F.normalize(new.mean, dim=0), inherit_scale=1.0)
    model.historical_sketches[0] = old
    model.historical_sketches[child] = new
    model.affine_gates[0] = derive_sketch_gate(old, new, diagonal_regularization=1e-4, target_old_fpr=0.1)
    model.bootstrap_complete = True
    model.bootstrap_parameter_hash_before = "same"
    model.bootstrap_parameter_hash_after = "same"
    path = tmp_path / "m3l2.pt"
    torch.save(model.checkpoint_payload(), path)
    restored, _ = OnlineAddressNativeCLM.load_checkpoint(path)
    tokens = torch.randint(0, 32, (2, 8))
    assert restored.address_state_metrics()["gate_count"] == 1
    assert restored.address_state_metrics()["sketch_count"] == 2
    assert torch.allclose(model(tokens)["logits"], restored(tokens)["logits"], atol=1e-7, rtol=0.0)


def test_affine_gate_separates_shifted_query_state() -> None:
    old = _acc(-2.0).to_sketch(rank=16, diagonal_regularization=1e-4)
    new = _acc(2.0).to_sketch(rank=16, diagonal_regularization=1e-4)
    gate = derive_sketch_gate(old, new, diagonal_regularization=1e-4, target_old_fpr=0.1)
    weight = gate["weight"]
    assert isinstance(weight, torch.Tensor)
    old_score = F.normalize(old.mean, dim=0).dot(weight) + float(gate["bias"])
    new_score = F.normalize(new.mean, dim=0).dot(weight) + float(gate["bias"])
    assert new_score > old_score


def test_protocol_rank_is_exactly_32() -> None:
    assert M3L2AddressConfig().rank == 32
    assert M3L2AddressConfig().maximum_persistent_bytes_per_cell == 52360
