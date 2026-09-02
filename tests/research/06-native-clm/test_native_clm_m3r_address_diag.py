from __future__ import annotations

import torch
import torch.nn.functional as F

from minicells.native_clm_m3r import LineageNativeCLM
from minicells.native_clm_m3r_address_diag import (
    AddressDiagnosticConfig,
    _auc,
    _birth_domain,
    _edge_eligible_mask,
    _forward_boundary_features,
    _root_and_path_to_parent,
    aggregate_diagnostic,
)
from minicells.native_clm_v0 import NativeCLMConfig


def _tiny_model() -> LineageNativeCLM:
    config = NativeCLMConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=8,
        n_layers=2,
        n_heads=2,
        d_ff=16,
        dropout=0.0,
        initial_cells=2,
        active_cells=1,
        cellular_layer_index=0,
        certificate_max_rank=4,
        tie_embeddings=False,
    )
    return LineageNativeCLM(config)


def test_birth_domain_and_auc() -> None:
    assert _birth_domain(1) == "B"
    assert _birth_domain(400) == "B"
    assert _birth_domain(401) == "C"
    assert _birth_domain(800) == "C"
    assert _birth_domain(801) == "D"
    scores = torch.tensor([0.0, 1.0, 2.0, 3.0])
    labels = torch.tensor([0, 0, 1, 1])
    assert _auc(scores, labels) == 1.0


def test_lineage_path_and_edge_conditioning() -> None:
    model = _tiny_model()
    with torch.no_grad():
        model.cellular.cells[0].route_key.copy_(torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]))
        model.cellular.cells[1].route_key.copy_(torch.tensor([0.0, 0, 0, 1, 0, 0, 0, 0]))
    child = model.spawn_cell(
        parent_id=0,
        route_key=torch.tensor([0.0, 1, 0, 0, 0, 0, 0, 0]),
        inherit_scale=1.0,
    )
    grandchild = model.spawn_cell(
        parent_id=child,
        route_key=torch.tensor([0.0, 0, 1, 0, 0, 0, 0, 0]),
        inherit_scale=1.0,
    )
    assert child == 2
    assert grandchild == 3
    assert _root_and_path_to_parent(model, child) == (0, [0, 2])

    query = F.normalize(
        torch.tensor([[[0.1, 0.9, 0.0, 0, 0, 0, 0, 0], [0.9, 0.1, 0.0, 0, 0, 0, 0, 0]]]),
        dim=-1,
    )
    root_idx = torch.tensor([[[0], [0]]])
    eligible = _edge_eligible_mask(model, query, root_idx, path_to_parent=[0, 2])
    assert eligible.tolist() == [[True, False]]


def test_boundary_write_left_feature_has_expected_shape() -> None:
    model = _tiny_model()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokens = torch.randint(0, model.config.vocab_size, (2, model.config.max_seq_len))
    targets = torch.randint(0, model.config.vocab_size, (2, model.config.max_seq_len))
    features = _forward_boundary_features(model, tokens, targets)
    assert features["query"].shape == (2, model.config.max_seq_len, model.config.d_model)
    assert features["write_input"].shape == features["query"].shape
    assert features["write_left"].shape == features["query"].shape
    assert features["root_idx"].shape == (2, model.config.max_seq_len, 1)
    assert torch.isfinite(features["write_left"]).all()


def _edge(query_auc: float, write_auc: float) -> dict:
    probes = {
        "query": {"auc": query_auc},
        "write_input": {"auc": write_auc},
        "write_left": {"auc": write_auc},
        "write_pair": {"auc": write_auc},
        "certificate_residual": {"auc": write_auc},
    }
    return {
        "valid": True,
        "current_cosine_auc": 0.6,
        "probes": probes,
    }


def test_aggregate_prefers_query_geometry_when_registered_rule_passes() -> None:
    config = AddressDiagnosticConfig()
    summaries = [
        {"seed": 73611, "edges": [_edge(0.91, 0.92) for _ in range(8)]},
        {"seed": 73612, "edges": [_edge(0.90, 0.93) for _ in range(8)]},
        {"seed": 73613, "edges": [_edge(0.89, 0.94) for _ in range(8)]},
    ]
    result = aggregate_diagnostic(
        summaries,
        config=config,
        parent_protocol_sha256="p",
        parent_data_manifest_sha256="d",
        hf_revision="h",
    )
    assert result["classification"] == "QUERY_GEOMETRY_SEPARABLE"


def test_aggregate_can_select_write_effect_geometry() -> None:
    config = AddressDiagnosticConfig()
    summaries = [
        {"seed": 73611, "edges": [_edge(0.70, 0.91) for _ in range(8)]},
        {"seed": 73612, "edges": [_edge(0.72, 0.90) for _ in range(8)]},
        {"seed": 73613, "edges": [_edge(0.71, 0.89) for _ in range(8)]},
    ]
    result = aggregate_diagnostic(
        summaries,
        config=config,
        parent_protocol_sha256="p",
        parent_data_manifest_sha256="d",
        hf_revision="h",
    )
    assert result["classification"] == "WRITE_EFFECT_GEOMETRY_SEPARABLE"
