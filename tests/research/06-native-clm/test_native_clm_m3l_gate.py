from __future__ import annotations

import torch

from minicells.native_clm_m3l_gate import (
    M3LQuerySketchConfig,
    _edge_ownership_metadata,
    _group_split,
    _score_gate,
    _woodbury_solve_pooled,
    aggregate_m3l_diagnostic,
    derive_sketch_gate,
    fit_low_rank_sketch,
)
from minicells.native_clm_m3r import LineageNativeCLM
from minicells.native_clm_m3r_address_diag import _auc
from minicells.native_clm_v0 import NativeCLMConfig


def _tiny_model() -> LineageNativeCLM:
    return LineageNativeCLM(
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


def test_edge_ownership_follows_parent_lifetime_not_global_A_anchor() -> None:
    model = _tiny_model()
    child_b = model.spawn_cell(parent_id=0, route_key=torch.randn(16), inherit_scale=1.0)
    child_c = model.spawn_cell(parent_id=child_b, route_key=torch.randn(16), inherit_scale=1.0)
    root_c = model.spawn_cell(parent_id=1, route_key=torch.randn(16), inherit_scale=1.0)
    events = [
        {"parent_id": 0, "child_id": child_b, "global_step": 50},
        {"parent_id": child_b, "child_id": child_c, "global_step": 450},
        {"parent_id": 1, "child_id": root_c, "global_step": 550},
    ]
    edges = _edge_ownership_metadata(model, events)
    by_child = {edge["child_id"]: edge for edge in edges}
    assert by_child[child_b]["old_domains"] == ["A"]
    assert by_child[child_b]["current_domain"] == "B"
    assert by_child[child_c]["parent_birth_domain"] == "B"
    assert by_child[child_c]["old_domains"] == ["B"]
    assert by_child[child_c]["current_domain"] == "C"
    assert by_child[root_c]["old_domains"] == ["A", "B"]
    assert by_child[root_c]["current_domain"] == "C"


def test_group_split_never_leaks_one_sequence_between_train_and_test() -> None:
    queries = torch.randn(24, 4)
    groups = torch.tensor([0] * 6 + [1] * 6 + [2] * 6 + [3] * 6)
    train, test = _group_split(queries, groups, train_fraction=0.5, seed=11)
    assert train.size(0) == 12
    assert test.size(0) == 12
    train_rows = {tuple(row.tolist()) for row in train}
    test_rows = {tuple(row.tolist()) for row in test}
    assert train_rows.isdisjoint(test_rows)


def test_woodbury_solver_matches_dense_pooled_covariance() -> None:
    torch.manual_seed(3)
    old = torch.randn(512, 12)
    current = torch.randn(512, 12) + 0.5
    old_sketch = fit_low_rank_sketch(
        old, rank=4, diagonal_regularization=1e-4, device=torch.device("cpu")
    )
    current_sketch = fit_low_rank_sketch(
        current, rank=4, diagonal_regularization=1e-4, device=torch.device("cpu")
    )
    delta = current_sketch.mean - old_sketch.mean
    solved = _woodbury_solve_pooled(
        old_sketch,
        current_sketch,
        delta,
        diagonal_regularization=1e-4,
    )
    old_cov = torch.diag(old_sketch.residual_variance)
    old_cov = old_cov + old_sketch.basis @ torch.diag(old_sketch.eigenvalues) @ old_sketch.basis.T
    current_cov = torch.diag(current_sketch.residual_variance)
    current_cov = (
        current_cov
        + current_sketch.basis
        @ torch.diag(current_sketch.eigenvalues)
        @ current_sketch.basis.T
    )
    dense = 0.5 * (old_cov + current_cov) + 1e-4 * torch.eye(12)
    expected = torch.linalg.solve(dense, delta)
    assert torch.allclose(solved, expected, atol=2e-4, rtol=2e-4)


def test_sketch_gate_separates_shifted_query_distributions_without_old_replay() -> None:
    torch.manual_seed(17)
    width = 16
    direction = torch.randn(width)
    direction = direction / direction.norm()
    old_train = torch.randn(2048, width) * 0.25 - 0.7 * direction
    current_train = torch.randn(2048, width) * 0.25 + 0.7 * direction
    old_test = torch.randn(1024, width) * 0.25 - 0.7 * direction
    current_test = torch.randn(1024, width) * 0.25 + 0.7 * direction
    old_sketch = fit_low_rank_sketch(
        old_train, rank=4, diagonal_regularization=1e-4, device=torch.device("cpu")
    )
    current_sketch = fit_low_rank_sketch(
        current_train, rank=4, diagonal_regularization=1e-4, device=torch.device("cpu")
    )
    gate = derive_sketch_gate(
        old_sketch,
        current_sketch,
        diagonal_regularization=1e-4,
        target_old_fpr=0.1,
    )
    old_scores = _score_gate(old_test, gate, torch.device("cpu"))
    current_scores = _score_gate(current_test, gate, torch.device("cpu"))
    labels = torch.cat([torch.zeros(len(old_scores)), torch.ones(len(current_scores))])
    auc = _auc(torch.cat([old_scores, current_scores]), labels)
    threshold = float(gate["threshold"])
    old_fpr = float((old_scores > threshold).float().mean())
    current_tpr = float((current_scores > threshold).float().mean())
    assert auc > 0.95
    assert old_fpr < 0.25
    assert current_tpr > 0.75


def _edge(*, sketch_auc: float, oracle_auc: float, old_fpr: float, current_tpr: float) -> dict:
    return {
        "valid": True,
        "current_cosine_auc": 0.52,
        "offline_oracle_auc": oracle_auc,
        "sketch_gate_auc": sketch_auc,
        "normalized_oracle_excess_recovery": (sketch_auc - 0.5) / (oracle_auc - 0.5),
        "sketch_gate_old_fpr": old_fpr,
        "sketch_gate_current_tpr": current_tpr,
        "historical_sketch_bytes": 4096,
    }


def test_registered_aggregate_can_select_query_sketch_gate_feasible() -> None:
    summaries = []
    for seed in (73611, 73612, 73613):
        summaries.append(
            {
                "seed": seed,
                "edges": [
                    _edge(sketch_auc=0.94, oracle_auc=0.96, old_fpr=0.12, current_tpr=0.82)
                    for _ in range(8)
                ],
            }
        )
    result = aggregate_m3l_diagnostic(
        summaries,
        config=M3LQuerySketchConfig(),
        parent_m3r_hf_revision="rev",
        parent_address_commit="commit",
    )
    assert result["classification"] == "QUERY_SKETCH_GATE_FEASIBLE"
    assert result["sketch_gate"]["median"] == 0.94
    assert result["new_formal_seeds_consumed"] is False
