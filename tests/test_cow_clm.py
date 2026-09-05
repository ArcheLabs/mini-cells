from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from minicells.cow_clm import (
    COWCLMError,
    COWRuntime,
    ExpertSite,
    apply_cell_artifact,
    export_cell,
    load_cell_artifact,
    save_cell_artifact,
    summarize_router_logits,
    top_expert_sites,
)


class _ToyExperts(nn.Module):
    def __init__(self, experts: int = 4, width: int = 4, intermediate: int = 6) -> None:
        super().__init__()
        self.num_experts = experts
        self.gate_up_proj = nn.Parameter(torch.randn(experts, 2 * intermediate, width))
        self.down_proj = nn.Parameter(torch.randn(experts, width, intermediate))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        for token in range(hidden_states.shape[0]):
            for rank in range(top_k_index.shape[1]):
                expert = int(top_k_index[token, rank].item())
                gate, up = F.linear(hidden_states[token], self.gate_up_proj[expert]).chunk(2)
                value = F.silu(gate) * up
                value = F.linear(value, self.down_proj[expert])
                final[token] += value * top_k_weights[token, rank]
        return final


class _ToyMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = _ToyExperts()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        flat = hidden.reshape(-1, hidden.shape[-1])
        first = torch.arange(flat.shape[0], device=hidden.device) % self.experts.num_experts
        top_k = first[:, None]
        weights = torch.ones_like(top_k, dtype=hidden.dtype)
        return self.experts(flat, top_k, weights).reshape_as(hidden)


class _ToyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block_sparse_moe = _ToyMoE()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.block_sparse_moe(hidden)


class _ToyBackbone(nn.Module):
    def __init__(self, layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ToyLayer() for _ in range(layers)])


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ToyBackbone()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden


def _runtime(seed: int = 1) -> tuple[_ToyModel, COWRuntime]:
    torch.manual_seed(seed)
    model = _ToyModel()
    runtime = COWRuntime(
        model,
        foundation_model_id="toy-granite",
        foundation_revision="root-v1",
    )
    return model, runtime


def test_empty_fork_is_exact_and_parent_object_graph_is_unchanged() -> None:
    model, runtime = _runtime()
    hidden = torch.randn(2, 3, 4)
    baseline = model(hidden).detach().clone()
    root_experts = model.model.layers[0].block_sparse_moe.experts
    runtime.fork_empty("empty")
    with runtime.activate("empty"):
        observed = model(hidden).detach().clone()
    assert torch.equal(observed, baseline)
    assert model.model.layers[0].block_sparse_moe.experts is root_experts
    runtime.assert_foundation_unchanged()


def test_expert_slice_birth_is_exact_and_only_selected_slice_is_private() -> None:
    model, runtime = _runtime()
    hidden = torch.randn(2, 4, 4)
    baseline = model(hidden).detach().clone()
    cell = runtime.fork_experts("cell-a", [ExpertSite(0, 1)])
    assert runtime.private_parameter_count("cell-a") == (
        model.model.layers[0].block_sparse_moe.experts.gate_up_proj[1].numel()
        + model.model.layers[0].block_sparse_moe.experts.down_proj[1].numel()
    )
    assert all(parameter.requires_grad for parameter in runtime.private_parameters("cell-a"))
    with runtime.activate(cell.cell_id):
        observed = model(hidden).detach().clone()
    assert torch.equal(observed, baseline)
    runtime.assert_foundation_unchanged()


def test_training_private_slice_changes_cell_view_but_never_parent() -> None:
    model, runtime = _runtime(seed=2)
    hidden = torch.randn(2, 4, 4)
    root_before = model(hidden).detach().clone()
    runtime.fork_experts("cell-a", [(0, 1), (1, 1)])
    optimizer = torch.optim.Adam(runtime.private_parameters("cell-a"), lr=0.03)

    with runtime.activate("cell-a"):
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = model(hidden).square().mean()
            loss.backward()
            optimizer.step()
        cell_output = model(hidden).detach().clone()

    root_after = model(hidden).detach().clone()
    assert not torch.equal(cell_output, root_before)
    assert torch.equal(root_after, root_before)
    runtime.assert_foundation_unchanged()


def test_activation_is_one_complete_ticket_and_nested_views_are_rejected() -> None:
    _model, runtime = _runtime()
    runtime.fork_experts("cell-a", [(0, 1)])
    runtime.fork_experts("cell-b", [(1, 2)])
    with (
        runtime.activate("cell-a"),
        pytest.raises(COWCLMError, match="nested"),
        runtime.activate("cell-b"),
    ):
        pass


def test_artifact_roundtrip_reconstructs_same_cell_view() -> None:
    model, runtime = _runtime(seed=3)
    hidden = torch.randn(2, 4, 4)
    runtime.fork_experts("cell-a", [(0, 1), (1, 2)])
    with torch.no_grad():
        for parameter in runtime.private_parameters("cell-a"):
            parameter.normal_(mean=0.0, std=0.02)
    with runtime.activate("cell-a"):
        expected = model(hidden).detach().clone()
    artifact = export_cell(runtime, "cell-a")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cell.pt"
        save_cell_artifact(path, artifact)
        loaded = load_cell_artifact(path)

    fresh_model, fresh_runtime = _runtime(seed=3)
    apply_cell_artifact(fresh_runtime, loaded)
    with fresh_runtime.activate("cell-a"):
        observed = fresh_model(hidden).detach().clone()
    assert torch.equal(observed, expected)
    assert export_cell(fresh_runtime, "cell-a").digest() == artifact.digest()
    fresh_runtime.assert_foundation_unchanged()


def test_wrong_foundation_rejects_artifact() -> None:
    _model, runtime = _runtime(seed=4)
    runtime.fork_experts("cell-a", [(0, 0)])
    artifact = export_cell(runtime, "cell-a")
    other_model = _ToyModel()
    other = COWRuntime(
        other_model,
        foundation_model_id="toy-granite",
        foundation_revision="different",
    )
    with pytest.raises(COWCLMError, match="parent digest"):
        apply_cell_artifact(other, artifact)


def test_v01_rejects_deeper_lineage_instead_of_inventing_merge_semantics() -> None:
    _model, runtime = _runtime()
    runtime.fork_experts("cell-a", [(0, 1)])
    with pytest.raises(COWCLMError, match="direct forks"):
        runtime.fork_experts("cell-b", [(1, 2)], parent_id="cell-a")


def test_router_trace_ranks_real_topk_activation_sites() -> None:
    logits = (
        torch.tensor([[[8.0, 1.0, 0.0], [7.0, 6.0, 0.0]]]),
        torch.tensor([[[0.0, 1.0, 9.0], [0.0, 1.0, 8.0]]]),
    )
    stats = summarize_router_logits(logits, top_k=1)
    sites = top_expert_sites(stats, 2)
    assert sites == (ExpertSite(0, 0), ExpertSite(1, 2))


def test_router_trace_respects_padding_for_rank3_logits() -> None:
    logits = (
        torch.tensor(
            [
                [[9.0, 0.0], [0.0, 9.0], [0.0, 9.0]],
                [[9.0, 0.0], [9.0, 0.0], [0.0, 9.0]],
            ]
        ),
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    stats = summarize_router_logits(logits, top_k=1, attention_mask=mask)
    by_site = {item.site: item.hits for item in stats}
    assert by_site[ExpertSite(0, 0)] == 2
    assert by_site[ExpertSite(0, 1)] == 1