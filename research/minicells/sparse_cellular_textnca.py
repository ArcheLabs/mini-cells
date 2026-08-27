from __future__ import annotations

import copy
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .clm_compute import CLMComputeStats, _StepStats, aggregate_compute_stats
from .clm_routing import (
    CLMRoutingConfig,
    straight_through_threshold,
    straight_through_topk,
)
from .language_models import LanguageModelOutput, NCAStage, TextNCALM


def _partitions(width: int, count: int) -> tuple[slice, ...]:
    if count > width:
        raise ValueError("num_programs cannot exceed the FFN hidden width")
    base, remainder = divmod(width, count)
    result: list[slice] = []
    start = 0
    for index in range(count):
        size = base + (index < remainder)
        result.append(slice(start, start + size))
        start += size
    return tuple(result)


class LocalReceptor(nn.Module):
    """A pointwise receptor: it cannot pool across cells or examples."""

    def __init__(self, dim: int, config: CLMRoutingConfig) -> None:
        super().__init__()
        input_dim = dim + config.phenotype_dim
        self.norm = nn.LayerNorm(input_dim)
        self.in_proj = nn.Linear(input_dim, config.receptor_dim)
        self.out_proj = nn.Linear(config.receptor_dim, 1 + config.num_programs)
        nn.init.zeros_(self.out_proj.weight)
        with torch.no_grad():
            self.out_proj.bias.fill_(8.0)

    def forward(self, perception: torch.Tensor, phenotype: torch.Tensor | None) -> torch.Tensor:
        if phenotype is not None:
            perception = torch.cat((perception, phenotype), dim=-1)
        return self.out_proj(F.silu(self.in_proj(self.norm(perception))))


class SegmentedUpdate(nn.Module):
    """The original Linear-GELU-Linear FFN, sliced without copying parameters."""

    def __init__(self, ffn: nn.Sequential, num_programs: int) -> None:
        super().__init__()
        if not (len(ffn) == 3 and isinstance(ffn[0], nn.Linear) and
                isinstance(ffn[1], nn.GELU) and isinstance(ffn[2], nn.Linear)):
            raise TypeError("exact segmentation requires Linear -> GELU -> Linear")
        self.in_proj = ffn[0]
        self.activation = ffn[1]
        self.out_proj = ffn[2]
        self.partitions = _partitions(self.in_proj.out_features, num_programs)

    def dense(self, inputs: torch.Tensor) -> torch.Tensor:
        # Retain the original operation order for strict dense and gradient parity.
        return self.out_proj(self.activation(self.in_proj(inputs)))

    def _program(self, inputs: torch.Tensor, index: int) -> torch.Tensor:
        part = self.partitions[index]
        hidden = F.linear(inputs, self.in_proj.weight[part], self.in_proj.bias[part])
        return F.linear(self.activation(hidden), self.out_proj.weight[:, part], None)

    def routed(self, inputs: torch.Tensor, gates: torch.Tensor, backend: str) -> torch.Tensor:
        output = self.out_proj.bias.expand(*inputs.shape[:-1], -1).clone()
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        flat_gates = gates.reshape(-1, gates.shape[-1])
        flat_output = output.reshape(-1, output.shape[-1])
        for index in range(len(self.partitions)):
            if backend == "masked_dense":
                contribution = self._program(inputs, index)
                output = output + contribution * gates[..., index, None]
                continue
            mask = flat_gates[:, index].detach().bool()
            active = mask.nonzero(as_tuple=False).squeeze(-1)
            if active.numel() == 0:
                continue
            contribution = self._program(flat_inputs.index_select(0, active), index)
            contribution = contribution * flat_gates.index_select(0, active)[:, index, None]
            flat_output.index_add_(0, active, contribution)
        return output


class SparseNCAStage(nn.Module):
    def __init__(self, source: NCAStage, config: CLMRoutingConfig) -> None:
        super().__init__()
        self.iterations = source.iterations
        self.norm_attention = source.norm_attention
        self.norm_ffn = source.norm_ffn
        self.attention = source.attention
        self.update = SegmentedUpdate(source.ffn, config.num_programs)
        self.step_embedding = source.step_embedding
        self.gru = source.gru
        self.config = config
        self.receptor = LocalReceptor(self.gru.hidden_size, config).to(
            device=self.step_embedding.device, dtype=self.step_embedding.dtype
        )
        self.register_buffer("program_usage_ema", torch.zeros(config.num_programs))

    def set_config(self, config: CLMRoutingConfig) -> None:
        self.config = config

    def _gates(
        self, perception: torch.Tensor, phenotype: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (*perception.shape[:-1],)
        if self.config.routing_mode == "dense":
            # The receptor remains a real, accounted local cost even while its result is ignored.
            self.receptor(perception, phenotype)
            return (
                perception.new_ones(shape),
                perception.new_ones((*shape, self.config.num_programs)),
            )
        logits = self.receptor(perception, phenotype)
        cell_soft = logits[..., 0].sigmoid()
        program_soft = logits[..., 1:].sigmoid()
        cell = (
            perception.new_ones(shape)
            if self.config.routing_mode in ("soft_program", "hard_program")
            else cell_soft
        )
        if self.config.routing_mode == "hard":
            cell = straight_through_threshold(cell_soft, self.config.cell_threshold)
        if self.config.routing_mode in ("hard_program", "soft_cell_hard_program", "hard"):
            k = self.config.program_top_k or self.config.num_programs
            programs = straight_through_topk(program_soft, k)
        else:
            programs = program_soft
        return cell, programs

    def forward(
        self, state: torch.Tensor, phenotype: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[_StepStats]]:
        batch, length, dim = state.shape
        stats: list[_StepStats] = []
        for step in range(self.iterations):
            conditioned = state + self.step_embedding[step].view(1, 1, dim)
            attention_delta = self.attention(self.norm_attention(conditioned))
            candidate_state = state + attention_delta
            perception = self.norm_ffn(candidate_state)
            cell_gate, program_gate = self._gates(perception, phenotype)
            if self.config.routing_mode == "dense":
                ffn_delta = self.update.dense(perception)
            else:
                executor_gates = program_gate
                if self.config.execution_backend == "sparse_dispatch":
                    executor_gates = program_gate * cell_gate[..., None]
                ffn_delta = self.update.routed(
                    perception, executor_gates, self.config.execution_backend
                )
            proposal = attention_delta + ffn_delta
            previous = state
            if (
                self.config.execution_backend == "sparse_dispatch"
                and self.config.routing_mode == "hard"
            ):
                flat_gate = cell_gate.reshape(-1)
                active = flat_gate.detach().bool().nonzero(as_tuple=False).squeeze(-1)
                next_flat = previous.reshape(-1, dim).clone()
                if active.numel():
                    updated = self.gru(
                        proposal.reshape(-1, dim).index_select(0, active),
                        previous.reshape(-1, dim).index_select(0, active),
                    )
                    next_flat.index_copy_(0, active, updated)
                state = next_flat.view(batch, length, dim)
            else:
                updated = self.gru(proposal.reshape(-1, dim), previous.reshape(-1, dim)).view(
                    batch, length, dim
                )
                state = previous + cell_gate[..., None] * (updated - previous)
            with torch.no_grad():
                usage = (cell_gate[..., None] * program_gate).detach().float().mean((0, 1))
                self.program_usage_ema.mul_(0.99).add_(usage.to(self.program_usage_ema), alpha=0.01)
            cells = batch * length
            hidden = self.update.in_proj.out_features
            receptor_input = dim + self.config.phenotype_dim
            receptor_flops = cells * (
                5 * receptor_input + 2 * receptor_input * self.config.receptor_dim
                + 4 * self.config.receptor_dim
                + 2 * self.config.receptor_dim * (1 + self.config.num_programs)
            )
            dense_ffn_flops = cells * (2 * dim * hidden + 2 * hidden * dim)
            dense_gru_flops = cells * 12 * dim * dim
            dense_flops = dense_ffn_flops + dense_gru_flops
            routed_program_fraction = float(
                (cell_gate[..., None] * program_gate).detach().mean()
            )
            cell_fraction = float(cell_gate.detach().mean())
            active_flops = round(
                dense_ffn_flops * routed_program_fraction + dense_gru_flops * cell_fraction
            )
            stats.append(_StepStats(cell_gate, program_gate, dense_flops, receptor_flops,
                                    active_flops))
        return state, stats


class SparseCellularTextNCA(nn.Module):
    def __init__(self, source: TextNCALM, config: CLMRoutingConfig) -> None:
        super().__init__()
        copied = copy.deepcopy(source)
        self.max_context = copied.max_context
        self.stage_supervision = copied.stage_supervision
        self.token_embedding = copied.token_embedding
        self.position_embedding = copied.position_embedding
        self.stages = nn.ModuleList([SparseNCAStage(stage, config) for stage in copied.stages])
        self.final_norm = copied.final_norm
        self.lm_head = copied.lm_head
        self.routing_config = config
        self._routing_ratios: tuple[torch.Tensor, torch.Tensor] | None = None
        self.conversion_metadata: dict[str, object] = {
            "source_model": "TextNCALM",
            "num_programs": config.num_programs,
            "partition_sizes": [
                part.stop - part.start for part in self.stages[0].update.partitions
            ],
            "dense_equivalence_verified": False,
        }

    def set_routing_mode(self, mode: str) -> None:
        self._replace_config(routing_mode=mode)

    def set_execution_backend(self, backend: str) -> None:
        self._replace_config(execution_backend=backend)

    def set_program_top_k(self, top_k: int | None) -> None:
        self._replace_config(program_top_k=top_k)

    def _replace_config(self, **changes: object) -> None:
        config = replace(self.routing_config, **changes)
        self.routing_config = config
        for stage in self.stages:
            stage.set_config(config)

    @property
    def program_usage_ema(self) -> torch.Tensor:
        return torch.stack([stage.program_usage_ema for stage in self.stages]).mean(0)

    def program_activity(self) -> torch.Tensor:
        return self.program_usage_ema.detach().clone()

    def routing_ratios(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable (program, cell) ratios from the most recent forward pass."""
        if self._routing_ratios is None:
            raise RuntimeError("routing ratios are available after a forward pass")
        return self._routing_ratios

    def get_extra_state(self) -> dict[str, object]:
        return {
            "routing_config": self.routing_config.__dict__,
            "conversion_metadata": self.conversion_metadata,
        }

    def set_extra_state(self, state: dict[str, object]) -> None:
        config = CLMRoutingConfig(**state["routing_config"])
        self.routing_config = config
        self.conversion_metadata = dict(state["conversion_metadata"])
        for stage in self.stages:
            stage.set_config(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        phenotype: torch.Tensor | None = None,
        return_stats: bool = False,
    ) -> LanguageModelOutput | tuple[LanguageModelOutput, CLMComputeStats]:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.max_context:
            raise ValueError(f"input_ids must be [batch, <= {self.max_context}]")
        if self.routing_config.phenotype_dim == 0 and phenotype is not None:
            raise ValueError("phenotype was provided but phenotype_dim is zero")
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)
        state = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        intermediate: list[torch.Tensor] = []
        all_stats: list[_StepStats] = []
        for index, stage in enumerate(self.stages):
            state, stage_stats = stage(state, phenotype)
            all_stats.extend(stage_stats)
            if self.stage_supervision and index < len(self.stages) - 1:
                intermediate.append(self.lm_head(self.final_norm(state)))
        logits = self.lm_head(self.final_norm(state))
        self._routing_ratios = (
            torch.stack([item.program_gate.mean() for item in all_stats]).mean(),
            torch.stack([item.cell_gate.mean() for item in all_stats]).mean(),
        )
        stage_logits = tuple([*intermediate, logits]) if self.stage_supervision else ()
        output = LanguageModelOutput(logits, stage_logits)
        if return_stats:
            return output, aggregate_compute_stats(all_stats, self.routing_config.num_programs)
        return output

    def cell_activity(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, stats = self(input_ids, return_stats=True)
        return stats.activation_by_nca_step
