from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from .clm_routing import straight_through_topk
from .clm_v2_compute import CLMv2Stats, linear_mlp_flops
from .language_models import LanguageModelOutput, NCAStage, TextNCALM


@dataclass(frozen=True)
class CLMv2Config:
    num_programs: int = 12
    expert_hidden_dim: int = 64
    shared_hidden_dim: int = 128
    receptor_dim: int = 32
    top_k: int = 6
    scaffold_alpha: float = 1.0
    execution_backend: str = "masked_dense"

    def __post_init__(self) -> None:
        if self.num_programs != 12:
            raise ValueError("CLM v2 Validation 001 fixes num_programs at 12")
        if self.expert_hidden_dim != 64 or self.shared_hidden_dim != 128:
            raise ValueError("CLM v2 fixes expert/shared hidden dimensions at 64/128")
        if not 1 <= self.top_k <= self.num_programs:
            raise ValueError("top_k must be in [1, num_programs]")
        if not 0 <= self.scaffold_alpha <= 1:
            raise ValueError("scaffold_alpha must be in [0, 1]")
        if self.execution_backend not in ("masked_dense", "sparse_dispatch"):
            raise ValueError("unknown execution backend")


class OvercompleteProgram(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(hidden_dim, dim)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.activation(self.in_proj(inputs)))


class LocalProgramReceptor(nn.Module):
    def __init__(self, dim: int, receptor_dim: int, num_programs: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, receptor_dim)
        self.out_proj = nn.Linear(receptor_dim, num_programs)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, perception: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.silu(self.in_proj(self.norm(perception))))


class OvercompleteProgramBank(nn.Module):
    def __init__(self, dim: int, config: CLMv2Config) -> None:
        super().__init__()
        self.dim = dim
        self.config = config
        self.shared = OvercompleteProgram(dim, config.shared_hidden_dim)
        self.programs = nn.ModuleList(
            [OvercompleteProgram(dim, config.expert_hidden_dim)
             for _ in range(config.num_programs)]
        )
        self.receptor = LocalProgramReceptor(dim, config.receptor_dim, config.num_programs)

    def set_config(self, config: CLMv2Config) -> None:
        self.config = config

    def route(self, perception: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.receptor(perception)
        probabilities = logits.sigmoid()
        gates = straight_through_topk(probabilities, self.config.top_k)
        return gates, probabilities, logits

    def forward(
        self, perception: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gates, probabilities, logits = self.route(perception)
        output = self.shared(perception)
        if self.config.execution_backend == "masked_dense":
            for index, program in enumerate(self.programs):
                output = output + program(perception) * gates[..., index, None]
        else:
            flat_inputs = perception.reshape(-1, perception.shape[-1])
            flat_gates = gates.reshape(-1, gates.shape[-1])
            flat_output = output.reshape(-1, output.shape[-1])
            for index, program in enumerate(self.programs):
                active = flat_gates[:, index].detach().bool().nonzero(as_tuple=False).squeeze(-1)
                if active.numel() == 0:
                    continue
                contribution = program(flat_inputs.index_select(0, active))
                contribution = contribution * flat_gates.index_select(0, active)[:, index, None]
                flat_output.index_add_(0, active, contribution)
        return output, gates, probabilities, logits


@dataclass
class _StageTelemetry:
    relative_mse: list[torch.Tensor]
    cosine: list[torch.Tensor]
    gates: list[torch.Tensor]
    probabilities: list[torch.Tensor]
    logits: list[torch.Tensor]
    cells: int = 0
    scaffold_calls: int = 0


class ScaffoldedNCAStage(nn.Module):
    def __init__(self, source: NCAStage, config: CLMv2Config) -> None:
        super().__init__()
        self.iterations = source.iterations
        self.norm_attention = source.norm_attention
        self.norm_ffn = source.norm_ffn
        self.attention = source.attention
        self.dense_scaffold = source.ffn
        self.step_embedding = source.step_embedding
        self.gru = source.gru
        self.program_bank = OvercompleteProgramBank(self.gru.hidden_size, config)
        self.config = config

    def set_config(self, config: CLMv2Config) -> None:
        self.config = config
        self.program_bank.set_config(config)

    def forward(
        self, state: torch.Tensor, *, return_local_imitation: bool
    ) -> tuple[torch.Tensor, _StageTelemetry]:
        batch, length, dim = state.shape
        telemetry = _StageTelemetry([], [], [], [], [], batch * length * self.iterations)
        alpha = self.config.scaffold_alpha
        for step in range(self.iterations):
            conditioned = state + self.step_embedding[step].view(1, 1, dim)
            attention_delta = self.attention(self.norm_attention(conditioned))
            perception = self.norm_ffn(state + attention_delta)
            need_scaffold = alpha > 0 or return_local_imitation
            need_bank = alpha < 1 or return_local_imitation
            dense_delta = None
            conditional_delta = None
            if need_scaffold:
                dense_delta = self.dense_scaffold(perception)
                telemetry.scaffold_calls += 1
            if need_bank:
                conditional_delta, gates, probabilities, logits = self.program_bank(perception)
                telemetry.gates.append(gates)
                telemetry.probabilities.append(probabilities)
                telemetry.logits.append(logits)
            if return_local_imitation:
                assert dense_delta is not None and conditional_delta is not None
                target = dense_delta.detach()
                numerator = (conditional_delta - target).square().mean()
                denominator = target.square().mean().add(1e-8)
                telemetry.relative_mse.append(numerator / denominator)
                telemetry.cosine.append(
                    F.cosine_similarity(conditional_delta.flatten(0, -2),
                                        target.flatten(0, -2), dim=-1).mean()
                )
            if alpha == 1:
                assert dense_delta is not None
                ffn_delta = dense_delta
            elif alpha == 0:
                assert conditional_delta is not None
                ffn_delta = conditional_delta
            else:
                assert dense_delta is not None and conditional_delta is not None
                ffn_delta = alpha * dense_delta + (1 - alpha) * conditional_delta
            proposal = attention_delta + ffn_delta
            state = self.gru(
                proposal.reshape(batch * length, dim), state.reshape(batch * length, dim)
            ).view(batch, length, dim)
        return state, telemetry


class OvercompleteCellularTextNCA(nn.Module):
    def __init__(self, source: TextNCALM, config: CLMv2Config) -> None:
        super().__init__()
        copied = copy.deepcopy(source)
        self.max_context = copied.max_context
        self.stage_supervision = copied.stage_supervision
        self.token_embedding = copied.token_embedding
        self.position_embedding = copied.position_embedding
        self.stages = nn.ModuleList([ScaffoldedNCAStage(stage, config) for stage in copied.stages])
        self.final_norm = copied.final_norm
        self.lm_head = copied.lm_head
        self.config = config
        self.scaffold_provenance: dict[str, object] = {"source_model": "TextNCALM"}

    def _replace_config(self, **changes: object) -> None:
        config = replace(self.config, **changes)
        self.config = config
        for stage in self.stages:
            stage.set_config(config)

    def set_scaffold_alpha(self, alpha: float) -> None:
        self._replace_config(scaffold_alpha=alpha)

    def set_program_top_k(self, top_k: int) -> None:
        self._replace_config(top_k=top_k)

    def set_execution_backend(self, backend: str) -> None:
        self._replace_config(execution_backend=backend)

    def freeze_scaffold(self) -> None:
        for stage in self.stages:
            stage.dense_scaffold.requires_grad_(False)

    def freeze_inherited_backbone(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for stage in self.stages:
            stage.program_bank.requires_grad_(True)

    def unfreeze_sparse_backbone(self) -> None:
        self.requires_grad_(True)
        self.freeze_scaffold()

    def sparse_parameters(self):
        for stage in self.stages:
            yield from stage.program_bank.parameters()

    def get_extra_state(self) -> dict[str, object]:
        return {"config": asdict(self.config), "scaffold_provenance": self.scaffold_provenance}

    def set_extra_state(self, state: dict[str, object]) -> None:
        self.scaffold_provenance = dict(state["scaffold_provenance"])
        self._replace_config(**state["config"])

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_stats: bool = False,
        return_local_imitation: bool = False,
    ) -> LanguageModelOutput | tuple[LanguageModelOutput, CLMv2Stats]:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.max_context:
            raise ValueError(f"input_ids must be [batch, <= {self.max_context}]")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        state = self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        intermediate: list[torch.Tensor] = []
        telemetry: list[_StageTelemetry] = []
        for index, stage in enumerate(self.stages):
            state, stage_telemetry = stage(
                state, return_local_imitation=return_local_imitation
            )
            telemetry.append(stage_telemetry)
            if self.stage_supervision and index < len(self.stages) - 1:
                intermediate.append(self.lm_head(self.final_norm(state)))
        logits = self.lm_head(self.final_norm(state))
        stage_logits = tuple([*intermediate, logits]) if self.stage_supervision else ()
        output = LanguageModelOutput(logits, stage_logits)
        if not (return_stats or return_local_imitation):
            return output
        stats = self._stats(telemetry, logits)
        return output, stats

    def _stats(self, telemetry: list[_StageTelemetry], reference: torch.Tensor) -> CLMv2Stats:
        gates = [item for stage in telemetry for item in stage.gates]
        probabilities = [item for stage in telemetry for item in stage.probabilities]
        logits = [item for stage in telemetry for item in stage.logits]
        relative = [item for stage in telemetry for item in stage.relative_mse]
        cosine = [item for stage in telemetry for item in stage.cosine]
        zero = reference.new_zeros(())
        if gates:
            flat_gates = torch.cat([item.reshape(-1, self.config.num_programs) for item in gates])
            flat_probabilities = torch.cat(
                [item.reshape(-1, self.config.num_programs) for item in probabilities]
            )
            flat_logits = torch.cat([item.reshape(-1, self.config.num_programs) for item in logits])
            usage = flat_gates.mean(0)
            soft_usage = flat_probabilities.mean(0)
            coactivation = flat_gates.T @ flat_gates / flat_gates.shape[0]
            balance = (usage - self.config.top_k / self.config.num_programs).square().mean()
            z_loss = flat_logits.square().mean()
            logit_variance = flat_logits.var(0, unbiased=False).mean()
        else:
            usage = reference.new_zeros(self.config.num_programs)
            soft_usage = usage
            coactivation = reference.new_zeros(self.config.num_programs, self.config.num_programs)
            balance = z_loss = logit_variance = zero
        cells = sum(stage.cells for stage in telemetry)
        dim = self.token_embedding.embedding_dim
        shared_flops = linear_mlp_flops(cells, dim, self.config.shared_hidden_dim, dim)
        expert_flops = linear_mlp_flops(
            cells, dim, self.config.expert_hidden_dim * self.config.top_k, dim
        )
        receptor_flops = cells * (
            5 * dim + 2 * dim * self.config.receptor_dim
            + 4 * self.config.receptor_dim
            + 2 * self.config.receptor_dim * self.config.num_programs
        )
        scaffold_flops = sum(
            linear_mlp_flops(
                stage_telemetry.cells,
                dim,
                stage.dense_scaffold[0].out_features,
                dim,
            )
            for stage, stage_telemetry in zip(self.stages, telemetry)
            if stage_telemetry.scaffold_calls
        )
        shared_parameters = sum(
            parameter.numel() for stage in self.stages
            for parameter in stage.program_bank.shared.parameters()
        )
        expert_parameters = sum(
            parameter.numel() for stage in self.stages
            for program in stage.program_bank.programs for parameter in program.parameters()
        )
        receptor_parameters = sum(
            parameter.numel() for stage in self.stages
            for parameter in stage.program_bank.receptor.parameters()
        )
        genome_parameters = shared_parameters + expert_parameters + receptor_parameters
        active_parameters = (
            shared_parameters
            + round(expert_parameters * self.config.top_k / self.config.num_programs)
            + receptor_parameters
        )
        return CLMv2Stats(
            torch.stack(relative).mean() if relative else zero,
            torch.stack(cosine).mean() if cosine else zero,
            balance,
            z_loss,
            usage,
            soft_usage,
            coactivation,
            logit_variance,
            self.config.top_k / self.config.num_programs,
            self.config.shared_hidden_dim
            + self.config.num_programs * self.config.expert_hidden_dim,
            self.config.shared_hidden_dim + self.config.top_k * self.config.expert_hidden_dim,
            genome_parameters,
            active_parameters,
            shared_flops,
            expert_flops,
            receptor_flops,
            scaffold_flops,
            shared_flops + expert_flops + receptor_flops,
        )
