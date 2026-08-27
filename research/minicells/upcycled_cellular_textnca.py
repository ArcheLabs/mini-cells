from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from .language_models import LanguageModelOutput, NCAStage, TextNCALM


@dataclass(frozen=True)
class UpcyclingConfig:
    num_experts: int = 4
    top_k: int = 1
    router_scale: float = 4.0
    execution_backend: str = "masked_dense"

    def __post_init__(self) -> None:
        if self.num_experts < 2:
            raise ValueError("upcycling requires at least two experts")
        if self.top_k != 1:
            raise ValueError("CLM Upcycling Study 001 fixes top_k at 1")
        if self.router_scale <= 0:
            raise ValueError("router_scale must be positive")
        if self.execution_backend not in ("masked_dense", "sparse_dispatch"):
            raise ValueError("unknown execution backend")


@dataclass(frozen=True)
class UpcycledStats:
    program_usage: torch.Tensor
    program_coactivation: torch.Tensor
    router_logit_variance: torch.Tensor
    balance_loss: torch.Tensor
    usage_entropy: torch.Tensor
    total_expert_parameters: int
    active_expert_parameters: int
    router_parameters: int


class PrototypeRouter(nn.Module):
    """Strictly pointwise local cosine-prototype router.

    Random and geometry-aware arms use the exact same router parameterization.
    The only difference is the initial prototype values.
    """

    def __init__(self, dim: int, num_experts: int, scale: float) -> None:
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.scale = float(scale)
        prototypes = torch.randn(num_experts, dim)
        self.prototypes = nn.Parameter(F.normalize(prototypes, dim=-1))

    @torch.no_grad()
    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        if prototypes.shape != self.prototypes.shape:
            raise ValueError(
                f"expected prototypes {tuple(self.prototypes.shape)}, got {tuple(prototypes.shape)}"
            )
        self.prototypes.copy_(F.normalize(prototypes.to(self.prototypes), dim=-1))

    def forward(self, perception: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(perception, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        return self.scale * normalized @ prototypes.T


def straight_through_top1(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = logits.softmax(dim=-1)
    indices = probabilities.argmax(dim=-1, keepdim=True)
    hard = torch.zeros_like(probabilities).scatter_(-1, indices, 1.0)
    gates = hard + probabilities - probabilities.detach()
    return gates, probabilities


class UpcycledProgramBank(nn.Module):
    def __init__(self, dense_ffn: nn.Module, dim: int, config: UpcyclingConfig) -> None:
        super().__init__()
        self.config = config
        # Knowledge inheritance invariant: every expert begins as an exact copy of
        # the pretrained dense FFN. The routed model is therefore function-preserving
        # for any top-1 assignment at initialization.
        self.experts = nn.ModuleList(
            [copy.deepcopy(dense_ffn) for _ in range(config.num_experts)]
        )
        self.router = PrototypeRouter(dim, config.num_experts, config.router_scale)

    def set_config(self, config: UpcyclingConfig) -> None:
        self.config = config

    def route(self, perception: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(perception)
        gates, probabilities = straight_through_top1(logits)
        return gates, probabilities, logits

    def forward(
        self, perception: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gates, probabilities, logits = self.route(perception)
        if self.config.execution_backend == "masked_dense":
            output = torch.zeros_like(perception)
            for index, expert in enumerate(self.experts):
                output = output + expert(perception) * gates[..., index, None]
        else:
            flat_inputs = perception.reshape(-1, perception.shape[-1])
            flat_gates = gates.reshape(-1, gates.shape[-1])
            flat_output = torch.zeros_like(flat_inputs)
            for index, expert in enumerate(self.experts):
                active = flat_gates[:, index].detach().bool().nonzero(as_tuple=False).squeeze(-1)
                if active.numel() == 0:
                    continue
                contribution = expert(flat_inputs.index_select(0, active))
                contribution = contribution * flat_gates.index_select(0, active)[:, index, None]
                flat_output.index_add_(0, active, contribution)
            output = flat_output.view_as(perception)
        return output, gates, probabilities, logits


@dataclass
class _StageTelemetry:
    gates: list[torch.Tensor]
    probabilities: list[torch.Tensor]
    logits: list[torch.Tensor]


class UpcycledNCAStage(nn.Module):
    def __init__(self, source: NCAStage, config: UpcyclingConfig) -> None:
        super().__init__()
        self.iterations = source.iterations
        self.norm_attention = source.norm_attention
        self.norm_ffn = source.norm_ffn
        self.attention = source.attention
        self.step_embedding = source.step_embedding
        self.gru = source.gru
        self.program_bank = UpcycledProgramBank(source.ffn, self.gru.hidden_size, config)
        self.config = config

    def set_config(self, config: UpcyclingConfig) -> None:
        self.config = config
        self.program_bank.set_config(config)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, _StageTelemetry]:
        batch, length, dim = state.shape
        telemetry = _StageTelemetry([], [], [])
        for step in range(self.iterations):
            conditioned = state + self.step_embedding[step].view(1, 1, dim)
            attention_delta = self.attention(self.norm_attention(conditioned))
            perception = self.norm_ffn(state + attention_delta)
            ffn_delta, gates, probabilities, logits = self.program_bank(perception)
            telemetry.gates.append(gates)
            telemetry.probabilities.append(probabilities)
            telemetry.logits.append(logits)
            proposal = attention_delta + ffn_delta
            state = self.gru(
                proposal.reshape(batch * length, dim), state.reshape(batch * length, dim)
            ).view(batch, length, dim)
        return state, telemetry


class UpcycledCellularTextNCA(nn.Module):
    def __init__(self, source: TextNCALM, config: UpcyclingConfig | None = None) -> None:
        super().__init__()
        config = config or UpcyclingConfig()
        copied = copy.deepcopy(source)
        self.max_context = copied.max_context
        self.stage_supervision = copied.stage_supervision
        self.token_embedding = copied.token_embedding
        self.position_embedding = copied.position_embedding
        self.stages = nn.ModuleList(
            [UpcycledNCAStage(stage, config) for stage in copied.stages]
        )
        self.final_norm = copied.final_norm
        self.lm_head = copied.lm_head
        self.config = config
        self.provenance: dict[str, object] = {
            "source_model": "TextNCALM",
            "initialization": "dense_ffn_copy",
        }

    def _replace_config(self, **changes: object) -> None:
        config = replace(self.config, **changes)
        self.config = config
        for stage in self.stages:
            stage.set_config(config)

    def set_execution_backend(self, backend: str) -> None:
        self._replace_config(execution_backend=backend)

    @torch.no_grad()
    def set_router_prototypes(self, stage_prototypes: list[torch.Tensor]) -> None:
        if len(stage_prototypes) != len(self.stages):
            raise ValueError("one prototype tensor is required per NCA stage")
        for stage, prototypes in zip(self.stages, stage_prototypes):
            stage.program_bank.router.set_prototypes(prototypes)
        self.provenance["router_initialization"] = "geometry"

    def get_extra_state(self) -> dict[str, object]:
        return {"config": asdict(self.config), "provenance": self.provenance}

    def set_extra_state(self, state: dict[str, object]) -> None:
        self.provenance = dict(state["provenance"])
        self._replace_config(**state["config"])

    def forward(
        self, input_ids: torch.Tensor, *, return_stats: bool = False
    ) -> LanguageModelOutput | tuple[LanguageModelOutput, UpcycledStats]:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.max_context:
            raise ValueError(f"input_ids must be [batch, <= {self.max_context}]")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        state = self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        telemetry: list[_StageTelemetry] = []
        intermediate: list[torch.Tensor] = []
        for index, stage in enumerate(self.stages):
            state, stage_telemetry = stage(state)
            telemetry.append(stage_telemetry)
            if self.stage_supervision and index < len(self.stages) - 1:
                intermediate.append(self.lm_head(self.final_norm(state)))
        logits = self.lm_head(self.final_norm(state))
        stage_logits = tuple([*intermediate, logits]) if self.stage_supervision else ()
        output = LanguageModelOutput(logits, stage_logits)
        if not return_stats:
            return output
        return output, self._stats(telemetry, logits)

    def _stats(self, telemetry: list[_StageTelemetry], reference: torch.Tensor) -> UpcycledStats:
        gates = [item for stage in telemetry for item in stage.gates]
        logits = [item for stage in telemetry for item in stage.logits]
        if gates:
            flat_gates = torch.cat(
                [item.reshape(-1, self.config.num_experts) for item in gates], dim=0
            )
            flat_logits = torch.cat(
                [item.reshape(-1, self.config.num_experts) for item in logits], dim=0
            )
            usage = flat_gates.mean(0)
            coactivation = flat_gates.T @ flat_gates / flat_gates.shape[0]
            balance = (usage - 1.0 / self.config.num_experts).square().mean()
            variance = flat_logits.var(0, unbiased=False).mean()
            frequency = usage / usage.sum().clamp_min(1e-8)
            nonzero = frequency > 0
            entropy = -(
                frequency[nonzero] * frequency[nonzero].log()
            ).sum() / torch.log(reference.new_tensor(float(self.config.num_experts)))
        else:
            usage = reference.new_zeros(self.config.num_experts)
            coactivation = reference.new_zeros(self.config.num_experts, self.config.num_experts)
            balance = variance = entropy = reference.new_zeros(())
        expert_parameters = sum(
            parameter.numel()
            for stage in self.stages
            for expert in stage.program_bank.experts
            for parameter in expert.parameters()
        )
        router_parameters = sum(
            parameter.numel()
            for stage in self.stages
            for parameter in stage.program_bank.router.parameters()
        )
        active_parameters = round(expert_parameters / self.config.num_experts)
        return UpcycledStats(
            usage,
            coactivation,
            variance,
            balance,
            entropy,
            expert_parameters,
            active_parameters,
            router_parameters,
        )


def convert_textnca_to_upcycled(
    source: TextNCALM, *, config: UpcyclingConfig | None = None
) -> UpcycledCellularTextNCA:
    return UpcycledCellularTextNCA(source, config)
