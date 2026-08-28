"""MiniCells CLM-0.3 progressive-growth model.

This module is additive to the CLM-0.1 implementation.  The recurrent
substrate and the four-way root routers are copied unchanged; births only add
an expert and a local binary router below an existing root leaf.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .growth_checkpoint import add_fresh_parameter_group, add_newborn_parameters
from .growth_pressure import (
    MIN_ROUTED_PERCEPTIONS,
    PREFERRED_ROUTED_PERCEPTIONS,
    cosine_kmeans_2,
)
from .growth_router import HierarchicalGrowthRouter, clone_module
from .language_models import LanguageModelOutput, TextNCALM
from .upcycled_cellular_textnca import UpcycledCellularTextNCA, UpcycledNCAStage, convert_textnca_to_upcycled


GROWTH_BIRTH_TOKENS = (500_000, 1_000_000)
GROWTH_TOTAL_TOKENS = 1_500_000
FORMAL_ARMS = ("fixed4", "pressure_growth", "random_growth")
REPLICATE_SEEDS = (55031, 55032, 55033)


def replicate_seed(replicate: int) -> int:
    try:
        return REPLICATE_SEEDS[replicate]
    except IndexError as exc:
        raise ValueError("replicate must be 0, 1, or 2") from exc


def phase_for_tokens(consumed_tokens: int, *, complete_tokens: int = GROWTH_TOTAL_TOKENS) -> str:
    if consumed_tokens >= complete_tokens:
        return "complete"
    if consumed_tokens >= GROWTH_BIRTH_TOKENS[1]:
        return "post_birth_2"
    if consumed_tokens >= GROWTH_BIRTH_TOKENS[0]:
        return "post_birth_1"
    return "pre_birth_1"


def next_growth_event(
    consumed_tokens: int, growth_history: list[dict[str, Any]]
) -> tuple[int, int] | None:
    completed = len(growth_history)
    if completed >= len(GROWTH_BIRTH_TOKENS):
        return None
    scheduled_token = GROWTH_BIRTH_TOKENS[completed]
    return (completed, scheduled_token) if consumed_tokens >= scheduled_token else None


def stop_target(target_tokens: int, stop_after_tokens: int | None, tokens_per_step: int) -> int:
    """Return the first step boundary at/after the requested stop, capped by budget."""

    if target_tokens < 0 or tokens_per_step <= 0:
        raise ValueError("target tokens must be non-negative and tokens_per_step must be positive")
    requested = target_tokens if stop_after_tokens is None else min(target_tokens, stop_after_tokens)
    return min(target_tokens, ((requested + tokens_per_step - 1) // tokens_per_step) * tokens_per_step)


@dataclass(frozen=True)
class Lineage:
    expert_id: str
    stage: int
    parent_id: str | None
    birth_token: int
    generation: int
    origin: str = "progressive_growth"
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "stage": self.stage,
            "parent_id": self.parent_id,
            "birth_token": self.birth_token,
            "generation": self.generation,
            "origin": self.origin,
            "active": self.active,
        }


@dataclass
class GrowthStageTelemetry:
    gates: list[torch.Tensor]
    root_indices: list[torch.Tensor]
    root_probabilities: list[torch.Tensor]
    split_choices: list[dict[str, torch.Tensor]]
    perceptions: list[torch.Tensor]
    states: list[torch.Tensor]


@dataclass(frozen=True)
class GrowthStats:
    usage: dict[str, torch.Tensor]
    route_counts: dict[str, int]
    split_entropy: dict[str, float]
    root_usage: tuple[torch.Tensor, ...]
    root_routes: tuple[torch.Tensor, ...]
    recurrent_states: tuple[torch.Tensor, ...] = ()


class GrowthProgramBank(nn.Module):
    """Stable-ID expert bank with a hierarchical root/split router."""

    def __init__(self, source: UpcycledNCAStage, stage: int) -> None:
        super().__init__()
        source_bank = source.program_bank
        if source_bank.config.num_experts != 4 or source_bank.config.top_k != 1:
            raise ValueError("CLM-0.3 requires the four-expert top-1 CLM-0.1 root")
        ids = [f"s{stage}-e{index}" for index in range(4)]
        self.experts = nn.ModuleDict({expert_id: copy.deepcopy(expert)
                                      for expert_id, expert in zip(ids, source_bank.experts)})
        self.router = HierarchicalGrowthRouter(
            stage, copy.deepcopy(source_bank.router), source.gru.hidden_size,
            root_expert_count=4, router_scale=source_bank.config.router_scale,
        )
        self.stage = stage
        self.next_expert_index = 4
        self.parent_by_child: dict[str, str] = {}
        self.split_by_child: dict[str, str] = {}
        self.last_route_counts: dict[str, int] = {expert_id: 0 for expert_id in ids}
        self.last_perceptions: dict[str, list[torch.Tensor]] = {expert_id: [] for expert_id in ids}
        self.collect_pressure = False
        self.pressure_perception_cap = PREFERRED_ROUTED_PERCEPTIONS

    @property
    def expert_ids(self) -> tuple[str, ...]:
        return self.router.expert_ids

    @property
    def root_expert_count(self) -> int:
        return self.router.root_expert_count

    def _record_routes(self, gates: dict[str, torch.Tensor], perception: torch.Tensor) -> None:
        if not self.collect_pressure:
            return
        for expert_id in self.expert_ids:
            hard = gates[expert_id].detach().reshape(-1) >= 0.5
            self.last_route_counts[expert_id] = self.last_route_counts.get(expert_id, 0) + int(hard.sum())
            if hard.any():
                stored = sum(item.shape[0] for item in self.last_perceptions[expert_id])
                remaining = max(0, self.pressure_perception_cap - stored)
                if remaining:
                    selected = perception.detach().reshape(-1, perception.shape[-1])[hard]
                    self.last_perceptions[expert_id].append(selected[:remaining].cpu())

    def reset_pressure_window(self) -> None:
        self.last_route_counts = {expert_id: 0 for expert_id in self.expert_ids}
        self.last_perceptions = {expert_id: [] for expert_id in self.expert_ids}

    def begin_pressure_collection(self, *, cap: int = PREFERRED_ROUTED_PERCEPTIONS) -> None:
        if cap < MIN_ROUTED_PERCEPTIONS:
            raise ValueError("pressure collection cap is below the minimum sample requirement")
        self.reset_pressure_window()
        self.pressure_perception_cap = int(cap)
        self.collect_pressure = True

    def end_pressure_collection(self) -> None:
        self.collect_pressure = False

    @torch.no_grad()
    def add_birth(
        self,
        parent_id: str,
        perceptions: torch.Tensor,
        *,
        child_id: str | None = None,
        split_id: str | None = None,
        precomputed_prototypes: torch.Tensor | None = None,
    ) -> tuple[str, nn.Module, nn.Module, torch.Tensor]:
        if perceptions.ndim != 2 or (precomputed_prototypes is None and perceptions.shape[0] < MIN_ROUTED_PERCEPTIONS):
            raise RuntimeError("NO_ELIGIBLE_GROWTH_PARENT")
        if parent_id not in self.experts or parent_id not in self.expert_ids:
            raise KeyError(f"parent is not an active leaf: {parent_id}")
        child_id = child_id or f"s{self.stage}-e{self.next_expert_index}"
        split_id = split_id or f"s{self.stage}-split{len(self.router.split_ids)}"
        if child_id in self.experts:
            raise ValueError(f"expert already exists: {child_id}")
        parent = self.experts[parent_id]
        child = clone_module(parent)
        centroids = precomputed_prototypes if precomputed_prototypes is not None else cosine_kmeans_2(perceptions)
        self.experts[child_id] = child
        split_router = self.router.add_split(parent_id, child_id, split_id, centroids)
        self.parent_by_child[child_id] = parent_id
        self.split_by_child[child_id] = split_id
        self.next_expert_index += 1
        self.last_route_counts[child_id] = 0
        self.last_perceptions[child_id] = []
        return child_id, parent, split_router, centroids

    @torch.no_grad()
    def remove_birth(self, child_id: str, split_id: str, previous_structure: dict[str, Any]) -> None:
        if child_id not in self.experts:
            return
        del self.experts[child_id]
        self.router.split_routers.pop(split_id, None)
        self.router.restore_structure(previous_structure)
        self.parent_by_child.pop(child_id, None)
        self.split_by_child.pop(child_id, None)
        self.last_route_counts.pop(child_id, None)
        self.last_perceptions.pop(child_id, None)

    def route(
        self,
        perception: torch.Tensor,
        *,
        merge_back_child: str | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        gates_by_id, root_indices, choices, root_probabilities = (
            self.router.route_with_details(perception)
        )
        if merge_back_child is not None and merge_back_child in self.parent_by_child:
            parent_id = self.parent_by_child[merge_back_child]
            child_gate = gates_by_id[merge_back_child]
            gates_by_id[parent_id] = gates_by_id[parent_id] + child_gate
            gates_by_id[merge_back_child] = torch.zeros_like(child_gate)
        gates = torch.stack([gates_by_id[expert_id] for expert_id in self.expert_ids], dim=-1)
        self._record_routes(gates_by_id, perception)
        return gates, root_indices, root_probabilities, choices, gates_by_id

    def forward(
        self,
        perception: torch.Tensor,
        *,
        backend: str = "masked_dense",
        merge_back_child: str | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        gates, root_indices, root_probabilities, choices, gates_by_id = self.route(
            perception, merge_back_child=merge_back_child
        )
        if backend not in ("masked_dense", "sparse_dispatch"):
            raise ValueError(f"unknown execution backend: {backend}")
        output = torch.zeros_like(perception)
        if backend == "masked_dense":
            for index, expert_id in enumerate(self.expert_ids):
                output = output + self.experts[expert_id](perception) * gates[..., index, None]
        else:
            flat_inputs = perception.reshape(-1, perception.shape[-1])
            flat_gates = gates.reshape(-1, gates.shape[-1])
            flat_output = torch.zeros_like(flat_inputs)
            for index, expert_id in enumerate(self.expert_ids):
                active = flat_gates[:, index].detach().bool().nonzero(as_tuple=False).squeeze(-1)
                if active.numel() == 0:
                    continue
                contribution = self.experts[expert_id](flat_inputs.index_select(0, active))
                contribution = contribution * flat_gates.index_select(0, active)[:, index, None]
                flat_output.index_add_(0, active, contribution)
            output = flat_output.view_as(perception)
        return output, gates, root_indices, root_probabilities, choices, gates_by_id

    def split_diagnostics(self, telemetry: GrowthStageTelemetry) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for choices in telemetry.split_choices:
            for split_id, indices in choices.items():
                counts = torch.bincount(indices.reshape(-1), minlength=2).float()
                probabilities = counts / counts.sum().clamp_min(1)
                entropy = float(-(probabilities[probabilities > 0] * probabilities[probabilities > 0].log()).sum())
                current = result.setdefault(split_id, {"left": 0.0, "right": 0.0, "entropy": 0.0, "updates": 0.0})
                current["left"] += float(counts[0])
                current["right"] += float(counts[1])
                current["entropy"] += entropy
                current["updates"] += 1
        for row in result.values():
            row["entropy"] /= max(1.0, row["updates"])
        return result


class GrowthNCAStage(nn.Module):
    def __init__(self, source: UpcycledNCAStage, stage: int) -> None:
        super().__init__()
        self.iterations = source.iterations
        self.norm_attention = copy.deepcopy(source.norm_attention)
        self.norm_ffn = copy.deepcopy(source.norm_ffn)
        self.attention = copy.deepcopy(source.attention)
        self.step_embedding = copy.deepcopy(source.step_embedding)
        self.gru = copy.deepcopy(source.gru)
        self.program_bank = GrowthProgramBank(source, stage)
        self.config = source.config

    def forward(
        self,
        state: torch.Tensor,
        *,
        backend: str,
        return_debug: bool = False,
        merge_back_child: str | None = None,
    ) -> tuple[torch.Tensor, GrowthStageTelemetry]:
        batch, length, dim = state.shape
        telemetry = GrowthStageTelemetry([], [], [], [], [], [])
        for step in range(self.iterations):
            conditioned = state + self.step_embedding[step].view(1, 1, dim)
            attention_delta = self.attention(self.norm_attention(conditioned))
            perception = self.norm_ffn(state + attention_delta)
            ffn_delta, gates, root_indices, root_probabilities, choices, _ = self.program_bank(
                perception, backend=backend, merge_back_child=merge_back_child
            )
            proposal = attention_delta + ffn_delta
            state = self.gru(
                proposal.reshape(batch * length, dim), state.reshape(batch * length, dim)
            ).view(batch, length, dim)
            telemetry.gates.append(gates)
            telemetry.root_indices.append(root_indices)
            telemetry.root_probabilities.append(root_probabilities)
            telemetry.split_choices.append(choices)
            if return_debug:
                telemetry.perceptions.append(perception.detach().clone())
                telemetry.states.append(state.detach().clone())
        return state, telemetry


class ProgressiveGrowthCLM(nn.Module):
    """CLM-0.3 organism with scheduled, explicit progressive births."""

    def __init__(
        self,
        source: UpcycledCellularTextNCA | TextNCALM,
        *,
        base_model_sha256: str | None = None,
    ) -> None:
        super().__init__()
        if isinstance(source, TextNCALM):
            source = convert_textnca_to_upcycled(source)
        if not isinstance(source, UpcycledCellularTextNCA):
            raise TypeError("source must be TextNCALM or UpcycledCellularTextNCA")
        if source.config.num_experts != 4 or source.config.top_k != 1:
            raise ValueError("CLM-0.3 starts from exactly four top-1 experts per stage")
        self.max_context = source.max_context
        self.stage_supervision = source.stage_supervision
        self.token_embedding = copy.deepcopy(source.token_embedding)
        self.position_embedding = copy.deepcopy(source.position_embedding)
        self.stages = nn.ModuleList([GrowthNCAStage(stage, index) for index, stage in enumerate(source.stages)])
        self.final_norm = copy.deepcopy(source.final_norm)
        self.lm_head = copy.deepcopy(source.lm_head)
        self.base_model_sha256 = base_model_sha256
        self.growth_history: list[dict[str, Any]] = []
        self._lineages = [
            Lineage(f"s{stage}-e{expert}", stage, None, 0, 0, "clm-0.1")
            for stage in range(len(self.stages)) for expert in range(4)
        ]

    @classmethod
    def from_clm01_release(cls, path: str, *, device: str | torch.device = "cpu") -> "ProgressiveGrowthCLM":
        from .clm_release import MODEL_FORMAT, build_release_model
        from .growth_checkpoint import verify_base_release_hash
        from pathlib import Path
        root = Path(path)
        observed = verify_base_release_hash(root / "model.pt")
        checkpoint = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
        if checkpoint.get("format") != MODEL_FORMAT:
            raise RuntimeError(f"unsupported CLM-0.1 model format: {checkpoint.get('format')!r}")
        released = build_release_model(
            num_experts=int(checkpoint["num_experts"]),
            router_scale=float(checkpoint["router_scale"]),
        )
        released.load_state_dict(checkpoint["model_state"], strict=True)
        return cls(released.to(device), base_model_sha256=observed)

    @property
    def expert_count(self) -> int:
        return sum(len(stage.program_bank.expert_ids) for stage in self.stages)

    def expert_counts_by_stage(self) -> list[int]:
        return [len(stage.program_bank.expert_ids) for stage in self.stages]

    def lineage_metadata(self) -> list[dict[str, Any]]:
        return [lineage.to_dict() for lineage in self._lineages]

    def growth_structure(self) -> dict[str, Any]:
        return {
            "stages": [stage.program_bank.router.structure() for stage in self.stages],
            "next_expert_indices": [stage.program_bank.next_expert_index for stage in self.stages],
        }

    def restore_growth_structure(
        self, structure: dict[str, Any], lineages: list[dict[str, Any]] | None = None
    ) -> None:
        if len(structure.get("stages", [])) != len(self.stages):
            raise ValueError("growth checkpoint stage count mismatch")
        # The history contains the geometry prototypes needed to reconstruct
        # every dynamic module before strict state loading.
        for event in self.growth_history:
            stage = self.stages[int(event["stage"])]
            child = str(event["child"])
            if child not in stage.program_bank.experts:
                stage.program_bank.add_birth(
                    str(event["parent"]),
                    torch.tensor(event["split_prototypes"]),
                    child_id=child,
                    split_id=str(event["split_id"]),
                    precomputed_prototypes=torch.tensor(event["split_prototypes"]),
                )
        for index, (stage, saved) in enumerate(zip(self.stages, structure["stages"])):
            stage.program_bank.router.restore_structure(saved)
            if index < len(structure.get("next_expert_indices", [])):
                stage.program_bank.next_expert_index = int(structure["next_expert_indices"][index])
        if lineages is not None:
            self._lineages = [Lineage(**item) for item in lineages]

    def _stats(self, telemetry: list[GrowthStageTelemetry]) -> GrowthStats:
        usage: dict[str, torch.Tensor] = {}
        route_counts: dict[str, int] = {}
        for stage in self.stages:
            bank = stage.program_bank
            all_gates = [item for item in telemetry[bank.stage].gates]
            flat = torch.cat([item.reshape(-1, item.shape[-1]) for item in all_gates])
            for index, expert_id in enumerate(bank.expert_ids):
                usage[expert_id] = flat[:, index].detach().float().mean().cpu()
                route_counts[expert_id] = bank.last_route_counts.get(expert_id, 0)
        split_entropy: dict[str, float] = {}
        for stage, item in zip(self.stages, telemetry):
            for split_id, values in stage.program_bank.split_diagnostics(item).items():
                split_entropy[split_id] = values["entropy"]
        root_usage = tuple(
            torch.cat(
                [item.reshape(-1, item.shape[-1]) for item in stage_telemetry.root_probabilities]
            ).mean(0)
            for stage_telemetry in telemetry
        )
        root_routes = tuple(
            torch.stack(stage_telemetry.root_indices, dim=0).detach().cpu()
            for stage_telemetry in telemetry
        )
        return GrowthStats(usage, route_counts, split_entropy, root_usage, root_routes)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        execution_backend: str = "masked_dense",
        return_stats: bool = False,
        return_debug: bool = False,
        merge_back: tuple[int, str] | None = None,
    ) -> LanguageModelOutput | tuple[LanguageModelOutput, GrowthStats] | tuple[LanguageModelOutput, GrowthStats, tuple[torch.Tensor, ...]]:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.max_context:
            raise ValueError(f"input_ids must be [batch, <= {self.max_context}]")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        state = self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        telemetry: list[GrowthStageTelemetry] = []
        intermediate: list[torch.Tensor] = []
        for index, stage in enumerate(self.stages):
            child = merge_back[1] if merge_back is not None and merge_back[0] == index else None
            state, stage_telemetry = stage(
                state, backend=execution_backend, return_debug=return_debug,
                merge_back_child=child,
            )
            telemetry.append(stage_telemetry)
            if self.stage_supervision and index < len(self.stages) - 1:
                intermediate.append(self.lm_head(self.final_norm(state)))
        logits = self.lm_head(self.final_norm(state))
        output = LanguageModelOutput(logits, tuple([*intermediate, logits]) if self.stage_supervision else ())
        if not return_stats and not return_debug:
            return output
        stats = self._stats(telemetry)
        states = tuple(state for stage in telemetry for state in stage.states)
        return (output, stats, states) if return_debug else (output, stats)

    @torch.no_grad()
    def birth(
        self,
        *,
        stage: int,
        parent_id: str,
        routed_perceptions: torch.Tensor,
        token: int,
        validation_inputs: torch.Tensor | None = None,
        validation_targets: torch.Tensor | None = None,
        selection_method: str = "pressure",
        pressure: dict[str, float] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        from .growth_validation import capture_execution, compare_captures
        if not 0 <= stage < len(self.stages):
            raise IndexError("invalid growth stage")
        bank = self.stages[stage].program_bank
        before_structure = copy.deepcopy(bank.router.structure())
        before = capture_execution(self, validation_inputs) if validation_inputs is not None else None
        child_id, parent, split_router, prototypes = bank.add_birth(parent_id, routed_perceptions)
        if optimizer is not None:
            add_newborn_parameters(optimizer, parent.parameters(), bank.experts[child_id].parameters())
            # A split router intentionally starts with a fresh optimizer state.
            add_fresh_parameter_group(optimizer, split_router.parameters())
        parity = compare_captures(
            before, capture_execution(self, validation_inputs), validation_targets=validation_targets
        ) if before is not None else {
            "status": "CLM_GROWTH_EQUIVALENCE",
            "ppl_ratio": 1.0,
            "max_logits_abs_diff": 0.0,
            "max_recurrent_state_abs_diff": 0.0,
            "non_parent_root_routes_unchanged": True,
            "child_parameters_equal_parent": True,
        }
        event = {
            "birth_index": len(self.growth_history) + 1,
            "token": int(token),
            "stage": int(stage),
            "parent": parent_id,
            "child": child_id,
            "split_id": bank.split_by_child[child_id],
            "parent_generation": next(item.generation for item in self._lineages if item.expert_id == parent_id),
            "child_generation": next(item.generation for item in self._lineages if item.expert_id == parent_id) + 1,
            "selection_method": selection_method,
            "split_init": "geometry",
            "split_prototypes": prototypes.cpu().tolist(),
            "pressure": pressure or {},
            "parity": parity,
        }
        if parity["status"] != "CLM_GROWTH_EQUIVALENCE":
            bank.remove_birth(child_id, bank.split_by_child[child_id], before_structure)
            raise RuntimeError("CLM growth equivalence failed; replicate aborted")
        event["parity"]["child_parameters_equal_parent"] = all(
            torch.equal(parent_parameter, child_parameter)
            for parent_parameter, child_parameter in zip(
                bank.experts[parent_id].parameters(), bank.experts[child_id].parameters()
            )
        )
        if not event["parity"]["child_parameters_equal_parent"]:
            bank.remove_birth(child_id, bank.split_by_child[child_id], before_structure)
            raise RuntimeError("CLM growth child initialization failed; replicate aborted")
        self._lineages.append(Lineage(child_id, stage, parent_id, int(token), event["child_generation"]))
        self.growth_history.append(event)
        return event


def build_progressive_growth_model(
    source: UpcycledCellularTextNCA | TextNCALM, *, base_model_sha256: str | None = None
) -> ProgressiveGrowthCLM:
    return ProgressiveGrowthCLM(source, base_model_sha256=base_model_sha256)


# Friendly public aliases used by experiment notebooks.
CLMProgressiveGrowth = ProgressiveGrowthCLM
build_growth_model = build_progressive_growth_model
