from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .language_models import LanguageModelOutput, LocalCausalSelfAttention, RMSNorm, TransformerLM, count_parameters


INITIAL_CELLS = 4
MAX_CELLS = 12
ACTIVITY_BUDGET = 3.0
ACTIVITY_RATE = 0.40
EDGE_COUPLING = 0.08
STEP_SIZE = 0.50
STABILITY_WEIGHT = 0.10
FORK_PERTURB = 0.02


@dataclass(frozen=True)
class OrganismVariant:
    code: str
    name: str
    structural_plasticity: bool


VARIANTS = (
    OrganismVariant("F", "fixed-cellular-lm", False),
    OrganismVariant("G", "growing-cellular-lm", True),
)
VARIANT_BY_CODE = {variant.code: variant for variant in VARIANTS}


@dataclass
class OrganismDiagnostics:
    alive_indices: torch.Tensor
    activity: torch.Tensor
    cell_states: torch.Tensor
    adjacency: torch.Tensor
    activity_trace: torch.Tensor
    reaction_rms_trace: torch.Tensor
    diffusion_rms_trace: torch.Tensor
    residual_trace: torch.Tensor

    @property
    def effective_active_fraction(self) -> torch.Tensor:
        weights = self.activity.float()
        participation = weights.sum(dim=-1).square() / weights.square().sum(dim=-1).clamp_min(1e-12)
        return (participation / weights.shape[-1]).mean()


@dataclass
class OrganismForward:
    output: LanguageModelOutput
    stability_loss: torch.Tensor
    diagnostics: OrganismDiagnostics | None = None


@dataclass(frozen=True)
class StructuralProbe:
    edge_utility: torch.Tensor
    pressure: torch.Tensor
    conflict: torch.Tensor
    split_direction: torch.Tensor
    loss: float


@dataclass(frozen=True)
class StructuralEvent:
    step: int
    event: str
    receiver: int | None = None
    source: int | None = None
    parent: int | None = None
    child: int | None = None
    utility_score: float | None = None
    pressure_score: float | None = None
    conflict: float | None = None

    def as_dict(self) -> dict[str, int | float | str | None]:
        return {
            "step": self.step,
            "event": self.event,
            "receiver": self.receiver,
            "source": self.source,
            "parent": self.parent,
            "child": self.child,
            "utility_score": self.utility_score,
            "pressure_score": self.pressure_score,
            "conflict": self.conflict,
        }


def variant_by_code(code: str) -> OrganismVariant:
    try:
        return VARIANT_BY_CODE[code.upper()]
    except KeyError as exc:
        raise ValueError(f"unknown organism variant: {code}") from exc


def _relative_residual(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    delta_rms = (after.float() - before.float()).square().mean().add(1e-12).sqrt()
    state_rms = before.float().square().mean().add(1e-6).sqrt().detach()
    return delta_rms / state_rms


def _initial_chain(cells: int, max_cells: int, *, device: torch.device | None = None) -> torch.Tensor:
    adjacency = torch.zeros(max_cells, max_cells, dtype=torch.bool, device=device)
    for left in range(cells - 1):
        right = left + 1
        adjacency[left, right] = True
        adjacency[right, left] = True
    return adjacency


class SharedCellRule(nn.Module):
    def __init__(self, *, dim: int, heads: int, ffn_dim: int, window: int) -> None:
        super().__init__()
        self.norm_attention = RMSNorm(dim)
        self.attention = LocalCausalSelfAttention(dim, heads, window)
        self.norm_ffn = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )
        self.memory_film = nn.Linear(dim, 2 * dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        with torch.no_grad():
            hidden = self.gru.hidden_size
            self.gru.bias_ih[hidden : 2 * hidden].fill_(1.0)
            self.gru.bias_hh[hidden : 2 * hidden].fill_(1.0)

    def forward(self, state: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch, length, cells, dim = state.shape
        scale, shift = self.memory_film(memory).chunk(2, dim=-1)
        conditioned = state * (1.0 + 0.10 * torch.tanh(scale)[None, None, :, :])
        conditioned = conditioned + 0.10 * shift[None, None, :, :]
        rows = conditioned.permute(0, 2, 1, 3).reshape(batch * cells, length, dim)
        horizontal = self.attention(self.norm_attention(rows))
        horizontal = horizontal.reshape(batch, cells, length, dim).permute(0, 2, 1, 3).contiguous()
        candidate = state + horizontal
        ffn_delta = self.ffn(self.norm_ffn(candidate))
        proposal = horizontal + ffn_delta
        proposed = self.gru(
            proposal.reshape(batch * length * cells, dim),
            state.reshape(batch * length * cells, dim),
        ).view(batch, length, cells, dim)
        return proposed - state


class GrowingCellularLM(nn.Module):
    """A language model whose cells share one genome while phenotype and graph can grow."""

    def __init__(
        self,
        *,
        vocab_size: int,
        variant: OrganismVariant,
        max_context: int = 128,
        dim: int = 96,
        heads: int = 4,
        ffn_dim: int = 384,
        iterations: int = 6,
        attention_window: int = 32,
        initial_cells: int = INITIAL_CELLS,
        max_cells: int = MAX_CELLS,
    ) -> None:
        super().__init__()
        if initial_cells < 2 or initial_cells > max_cells:
            raise ValueError("initial_cells must be in [2, max_cells]")
        self.variant = variant
        self.max_context = max_context
        self.dim = dim
        self.iterations = iterations
        self.initial_cells = initial_cells
        self.max_cells = max_cells
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_context, dim)
        self.cell_memory = nn.Parameter(torch.zeros(max_cells, dim))
        self.rule = SharedCellRule(dim=dim, heads=heads, ffn_dim=ffn_dim, window=attention_window)
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        alive = torch.zeros(max_cells, dtype=torch.bool)
        alive[:initial_cells] = True
        adjacency = _initial_chain(initial_cells, max_cells)
        self.register_buffer("alive_mask", alive)
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("protected_edges", adjacency.clone())
        parent = torch.full((max_cells,), -1, dtype=torch.long)
        parent[:initial_cells] = torch.arange(initial_cells)
        birth = torch.full((max_cells,), -1, dtype=torch.long)
        birth[:initial_cells] = 0
        self.register_buffer("parent", parent)
        self.register_buffer("birth_step", birth)
        self.apply(self._init_weights)
        nn.init.normal_(self.cell_memory, mean=0.0, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def alive_count(self) -> int:
        return int(self.alive_mask.sum().item())

    @property
    def edge_count(self) -> int:
        alive = self.alive_mask
        return int(self.adjacency[alive][:, alive].sum().item())

    def genome_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if name != "cell_memory":
                yield parameter

    def freeze_genome(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name == "cell_memory")

    @staticmethod
    def _replicator_activity(activity: torch.Tensor, reaction: torch.Tensor) -> torch.Tensor:
        drive = reaction.float().square().mean(dim=-1).add(1e-8).sqrt()
        mean = drive.mean(dim=-1, keepdim=True)
        std = drive.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        fitness = (drive - mean) / std
        growth = torch.exp(ACTIVITY_RATE * fitness).clamp_max(20.0)
        updated = activity.float() * growth
        return ACTIVITY_BUDGET * updated / updated.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _diffusion(state: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        source = torch.einsum("rs,blsd->blrd", weights, state)
        row_mass = weights.sum(dim=-1, keepdim=True).view(1, 1, weights.shape[0], 1)
        return source - row_mass * state

    def _graph_weights(self, alive_indices: torch.Tensor, edge_probe: torch.Tensor | None) -> torch.Tensor:
        adjacency = self.adjacency.index_select(0, alive_indices).index_select(1, alive_indices)
        scores = adjacency.to(dtype=self.cell_memory.dtype)
        if edge_probe is not None:
            probe = edge_probe.index_select(0, alive_indices).index_select(1, alive_indices)
            eye = torch.eye(len(alive_indices), device=probe.device, dtype=torch.bool)
            scores = scores + probe.masked_fill(eye, 0.0).to(scores.dtype)
        return EDGE_COUPLING * scores / scores.sum(dim=-1, keepdim=True).clamp_min(1.0)

    def forward_variable(
        self,
        input_ids: torch.Tensor,
        *,
        iterations: int | None = None,
        edge_probe: torch.Tensor | None = None,
        collect_observability: bool = False,
    ) -> OrganismForward:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.max_context:
            raise ValueError(f"input_ids must be [batch, <= {self.max_context}]")
        steps = self.iterations if iterations is None else int(iterations)
        if steps < 1:
            raise ValueError("iterations must be positive")
        alive_indices = torch.nonzero(self.alive_mask, as_tuple=False).flatten()
        if alive_indices.numel() < 1 or int(alive_indices[0]) != 0:
            raise RuntimeError("interface cell 0 must remain alive")
        memory = self.cell_memory.index_select(0, alive_indices)
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)
        token_state = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        state = memory[None, None, :, :].expand(batch, length, -1, -1).clone()
        state[:, :, 0, :] = state[:, :, 0, :] + token_state
        activity = torch.full(
            (batch, length, len(alive_indices)),
            ACTIVITY_BUDGET / len(alive_indices),
            device=input_ids.device,
            dtype=torch.float32,
        )
        weights = self._graph_weights(alive_indices, edge_probe).to(device=input_ids.device)
        activity_trace: list[torch.Tensor] = []
        reaction_trace: list[torch.Tensor] = []
        diffusion_trace: list[torch.Tensor] = []
        residual_trace: list[torch.Tensor] = []
        last_before = state
        for _ in range(steps):
            last_before = state
            reaction = self.rule(state, memory)
            activity = self._replicator_activity(activity, reaction)
            diffusion = self._diffusion(state, weights)
            relative = activity / (ACTIVITY_BUDGET / len(alive_indices))
            gain = relative.clamp(0.05, 3.0).unsqueeze(-1).to(state.dtype)
            state = state + STEP_SIZE * gain * (reaction + diffusion)
            if collect_observability:
                activity_trace.append(activity.detach())
                reaction_trace.append(reaction.float().square().mean().sqrt().detach())
                diffusion_trace.append(diffusion.float().square().mean().sqrt().detach())
                residual_trace.append(_relative_residual(last_before, state).detach())
        logits = self.lm_head(self.final_norm(state[:, :, 0, :]))
        stability = _relative_residual(last_before, state)
        diagnostics = None
        if collect_observability:
            diagnostics = OrganismDiagnostics(
                alive_indices=alive_indices.detach(),
                activity=activity,
                cell_states=state,
                adjacency=self.adjacency.detach().clone(),
                activity_trace=torch.stack(activity_trace),
                reaction_rms_trace=torch.stack(reaction_trace),
                diffusion_rms_trace=torch.stack(diffusion_trace),
                residual_trace=torch.stack(residual_trace),
            )
        return OrganismForward(LanguageModelOutput(logits), stability, diagnostics)

    def forward(self, input_ids: torch.Tensor) -> LanguageModelOutput:
        return self.forward_variable(input_ids).output

    @torch.no_grad()
    def connect(self, receiver: int, source: int) -> bool:
        if receiver == source or not bool(self.alive_mask[receiver]) or not bool(self.alive_mask[source]):
            return False
        if bool(self.adjacency[receiver, source]):
            return False
        self.adjacency[receiver, source] = True
        return True

    @torch.no_grad()
    def prune(self, receiver: int, source: int) -> bool:
        if not bool(self.adjacency[receiver, source]) or bool(self.protected_edges[receiver, source]):
            return False
        self.adjacency[receiver, source] = False
        return True

    @torch.no_grad()
    def fork_cell(self, parent: int, *, step: int, direction: torch.Tensor) -> int | None:
        if not self.variant.structural_plasticity or parent == 0 or not bool(self.alive_mask[parent]):
            return None
        inactive = torch.nonzero(~self.alive_mask, as_tuple=False).flatten()
        if inactive.numel() == 0:
            return None
        child = int(inactive[0].item())
        vector = direction.to(device=self.cell_memory.device, dtype=self.cell_memory.dtype)
        vector = vector / vector.norm().clamp_min(1e-8)
        center = self.cell_memory[parent].clone()
        self.cell_memory[parent].copy_(center - FORK_PERTURB * vector)
        self.cell_memory[child].copy_(center + FORK_PERTURB * vector)
        self.alive_mask[child] = True
        self.parent[child] = parent
        self.birth_step[child] = int(step)
        self.adjacency[parent, child] = True
        self.adjacency[child, parent] = True
        self.protected_edges[parent, child] = True
        self.protected_edges[child, parent] = True
        return child

    @torch.no_grad()
    def copy_tissue_from(self, donor: "GrowingCellularLM", cells: Iterable[int]) -> None:
        selected = sorted(set(int(cell) for cell in cells))
        for cell in selected:
            if cell <= 0 or cell >= self.max_cells:
                raise ValueError("transplanted cells must be non-interface cells inside organism")
            self.cell_memory[cell].copy_(donor.cell_memory[cell])
            self.alive_mask[cell] = donor.alive_mask[cell]
            self.parent[cell] = donor.parent[cell]
            self.birth_step[cell] = donor.birth_step[cell]
        boundary = sorted(set([0, *selected]))
        for receiver in boundary:
            for source in boundary:
                if receiver in selected or source in selected:
                    self.adjacency[receiver, source] = donor.adjacency[receiver, source]
                    self.protected_edges[receiver, source] = donor.protected_edges[receiver, source]


class StructuralController:
    def __init__(
        self,
        *,
        max_cells: int = MAX_CELLS,
        ema_beta: float = 0.70,
        connect_score: float = 1.0,
        prune_score: float = -0.5,
        conflict_threshold: float = 0.35,
        pressure_score: float = 0.5,
        persistence: int = 2,
        fork_cooldown: int = 2,
    ) -> None:
        self.ema_beta = ema_beta
        self.connect_score = connect_score
        self.prune_score = prune_score
        self.conflict_threshold = conflict_threshold
        self.pressure_score = pressure_score
        self.persistence = persistence
        self.fork_cooldown = fork_cooldown
        self.utility_ema = torch.zeros(max_cells, max_cells)
        self.pressure_ema = torch.zeros(max_cells)
        self.conflict_ema = torch.zeros(max_cells)
        self.connect_streak = torch.zeros(max_cells, max_cells, dtype=torch.long)
        self.prune_streak = torch.zeros(max_cells, max_cells, dtype=torch.long)
        self.probes_since_fork = fork_cooldown
        self.events: list[StructuralEvent] = []

    @staticmethod
    def _zscore(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        selected = values[mask]
        if selected.numel() < 2:
            return torch.zeros_like(values)
        mean = selected.mean()
        std = selected.std(unbiased=False).clamp_min(1e-8)
        return (values - mean) / std

    def apply(self, model: GrowingCellularLM, probe: StructuralProbe, *, step: int) -> list[StructuralEvent]:
        if not model.variant.structural_plasticity:
            return []
        alive = model.alive_mask.detach().cpu()
        utility = probe.edge_utility.detach().float().cpu()
        pressure = probe.pressure.detach().float().cpu()
        conflict = probe.conflict.detach().float().cpu()
        beta = self.ema_beta
        self.utility_ema.mul_(beta).add_(utility, alpha=1.0 - beta)
        self.pressure_ema.mul_(beta).add_(pressure, alpha=1.0 - beta)
        self.conflict_ema.mul_(beta).add_(conflict, alpha=1.0 - beta)
        candidate = alive[:, None] & alive[None, :]
        candidate.fill_diagonal_(False)
        utility_z = self._zscore(self.utility_ema, candidate)
        pressure_mask = alive.clone()
        pressure_mask[0] = False
        pressure_z = self._zscore(self.pressure_ema, pressure_mask)
        new_events: list[StructuralEvent] = []

        absent = candidate & ~model.adjacency.detach().cpu()
        self.connect_streak = torch.where(
            absent & (utility_z > self.connect_score) & (self.utility_ema > 0),
            self.connect_streak + 1,
            torch.zeros_like(self.connect_streak),
        )
        if bool((self.connect_streak >= self.persistence).any()):
            scores = utility_z.masked_fill(self.connect_streak < self.persistence, float("-inf"))
            flat = int(scores.argmax().item())
            receiver, source = divmod(flat, model.max_cells)
            if model.connect(receiver, source):
                event = StructuralEvent(step=step, event="connect", receiver=receiver, source=source, utility_score=float(utility_z[receiver, source]))
                self.events.append(event)
                new_events.append(event)
                self.connect_streak[receiver, source] = 0

        learned = model.adjacency.detach().cpu() & ~model.protected_edges.detach().cpu() & candidate
        self.prune_streak = torch.where(
            learned & (utility_z < self.prune_score) & (self.utility_ema < 0),
            self.prune_streak + 1,
            torch.zeros_like(self.prune_streak),
        )
        if bool((self.prune_streak >= self.persistence).any()):
            scores = utility_z.masked_fill(self.prune_streak < self.persistence, float("inf"))
            flat = int(scores.argmin().item())
            receiver, source = divmod(flat, model.max_cells)
            if model.prune(receiver, source):
                event = StructuralEvent(step=step, event="prune", receiver=receiver, source=source, utility_score=float(utility_z[receiver, source]))
                self.events.append(event)
                new_events.append(event)
                self.prune_streak[receiver, source] = 0

        self.probes_since_fork += 1
        connect_happened = any(event.event == "connect" for event in new_events)
        can_fork = model.alive_count < model.max_cells and self.probes_since_fork >= self.fork_cooldown
        if can_fork and not connect_happened and bool(pressure_mask.any()):
            eligible = pressure_mask & (pressure_z > self.pressure_score) & (self.conflict_ema > self.conflict_threshold)
            if bool(eligible.any()):
                candidates = torch.nonzero(eligible, as_tuple=False).flatten()
                parent = int(candidates[pressure_z[candidates].argmax()].item())
                incoming = absent[parent]
                best_edge = float(utility_z[parent][incoming].max().item()) if bool(incoming.any()) else float("-inf")
                if best_edge <= self.connect_score:
                    child = model.fork_cell(parent, step=step, direction=probe.split_direction[parent])
                    if child is not None:
                        event = StructuralEvent(
                            step=step,
                            event="fork",
                            parent=parent,
                            child=child,
                            pressure_score=float(pressure_z[parent]),
                            conflict=float(self.conflict_ema[parent]),
                            utility_score=best_edge,
                        )
                        self.events.append(event)
                        new_events.append(event)
                        self.probes_since_fork = 0
        return new_events


def make_structural_probe(model: GrowingCellularLM, microbatches: list[tuple[torch.Tensor, torch.Tensor]], *, loss_fn) -> StructuralProbe:
    if not microbatches:
        raise ValueError("microbatches must not be empty")
    device = model.cell_memory.device
    edge_probe = torch.zeros(model.max_cells, model.max_cells, device=device, requires_grad=True)
    inputs = torch.cat([item[0] for item in microbatches], dim=0)
    targets = torch.cat([item[1] for item in microbatches], dim=0)
    result = model.forward_variable(inputs, edge_probe=edge_probe)
    loss = loss_fn(result.output.logits, targets)
    edge_grad = torch.autograd.grad(loss, edge_probe, retain_graph=False)[0]
    edge_utility = -edge_grad.detach()

    memory_gradients = []
    for inputs_mb, targets_mb in microbatches:
        result_mb = model.forward_variable(inputs_mb)
        loss_mb = loss_fn(result_mb.output.logits, targets_mb)
        grad = torch.autograd.grad(loss_mb, model.cell_memory, retain_graph=False)[0]
        memory_gradients.append(grad.detach())
    gradients = torch.stack(memory_gradients, dim=0).float()
    pressure = gradients.square().mean(dim=(0, 2)).sqrt()
    summed = gradients.sum(dim=0)
    numerator = summed.square().sum(dim=-1)
    denominator = gradients.square().sum(dim=-1).sum(dim=0) * gradients.shape[0]
    conflict = (1.0 - numerator / denominator.clamp_min(1e-12)).clamp(0.0, 1.0)
    directions = torch.zeros_like(model.cell_memory.detach().float())
    for cell in range(model.max_cells):
        centered = gradients[:, cell, :] - gradients[:, cell, :].mean(dim=0, keepdim=True)
        if float(centered.square().sum()) > 1e-12:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            directions[cell] = vh[0]
        else:
            directions[cell, cell % model.dim] = 1.0
    return StructuralProbe(edge_utility.cpu(), pressure.cpu(), conflict.cpu(), directions.cpu(), float(loss.detach()))


def build_cellular_model(vocab_size: int, variant_code: str, **kwargs: object) -> GrowingCellularLM:
    return GrowingCellularLM(vocab_size=vocab_size, variant=variant_by_code(variant_code), **kwargs)


def build_parameter_matched_small_transformer(vocab_size: int, target_parameters: int, *, max_context: int = 128) -> tuple[TransformerLM, dict[str, int | float]]:
    candidates: list[tuple[float, TransformerLM, int, int, int, int]] = []
    for dim in (80, 96, 112, 128):
        heads = 4
        if dim % heads:
            continue
        for layers in range(1, 7):
            for ffn_dim in (256, 320, 384, 448, 512):
                model = TransformerLM(vocab_size=vocab_size, max_context=max_context, dim=dim, heads=heads, ffn_dim=ffn_dim, layers=layers)
                parameters = count_parameters(model)
                error = abs(parameters - target_parameters) / max(1, target_parameters)
                candidates.append((error, model, dim, layers, ffn_dim, parameters))
    error, model, dim, layers, ffn_dim, parameters = min(candidates, key=lambda item: item[0])
    return model, {
        "dim": dim,
        "layers": layers,
        "ffn_dim": ffn_dim,
        "parameters": parameters,
        "target_parameters": target_parameters,
        "relative_parameter_error": error,
    }
