"""Copy-on-write Shadow Cell primitives for Validation 001 v2.

The implementation is intentionally a sidecar.  It reads the accepted model's
final features and adds a candidate contribution to the logits; it never joins
the accepted model's Top-K Cell router.  This keeps the architectural claim
under test separate from routing/read-ownership changes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .clm04mini.examples import ScoredTokenExample, collate_scored
from .clm04mini.model import TinyCLMDecoder
if TYPE_CHECKING:
    from .clm04mini.tokenizer import TokenizerBundle


GateMode = Literal["task_id", "input_only", "zero"]
MATURITY_GRID: tuple[float, ...] = (0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0)
SHADOW_ZERO_TOLERANCE = 1e-6
REALIZED_DELTA_TOLERANCE = 1e-6


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes()


def hash_accepted_state(model: nn.Module) -> str:
    """Hash parameter and buffer names, metadata, dtype, shape, and values."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


class AcceptedModelSnapshot:
    """Immutable-state witness used as a hard validity gate."""

    def __init__(self, model: nn.Module) -> None:
        self.hash_before = hash_accepted_state(model)

    def hash_state(self, model: nn.Module) -> str:
        return hash_accepted_state(model)

    def assert_unchanged(self, model: nn.Module) -> None:
        after = self.hash_state(model)
        if after != self.hash_before:
            raise AssertionError(
                "accepted model mutated during Shadow training: "
                f"before={self.hash_before} after={after}"
            )


class ShadowCell(nn.Module):
    """A full-width native linear Cell with exactly neutral birth."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.operator = nn.Linear(int(width), int(width), bias=False)
        nn.init.zeros_(self.operator.weight)

    def forward(
        self,
        hidden: torch.Tensor,
        maturity: float = 1.0,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.operator(hidden) * float(maturity)
        if gate is not None:
            while gate.ndim < output.ndim:
                gate = gate.unsqueeze(-1)
            output = output * gate
        return output

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.parameters())


class InputOnlyShadowGate(nn.Module):
    """Small input-only admission gate using accepted hidden state only."""

    def __init__(self, width: int, *, temperature: float = 4.0) -> None:
        super().__init__()
        self.key = nn.Parameter(torch.empty(int(width)))
        self.bias = nn.Parameter(torch.zeros(()))
        self.temperature = float(temperature)
        nn.init.normal_(self.key, mean=0.0, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        query = hidden.mean(dim=1)
        query = F.normalize(query, dim=-1, eps=1e-8)
        key = F.normalize(self.key, dim=-1, eps=1e-8)
        return (self.temperature * (query * key).sum(dim=-1) + self.bias).sigmoid()


class AcceptedModelChain(nn.Module):
    """Immutable accepted model plus previously committed sidecars."""

    def __init__(self, base: TinyCLMDecoder) -> None:
        super().__init__()
        self.base = base
        self.committed_shadows = nn.ModuleList()
        self.committed_gates = nn.ModuleList()
        self.committed_maturities: list[float] = []
        self.committed_gate_modes: list[GateMode] = []

    @property
    def cfg(self):
        return self.base.cfg

    @property
    def token_embedding(self):
        return self.base.token_embedding

    def forward_features(self, input_ids: torch.Tensor, address_ids: list[str]) -> torch.Tensor:
        return self.base.forward_features(input_ids, address_ids)

    def _committed_gate(self, index: int, hidden: torch.Tensor, address_ids: list[str]) -> torch.Tensor:
        mode = self.committed_gate_modes[index]
        if mode == "zero":
            return hidden.new_ones(hidden.size(0))
        if mode == "task_id":
            return hidden.new_tensor([
                float(str(address).startswith(("math/", "story/", "v2/math/", "v2/story/")))
                for address in address_ids
            ])
        return self.committed_gates[index](hidden)

    def forward(self, input_ids: torch.Tensor, address_ids: list[str]) -> torch.Tensor:
        with torch.no_grad():
            hidden = self.forward_features(input_ids, address_ids)
            logits = F.linear(hidden, self.token_embedding.weight)
            for index, shadow in enumerate(self.committed_shadows):
                gate = self._committed_gate(index, hidden, address_ids)
                contribution = shadow(hidden) * self.committed_maturities[index] * gate[:, None, None]
                logits = logits + F.linear(contribution, self.token_embedding.weight)
        return logits

    def base_routes(self, address_id: str | int) -> dict[str, list[int]]:
        return self.base.base_routes(address_id)

    def append(self, sidecar: "ShadowSidecar", maturity: float) -> "AcceptedModelChain":
        clone = AcceptedModelChain(self.base)
        clone.committed_shadows.extend([*self.committed_shadows])
        clone.committed_gates.extend([*self.committed_gates])
        clone.committed_maturities = list(self.committed_maturities)
        clone.committed_gate_modes = list(self.committed_gate_modes)
        clone.committed_shadows.append(sidecar.shadow)
        clone.committed_gates.append(sidecar.input_gate)
        clone.committed_maturities.append(float(maturity))
        clone.committed_gate_modes.append(sidecar.gate_mode)
        for parameter in clone.parameters():
            parameter.requires_grad_(False)
        clone.eval()
        return clone


class ShadowSidecar(nn.Module):
    """Accepted model plus an isolated Shadow contribution.

    ``accepted`` is a frozen reference.  ``forward`` computes the accepted
    logits exactly as usual and adds ``maturity * gate * Shadow(hidden)`` after
    the accepted output projection.  Consequently no Shadow parameter is ever
    a candidate in the accepted global router.
    """

    def __init__(self, accepted: nn.Module, *, gate_mode: GateMode = "input_only") -> None:
        super().__init__()
        self.accepted = accepted
        for parameter in self.accepted.parameters():
            parameter.requires_grad_(False)
        self.accepted.eval()
        width = int(accepted.cfg.d_model)
        self.shadow = ShadowCell(width)
        self.input_gate = InputOnlyShadowGate(width)
        self.gate_mode: GateMode = gate_mode
        self.new_domain_prefixes: tuple[str, ...] = ("math/", "story/", "v2/math/", "v2/story/")

    @property
    def shadow_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.shadow.parameters())

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.shadow.parameters()) + tuple(self.input_gate.parameters())

    def gate(self, hidden: torch.Tensor, address_ids: list[str], *, is_new: bool | None) -> torch.Tensor:
        if self.gate_mode == "zero":
            return hidden.new_ones(hidden.size(0))
        if self.gate_mode == "task_id":
            if is_new is not None:
                return hidden.new_full((hidden.size(0),), float(is_new))
            return hidden.new_tensor(
                [float(any(str(address).startswith(prefix) for prefix in self.new_domain_prefixes))
                 for address in address_ids]
            )
        return self.input_gate(hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        address_ids: list[str],
        *,
        maturity: float = 1.0,
        is_new: bool | None = None,
        return_gate: bool = False,
        gate_override: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if len(address_ids) != input_ids.size(0):
            raise ValueError("address_ids must align with input batch")
        with torch.no_grad():
            hidden = self.accepted.forward_features(input_ids, address_ids)
            accepted_logits = self.accepted(input_ids, address_ids)
        gate = self.gate(hidden, address_ids, is_new=is_new) if gate_override is None else gate_override.to(hidden)
        if gate.ndim != 1 or gate.size(0) != input_ids.size(0):
            raise ValueError("gate_override must have shape [batch]")
        contribution = self.shadow(hidden, maturity=float(maturity), gate=gate)
        logits = accepted_logits + F.linear(contribution, self.accepted.token_embedding.weight)
        if return_gate:
            return logits, gate
        return logits


def m0_equivalence_delta(
    sidecar: ShadowSidecar,
    input_ids: torch.Tensor,
    address_ids: list[str],
) -> float:
    with torch.no_grad():
        accepted = sidecar.accepted(input_ids, address_ids)
        candidate = sidecar(input_ids, address_ids, maturity=0.0)
        return float((accepted - candidate).abs().max().cpu())


def routing_signature(model: TinyCLMDecoder, address_ids: Iterable[str]) -> dict[str, dict[str, list[int]]]:
    return {str(address): model.base_routes(address) for address in address_ids}


def routing_is_preserved(
    accepted: TinyCLMDecoder, sidecar: ShadowSidecar, address_ids: Iterable[str]
) -> bool:
    # The sidecar has no route table by construction; this explicit comparison
    # is retained as a machine-checkable read-ownership invariant.
    addresses = list(address_ids)
    return routing_signature(accepted, addresses) == routing_signature(sidecar.accepted, addresses)


def _masked_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    selected = values[mask]
    if selected.numel() == 0:
        raise ValueError("no scored targets")
    return selected.mean()


def _mean_nll(
    sidecar: ShadowSidecar,
    examples: list[ScoredTokenExample],
    tokenizer: TokenizerBundle,
    device: torch.device,
    *,
    maturity: float,
    is_new: bool | None,
    batch_size: int,
    gate_overrides: Mapping[str, float] | None = None,
) -> float:
    if not examples:
        return 0.0
    sidecar.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(examples), int(batch_size)):
            batch = examples[start : start + int(batch_size)]
            x, y, mask, addresses = collate_scored(batch, pad_id=tokenizer.pad_id, device=device)
            override = None
            if gate_overrides is not None:
                override = torch.tensor(
                    [float(gate_overrides[address]) for address in addresses],
                    dtype=torch.float32,
                    device=device,
                )
            logits = sidecar(x, addresses, maturity=maturity, is_new=is_new, gate_override=override)
            per_token = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
            ).reshape_as(y)
            total += float(per_token[mask].sum().cpu())
            count += int(mask.sum())
    return total / float(max(1, count))


def evaluate_sidecar_metrics(
    sidecar: ShadowSidecar,
    examples: list[ScoredTokenExample],
    tokenizer: TokenizerBundle,
    device: torch.device,
    *,
    maturity: float,
    is_new: bool | None,
    batch_size: int = 32,
    gate_overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Evaluate a sidecar with the selected maturity and optional gate values."""
    if not examples:
        raise ValueError("cannot evaluate an empty sidecar example set")
    sidecar.eval()
    nll_total = 0.0
    correct = 0
    tokens = 0
    with torch.no_grad():
        for start in range(0, len(examples), int(batch_size)):
            batch = examples[start : start + int(batch_size)]
            x, y, mask, addresses = collate_scored(batch, pad_id=tokenizer.pad_id, device=device)
            override = None
            if gate_overrides is not None:
                override = torch.tensor(
                    [float(gate_overrides[address]) for address in addresses],
                    dtype=torch.float32,
                    device=device,
                )
            logits = sidecar(
                x, addresses, maturity=float(maturity), is_new=is_new, gate_override=override
            )
            values = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
            ).reshape_as(y)
            nll_total += float(values[mask].sum().cpu())
            correct += int((logits.argmax(dim=-1)[mask] == y[mask]).sum().cpu())
            tokens += int(mask.sum())
    return {"nll": nll_total / float(max(1, tokens)), "accuracy": correct / float(max(1, tokens))}


def evaluate_model_metrics(
    model: TinyCLMDecoder | AcceptedModelChain,
    examples: list[ScoredTokenExample],
    tokenizer: TokenizerBundle,
    device: torch.device,
    *,
    batch_size: int = 32,
) -> dict[str, float]:
    """Return behavioral NLL and answer-token accuracy for an accepted model."""
    if not examples:
        raise ValueError("cannot evaluate an empty example set")
    model.eval()
    nll_total = 0.0
    correct = 0
    tokens = 0
    with torch.no_grad():
        for start in range(0, len(examples), int(batch_size)):
            batch = examples[start : start + int(batch_size)]
            x, y, mask, addresses = collate_scored(batch, pad_id=tokenizer.pad_id, device=device)
            logits = model(x, addresses)
            values = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
            ).reshape_as(y)
            nll_total += float(values[mask].sum().cpu())
            prediction = logits.argmax(dim=-1)
            correct += int((prediction[mask] == y[mask]).sum().cpu())
            tokens += int(mask.sum())
    return {"nll": nll_total / float(max(1, tokens)), "accuracy": correct / float(max(1, tokens))}


def interpolate_models(
    parent: TinyCLMDecoder,
    direct: TinyCLMDecoder,
    maturity: float,
) -> TinyCLMDecoder:
    """Build the registered Direct-Interp control without mutating either endpoint."""
    if parent.cfg.to_dict() != direct.cfg.to_dict():
        raise ValueError("Direct-Interp endpoints must share one model configuration")
    candidate = TinyCLMDecoder(parent.cfg)
    parent_state = parent.state_dict()
    direct_state = direct.state_dict()
    value = float(maturity)
    state = {
        key: parent_state[key].detach().cpu() + value * (
            direct_state[key].detach().cpu() - parent_state[key].detach().cpu()
        )
        for key in parent_state
    }
    candidate.load_state_dict(state, strict=True)
    return candidate


def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().float().cpu()
    labels = labels.detach().long().cpu()
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    ordered_labels = labels[order]
    true_positive = torch.cumsum((ordered_labels == 1).float(), dim=0)
    false_positive = torch.cumsum((ordered_labels == 0).float(), dim=0)
    return float((true_positive[ordered_labels == 0] / positives).sum() / negatives)


def _gate_features(
    accepted: nn.Module,
    examples: list[ScoredTokenExample],
    tokenizer: TokenizerBundle,
    device: torch.device,
    *,
    batch_size: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    accepted.eval()
    with torch.no_grad():
        for start in range(0, len(examples), int(batch_size)):
            batch = examples[start : start + int(batch_size)]
            x, _, _, addresses = collate_scored(batch, pad_id=tokenizer.pad_id, device=device)
            hidden = accepted.forward_features(x, addresses).mean(dim=1)
            rows.append(hidden.detach())
    if not rows:
        raise ValueError("gate calibration requires examples")
    return torch.cat(rows, dim=0)


def calibrate_input_gate(
    sidecar: ShadowSidecar,
    old_calibration: list[ScoredTokenExample],
    new_calibration: list[ScoredTokenExample],
    old_eval: list[ScoredTokenExample],
    new_eval: list[ScoredTokenExample],
    tokenizer: TokenizerBundle,
    device: torch.device,
    *,
    steps: int = 200,
    batch_size: int = 128,
    seed: int = 0,
    learning_rate: float = 5e-2,
    weight_decay: float = 1e-4,
) -> dict[str, Any]:
    """Fit only the input gate on calibration splits and report held-out AUC."""
    if sidecar.gate_mode != "input_only":
        return {"gate_calibration_steps": 0, "gate_calibration_examples": 0, "gate_auc": 1.0}
    old = _gate_features(sidecar.accepted, old_calibration, tokenizer, device, batch_size=batch_size)
    new = _gate_features(sidecar.accepted, new_calibration, tokenizer, device, batch_size=batch_size)
    features = torch.cat([old, new], dim=0)
    labels = torch.cat([torch.zeros(old.size(0), device=device), torch.ones(new.size(0), device=device)])
    optimizer = torch.optim.AdamW(sidecar.input_gate.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    sidecar.input_gate.train()
    for _ in range(int(steps)):
        if features.size(0) <= int(batch_size):
            indices = torch.arange(features.size(0), device=device)
        else:
            indices = torch.randperm(features.size(0), generator=generator)[: int(batch_size)].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = sidecar.input_gate(features[indices].unsqueeze(1))
        loss = F.binary_cross_entropy(logits, labels[indices])
        loss.backward()
        optimizer.step()
    old_eval_features = _gate_features(sidecar.accepted, old_eval, tokenizer, device, batch_size=batch_size)
    new_eval_features = _gate_features(sidecar.accepted, new_eval, tokenizer, device, batch_size=batch_size)
    with torch.no_grad():
        scores = torch.cat([
            sidecar.input_gate(old_eval_features.unsqueeze(1)),
            sidecar.input_gate(new_eval_features.unsqueeze(1)),
        ])
    eval_labels = torch.cat([
        torch.zeros(old_eval_features.size(0), device=device),
        torch.ones(new_eval_features.size(0), device=device),
    ])
    return {
        "gate_calibration_steps": int(steps),
        "gate_calibration_examples": int(features.size(0)),
        "gate_auc": _roc_auc(scores, eval_labels),
    }


def build_activation_certificates(
    model: TinyCLMDecoder,
    historical_examples: list[ScoredTokenExample],
    tokenizer: TokenizerBundle,
    device: torch.device,
    *,
    batch_size: int = 32,
    rank: int = 4,
) -> dict[int, torch.Tensor]:
    """Build incoming-activation row-span certificates from retained history."""
    if not historical_examples:
        raise ValueError("historical certificate requires examples")
    captures: dict[str, list[torch.Tensor]] = {}
    hooks = []
    for name, module in model.named_modules():
        if not name.startswith("blocks.") or ".ff.base_cells." not in name:
            continue
        if not isinstance(module, nn.Linear):
            continue
        hooks.append(module.register_forward_pre_hook(
            lambda current, inputs, module_name=name: captures.setdefault(module_name, []).append(
                inputs[0].detach().reshape(-1, inputs[0].size(-1)).cpu()
            )
        ))
    try:
        with torch.no_grad():
            for start in range(0, len(historical_examples), int(batch_size)):
                batch = historical_examples[start : start + int(batch_size)]
                x, _, _, addresses = collate_scored(batch, pad_id=tokenizer.pad_id, device=device)
                model.forward_features(x, addresses)
    finally:
        for hook in hooks:
            hook.remove()
    certificates: dict[int, torch.Tensor] = {}
    for name, module in model.named_modules():
        if name not in captures or not isinstance(module, nn.Linear):
            continue
        rows = torch.cat(captures[name], dim=0)
        if rows.numel() == 0:
            certificates[id(module.weight)] = torch.empty(0, module.weight.size(1))
            continue
        max_rank = min(int(rank), rows.size(1), rows.size(0))
        q, _ = torch.linalg.qr(rows[: max_rank * 4].T.float(), mode="reduced")
        certificates[id(module.weight)] = q[:, :max_rank].T.contiguous().to(module.weight.dtype)
        certificates[id(module.bias)] = torch.empty(0, module.bias.numel(), dtype=module.bias.dtype)
    return certificates


def train_shadow(
    sidecar: ShadowSidecar,
    examples: list[ScoredTokenExample],
    tokenizer: TokenizerBundle,
    device: torch.device,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
) -> dict[str, Any]:
    """Train only the current candidate on current-domain examples."""
    if not examples:
        raise ValueError("Shadow training requires current-domain examples")
    accepted_snapshot = AcceptedModelSnapshot(sidecar.accepted)
    sidecar.accepted.eval().requires_grad_(False)
    for parameter in sidecar.parameters():
        parameter.requires_grad_(False)
    for parameter in sidecar.shadow.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        sidecar.shadow.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    training_tokens = 0
    for _ in range(int(steps)):
        if len(examples) <= int(batch_size):
            batch = examples
        else:
            indices = torch.randperm(len(examples), generator=generator)[: int(batch_size)].tolist()
            batch = [examples[index] for index in indices]
        x, y, mask, addresses = collate_scored(batch, pad_id=tokenizer.pad_id, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = sidecar(x, addresses, maturity=1.0, is_new=True)
        loss = _masked_loss(logits, y, mask)
        loss.backward()
        optimizer.step()
        training_tokens += int(mask.sum())
    changed = sum(
        float(parameter.detach().norm().cpu())
        for parameter in sidecar.shadow.parameters()
    )
    accepted_snapshot.assert_unchanged(sidecar.accepted)
    return {
        "optimizer_steps": int(steps),
        "training_tokens": int(training_tokens),
        "historical_examples_seen_by_optimizer": 0,
        "historical_examples_seen_by_candidate_trainer": 0,
        "shadow_parameter_change_norm": changed,
        "accepted_hash_before_training": accepted_snapshot.hash_before,
        "accepted_hash_after_training": hash_accepted_state(sidecar.accepted),
    }


def train_corrected_direct(
    model: TinyCLMDecoder,
    examples: list[ScoredTokenExample],
    tokenizer: TokenizerBundle,
    device: torch.device,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    certificate_rank: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    certificates: Mapping[int, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Corrected mature-Cell baseline using projected realized AdamW updates."""
    if not examples:
        raise ValueError("direct training requires current-domain examples")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = []
    seen: set[int] = set()
    for example in examples:
        for cell_id in model.base_cell_ids(example.address_id):
            for module in model.modules_for_cell_ids([cell_id]):
                if id(module) not in seen:
                    modules.append(module)
                    seen.add(id(module))
    parameters = [parameter for module in modules for parameter in module.parameters()]
    for parameter in parameters:
        parameter.requires_grad_(True)
    states: dict[int, dict[str, torch.Tensor]] = {}
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    violations: list[float] = []
    for _ in range(int(steps)):
        if len(examples) <= int(batch_size):
            batch = examples
        else:
            indices = torch.randperm(len(examples), generator=generator)[: int(batch_size)].tolist()
            batch = [examples[index] for index in indices]
        x, y, mask, addresses = collate_scored(batch, pad_id=tokenizer.pad_id, device=device)
        model.zero_grad(set_to_none=True)
        logits = model(x, addresses)
        loss = _masked_loss(logits, y, mask)
        loss.backward()
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is None:
                    continue
                if parameter.ndim == 2:
                    if certificates is None or id(parameter) not in certificates:
                        raise ValueError("corrected direct requires a historical activation certificate")
                    certificate = certificates[id(parameter)].to(parameter.device)
                else:
                    certificate = torch.empty(0, parameter.numel(), device=parameter.device, dtype=parameter.dtype)
                proposal, state, violation = corrected_adamw_proposal(
                    parameter, parameter.grad, states.get(id(parameter)), certificate,
                    learning_rate=learning_rate, weight_decay=weight_decay,
                )
                parameter.copy_(proposal)
                states[id(parameter)] = state
                violations.append(violation)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {
        "optimizer_steps": int(steps),
        "historical_examples_seen_by_optimizer": 0,
        "realized_update_violation_max": max(violations, default=0.0),
        "mutable_parameter_count": int(sum(parameter.numel() for parameter in parameters)),
    }


def evaluate_maturity_frontier(
    accepted_model: TinyCLMDecoder | ShadowSidecar,
    shadow: ShadowSidecar | None,
    maturity_grid: Iterable[float],
    old_eval_loader: list[ScoredTokenExample],
    new_eval_loader: list[ScoredTokenExample],
    gate_mode: GateMode,
    *,
    tokenizer: TokenizerBundle,
    device: torch.device,
    baseline_old_nll: float | None = None,
    baseline_new_nll: float | None = None,
    batch_size: int = 32,
) -> list[dict[str, float]]:
    """Evaluate every registered maturity without creating learner gradients."""
    if shadow is None:
        if not isinstance(accepted_model, ShadowSidecar):
            shadow = ShadowSidecar(accepted_model, gate_mode=gate_mode)
        else:
            shadow = accepted_model
    shadow.gate_mode = gate_mode
    if baseline_old_nll is None:
        baseline_old_nll = _mean_nll(shadow, old_eval_loader, tokenizer, device,
                                     maturity=0.0, is_new=False, batch_size=batch_size)
    if baseline_new_nll is None:
        baseline_new_nll = _mean_nll(shadow, new_eval_loader, tokenizer, device,
                                     maturity=0.0, is_new=True, batch_size=batch_size)
    baseline_new_accuracy = evaluate_sidecar_metrics(
        shadow, new_eval_loader, tokenizer, device,
        maturity=0.0, is_new=True, batch_size=batch_size,
    )["accuracy"]
    frontier: list[dict[str, float]] = []
    for maturity in maturity_grid:
        old_nll = _mean_nll(shadow, old_eval_loader, tokenizer, device,
                            maturity=float(maturity), is_new=False, batch_size=batch_size)
        new_nll = _mean_nll(shadow, new_eval_loader, tokenizer, device,
                            maturity=float(maturity), is_new=True, batch_size=batch_size)
        old_metrics = evaluate_sidecar_metrics(
            shadow, old_eval_loader, tokenizer, device,
            maturity=float(maturity), is_new=False, batch_size=batch_size,
        )
        new_metrics = evaluate_sidecar_metrics(
            shadow, new_eval_loader, tokenizer, device,
            maturity=float(maturity), is_new=True, batch_size=batch_size,
        )
        frontier.append({
            "maturity": float(maturity),
            "old_nll": old_nll,
            "new_nll": new_nll,
            "old_regression": max(0.0, old_nll - float(baseline_old_nll)),
            "new_gain": float(baseline_new_nll) - new_nll,
            "old_accuracy": old_metrics["accuracy"],
            "new_accuracy": new_metrics["accuracy"],
            "accuracy_gain": new_metrics["accuracy"] - baseline_new_accuracy,
        })
    return frontier


def select_oracle_maturity(
    frontier: Iterable[Mapping[str, float]],
    max_old_regression: float,
    min_new_gain: float,
) -> float | None:
    eligible = [
        row for row in frontier
        if float(row["old_regression"]) <= float(max_old_regression)
        and float(row["new_gain"]) >= float(min_new_gain)
    ]
    if not eligible:
        return None
    selected = sorted(
        eligible,
        key=lambda row: (-float(row["new_gain"]), float(row["old_regression"]), float(row["maturity"])),
    )[0]
    return float(selected["maturity"])


@dataclass(frozen=True)
class FunctionalSketch:
    """Bounded persistent old-history state; no raw examples are retained."""

    A: torch.Tensor
    B: torch.Tensor
    sample_count: int
    sketch_rank: int

    @property
    def bytes(self) -> int:
        return int(self.A.numel() * self.A.element_size() + self.B.numel() * self.B.element_size())

    def predict_damage(self, delta_weight: torch.Tensor, maturity: float) -> float:
        weight = delta_weight.detach().float().cpu()
        a = self.A.float().cpu()
        b = self.B.float().cpu()
        middle = weight @ a @ weight.T
        return max(0.0, float((b * middle).sum()) * float(maturity) ** 2)


@torch.no_grad()
def build_functional_sketch(
    sidecar: ShadowSidecar,
    old_examples: list[ScoredTokenExample],
    tokenizer: TokenizerBundle,
    device: torch.device,
    *,
    batch_size: int = 32,
    sketch_rank: int | None = None,
) -> FunctionalSketch:
    """Compress old features to A and output sensitivity to B, then discard them."""
    features: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    for start in range(0, len(old_examples), int(batch_size)):
        batch = old_examples[start : start + int(batch_size)]
        x, _, _, addresses = collate_scored(batch, pad_id=tokenizer.pad_id, device=device)
        hidden = sidecar.accepted.forward_features(x, addresses)
        gate = sidecar.gate(hidden, addresses, is_new=False)
        features.append(hidden.reshape(-1, hidden.size(-1)).cpu())
        gates.append(gate[:, None].expand(-1, hidden.size(1)).reshape(-1).cpu())
    if not features:
        width = sidecar.accepted.cfg.d_model
        return FunctionalSketch(torch.zeros(width, width), torch.eye(width), 0, 0)
    h = torch.cat(features)
    g = torch.cat(gates).float()
    weighted = h * g[:, None]
    A = weighted.T @ weighted / float(max(1, h.size(0)))
    output_weight = sidecar.accepted.token_embedding.weight.detach().float().cpu()
    B = output_weight.T @ output_weight / float(max(1, output_weight.size(0)))
    return FunctionalSketch(A, B, int(h.size(0)), int(sketch_rank or h.size(-1)))


def select_sketch_maturity(
    shadow: ShadowSidecar,
    historical_sketch: FunctionalSketch,
    maturity_grid: Iterable[float],
    current_domain_metrics: Mapping[float, float] | Iterable[Mapping[str, float]],
    *,
    max_predicted_damage: float = 0.2,
    min_new_gain: float = 0.0,
) -> float | None:
    if isinstance(current_domain_metrics, Mapping):
        gains = {float(key): float(value) for key, value in current_domain_metrics.items()}
    else:
        gains = {float(row["maturity"]): float(row["new_gain"]) for row in current_domain_metrics}
    eligible = []
    for maturity in maturity_grid:
        value = float(maturity)
        if historical_sketch.predict_damage(shadow.shadow.operator.weight, value) <= max_predicted_damage:
            if gains.get(value, float("-inf")) >= float(min_new_gain):
                eligible.append(value)
    return max(eligible) if eligible else None


def count_false_safe(
    selected_maturity: float | None,
    frontier: Iterable[Mapping[str, float]],
    max_old_regression: float,
) -> int:
    """Count a sketch decision that its hidden historical evaluator disproves."""
    if selected_maturity is None:
        return 0
    selected = next(
        (row for row in frontier if float(row["maturity"]) == float(selected_maturity)), None
    )
    return int(selected is not None and float(selected["old_regression"]) > float(max_old_regression))


def project_realized_delta(delta: torch.Tensor, certificate: torch.Tensor) -> torch.Tensor:
    """Project an already-realized matrix update away from certificate rows."""
    if delta.ndim != 2 or certificate.ndim != 2 or delta.size(1) != certificate.size(1):
        raise ValueError("delta and certificate must be [out,in] and [rank,in]")
    if certificate.numel() == 0:
        return delta.clone()
    q = certificate.to(device=delta.device, dtype=delta.dtype)
    return delta - (delta @ q.T) @ q


def realized_update_violation(delta: torch.Tensor, certificate: torch.Tensor) -> float:
    projected = delta @ certificate.to(delta).T if certificate.numel() else delta.new_zeros(())
    return float(projected.norm() / (delta.norm() + 1e-12))


def corrected_adamw_proposal(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    state: dict[str, torch.Tensor] | None,
    certificate: torch.Tensor,
    *,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], float]:
    """Compute AdamW including decay, then project the realized proposal."""
    state = state or {
        "step": torch.zeros((), dtype=parameter.dtype, device=parameter.device),
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.zeros_like(parameter),
    }
    step = state["step"] + 1
    exp_avg = state["exp_avg"] * beta1 + gradient * (1.0 - beta1)
    exp_avg_sq = state["exp_avg_sq"] * beta2 + gradient.square() * (1.0 - beta2)
    bias1 = 1.0 - beta1 ** int(step.item())
    bias2 = 1.0 - beta2 ** int(step.item())
    normalized = (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + eps)
    raw_delta = -float(learning_rate) * normalized - float(learning_rate * weight_decay) * parameter
    projected = project_realized_delta(raw_delta, certificate) if parameter.ndim == 2 else raw_delta
    next_state = {"step": step, "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}
    return parameter + projected, next_state, realized_update_violation(projected, certificate) if parameter.ndim == 2 else 0.0


def copy_on_write_artifact(
    accepted: nn.Module,
    sidecar: ShadowSidecar,
    selected_maturity: float | None,
    path: str | Path,
    *,
    phase: str,
    arm: str,
) -> dict[str, Any]:
    """Write a new accepted artifact without modifying the parent checkpoint."""
    snapshot = AcceptedModelSnapshot(accepted)
    payload = {
        "format": "minicells.shadow-cell-validation-001-v2.accepted-artifact.v1",
        "phase": str(phase),
        "arm": str(arm),
        "parent_accepted_sha256": snapshot.hash_before,
        "accepted_state_dict": {key: value.detach().cpu().clone() for key, value in accepted.state_dict().items()},
        "committed_shadow_state_dict": {
            key: value.detach().cpu().clone() for key, value in sidecar.shadow.state_dict().items()
        },
        "input_gate_state_dict": {
            key: value.detach().cpu().clone() for key, value in sidecar.input_gate.state_dict().items()
        },
        "gate_mode": sidecar.gate_mode,
        "selected_maturity": selected_maturity,
        "parent_provenance": {
            "accepted_type": type(accepted).__name__,
            "committed_shadow_count": len(getattr(accepted, "committed_shadows", [])),
            "committed_gate_count": len(getattr(accepted, "committed_gates", [])),
            "committed_maturities": list(getattr(accepted, "committed_maturities", [])),
            "committed_gate_modes": list(getattr(accepted, "committed_gate_modes", [])),
        },
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    snapshot.assert_unchanged(accepted)
    return {"path": str(target), "parent_accepted_sha256": snapshot.hash_before}


def synthetic_examples(*, vocab_size: int, domain: str, count: int, seed: int) -> list[ScoredTokenExample]:
    """Deterministic fallback data for smoke/preflight; formal data is injectable."""
    values: list[ScoredTokenExample] = []
    for index in range(int(count)):
        base = 4 + ((index + seed) % max(4, vocab_size - 8))
        tokens = (1, base, 2 + index % 3, 3 + (index + seed) % 5, 2)
        values.append(ScoredTokenExample(
            example_id=f"{domain}-{seed}-{index}",
            address_id=f"{domain}/example-{index:04d}",
            tokens=tokens,
            target_mask=(False, True, True, True),
            prompt_text="synthetic",
            answer_text=str(tokens[-1]),
        ))
    return values
