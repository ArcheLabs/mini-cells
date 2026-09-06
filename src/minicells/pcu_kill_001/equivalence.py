"""G0 algebra, activation, MoE, and end-to-end numerical verification."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cellular import CellPartition, CellularExpert, ExpertProjections, extract_expert_projections


@dataclass(frozen=True)
class EquivalenceMetrics:
    max_abs_error: float
    mean_abs_error: float
    relative_l2: float
    cosine: float
    top1_token_agreement: float | None = None
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        value = {
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "relative_l2": self.relative_l2,
            "cosine": self.cosine,
            "passed": self.passed,
        }
        if self.top1_token_agreement is not None:
            value["top1_token_agreement"] = self.top1_token_agreement
        return value


def compare_values(reference: Tensor, candidate: Tensor, tolerance: float = 1e-5, logits: bool = False) -> EquivalenceMetrics:
    difference = candidate.float() - reference.float()
    ref = reference.float()
    relative = float(difference.norm() / ref.norm().clamp_min(1e-12))
    cosine = float(F.cosine_similarity(candidate.float().reshape(1, -1), reference.float().reshape(1, -1)).item())
    agreement = float((candidate.argmax(-1) == reference.argmax(-1)).float().mean()) if logits else None
    passed = relative <= tolerance and (agreement is None or agreement == 1.0)
    return EquivalenceMetrics(float(difference.abs().max()), float(difference.abs().mean()), relative, cosine, agreement, passed)


def _expert_formula(projections: ExpertProjections, hidden: Tensor) -> Tensor:
    gate = F.linear(hidden, projections.gate_weight, projections.gate_bias)
    up = F.linear(hidden, projections.up_weight, projections.up_bias)
    return F.linear(F.silu(gate) * up, projections.down_weight, projections.down_bias)


@torch.no_grad()
def verify_expert_algebra(experts: nn.Module, expert_index: int, partition: CellPartition | None = None, vectors: int = 1024, seed: int = 26090501) -> EquivalenceMetrics:
    projections = extract_expert_projections(experts, expert_index)
    partition = partition or CellPartition(projections.intermediate_size, 4)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + int(expert_index))
    hidden = torch.randn(vectors, projections.hidden_size, generator=generator, dtype=torch.float32).to(projections.gate_weight.device)
    reference = _expert_formula(
        ExpertProjections(
            projections.gate_weight.float(), projections.up_weight.float(), projections.down_weight.float(),
            projections.gate_bias.float() if projections.gate_bias is not None else None,
            projections.up_bias.float() if projections.up_bias is not None else None,
            projections.down_bias.float() if projections.down_bias is not None else None,
        ), hidden,
    )
    cellular = CellularExpert(projections, partition).float()
    candidate = cellular(hidden)
    return compare_values(reference, candidate, tolerance=2e-6)


@torch.no_grad()
def verify_real_expert_activations(experts: nn.Module, expert_index: int, activations: Tensor, partition: CellPartition | None = None, tolerance: float = 2e-5) -> EquivalenceMetrics:
    projections = extract_expert_projections(experts, expert_index)
    cellular = CellularExpert(projections, partition or CellPartition(projections.intermediate_size, 4)).to(activations.device)
    reference = _expert_formula(projections, activations)
    return compare_values(reference, cellular(activations), tolerance=tolerance)


@torch.no_grad()
def verify_full_moe(original: nn.Module, cellular: nn.Module, layer_input: Tensor, tolerance: float = 2e-5) -> EquivalenceMetrics:
    """Verify the complete router+expert MoE block using its sequence contract.

    Granite's MoE block accepts ``[batch, sequence, hidden]`` and performs the
    token flattening internally before routing.  Expert-level verification is
    intentionally 2-D, but a synthetic full-MoE probe is often convenient as
    ``[tokens, hidden]``.  Promote that probe to a single sequence here rather
    than bypassing the real block contract.
    """
    if layer_input.ndim == 2:
        block_input = layer_input.unsqueeze(0)
    elif layer_input.ndim == 3:
        block_input = layer_input
    else:
        raise ValueError(
            "full MoE verification requires [tokens, hidden] or "
            "[batch, sequence, hidden] input"
        )
    reference = original(block_input)
    candidate = cellular(block_input)
    if isinstance(reference, tuple):
        reference = reference[0]
    if isinstance(candidate, tuple):
        candidate = candidate[0]
    return compare_values(reference, candidate, tolerance=tolerance)


def _get_logits(value: Any) -> Tensor:
    value = getattr(value, "logits", value)
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not isinstance(value, Tensor):
        raise TypeError("model output does not contain logits")
    return value


@torch.no_grad()
def verify_end_to_end(original: nn.Module, cellular: nn.Module, model_inputs: dict[str, Any], tolerance: float = 2e-5) -> EquivalenceMetrics:
    reference = _get_logits(original(**model_inputs))
    candidate = _get_logits(cellular(**model_inputs))
    return compare_values(reference, candidate, tolerance=tolerance, logits=True)
