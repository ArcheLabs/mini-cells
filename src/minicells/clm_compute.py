from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CLMComputeStats:
    dense_executor_flops: int
    receptor_flops: int
    active_executor_flops: int
    cell_active_ratio: float
    program_active_ratio: float
    effective_compute_ratio: float
    program_usage: torch.Tensor
    cell_usage: torch.Tensor
    program_coactivation: torch.Tensor
    activation_by_nca_step: torch.Tensor
    activation_entropy: float


@dataclass
class _StepStats:
    cell_gate: torch.Tensor
    program_gate: torch.Tensor
    dense_executor_flops: int
    receptor_flops: int
    active_executor_flops: int


def aggregate_compute_stats(steps: list[_StepStats], num_programs: int) -> CLMComputeStats:
    if not steps:
        empty = torch.zeros(num_programs)
        return CLMComputeStats(0, 0, 0, 0.0, 0.0, 0.0, empty, torch.empty(0),
                               torch.zeros(num_programs, num_programs), torch.empty(0), 0.0)
    cells = torch.cat([step.cell_gate.detach().reshape(-1) for step in steps])
    programs = torch.cat(
        [step.program_gate.detach().reshape(-1, num_programs) for step in steps], dim=0
    )
    dense = sum(step.dense_executor_flops for step in steps)
    receptor = sum(step.receptor_flops for step in steps)
    active = sum(step.active_executor_flops for step in steps)
    effective = (receptor + active) / dense if dense else 0.0
    eps = torch.finfo(cells.dtype).eps
    entropy = -(cells.clamp(eps, 1 - eps) * cells.clamp(eps, 1 - eps).log() +
                (1 - cells).clamp(eps, 1 - eps) * (1 - cells).clamp(eps, 1 - eps).log()).mean()
    by_step = torch.stack([step.cell_gate.detach().float().mean().cpu() for step in steps])
    return CLMComputeStats(
        dense, receptor, active, float(cells.float().mean()), float(programs.float().mean()),
        effective, programs.float().mean(0).cpu(), cells.float().cpu(),
        (programs.float().T @ programs.float() / programs.shape[0]).cpu(), by_step,
        float(entropy.float()),
    )
