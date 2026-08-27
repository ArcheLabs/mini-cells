from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CLMv2Stats:
    local_relative_mse: torch.Tensor
    local_cosine_similarity: torch.Tensor
    balance_loss: torch.Tensor
    router_z_loss: torch.Tensor
    program_usage: torch.Tensor
    soft_program_usage: torch.Tensor
    program_coactivation: torch.Tensor
    router_logit_variance: torch.Tensor
    active_program_ratio: float
    genome_hidden_equivalent: int
    active_hidden_equivalent: int
    genome_parameters: int
    active_parameters: int
    shared_flops: int
    expert_flops: int
    receptor_flops: int
    scaffold_flops: int
    final_inference_flops: int


def linear_mlp_flops(cells: int, input_dim: int, hidden_dim: int, output_dim: int) -> int:
    return cells * 2 * (input_dim * hidden_dim + hidden_dim * output_dim)
