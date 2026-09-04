from __future__ import annotations

from typing import Any

import torch

from minicells.moe_subexpert import MoeSubexpertError, validate_group_shapes


def identify_packed_expert_tensors(
    model: torch.nn.Module,
    layer_index: int,
) -> tuple[tuple[str, torch.nn.Parameter], tuple[str, torch.nn.Parameter]]:
    """Identify Granite gate/up and down tensors from their actual packed layout.

    Granite exposes a model-level ``intermediate_size`` that is not guaranteed to
    equal the per-expert intermediate width.  The expert width is therefore
    inferred from the two packed tensors themselves via ``validate_group_shapes``.
    """

    marker = f"layers.{layer_index}.block_sparse_moe."
    experts = int(model.config.num_local_experts)
    matches = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if marker in name and parameter.ndim == 3 and parameter.shape[0] == experts
    ]
    if len(matches) != 2:
        raise RuntimeError(
            f"expected two Granite packed expert tensors at layer {layer_index}, "
            f"found {[(name, list(parameter.shape)) for name, parameter in matches]}"
        )

    valid: list[
        tuple[
            tuple[str, torch.nn.Parameter],
            tuple[str, torch.nn.Parameter],
            int,
        ]
    ] = []
    for gate_up in matches:
        for down in matches:
            if gate_up[0] == down[0]:
                continue
            try:
                intermediate = validate_group_shapes(gate_up[1], down[1])
            except MoeSubexpertError:
                continue
            valid.append((gate_up, down, intermediate))

    if len(valid) != 1:
        details: list[dict[str, Any]] = [
            {
                "name": name,
                "shape": list(parameter.shape),
            }
            for name, parameter in matches
        ]
        raise RuntimeError(
            "could not uniquely infer Granite gate/up and down roles from packed "
            f"tensor geometry: tensors={details}, valid_orientations={len(valid)}"
        )

    gate_up, down, _intermediate = valid[0]
    return gate_up, down
