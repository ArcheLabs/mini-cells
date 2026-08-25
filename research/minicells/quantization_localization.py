from __future__ import annotations

import torch
from torch.nn import functional as F

Q88_FRAC_BITS = 8
PARAM_MIN = -8.0
PARAM_MAX = 8.0


def round_fixed_float(x: torch.Tensor, frac_bits: int) -> torch.Tensor:
    scale = float(1 << frac_bits)
    return torch.sign(x) * torch.floor(torch.abs(x) * scale + 0.5) / scale


def clip_param(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, PARAM_MIN, PARAM_MAX)


def q88_param(x: torch.Tensor) -> torch.Tensor:
    return round_fixed_float(clip_param(x), Q88_FRAC_BITS)


def transform_weight(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "fp32":
        return x
    if mode == "clip":
        return clip_param(x)
    if mode == "q88":
        return q88_param(x)
    raise ValueError(mode)


def _linear_variant(x, weight, bias, *, weight_mode: str, linear_frac_bits: int | None):
    w = transform_weight(weight, weight_mode)
    b = transform_weight(bias, weight_mode) if bias is not None else None
    y = F.linear(x, w, b)
    return round_fixed_float(y, linear_frac_bits) if linear_frac_bits is not None else y


@torch.no_grad()
def forward_variant(
    model,
    input_ids: torch.Tensor,
    *,
    weight_mode: str = "fp32",
    linear_frac_bits: int | None = None,
    state_frac_bits: int | None = None,
) -> torch.Tensor:
    embedded = F.embedding(
        input_ids,
        transform_weight(model.embedding.weight, weight_mode),
        padding_idx=model.embedding.padding_idx,
    )
    state = torch.zeros(
        (*input_ids.shape, model.hidden_dim),
        device=input_ids.device,
        dtype=model.embedding.weight.dtype,
    )
    for _ in range(model.iterations):
        padded = F.pad(state, (0, 0, model.radius, model.radius))
        views = [
            padded[:, offset:offset + model.num_cells]
            for offset in range(2 * model.radius + 1)
        ]
        neighborhood = torch.cat(views, dim=-1)
        update_input = torch.cat((neighborhood, embedded), dim=-1)
        hidden = F.relu(_linear_variant(
            update_input,
            model.update_in.weight,
            model.update_in.bias,
            weight_mode=weight_mode,
            linear_frac_bits=linear_frac_bits,
        ))
        delta = _linear_variant(
            hidden,
            model.update_out.weight,
            model.update_out.bias,
            weight_mode=weight_mode,
            linear_frac_bits=linear_frac_bits,
        )
        state = torch.clamp(state + model.residual_scale * delta, -1.0, 1.0)
        if state_frac_bits is not None:
            state = round_fixed_float(state, state_frac_bits)
    return _linear_variant(
        state,
        model.output.weight,
        model.output.bias,
        weight_mode=weight_mode,
        linear_frac_bits=linear_frac_bits,
    )


@torch.no_grad()
def metrics_from_logits(logits, target_ids, mask):
    pred = logits.argmax(dim=-1)
    valid = mask.bool()
    token_accuracy = (pred[valid] == target_ids[valid]).float().mean().item()
    exact = (((pred == target_ids) | (~valid)).all(dim=1)).float().mean().item()
    return {
        "token_accuracy": token_accuracy,
        "exact_sequence_accuracy": exact,
    }, pred


def q88_int(x: torch.Tensor) -> torch.Tensor:
    q = torch.sign(x) * torch.floor(torch.abs(x) * 256.0 + 0.5)
    return torch.clamp(q, -2048, 2048).to(torch.int64)


@torch.no_grad()
def integer_parameters(model):
    return {
        "embedding": q88_int(model.embedding.weight.detach().cpu()),
        "update_in_w": q88_int(model.update_in.weight.detach().cpu()),
        "update_in_b": q88_int(model.update_in.bias.detach().cpu()),
        "update_out_w": q88_int(model.update_out.weight.detach().cpu()),
        "update_out_b": q88_int(model.update_out.bias.detach().cpu()),
        "output_w": q88_int(model.output.weight.detach().cpu()),
        "output_b": q88_int(model.output.bias.detach().cpu()),
    }


def round_div_256_half_away(acc: torch.Tensor) -> torch.Tensor:
    mag = torch.div(acc.abs() + 128, 256, rounding_mode="floor")
    return torch.where(acc < 0, -mag, mag)


def int_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    acc = torch.matmul(x, w.t()) + b * 256
    return round_div_256_half_away(acc)


@torch.no_grad()
def exact_integer_forward(input_ids: torch.Tensor, params) -> torch.Tensor:
    ids = input_ids.to(device="cpu", dtype=torch.long)
    batch, num_cells = ids.shape
    state = torch.zeros((batch, num_cells, 16), dtype=torch.int64)
    embedded = params["embedding"][ids]
    for _ in range(4):
        zero = torch.zeros((batch, 2, 16), dtype=torch.int64)
        padded = torch.cat((zero, state, zero), dim=1)
        neighborhood = torch.cat(
            [padded[:, offset:offset + num_cells] for offset in range(5)],
            dim=-1,
        )
        x = torch.cat((neighborhood, embedded), dim=-1)
        hidden = int_linear(x, params["update_in_w"], params["update_in_b"]).clamp(0, 32767)
        delta = int_linear(hidden, params["update_out_w"], params["update_out_b"])
        state = (state + delta).clamp(-256, 256)
    logits = int_linear(state, params["output_w"], params["output_b"])
    return logits.argmax(dim=-1)


@torch.no_grad()
def evaluate_exact_integer(model, source_batch, examples: int, chunk_size: int = 32):
    params = integer_parameters(model)
    ids = source_batch.input_ids[:examples].detach().cpu()
    target = source_batch.target_ids[:examples].detach().cpu()
    mask = source_batch.mask[:examples].detach().cpu().bool()
    pred = torch.cat(
        [
            exact_integer_forward(ids[start:start + chunk_size], params)
            for start in range(0, examples, chunk_size)
        ],
        dim=0,
    )
    token_accuracy = (pred[mask] == target[mask]).float().mean().item()
    exact = (((pred == target) | (~mask)).all(dim=1)).float().mean().item()
    return {
        "token_accuracy": token_accuracy,
        "exact_sequence_accuracy": exact,
    }, pred, params


def flatten_integer_parameters(params) -> torch.Tensor:
    ordered = [
        params["embedding"],
        params["update_in_w"],
        params["update_in_b"],
        params["update_out_w"],
        params["update_out_b"],
        params["output_w"],
        params["output_b"],
    ]
    flat = torch.cat([x.reshape(-1) for x in ordered])
    if flat.numel() != 4476:
        raise ValueError(f"expected 4476 parameters, got {flat.numel()}")
    return flat
