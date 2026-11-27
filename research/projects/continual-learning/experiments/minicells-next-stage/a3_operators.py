"""Observe and independently replay executed nodes of the frozen first block."""
from collections import Counter
from contextlib import ExitStack, contextmanager
from functools import wraps
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

from common import write


def freeze(value, tensors, path):
    if torch.is_tensor(value):
        tensors[path] = value.detach().to("cpu", copy=True).contiguous()
        return {"tensor": path, "shape": list(value.shape), "dtype": str(value.dtype),
                "stride": list(value.stride()), "device_type": value.device.type}
    if isinstance(value, (list, tuple)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [freeze(v, tensors, f"{path}.{i}") for i, v in enumerate(value)]}
    if isinstance(value, dict):
        return {"kind": "dict", "items": {k: freeze(v, tensors, f"{path}.{k}")
                                         for k, v in value.items()}}
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, torch.dtype):
        return {"dtype_argument": str(value)}
    raise TypeError(f"Unsupported operator argument at {path}: {type(value)}")


def thaw(value, tensors, device):
    if not isinstance(value, dict):
        return value
    if "tensor" in value:
        source = tensors[value["tensor"]]
        target = device if value["device_type"] == "cuda" else "cpu"
        if list(source.stride()) == value["stride"]:
            return source.to(device=target, copy=True)
        array = torch.empty_strided(source.shape, value["stride"], dtype=source.dtype, device=target)
        return array.copy_(source)
    if "dtype_argument" in value:
        return getattr(torch, value["dtype_argument"].removeprefix("torch."))
    if value["kind"] == "dict":
        return {k: thaw(v, tensors, device) for k, v in value["items"].items()}
    items = [thaw(v, tensors, device) for v in value["items"]]
    return tuple(items) if value["kind"] == "tuple" else items


def entrypoints(backbone):
    profile = backbone._recurrent_execution
    block = backbone.model.model.layers[0]
    nodes = {
        "dispatch.stable_transform": (profile, "transform"),
        "reconstruct": (profile.ops, "reconstruct"),
        "fused_recurrent": (profile.ops, "fused_recurrent_gated_delta_rule"),
        "short_conv": (profile.convolution, "causal_conv1d"),
        "short_conv_update": (profile.convolution, "causal_conv1d_update"),
        "topk": (torch, "topk"),
        "shared_branch": (block.attn, "shared_o"),
    }
    for path, module in block.named_modules():
        if path and path != "attn" and not path.endswith(("conv1d", "k_proj", "v_proj", "a_proj", "b_proj")):
            if type(module).forward is not torch.nn.Module.forward:
                nodes[f"module.{path}"] = (module, "forward")
    return nodes


ATEN = {str(op): op for op in (torch.ops.aten.add.Tensor, torch.ops.aten.mul.Tensor,
                              torch.ops.aten.sigmoid.default, torch.ops.aten._softmax.default,
                              torch.ops.aten.softplus.default, torch.ops.aten.exp.default,
                              torch.ops.aten.neg.default, torch.ops.aten.div.Tensor,
                              torch.ops.aten.masked_fill.Scalar)}


class Observer(TorchDispatchMode):
    def __init__(self):
        self.calls = Counter()
        self.rows = []
        self.tensors = {}
        self.active = False

    def call(self, name, fn, args, kwargs, *, replayable=True, mutation=False):
        if not self.active:
            return fn(*args, **kwargs)
        key = f"{name}:{self.calls[name]}"
        self.calls[name] += 1
        identity = getattr(fn, "__qualname__", type(fn).__qualname__)
        row = dict(key=key, operator=name, kernel_identity=f"{fn.__module__}.{identity}",
                   layer=0, replayable=replayable)
        with _disable_current_modes():
            row["input"] = freeze((args, kwargs), self.tensors, f"{key}.input")
        result = fn(*args, **kwargs)
        with _disable_current_modes():
            row["output"] = freeze(result, self.tensors, f"{key}.output")
            if mutation:
                row["updated_arguments"] = freeze((args, kwargs), self.tensors, f"{key}.updated_arguments")
        self.rows.append(row)
        return result

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if name in ATEN:
            return self.call(name, func, args, kwargs or {})
        return func(*args, **(kwargs or {}))


@contextmanager
def instrument(backbone, observer):
    block = backbone.model.model.layers[0]
    original = block.forward
    with ExitStack() as scope:
        for name, (owner, attr) in entrypoints(backbone).items():
            fn = getattr(owner, attr)

            def hook(fn, name):
                @wraps(fn)
                def measured(*args, **kwargs):
                    return observer.call(name, fn, args, kwargs,
                                         replayable=name != "shared_branch",
                                         mutation=name in ("short_conv", "short_conv_update", "shared_branch"))
                return measured

            scope.enter_context(patch.object(owner, attr, hook(fn, name)))

        def forward(*args, **kwargs):
            observer.active = True
            try:
                observer.call("block.input", lambda x: x, (args[0],), {})
                with observer:
                    output = original(*args, **kwargs)
                observer.call("block.output", lambda x: x, (output[0],), {})
                return output
            finally:
                observer.active = False

        scope.enter_context(patch.object(block, "forward", forward))
        yield


def save_observations(path, observer):
    write(path.with_suffix(".json"), observer.rows)
    save_file(observer.tensors, path.with_suffix(".safetensors"))


def alignment(left, right, xs, ys, bounds, query_length=None):
    by_key = {row["key"]: row for row in right}
    first = None
    rows = []

    def metrics(a, b):
        if a.shape != b.shape or a.dtype != b.dtype:
            return {"shape_or_dtype_mismatch": True}
        if torch.equal(a, b):
            return dict(max_abs=0.0, max_relative=0.0, cosine=1.0, first_index=None, by_leading_index=[])
        x, y = a.double(), b.double()
        delta = (x - y).abs()
        token_axis = 1 if delta.ndim >= 2 and delta.shape[0] == 1 and delta.shape[1] == query_length else None
        per_token = (delta.movedim(token_axis, 0).reshape(query_length, -1).max(-1).values.tolist()
                     if token_axis is not None else None)
        norm = x.norm() * y.norm()
        return dict(max_abs=float(delta.max()), max_relative=float(
            (delta / torch.maximum(x.abs(), y.abs()).clamp_min(1e-12)).max()),
            cosine=float((x.flatten() @ y.flatten()) / norm) if norm else 0.0,
            first_index=(delta != 0).nonzero()[0].tolist(), by_query_token=per_token)

    sequence_aligned = True
    for index, row in enumerate(left):
        key = row["key"]
        sequence_aligned &= index < len(right) and key == right[index]["key"]
        other = by_key.get(key)
        if other is None:
            rows.append(dict(key=key, status="UNALIGNED"))
            continue
        summaries = {}
        at = lambda k: k.startswith(key + ".")
        for field in sorted({k for k in xs if at(k)} | {k for k in ys if at(k)}):
            summaries[field] = (metrics(xs[field], ys[field]) if field in xs and field in ys
                                else dict(shape_or_dtype_mismatch=True))
        numeric = lambda k: k.get("max_abs", float("inf"))
        inputs = {key: value for key, value in summaries.items() if ".input." in key}
        outputs = {key: value for key, value in summaries.items() if ".input." not in key}
        max_in = max((numeric(v) for v in inputs.values()), default=0.0)
        max_out = max((numeric(v) for v in outputs.values()), default=0.0)
        limit = bounds["query.layer.0"] if row["operator"] == "block.output" else 0.0
        structural_in = row.get("input") != other.get("input")
        if (first is None and sequence_aligned and row["replayable"] and not structural_in
                and max_in == 0 and max_out > limit):
            first_output = next((name for name, value in outputs.items() if numeric(value) > limit), None)
            detail = outputs[first_output]
            first = dict(operator=row["operator"], key=key, kernel_identity=row["kernel_identity"],
                         layer=0, output_error=max_out if max_out != float("inf") else None,
                         input_error=0.0, original_tolerance=limit,
                         max_relative_error=detail.get("max_relative"), first_output_field=first_output,
                         first_tensor_index=detail.get("first_index"),
                         query_token_index=next((i for i, v in enumerate(detail.get("by_query_token") or [])
                                                 if v > limit), None))
        rows.append(dict(key=key, status="UNALIGNED" if not sequence_aligned else
                         "INPUT_DIFFERENCE" if structural_in or max_in else
                         "FAIL" if max_out > limit else "PASS", operator=row["operator"],
                         original_tolerance=limit, kernel_identity=row["kernel_identity"],
                         input_structure=row.get("input"), output_structure=row["output"], tensors=summaries))
    for key in sorted(by_key.keys() - {row["key"] for row in left}):
        rows.append(dict(key=key, status="UNALIGNED"))
    return first, rows


def replay(backbone, operator, node, tensors):
    if operator in ATEN:
        fn = ATEN[operator]
    else:
        owner, name = entrypoints(backbone)[operator]
        fn = getattr(owner, name)
    args, kwargs = thaw(node, tensors, backbone.device)
    result = fn(*args, **kwargs)
    output = {}
    structure = freeze(result, output, "replay.output")
    if operator in ("short_conv", "short_conv_update"):
        return dict(result=structure, updated=freeze((args, kwargs), output, "replay.updated")), output
    return structure, output
