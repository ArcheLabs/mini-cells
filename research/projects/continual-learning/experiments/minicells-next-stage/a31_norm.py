"""Observe the actual Q/K normalization boundary without blocking the GPU stream."""
from contextlib import ExitStack, contextmanager
from hashlib import sha256
import inspect
import math
from pathlib import Path
from unittest.mock import patch

import torch


FIELDS = tuple(f"{axis}_{stage}" for axis in "QK" for stage in ("PRE_NORM", "POST_NORM", "RECURRENT"))


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def describe(tensor):
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), stride=list(tensor.stride()),
                storage_offset=tensor.storage_offset(), device=str(tensor.device),
                contiguous=tensor.is_contiguous(), layout=str(tensor.layout),
                memory_format="contiguous" if tensor.is_contiguous() else "strided",
                data_ptr_alignment_256=tensor.data_ptr() % 256)


def materialize(tensor, receipt):
    cpu = tensor.detach().to("cpu", copy=True).contiguous()
    return cpu, dict(receipt, sha256=sha256(tensor_bytes(cpu)).hexdigest())


def restore(source, metadata, device):
    if sha256(tensor_bytes(source)).hexdigest() != metadata["sha256"]:
        raise ValueError("Captured PRE_NORM tensor fails SHA-256 verification")
    if list(source.shape) != metadata["shape"] or str(source.dtype) != metadata["dtype"]:
        raise ValueError("Captured tensor shape or dtype changed")
    stride = tuple(metadata["stride"])
    offset = metadata["storage_offset"]
    end = sum((n - 1) * s for n, s in zip(source.shape, stride, strict=True))
    if offset < 0 or end + offset + 1 > 32_000_000:
        raise ValueError("Captured view is outside replay storage limits")
    buffer = torch.empty(end + offset + 1, dtype=source.dtype, device=device)
    view = buffer.as_strided(source.shape, stride, offset)
    view.copy_(source)
    current = describe(view)
    for key in ("shape", "dtype", "stride", "storage_offset", "contiguous", "memory_format"):
        if current[key] != metadata[key]:
            raise RuntimeError(f"Replay changed tensor {key}")
    if sha256(tensor_bytes(view)).hexdigest() != metadata["sha256"]:
        raise RuntimeError("Replay tensor bytes changed after device transfer")
    return view


def describe_norm(fn, module):
    target = inspect.unwrap(fn)
    file = Path(inspect.getsourcefile(target))
    _, line = inspect.getsourcelines(target)
    return dict(function=f"{target.__module__}.{target.__qualname__}", filename=file.name, line=line,
                signature=str(inspect.signature(fn)), source_file_sha256=sha256(file.read_bytes()).hexdigest(),
                forward_source_sha256=sha256(inspect.getsource(module.l2norm_fwd).encode()).hexdigest(),
                normalization_dimension=-1, accumulation_dtype="float32",
                accumulation_evidence="Pinned Triton source casts loaded values to tl.float32 before tl.sum.")


def config(module, tensor):
    dimension = tensor.shape[-1]
    rows = tensor.numel() // dimension
    kernel = module.l2norm_fwd_kernel if dimension <= 512 else module.l2norm_fwd_kernel1
    best = getattr(kernel, "best_config", None)
    block = getattr(best, "kwargs", None)
    bt = block.get("BT") if block else None
    return dict(kernel=getattr(kernel.fn, "__name__", type(kernel.fn).__name__),
                D=dimension, T=rows, NB=math.ceil(rows / 2048) if dimension <= 512 else None,
                BD=min(65536 // tensor.element_size(), 1 << (dimension - 1).bit_length()),
                BT=bt, launch_grid=([math.ceil(rows / bt)] if bt else [rows] if dimension > 512 else None),
                num_warps=getattr(best, "num_warps", None), num_stages=getattr(best, "num_stages", None),
                autocast=torch.is_autocast_enabled("cuda"))


class Boundary:
    def __init__(self, fn, module):
        self.fn, self.module = fn, module
        self.active = False
        self.complete = False
        self.norm_calls = 0
        self.buffers, self.calls, self.normalized = {}, {}, {}

    def remember(self, name, tensor):
        self.buffers[name] = (tensor.detach().clone(memory_format=torch.preserve_format), describe(tensor))

    def norm(self, *args, **kwargs):
        if not self.active or self.norm_calls >= 2:
            return self.fn(*args, **kwargs)
        bound = inspect.signature(self.fn).bind(*args, **kwargs)
        bound.apply_defaults()
        tensor = next(iter(bound.arguments.values()))
        axis = "QK"[self.norm_calls]
        self.norm_calls += 1
        self.remember(f"{axis}_PRE_NORM", tensor)
        output = self.fn(*args, **kwargs)
        self.remember(f"{axis}_POST_NORM", output)
        self.normalized[axis] = output
        self.calls[axis] = dict(arguments={k: str(v) if isinstance(v, torch.dtype) else v
                                          for k, v in list(bound.arguments.items())[1:]},
                                kernel=config(self.module, tensor),
                                stream=torch.cuda.current_stream().cuda_stream if tensor.is_cuda else None)
        return output

    def recur(self, original, **kwargs):
        if self.active and "K_RECURRENT" not in self.buffers:
            for axis in "QK":
                value = kwargs[axis.lower()]
                self.remember(f"{axis}_RECURRENT", value)
                source = self.normalized[axis]
                shift = value.storage_offset() - source.storage_offset()
                stride = source.stride(0)
                routed = (source.ndim == value.ndim + 1 and source.shape[1:] == value.shape
                          and source.stride()[1:] == value.stride()
                          and source.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
                          and stride > 0 and 0 <= shift < source.shape[0] * stride and shift % stride == 0)
                self.calls[axis]["first_recurrent_routed_slot"] = shift // stride if routed else None
        return original(**kwargs)

    def export(self):
        if set(self.buffers) != set(FIELDS):
            raise RuntimeError(f"Missing first-block tensor capture: {sorted(set(FIELDS) - self.buffers.keys())}")
        tensors, metadata = {}, {}
        for name, (value, receipt) in self.buffers.items():
            tensors[name], metadata[name] = materialize(value, receipt)
        return tensors, metadata


@contextmanager
def instrument(backbone, observer):
    ops = backbone._recurrent_execution.ops
    layer = backbone.model.model.layers[0]
    original_forward = layer.forward
    original_recur = ops.fused_recurrent_gated_delta_rule

    def forward(*args, **kwargs):
        if observer.complete:
            return original_forward(*args, **kwargs)
        observer.active = True
        try:
            return original_forward(*args, **kwargs)
        finally:
            observer.active = False
            observer.complete = True

    with ExitStack() as stack:
        stack.enter_context(patch.object(layer, "forward", forward))
        stack.enter_context(patch.object(ops, "l2_norm", observer.norm))
        stack.enter_context(patch.object(ops, "fused_recurrent_gated_delta_rule",
                                         lambda **args: observer.recur(original_recur, **args)))
        yield


def metrics(reference, observed):
    if reference.shape != observed.shape or reference.dtype != observed.dtype:
        raise ValueError("Cannot compare tensors with different shapes or dtype")
    equal = tensor_bytes(reference) == tensor_bytes(observed)
    x, y = reference.double(), observed.double()
    if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise ValueError("Nonfinite normalization result")
    if equal:
        return dict(bitwise_equal=True, max_absolute=0., max_relative=0., cosine=1., first_index=None)
    delta = (x-y).abs()
    denom = x.norm() * y.norm()
    nonzero = (x.view(torch.int64) != y.view(torch.int64)).nonzero()
    return dict(bitwise_equal=False, max_absolute=float(delta.max()),
                max_relative=float((delta / torch.maximum(x.abs(), y.abs()).clamp_min(1e-12)).max()),
                cosine=float((x.flatten() @ y.flatten()) / denom) if denom else 1.,
                first_index=nonzero[0].tolist() if len(nonzero) else None)
