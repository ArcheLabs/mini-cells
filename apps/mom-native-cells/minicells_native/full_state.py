"""Typed, pickle-free snapshots of the pinned FLA native Cache."""
from dataclasses import dataclass
import json
import math

import torch


MAX_TENSOR_BYTES = 150_000_000
MAX_STRUCTURE_BYTES = 500_000
FIELDS = {"recurrent_state", "conv_state", "attn_state", "ffn_state"}


def metadata(cache):
    length = len(cache)
    maximum = getattr(cache, "get_max_length", None)
    return {
        "sequence_length": cache.get_seq_length(),
        "per_layer_seen_tokens": [cache.get_seq_length(i) for i in range(length)] if length > 1
        else [cache.get_seq_length()] * length,
        "window_size": maximum() if maximum else None,
        "layer_count": length,
    }


@dataclass
class FullState:
    structure: dict
    tensors: dict

    @classmethod
    def capture(cls, backbone, cache):
        tensors = {}

        def encode(value, path):
            if torch.is_tensor(value):
                if value.layout != torch.strided:
                    raise ValueError(f"Unsupported Cache tensor layout at {path}")
                tensors[path] = value.detach().to(device="cpu", copy=True).contiguous()
                return {"tensor": path, "shape": list(value.shape), "stride": list(value.stride()),
                        "dtype": str(value.dtype), "layout": str(value.layout)}
            if value is None:
                return None
            if type(value) in (tuple, list):
                return {"kind": type(value).__name__,
                        "items": [encode(item, f"{path}.{i}") for i, item in enumerate(value)]}
            raise ValueError(f"Unsupported Cache field at {path}: {type(value).__name__}")

        layers = [{key: encode(layer[key], f"{i}.{key}") for key in sorted(layer)}
                  for i, layer in enumerate(cache.to_legacy_cache())]
        result = cls(dict(metadata(cache), memory_bank_count=backbone.bank_count, layers=layers), tensors)
        result.validate(backbone)
        return result

    def validate(self, backbone):
        info = self.structure
        if set(info) != {"sequence_length", "per_layer_seen_tokens", "window_size",
                         "layer_count", "memory_bank_count", "layers"}:
            raise ValueError("Invalid full-state metadata fields")
        n, length = info["layer_count"], info["sequence_length"]
        if (type(n) is not int or type(length) is not int or length < 0
                or n != len(info["layers"]) or n != (backbone.layer_count if length else 0)
                or info["memory_bank_count"] != backbone.bank_count
                or info["per_layer_seen_tokens"] != [length] * n or info["window_size"] is not None):
            raise ValueError("Invalid native Cache cursor, window, or coverage")
        seen = set()
        finite_checks = {}
        storage_bytes = 0

        def visit(node, path):
            nonlocal storage_bytes
            if node is None:
                return
            if "tensor" not in node:
                if set(node) != {"kind", "items"} or node["kind"] not in ("list", "tuple"):
                    raise ValueError(f"Invalid Cache structure at {path}")
                for i, item in enumerate(node["items"]):
                    visit(item, f"{path}.{i}")
                return
            if set(node) != {"tensor", "shape", "stride", "dtype", "layout"} or node["tensor"] != path:
                raise ValueError(f"Invalid tensor reference at {path}")
            seen.add(path)
            tensor = self.tensors[path]
            dims, strides = list(tensor.shape), node["stride"]
            if (dims != node["shape"] or len(strides) != len(dims)
                    or node["dtype"] != str(tensor.dtype) or node["layout"] != "torch.strided"
                    or any(type(s) is not int or s < 0 for s in strides)):
                raise ValueError(f"Invalid tensor metadata at {path}")
            span = 1
            for size, stride in sorted(zip(dims, strides), key=lambda pair: pair[1]):
                if size > 1:
                    if stride < span:
                        raise ValueError(f"Overlapping tensor layout at {path}")
                    span += (size - 1) * stride
            if span > max(1, math.prod(dims)) * 4:
                raise ValueError(f"Unbounded tensor stride at {path}")
            storage_bytes += span * tensor.element_size() if tensor.numel() else 0
            if storage_bytes > MAX_TENSOR_BYTES:
                raise ValueError("Full-state Cache exceeds the size limit")
            if tensor.is_floating_point():
                finite_checks[path] = torch.isfinite(tensor).all()

        for i, layer in enumerate(info["layers"]):
            if set(layer) != FIELDS:
                raise ValueError(f"Unexpected Cache fields at layer {i}: {sorted(layer)}")
            recurrent = layer["recurrent_state"]["items"]
            if len(recurrent) != backbone.bank_count or recurrent[-1] is None:
                raise ValueError(f"Missing recurrent bank coverage at layer {i}")
            conv = layer["conv_state"]["items"]
            if len(conv) != 3 or any(len(branch["items"]) != 2 for branch in conv):
                raise ValueError(f"Missing Q/K/V routed or shared Conv branch at layer {i}")
            for key, node in layer.items():
                visit(node, f"{i}.{key}")
            for bank, node in enumerate(recurrent):
                if node is not None:
                    tensor = self.tensors[f"{i}.recurrent_state.{bank}"]
                    if tensor.shape != backbone.recurrent_shape or tensor.dtype != torch.float32:
                        raise ValueError(f"Invalid recurrent tensor at layer {i} bank {bank}")
        if seen != set(self.tensors):
            raise ValueError("Unexpected unreferenced Cache tensors")
        if finite_checks and not bool(torch.stack(list(finite_checks.values())).all()):
            failed = next(path for path, ok in finite_checks.items() if not bool(ok))
            raise ValueError(f"Nonfinite tensor at {failed}")

    def restore(self, backbone, *, conv=True, cursor=True):
        def decode(node):
            if node is None:
                return None
            if "tensor" in node:
                tensor = self.tensors[node["tensor"]]
                if list(tensor.stride()) == node["stride"]:
                    return tensor.to(device=backbone.device, copy=True)
                target = torch.empty_strided(tensor.shape, node["stride"], dtype=tensor.dtype, device=backbone.device)
                return target.copy_(tensor)
            items = [decode(item) for item in node["items"]]
            return tuple(items) if node["kind"] == "tuple" else items

        layers = [{key: (tuple([None, None] for _ in range(3))
                        if key == "conv_state" and not conv else decode(node))
                   for key, node in layer.items()} for layer in self.structure["layers"]]
        position = self.structure["sequence_length"] if cursor else 0
        cache = backbone.cache_type.from_legacy_cache(layers, seen_tokens=position)
        expected = {k: self.structure[k] for k in metadata(cache)}
        if not cursor:
            expected.update(sequence_length=0, per_layer_seen_tokens=[0] * len(layers))
        if metadata(cache) != expected:
            raise RuntimeError("Restored Cache metadata differs from its snapshot")
        return cache

    def structure_bytes(self):
        return json.dumps(self.structure, sort_keys=True, separators=(",", ":")).encode()
