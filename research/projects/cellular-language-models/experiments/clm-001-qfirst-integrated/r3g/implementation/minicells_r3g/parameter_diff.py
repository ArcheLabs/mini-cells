"""Exact tensor and semantic hashes, with per-Cell numerical differences."""

import math
import torch
from .common import fingerprint


def tensors(old, new):
    if set(old) != set(new):
        raise RuntimeError("CELL_PARAMETER_KEYS_DIFFER")
    equal = changed = nonzero = total = 0
    squared = old_squared = maximum = 0.0
    details = {}
    for name in sorted(old):
        a, b = old[name].detach().cpu(), new[name].detach().cpu()
        if (
            a.shape != b.shape
            or a.dtype != b.dtype
            or not torch.isfinite(a).all()
            or not torch.isfinite(b).all()
        ):
            raise RuntimeError("INVALID_CELL_PARAMETER_TENSOR")
        identical = fingerprint(a) == fingerprint(b)
        equal += int(identical)
        changed += int(not identical)
        d = b.double() - a.double()
        count = int(torch.count_nonzero(d))
        size = a.numel()
        ss = float(d.square().sum())
        mx = float(d.abs().max()) if size else 0.0
        nonzero += count
        total += size
        squared += ss
        old_squared += float(a.double().square().sum())
        maximum = max(maximum, mx)
        details[name] = {
            "bitwise_equal": identical,
            "changed_parameter_count": count,
            "l2_difference": math.sqrt(ss),
            "max_abs_difference": mx,
            "rms_difference": math.sqrt(ss / size) if size else 0.0,
        }
    return {
        "old_parameter_hash": fingerprint(old),
        "new_parameter_hash": fingerprint(new),
        "bitwise_equal": equal == len(old),
        "identical_tensor_count": equal,
        "different_tensor_count": changed,
        "changed_parameter_count": nonzero,
        "parameter_count": total,
        "max_abs_difference": maximum,
        "l2_difference": math.sqrt(squared),
        "rms_difference": math.sqrt(squared / total) if total else 0.0,
        "relative_l2_difference": math.sqrt(squared / old_squared) if old_squared else None,
        "relative_parameter_change_fraction": nonzero / total if total else 0.0,
        "tensors": details,
    }


def cells(old_state, new_state):
    old = {k: v for k, v in old_state.items() if k.startswith("cells.")}
    new = {k: v for k, v in new_state.items() if k.startswith("cells.")}
    per_cell = []
    for i in range(6):
        prefix = "cells." + str(i) + "."
        a, b = (
            {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
            for state in (old, new)
        )
        if not a:
            raise RuntimeError("CELL_PARAMETER_GROUP_MISSING")
        per_cell.append({"cell_id": i, **tensors(a, b)})
    return {
        "all_cells_bitwise_equal": all(r["bitwise_equal"] for r in per_cell),
        "overall": tensors(old, new),
        "cells": per_cell,
    }
