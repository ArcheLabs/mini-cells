"""A/B execution traces and SDK lifecycle probes for the full-state bridge."""
from contextlib import ExitStack

import torch
from safetensors.torch import load_file, save_file

from minicells_native.full_state import FullState
from common import canonical_hash, write
from m1_reference_capacity import digest_file
from m1_value_swap import step_prefix
from m1_bridge import scores


def live(backbone, writes):
    cache = backbone.stream({})
    for tokens in writes:
        step_prefix(backbone, cache, tokens)
    return cache


def trace(backbone, factory, query, *, primary_cache=None):
    cache = factory() if primary_cache is None else primary_cache
    previous = FullState.capture(backbone, cache)
    tensors = {f"pre.{key}": value for key, value in previous.tensors.items()}
    prefix = query["tokens"]
    candidate = next(iter(query["candidates"].values()))
    with ExitStack() as hooks:
        for index, layer in enumerate(backbone.model.model.layers):
            def observe(_module, _inputs, output, index=index):
                hidden = output[0] if isinstance(output, tuple) else output
                tensors[f"query.layer.{index}"] = hidden[0].detach().to("cpu", copy=True).contiguous()
            handle = layer.register_forward_hook(observe)
            hooks.callback(handle.remove)
        hidden = backbone.step(prefix + candidate[:-1], [cache], [1.0])
    tensors["query.hidden"] = hidden[0].detach().to("cpu", copy=True).contiguous()
    tensors["query.logits"] = backbone.model.lm_head(hidden[0]).detach().to("cpu", copy=True).contiguous()
    final = FullState.capture(backbone, cache)
    tensors.update({f"post.{key}": value for key, value in final.tensors.items()})
    logp = scores(backbone, factory, query)
    tensors["query.candidate_logp"] = torch.tensor(list(logp.values()), dtype=torch.float64)
    return dict(pre=previous.structure, post=final.structure), tensors


def compare(reference, observed, bounds=None):
    structures_a, a = reference
    structures_b, b = observed
    differences = [key for key in ("pre", "post") if structures_a[key] != structures_b[key]]
    rows = []
    def first_difference(left, right, path):
        if isinstance(left, dict) and isinstance(right, dict):
            children = ((key, left.get(key), right.get(key)) for key in sorted(left.keys() | right.keys()))
        elif isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
            children = ((str(i), a, b) for i, (a, b) in enumerate(zip(left, right, strict=True)))
        else:
            return path
        return next((first_difference(a, b, f"{path}.{key}") for key, a, b in children if a != b), path)

    failures = [first_difference(structures_a[key], structures_b[key], f"structures.{key}")
                for key in differences]
    first_token, first_layer, first_state = None, None, None
    phases = {"pre": 0, "query": 1, "post": 2}
    for key in sorted(a.keys() | b.keys(), key=lambda path: (phases[path.split(".", 1)[0]], path)):
        left, right = a.get(key), b.get(key)
        if left is None or right is None or left.shape != right.shape or left.dtype != right.dtype:
            failures.append(key)
            if first_state is None and key.startswith(("pre.", "post.")):
                first_state = key
            rows.append(dict(field=key, status="STRUCTURE_MISMATCH"))
            continue
        equal = torch.equal(left, right)
        if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
            raise ValueError(f"Nonfinite execution trace: {key}")
        if equal:
            absolute, relative, cosine = 0.0, 0.0, 1.0
        else:
            x, y = left.double(), right.double()
            delta = (x - y).abs()
            absolute = float(delta.max())
            relative = float((delta / torch.maximum(x.abs(), y.abs()).clamp_min(1e-12)).max())
            denom = x.norm() * y.norm()
            cosine = float((x.flatten() @ y.flatten()) / denom) if denom else 0.0
            if key.startswith("query.") and left.ndim == 2:
                token = int((delta != 0).any(dim=-1).nonzero()[0])
                first_token = token if first_token is None else min(token, first_token)
            if key.startswith("query.layer."):
                layer = int(key.rsplit(".", 1)[1])
                first_layer = layer if first_layer is None else min(layer, first_layer)
            if key.startswith(("pre.", "post.")) and first_state is None:
                first_state = key
        limit = (bounds or {}).get(key, 0.0)
        if absolute > limit:
            failures.append(key)
        rows.append(dict(field=key, max_absolute=absolute, max_relative=relative,
                         cosine=cosine, bound=limit, status="PASS" if absolute <= limit else "FAIL"))
    if first_layer is None and first_state is not None:
        parts = first_state.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            first_layer = int(parts[1])
    if first_state is None and differences:
        first_state = failures[0]
    return dict(passed=not failures, failures=failures, first_divergent_token=first_token,
                first_divergent_layer=first_layer, first_divergent_state_field=first_state), rows


def envelope(reference, repeat):
    summary, rows = compare(reference, repeat)
    if any(row["status"] == "STRUCTURE_MISMATCH" for row in rows) or any(
            key.startswith("structures.") for key in summary["failures"]):
        raise RuntimeError("Independent live repeats disagree on Cache structure")
    return {row["field"]: row["max_absolute"] for row in rows}


def save_trace(path, result):
    state, tensors = result
    write(path.with_suffix(".json"), state)
    save_file(tensors, path.with_suffix(".safetensors"))


def load_trace(path):
    import json

    return json.loads(path.with_suffix(".json").read_text()), load_file(path.with_suffix(".safetensors"))


def ensure_readonly(mc, cell, action):
    file = mc.cells._store.path(cell.id)
    sha = digest_file(file)
    result = action()
    if sha != digest_file(file):
        raise RuntimeError(f"Query modified the persisted Cell {cell.id}")
    return result


def restored_trace(mc, cell, query):
    state = mc.cells._state(cell.id)
    return ensure_readonly(mc, cell, lambda: trace(mc._backbone, lambda: state.restore(mc._backbone), query))


def check_restored_tensors(mc, cell, expected=None):
    backbone = mc._backbone
    state = mc.cells._state(cell.id)
    restored = FullState.capture(backbone, state.restore(backbone))
    for reference in (state,) if expected is None else (state, expected):
        if restored.structure != reference.structure or restored.tensors.keys() != reference.tensors.keys():
            raise RuntimeError("Decoded bundle changed Cache structure or tensor names")
        for key, tensor in reference.tensors.items():
            other = restored.tensors[key]
            if tensor.dtype != other.dtype or not torch.equal(tensor, other):
                raise RuntimeError(f"Decoded bundle changed tensor {key}")
    return dict(cursor=restored.structure["sequence_length"],
                layer_count=restored.structure["layer_count"],
                state_schema_sha256=canonical_hash(restored.structure),
                tensors=len(restored.tensors))


def cache_audit(structure):
    rows = []
    for index, layer in enumerate(structure["layers"]):
        recurrent = [node is not None for node in layer["recurrent_state"]["items"]]
        conv = [[node is not None for node in projection["items"]] for projection in layer["conv_state"]["items"]]
        rows.append(dict(layer=index, recurrent_banks=recurrent, qkv_routed_shared=conv,
                         attn_nonempty=layer["attn_state"] is not None,
                         ffn_nonempty=layer["ffn_state"] is not None))
    return rows


def compare_coverage(live_structure, restored_structure):
    live_layers = cache_audit(live_structure)
    restored_layers = cache_audit(restored_structure)
    failures = []
    if len(live_layers) != len(restored_layers):
        failures.append("layer_count")
    inactive = []
    for live_layer, restored in zip(live_layers, restored_layers):
        layer = live_layer["layer"]
        expected = live_layer["recurrent_banks"]
        actual = restored["recurrent_banks"]
        if len(expected) != len(actual):
            failures.append(f"{layer}.recurrent_state.bank_count")
        for bank, (present, recovered) in enumerate(zip(expected, actual)):
            field = f"{layer}.recurrent_state.{bank}"
            if present != recovered:
                failures.append(field)
            if not present:
                inactive.append(field)
        if not expected or not expected[-1]:
            failures.append(f"{layer}.recurrent_state.shared")
        for qkv, (live_branches, restored_branches) in enumerate(zip(
                live_layer["qkv_routed_shared"], restored["qkv_routed_shared"], strict=True)):
            for branch, (present, recovered) in enumerate(zip(live_branches, restored_branches, strict=True)):
                if not present or not recovered:
                    failures.append(f"{layer}.conv_state.{qkv}.{branch}")
        for field in ("attn_nonempty", "ffn_nonempty"):
            if live_layer[field] != restored[field]:
                failures.append(f"{layer}.{field}")
    return dict(passed=not failures, failures=failures, inactive_live_routed_banks=inactive)
