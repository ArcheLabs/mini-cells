"""Exact resident inventory and actual module-call weight work, isolated from training."""

import time
import statistics
import torch
from torch import nn
from pd001.accounting import TrainingGemmCounter
from .common import hash_accepted_state


def parameters(model):
    groups = {}
    for name, p in model.named_parameters():
        key = (
            "connection"
            if name.startswith("controller.")
            else ("cell_pool" if name.startswith("cells.") else "shared")
        )
        groups[key] = groups.get(key, 0) + p.numel()
    total = sum(p.numel() for p in model.parameters())
    reserved = sum(p.numel() for p in model.cells[model.registry.cells].parameters())
    return {
        "total_parameters": total,
        "by_module_group": groups,
        "reserved_parameters": reserved,
        "accepted_initial_resident_parameters": total - reserved,
        "includes_critic_and_reserved_slot": True,
        "dtype": "float32",
        "resident_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
    }


@torch.no_grad()
def profile(model, x, *, repeats=3):
    device = x.device
    model.eval()
    before = hash_accepted_state(model)
    calls = {}
    unique = {}
    handles = []
    prior_rows = set()
    step_rows = set()

    def hook(name):
        def record(module, inputs, output):
            if module is model.controller:
                prior_rows.update(inputs[1].reshape(-1).tolist())
                step_rows.add(int(inputs[2]))
                count = inputs[0].shape[0]
                calls["controller.step_rows"] = (
                    calls.get("controller.step_rows", 0)
                    + count * model.controller.step.embedding_dim
                )
                calls["controller.edge_bias_rows"] = (
                    calls.get("controller.edge_bias_rows", 0) + count * model.registry.actions
                )
                calls["controller.target_table"] = (
                    calls.get("controller.target_table", 0)
                    + count * model.controller.target.weight.numel()
                )
                return
            own = list(module.parameters(recurse=False))
            if not own:
                return
            if isinstance(module, nn.Embedding):
                ids = set(inputs[0].reshape(-1).tolist())
                rows = unique.setdefault(name, set())
                rows.update(ids)
                calls[name] = calls.get(name, 0) + inputs[0].numel() * module.embedding_dim
            else:
                weight = sum(p.numel() for p in own)
                instances = inputs[0].numel() // inputs[0].shape[-1]
                unique[name] = weight
                calls[name] = calls.get(name, 0) + instances * weight

        return record

    for name, module in model.named_modules():
        if any(True for _ in module.parameters(recurse=False)):
            handles.append(module.register_forward_hook(hook(name)))
    gen = torch.Generator(device=device).manual_seed(model.experiment_config["evaluation"]["seed"])
    with TrainingGemmCounter() as counter:
        out = model(x, policy="greedy", generator=gen)
    for h in handles:
        h.remove()
    # Tied output projection is functional, outside Linear hooks; count its full weight per output token once.
    tokens = x.numel()
    calls["tied_output_projection"] = tokens * model.embedding.weight.numel()
    unique_count = 0
    for name, value in unique.items():
        module = dict(model.named_modules())[name]
        if name == "embedding":
            continue  # all vocabulary rows already touched by tied output
        unique_count += len(value) * module.embedding_dim if isinstance(value, set) else value
    unique_count += model.embedding.weight.numel()
    if prior_rows:
        unique_count += (
            len(prior_rows) * model.registry.actions
            + model.controller.target.weight.numel()
            + len(step_rows) * model.controller.step.embedding_dim
        )
    timings = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        model(x, policy="greedy", generator=gen)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - begin) * 1000)
    assert hash_accepted_state(model) == before
    return {
        "parameter_inventory": parameters(model),
        "unique_touched_parameters_window": unique_count,
        "executed_parameter_work_per_output_token": sum(calls.values()) / tokens,
        "module_parameter_work": calls,
        "scope": "actual executed leaf-module weights plus functional tied head; embeddings charge selected rows; includes critic currently executed",
        "p50_forward_ms": statistics.median(timings),
        "repeats": repeats,
        "device": str(device),
        "observed_gemm_flops": counter.flops,
        "observed_gemm_by_operator": counter.by_operator,
        "flops_scope": "offline executed mm/addmm/bmm lower bound, includes tied head and Context QK/AV, excludes non-GEMM",
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "actual_stop_rate": float(out.actual_stop.float().mean()),
        "forced_end_rate": float(out.forced_end.float().mean()),
    }
