"""CPU-only Gate-0 contracts, capacity math and synthetic Gate-1 mechanism test."""
from __future__ import annotations

from typing import Any

import torch

from .accounting import ObservedGemmFlops, execution_summary, unique_weight_counts
from .models import ARMS, make_model
from .profiling import check_decode_parity
from .synthetic import run_synthetic_composition_gate
from .topology import validate_adjacency


def estimated_parameter_counts(config: dict[str, Any]) -> dict[str, int]:
    model, data = config["model"], config["data"]
    width, layers = int(model["width"]), int(model["layers"])
    cell_count = int(model["cell_count"])
    common = ((int(data["vocab_size"]) + int(data["sequence_length"])) * width +
              layers * (4 * width * width + 4 * width) + 2 * width)
    moe_ffn = layers * int(model["moe_experts"]) * 2 * width * int(model["moe_hidden"])
    cell_ffn = layers * cell_count * 2 * width * int(model["cell_hidden"])
    moe_router = layers * width * int(model["moe_experts"])
    parallel_router = layers * width * cell_count
    r = int(model["router_width"])
    state_router = layers * ((width + r) * r + (4 * cell_count + 2) * r)
    return {"M0": common + moe_ffn + moe_router,
            "P0": common + cell_ffn + parallel_router,
            "S0": common + cell_ffn + state_router,
            "R0": common + cell_ffn + state_router,
            "G0": common + cell_ffn + state_router}


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    model, data, training = config["model"], config["data"], config["training"]
    positive = ("width", "layers", "heads", "cell_count", "cell_hidden", "cell_visits",
                "graph_out_degree", "router_width", "moe_experts", "moe_hidden")
    if any(int(model[key]) <= 0 for key in positive):
        raise ValueError("model dimensions must be positive")
    if int(model["width"]) % int(model["heads"]):
        raise ValueError("width must be divisible by attention heads")
    if int(model["graph_out_degree"]) > int(model["cell_count"]):
        raise ValueError("graph out-degree exceeds Cell count")
    if int(model["cell_visits"]) != 4:
        raise ValueError("the proposed fixed sequential budget is four visits")
    if int(model["moe_experts"]) * int(model["moe_hidden"]) != int(model["cell_count"]) * int(model["cell_hidden"]):
        raise ValueError("M0 and fine-grained arms do not have matched FFN parameter pools")
    expected_steps = int(data["train_tokens_per_arm"]) // (
        int(data["batch_size"]) * int(data["sequence_length"]))
    if expected_steps != int(training["steps"]):
        raise ValueError("steps and train-token budget do not agree")
    counts = estimated_parameter_counts(config)
    tolerance = float(config["proposed_gates"]["capacity_tolerance_fraction"])
    capacity_error = max(counts.values()) / min(counts.values()) - 1.0
    if capacity_error > tolerance:
        raise ValueError(f"total parameter mismatch {capacity_error:.6%} exceeds {tolerance:.6%}")
    return {"parameter_counts_estimate": counts,
            "total_parameter_capacity_spread": capacity_error,
            "parameter_tolerance_fraction": tolerance,
            "ffn_pool_per_layer": 2 * int(model["width"]) *
                                  int(model["moe_hidden"]) * int(model["moe_experts"]),
            "steps": expected_steps,
            "tokens_per_arm": int(data["train_tokens_per_arm"])}


def run_cpu_preflight(config: dict[str, Any]) -> dict[str, Any]:
    capacity = validate_config(config)
    tiny = {"width": 16, "layers": 2, "heads": 4, "vocab_size": 32,
            "max_sequence_length": 8, "sequence_length": 4,
            "moe_experts": 4, "moe_hidden": 8,
            "cell_count": 8, "cell_hidden": 4, "cell_visits": 4,
            "graph_out_degree": 4, "router_width": 4,
            "residual_scale": 0.25, "top2_backup_gradient": True}
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    arms = {}
    for arm in ARMS:
        torch.manual_seed(9035)
        model = make_model(tiny, arm, seed=123).cpu()
        model.set_collect_trace(True)
        model.eval()
        with torch.no_grad(), ObservedGemmFlops(model) as flops:
            eval_logits, _, _ = model(tokens)
        measured = flops.total
        parity = check_decode_parity(model, tokens)
        model.set_collect_trace(True)
        with torch.no_grad():
            eval_logits, _, _ = model(tokens)
        touched, _ = unique_weight_counts(model)
        if eval_logits.shape != (2, 4, 32) or measured <= 0 or not parity["passed"]:
            raise AssertionError(f"Gate-0 forward/decode/GEMM check failed for {arm}")
        if not torch.isfinite(touched).all() or torch.any(touched <= 0):
            raise AssertionError(f"active-weight accounting failed for {arm}")
        if arm in {"R0", "G0"}:
            for block in model.blocks:
                validate_adjacency(block.ffn.adjacency, tiny["graph_out_degree"])
            for layer, trace in enumerate(model.last_execution):
                assert trace is not None
                paths, candidates = trace["paths"], trace["candidates"]
                for step in range(tiny["cell_visits"]):
                    sources = torch.full_like(paths[:, :, step], tiny["cell_count"]) if step == 0 else paths[:, :, step - 1]
                    expected = model.blocks[layer].ffn.adjacency[sources]
                    if not torch.equal(expected, torch.nn.functional.one_hot(
                            candidates[:, :, step], tiny["cell_count"]).bool().sum(-2).bool()):
                        # Candidate ids are an ordered list; compare as sorted sets below.
                        actual = torch.zeros_like(expected)
                        actual.scatter_(2, candidates[:, :, step], True)
                        if not torch.equal(expected, actual):
                            raise AssertionError("graph candidate restriction did not apply in forward")

        if arm in {"S0", "R0", "G0"}:
            eval_logits = eval_logits.detach()
            calls = [0] * tiny["cell_count"]
            hooks = []
            for cell_index, cell in enumerate(model.blocks[0].ffn.cells):
                hooks.append(cell.compute.register_forward_hook(
                    lambda _module, _inputs, _output, i=cell_index:
                        calls.__setitem__(i, calls[i] + 1)))
            model.train()
            model.zero_grad(set_to_none=True)
            train_logits, auxiliary, _ = model(tokens)
            (train_logits.square().mean() + auxiliary * 0.01).backward()
            for hook in hooks:
                hook.remove()
            if not torch.allclose(train_logits, eval_logits, atol=2e-6, rtol=2e-5):
                raise AssertionError(f"top-2 backup changed hard forward for {arm}")
            router = model.blocks[0].ffn.router
            if router.query.weight.grad is None or not torch.isfinite(router.query.weight.grad).all():
                raise AssertionError(f"router gradient missing or non-finite for {arm}")
            assert model.last_execution[0] is not None
            backup = model.last_execution[0]["backup_ids"]
            expected_calls = [0] * tiny["cell_count"]
            for hop in range(tiny["cell_visits"]):
                for slot in range(2):
                    for cell in range(tiny["cell_count"]):
                        if torch.any(backup[:, :, hop, slot] == cell):
                            expected_calls[cell] += 1
            if calls != expected_calls:
                raise AssertionError("a backup Cell was skipped or an unselected Cell was computed")
        arms[arm] = {"forward_shape": list(eval_logits.shape),
                     "observed_gemm_flops": measured,
                     "decode_parity": parity["passed"],
                     "active_weight_accounting": "finite"}
        del model

    synthetic = run_synthetic_composition_gate()
    return {"status": "CPU_PREFLIGHT_PASSED", "scientific_decision": False,
            "capacity": capacity, "arms": arms, "synthetic_gate": synthetic,
            "device": "cpu",
            "note": "engineering checks only; full training remains blocked until proposal protocol is registered"}
