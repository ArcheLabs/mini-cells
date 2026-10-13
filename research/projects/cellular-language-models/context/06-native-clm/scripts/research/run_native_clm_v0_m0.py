#!/usr/bin/env python3
"""Native CLM v0 M0 architecture/execution smoke.

M0 is an engineering gate. It verifies that the first token-predictive Native CLM
runtime can forward/backward, route sparsely, project Cell gradients through the
certificate, spawn a child, checkpoint a dynamic Cell set, reload it, and generate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minicells.native_clm_v0 import ByteTokenizer, NativeCLM, NativeCLMConfig


def run_m0(*, output_dir: Path) -> dict:
    torch.manual_seed(76001)
    config = NativeCLMConfig(
        vocab_size=256,
        max_seq_len=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        initial_cells=4,
        active_cells=2,
        cellular_layer_index=0,
        route_temperature=0.7,
        certificate_max_rank=8,
        tie_embeddings=True,
    )
    model = NativeCLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    text = (
        "Native CLM M0 routes token states through sparse persistent Cells. "
        "The smoke test checks execution, safe writes, growth, and checkpointing."
    )
    encoded = ByteTokenizer.encode(text)
    tokens = torch.tensor(encoded[:32], dtype=torch.long)[None, :].repeat(2, 1)
    targets = torch.roll(tokens, shifts=-1, dims=1)

    before = model(tokens, targets, return_info=True)
    loss_before = float(before["loss"].detach())
    before["loss"].backward()

    groups = model.parameter_groups()
    router_has_grad = any(
        p.grad is not None and bool(torch.any(p.grad != 0)) for p in groups["router"]
    )
    cells_have_grad = any(
        p.grad is not None and bool(torch.any(p.grad != 0)) for p in groups["cells"]
    )

    added_certificate = model.update_certificates(before["cell_info"])
    projection_ratios = model.project_cell_gradients_()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        route_state = before["cell_info"]["route_input"].mean(dim=(0, 1))
        route_key = model.cellular.query_proj(route_state)
    child_id = model.spawn_cell(parent_id=0, route_key=route_key, inherit_scale=0.25)
    child = model.cellular.cells[child_id]
    optimizer.add_param_group({"params": [child.weight, child.route_key], "lr": 1e-3})
    child_weight_before = child.weight.detach().clone()

    after = model(tokens, targets, return_info=True)
    loss_after = float(after["loss"].detach())
    after["loss"].backward()
    model.project_cell_gradients_()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    child_optimizer_update = not torch.equal(child_weight_before, child.weight.detach())

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "m0-dynamic-checkpoint.pt"
    model.save_checkpoint(checkpoint, extra={"milestone": "M0", "child_id": child_id})
    restored, extra = NativeCLM.load_checkpoint(checkpoint)
    restored_output = restored(tokens, targets, return_info=True)

    prompt = torch.tensor([ByteTokenizer.encode("Cell")], dtype=torch.long)
    generated = restored.generate(prompt, max_new_tokens=4, temperature=0.0)

    gates = {
        "forward_loss_finite": torch.isfinite(before["loss"]).item(),
        "backward_reaches_router": router_has_grad,
        "backward_reaches_cells": cells_have_grad,
        "sparse_execution": before["cell_info"]["active_fraction_vs_dense"] == 0.5,
        "certificate_state_updates": added_certificate > 0,
        "certificate_projection_executes": len(projection_ratios) == 4,
        "dynamic_spawn_executes": child_id == 4 and model.cell_count == 5,
        "spawned_runtime_remains_sparse": after["cell_info"]["active_fraction_vs_dense"] <= 0.4,
        "spawned_cell_can_join_optimizer": child_optimizer_update,
        "dynamic_checkpoint_roundtrip": restored.cell_count == 5 and extra.get("child_id") == 4,
        "restored_forward_finite": torch.isfinite(restored_output["loss"]).item(),
        "generation_executes": generated.shape[1] == prompt.shape[1] + 4,
    }
    passed = all(gates.values())
    decision = {
        "format": "minicells.native-clm-v0.m0-smoke.v1",
        "status": (
            "NATIVE_CLM_V0_M0_EXECUTION_SMOKE_PASS"
            if passed
            else "NATIVE_CLM_V0_M0_EXECUTION_SMOKE_FAIL"
        ),
        "scientific_decision": False,
        "seed": 76001,
        "config": config.__dict__,
        "parameter_count": model.parameter_count(),
        "initial_cells": 4,
        "final_cells": model.cell_count,
        "active_cells": config.active_cells,
        "loss_before": loss_before,
        "loss_after_one_step": loss_after,
        "certificate_vectors_added": added_certificate,
        "projection_ratios": projection_ratios,
        "gates": gates,
        "pass": passed,
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "RESULTS.md").write_text(
        "# Native CLM v0 M0 — Architecture / execution smoke\n\n"
        f"- Status: `{decision['status']}`\n"
        f"- Dynamic Cells: `4 -> {model.cell_count}`\n"
        f"- Active Cells/token: `{config.active_cells}`\n"
        f"- Parameter count after spawn: `{decision['parameter_count']['total']:,}`\n"
        "- Scientific decision: `False` — M0 is an engineering execution gate.\n",
        encoding="utf-8",
    )
    checkpoint.unlink(missing_ok=True)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments/native-clm-v0-m0-execution-smoke"),
    )
    args = parser.parse_args()
    decision = run_m0(output_dir=args.output_dir)
    print(json.dumps({k: decision[k] for k in ("status", "pass", "final_cells")}, indent=2))
    return 0 if decision["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
