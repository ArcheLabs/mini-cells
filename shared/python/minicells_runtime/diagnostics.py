"""Frozen-plan lightweight probes and explicit counterfactual cost estimates."""

import torch


@torch.no_grad()
def router_scores(model, frozen):
    rows = []
    for item in frozen:
        state = item["state"]
        legal = model.registry.legal(
            state.depth,
            model.config["budget"],
            model.config["minimum_actions"],
            state.hidden.device,
        )
        q, _ = model.controller(
            state.hidden[None],
            torch.tensor([state.previous], device=state.hidden.device),
            state.depth,
            legal,
            model.registry.version,
        )
        values = q[0].detach().cpu().tolist()
        allowed = [a for a in range(len(values)) if bool(legal[a])]
        cells = [a for a in allowed if a < model.registry.slots]
        rows.append(
            {
                "depth": state.depth,
                "previous": state.previous,
                "group": item["group_feature"],
                "ranked_legal_actions": sorted(allowed, key=lambda a: values[a]),
                "q": values,
                "stop_minus_best_cell": values[model.registry.stop] - min(values[a] for a in cells)
                if cells
                else None,
                "stop_minus_ctx": values[model.registry.stop] - values[model.registry.ctx],
                "stop_legal": bool(legal[model.registry.stop]),
                "state_source": "FROZEN_STEP_ZERO_BANK",
            }
        )
    return rows


def resource_estimate(root, c, arms):
    from pathlib import Path
    from clm001_r3.report import verified

    root = Path(root)
    costs = {}
    for arm, progress in arms.items():
        paths = sorted((root / "arms" / arm / "diagnostics").glob("snapshot-step-*.json"))
        paths = [p for p in paths if not p.name.endswith(".sha256.json")]
        rows = [verified(p) for p in paths]
        sampled = {False: [], True: []}
        for row in rows:
            sampled[row["step"] in c["evaluation"].get("counterfactual_snapshots", [])].append(
                row["elapsed_seconds"]
            )
        future = [v for v in c["evaluation"]["snapshots"] if v > progress["last_committed_step"]]
        missing = []
        estimate = 0.0
        for step in future:
            high = step in c["evaluation"].get("counterfactual_snapshots", [])
            values = sampled[high]
            if not values:
                missing.append(step)
            else:
                estimate += max(values)
        costs[arm] = {
            "observed_snapshot_seconds": sum(r["elapsed_seconds"] for r in rows),
            "remaining_snapshot_seconds_estimate": estimate if not missing else None,
            "unmeasured_snapshot_steps": missing,
            "observed_snapshot_count": len(rows),
            "frozen_plan": c["evaluation"]["snapshots"],
            "counterfactual_plan": c["evaluation"].get("counterfactual_snapshots", []),
        }
    return {
        "scope": "MEASURED_DIAGNOSTIC_COST_NOT_FULL_TRAINING_FORECAST",
        "arms": costs,
        "diagnostics_enabled": c["evaluation"]["periodic_diagnostics_enabled"],
        "output_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        "safety_factor": c["resources"]["projection_safety_factor"],
        "no_runtime_kill": True,
        "scientific_decision": False,
    }
