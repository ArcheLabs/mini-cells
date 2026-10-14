"""Trace diagnostics distinguish repetitions from meaningful terminal effects."""

from collections import Counter
import numpy as np
import torch


def routing(out, cells):
    if out.log_prob.numel() == 0:
        return {"scope": "native MoE top-2 routing", "adaptive_stop": False}
    paths = out.actions.detach().reshape(-1, out.actions.shape[-1]).cpu().tolist()
    usage, connections = Counter(), Counter()
    repeats = transitions = reads = 0
    signatures = []
    for row in paths:
        previous = None
        path = [a for a in row if a >= 0]
        signatures.append(tuple(path))
        for a in path:
            if a < cells:
                usage[a] += 1
            reads += a == cells
            if previous is not None:
                connections[f"{previous}->{a}"] += 1
                if a < cells and previous < cells:
                    repeats += previous == a
                    transitions += 1
            previous = a
    lengths = out.lengths.detach().cpu().numpy().ravel()
    mask = out.decision_mask
    return {
        "lengths": {
            k: float(v)
            for k, v in zip(
                ("mean", "p50", "p90", "p99", "max"),
                (lengths.mean(), *np.percentile(lengths, [50, 90, 99]), lengths.max()),
            )
        },
        "forced_stop_fraction": float(out.forced_stop.float().mean()),
        "stop_probability": float(out.stop_prob[mask].detach().mean()) if mask.any() else 0.0,
        "self_loop_ratio": repeats / max(1, transitions),
        "cross_cell_transitions": transitions - repeats,
        "ctx_read_calls": reads,
        "cell_usage": dict(usage),
        "unused_cell_fraction": sum(usage[i] == 0 for i in range(cells)) / cells,
        "path_unique_fraction": len(set(signatures)) / len(paths),
        "predecessor_distribution": dict(connections),
        "entropy_per_token": float((out.entropy * mask).detach().sum(-1).mean()),
    }


def gradients(model, before, outputs):
    groups = {key: [] for key in ("cell", "connection", "read", "head", "critic", "shared")}
    updates = {key: [] for key in groups}
    for name, p in model.named_parameters():
        key = (
            "critic"
            if ".critic." in name
            else "connection"
            if name.startswith("controller.")
            else "cell"
            if name.startswith(("cells.", "blocks."))
            else "read"
            if name.startswith(("context.", "memory."))
            else "head"
            if name.startswith("embedding.")
            else "shared"
        )
        if p.grad is not None:
            groups[key].append(p.grad.detach().float().square().sum())
        updates[key].append((p.detach() - before[name]).float().square().sum())
    norms = {
        f"grad_norm_{key}": float(torch.stack(v).sum().sqrt()) if v else 0.0 for key, v in groups.items()
    }
    norms["update_norms"] = {
        key: float(torch.stack(v).sum().sqrt()) if v else 0.0 for key, v in updates.items()
    }
    records = [r for o in outputs if hasattr(o, "state_records") for r in o.state_records]
    positions = {}
    for r in records:
        row = positions.setdefault(r["step"], {"state": [], "delta": [], "gradient": []})
        row["state"].append(float(r["before"].detach().norm()))
        row["delta"].append(float(r["delta"].detach().norm()))
        if r["router_state"].grad is not None:
            row["gradient"].append(float(r["router_state"].grad.norm()))
    norms["path_position_diagnostics"] = {
        "steps": sorted(positions),
        "state_norm_mean": [
            sum(positions[t]["state"]) / len(positions[t]["state"]) for t in sorted(positions)
        ],
        "update_norm_mean": [
            sum(positions[t]["delta"]) / len(positions[t]["delta"]) for t in sorted(positions)
        ],
        "policy_gradient_norm_mean": [
            sum(positions[t]["gradient"]) / max(1, len(positions[t]["gradient"])) for t in sorted(positions)
        ],
    }
    norms["nonfinite"] = any(
        p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
    )
    return norms


def combined_routing(outputs, cells):
    from types import SimpleNamespace

    fields = ("actions", "lengths", "forced_stop", "stop_prob", "decision_mask", "entropy", "log_prob")
    combined = SimpleNamespace(
        **{key: torch.cat([getattr(o, key).detach() for o in outputs], dim=0) for key in fields}
    )
    return routing(combined, cells)
