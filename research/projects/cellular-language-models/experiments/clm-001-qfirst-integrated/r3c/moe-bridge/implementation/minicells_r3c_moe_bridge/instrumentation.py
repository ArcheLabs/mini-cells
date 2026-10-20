"""Small on-device summaries; matrix counters run only in isolated profiling."""

import torch
from torch import nn


def norm(grads):
    values = [g.detach().square().sum() for g in grads if g is not None]
    return torch.stack(values).sum().sqrt() if values else torch.tensor(0.0)


def routing_summary(out, c):
    report = []
    for t, s in enumerate(out.stages):
        p = (s.logits / c["routing"]["temperature"]).softmax(-1).detach()
        report.append(
            {
                "stage": t,
                "selected_counts": out.operator_rows[t].detach().cpu().tolist(),
                "mean_full_probability": p.mean(0).cpu().tolist(),
                "mean_entropy": float(-(p * p.clamp_min(1e-30).log()).sum(-1).mean()),
                "mean_weights": s.weights.detach().mean(0).cpu().tolist(),
                "dead_cell_rate": float((out.operator_rows[t] == 0).float().mean()),
                "top2_overlap_previous": None
                if t == 0
                else float(
                    (s.indices[:, :, None] == out.stages[t - 1].indices[:, None, :])
                    .any(-1)
                    .float()
                    .mean()
                ),
            }
        )
    return report


class MatmulAudit:
    def __init__(self, model):
        self.model = model
        self.flops = 0
        self.rows = {}
        self.handles = []

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):

                def hook(m, args, out, name=name):
                    rows = args[0].numel() // m.in_features
                    self.flops += 2 * rows * m.in_features * m.out_features
                    self.rows[name] = self.rows.get(name, 0) + rows

                self.handles.append(module.register_forward_hook(hook))
        return self

    def finish(self, out):
        b, n, v = out.logits.shape
        d = self.model.c["model"]["width"]
        memory = out.memory.key.shape[1]
        return {
            "forward_matmul_flops": self.flops + 2 * b * n * d * v + 4 * b * n * memory * d,
            "per_token_forward_matmul_flops": (
                self.flops + 2 * b * n * d * v + 4 * b * n * memory * d
            )
            / (b * n),
            "cell_rows": sum(
                v for k, v in self.rows.items() if k.startswith("cells.") and k.endswith("up")
            ),
            "context_rows": self.rows.get("context.query", 0),
            "gate_rows": self.rows.get("gate.net.0", 0),
            "scope": "executed linear, tied head, dense QK/AV matmuls; excludes norms/GELU/softmax/dispatch/optimizer",
            "hardware_kernel_flops": "NOT_MEASURED",
        }

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()


def parameters(model):
    total = sum(p.numel() for p in model.parameters())
    gate = sum(p.numel() for p in model.gate.parameters())
    return {
        "nominal_parameters": total,
        "gate_parameters": gate,
        "core_parameters": total - gate,
        "task_trainable_parameters": total - gate if model.arm in ("A", "B", "D") else total,
        "any_loss_trainable_parameters": (
            total - gate
            if model.arm in ("A", "B")
            or (model.arm == "D" and model.c["balance"]["coefficient"] == 0)
            else None
            if model.arm == "D" and model.c["balance"]["coefficient"] is None
            else total
        ),
        "requires_grad_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "cells": len(model.cells),
        "shared_across_stages": True,
        "note": "participating parameters describe reachable loss paths, not weights merely declared requires_grad",
    }
