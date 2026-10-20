"""Joint CE and balance update. Exact component gradients without a third backward pass."""

import copy
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .common import canonical
from .instrumentation import norm, routing_summary


class Trainer:
    def __init__(self, model, balance_coefficient=None):
        self.model = model
        c = model.c
        t = c["training"]
        self.coefficient = (
            c["balance"]["coefficient"] if balance_coefficient is None else balance_coefficient
        )
        if self.coefficient is None:
            raise ValueError(
                "balance coefficient must be selected on Train/Calibration before real training"
            )
        self.route_rng = torch.Generator(device="cpu").manual_seed(model.seed + 2001)
        self.data_rng = np.random.default_rng(model.seed + 1001)
        self.step = 0
        self.data_cursor = 0
        self.named = list(model.named_parameters())
        self.gate_ids = [i for i, (n, p) in enumerate(self.named) if n.startswith("gate.")]
        self.opt = torch.optim.AdamW(
            [
                {
                    "params": [p for n, p in self.named if not n.startswith("gate.")],
                    "lr": t["learning_rate"],
                },
                {"params": list(model.gate.parameters()), "lr": t["connection_learning_rate"]},
            ],
            weight_decay=t["weight_decay"],
        )
        self.optimizer_identity = canonical(
            {
                "type": "AdamW",
                "betas": [0.9, 0.999],
                "epsilon": 1e-8,
                "weight_decay": t["weight_decay"],
                "core_lr": t["learning_rate"],
                "gate_lr": t["connection_learning_rate"],
            }
        )

    def update(self, batches, *, role="train"):
        if role != "train":
            raise ValueError("only Train may update parameters")
        m = self.model
        c = m.c
        t = c["training"]
        expected = t["effective_batch"] // t["microbatch"]
        if len(batches) != expected:
            raise ValueError("effective batch changed")
        m.train()
        self.opt.zero_grad(set_to_none=True)
        params = [p for n, p in self.named]
        rows = []
        task_sums = [None] * len(params)
        balance_sums = [None] * len(params)
        begin = time.perf_counter()
        losses = []
        balances = []
        stage_task = []
        stage_balance = []
        counts = []
        active = m.arm in ("C", "D") and self.coefficient != 0
        for x, y in batches:
            if (
                x.shape != y.shape
                or x.shape[0] != t["microbatch"]
                or x.shape[1] != t["sequence_length"]
            ):
                raise ValueError("training token shape changed")
            out = m(x, route_rng=self.route_rng)
            ce = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), y.reshape(-1))
            targets = params + [stage.delta for stage in out.stages]
            task = torch.autograd.grad(ce, targets, allow_unused=True, retain_graph=active)
            bal = (
                torch.autograd.grad(out.balance, targets, allow_unused=True)
                if active
                else (None,) * len(targets)
            )
            for i in range(len(params)):
                if task[i] is not None:
                    task_sums[i] = (
                        task[i] / expected
                        if task_sums[i] is None
                        else task_sums[i] + task[i] / expected
                    )
                if bal[i] is not None:
                    balance_sums[i] = (
                        bal[i] / expected
                        if balance_sums[i] is None
                        else balance_sums[i] + bal[i] / expected
                    )
            stage_task.append(torch.stack([norm([v]).to(ce) for v in task[len(params) :]]))
            stage_balance.append(torch.stack([norm([v]).to(ce) for v in bal[len(params) :]]))
            losses.append(ce.detach())
            balances.append(out.balance.detach())
            counts.append(out.operator_rows.detach())
            rows.append(routing_summary(out, c))
        for i, p in enumerate(params):
            a, b = task_sums[i], balance_sums[i]
            p.grad = (
                a if b is None else self.coefficient * b if a is None else a + self.coefficient * b
            )
        tensors = [p.grad for p in params if p.grad is not None]
        if not bool(
            torch.stack([torch.isfinite(g).all() for g in tensors + losses + balances]).all()
        ):
            raise FloatingPointError("nonfinite joint gradient; no optimizer update")
        gate_task = norm([task_sums[i] for i in self.gate_ids])
        gate_balance = norm([balance_sums[i] for i in self.gate_ids])
        cell_norms = [norm([p.grad for p in cell.parameters()]) for cell in m.cells]
        scale = min(1.0, (self.step + 1) / max(1, t["warmup_steps"]))
        clip = []
        for group, lr in zip(
            self.opt.param_groups, (t["learning_rate"], t["connection_learning_rate"])
        ):
            group["lr"] = lr * scale
            before = torch.nn.utils.clip_grad_norm_(group["params"], t["clip"])
            clip.append(min(1.0, t["clip"] / (float(before) + 1e-12)))
        self.opt.step()
        self.step += 1
        tokens = sum(x.numel() for x, y in batches)
        self.data_cursor += tokens
        return {
            "arm": m.arm,
            "seed": m.seed,
            "step": self.step,
            "train_tokens": self.data_cursor,
            "ce": float(torch.stack(losses).mean()),
            "balance_loss": float(torch.stack(balances).mean()),
            "balance_coefficient": self.coefficient if m.arm in ("C", "D") else 0.0,
            "gate_task_grad_norm": float(gate_task),
            "gate_balance_grad_norm": float(gate_balance),
            "gate_balance_weighted_grad_norm": float(gate_balance) * self.coefficient,
            "cell_grad_norms": [float(v) for v in cell_norms],
            "stage_task_grad_norms": torch.stack(stage_task).mean(0).cpu().tolist(),
            "stage_balance_grad_norms": torch.stack(stage_balance).mean(0).cpu().tolist(),
            "clip_ratios": clip,
            "routing_microbatches": rows,
            "stage_cell_counts": torch.stack(counts).sum(0).cpu().tolist(),
            "cell_calls_per_token": 8,
            "context_calls_per_token": 1,
            "optimizer_sha": self.optimizer_identity,
            "optimizer_sha_scope": "algorithm/hyperparameters; complete optimizer state attested at Checkpoint only",
            "optimizer_state_sha256": "NOT_MEASURED_PER_UPDATE",
            "training_matmul_flops": "NOT_MEASURED",
            "wall_seconds": time.perf_counter() - begin,
            "cuda_time_ms": "NOT_MEASURED",
            "peak_gpu_bytes": torch.cuda.max_memory_allocated()
            if next(m.parameters()).is_cuda
            else "NOT_MEASURED",
            "confirmation": "SEALED",
            "scientific_decision": False,
        }

    def state(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "step": self.step,
            "data_cursor": self.data_cursor,
            "route_rng": self.route_rng.get_state(),
            "data_rng": copy.deepcopy(self.data_rng.bit_generator.state),
            "global_cpu_rng": torch.get_rng_state(),
            "global_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "balance_coefficient": self.coefficient,
            "optimizer_identity": self.optimizer_identity,
        }

    def restore(self, state):
        if (
            state["balance_coefficient"] != self.coefficient
            or state["optimizer_identity"] != self.optimizer_identity
        ):
            raise ValueError("optimizer/coefficient identity mismatch")
        self.model.load_state_dict(state["model"], strict=True)
        self.opt.load_state_dict(state["optimizer"])
        self.step = state["step"]
        self.data_cursor = state["data_cursor"]
        self.route_rng.set_state(state["route_rng"].cpu())
        self.data_rng.bit_generator.state = state["data_rng"]
        torch.set_rng_state(state["global_cpu_rng"].cpu())
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        if state["global_cuda_rng"]:
            if not torch.cuda.is_available():
                raise ValueError("CUDA checkpoint requires compatible CUDA restore")
            torch.cuda.set_rng_state_all(state["global_cuda_rng"])
