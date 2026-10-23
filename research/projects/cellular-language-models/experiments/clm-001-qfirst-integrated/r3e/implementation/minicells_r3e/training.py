"""Executed-path conditional CE; finite Cell gradients; no Development access."""

import copy
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from minicells_r3c_moe_bridge.data import train_batches
from .common import canonical, fingerprint


class Trainer:
    def __init__(self, model, c, arm):
        self.model = model
        self.c = c
        self.arm = arm
        self.step = 0
        self.data_cursor = 0
        self.policy_updates = 0
        self.connection_version = 0
        self.proposals = []
        self.curve = []
        self.pending = None
        self.data_rng = np.random.default_rng(model.seed + c["training"]["data_seed_offset"])
        self.route_rng = torch.Generator().manual_seed(
            model.seed + c["training"]["route_seed_offset"]
        )
        self.cf_rng = torch.Generator().manual_seed(model.seed + c["training"]["cf_seed_offset"])
        q = c["engine"]["training"]
        self.cell_opt = torch.optim.AdamW(
            model.cells.parameters(), lr=q["learning_rate"], weight_decay=q["weight_decay"]
        )
        self.policy_opt = None
        self.opt_identity = canonical({"cell": q, "experiment": c["experiment_id"], "arm": arm})
        self.frozen_sha = fingerprint(
            {
                n: p
                for n, p in model.named_parameters()
                if not n.startswith(("cells.", "connection."))
            }
        )
        self.cost = {
            "main_updates": 0,
            "main_tokens": 0,
            "discarded_adaptation_updates": 0,
            "discarded_adaptation_tokens": 0,
            "policy_optimizer_updates": 0,
            "elapsed_update_seconds": 0.0,
            "work": {},
        }

    def epsilon(self):
        q = self.c["training"]
        f = self.step / max(1, q["steps"] - 1)
        return q["epsilon_initial"] + (q["epsilon_final"] - q["epsilon_initial"]) * f

    def update(self, data, batches=None):
        q = self.c["engine"]["training"]
        m = self.model
        if batches is None:
            batches = train_batches(
                data, self.c["engine"], self.data_rng, next(m.parameters()).device
            )
        if len(batches) != q["effective_batch"] // q["microbatch"]:
            raise RuntimeError("BATCH_BUDGET_CHANGED")
        begin = time.perf_counter()
        self.cell_opt.zero_grad(set_to_none=True)
        losses = []
        counts = []
        for x, y in batches:
            out = m(x, epsilon=self.epsilon(), route_rng=self.route_rng)
            loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), y.reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError("NONFINITE_CE")
            (loss / len(batches)).backward()
            losses.append(loss.detach())
            counts.append(out.operator_rows)
        grad = []
        for cell in m.cells:
            values = [p.grad.square().sum() for p in cell.parameters() if p.grad is not None]
            grad.append(
                torch.stack(values).sum().sqrt()
                if values
                else torch.tensor(0.0, device=loss.device)
            )
        g = torch.stack(grad)
        if not torch.isfinite(g).all() or not (g > 0).any():
            raise FloatingPointError("INVALID_CELL_GRADIENT")
        torch.nn.utils.clip_grad_norm_(m.cells.parameters(), q["clip"])
        for group in self.cell_opt.param_groups:
            group["lr"] = q["learning_rate"] * min(1, (self.step + 1) / q["warmup_steps"])
        self.cell_opt.step()
        self.step += 1
        tokens = sum(x.numel() for x, y in batches)
        self.data_cursor += tokens
        if tokens != self.c["training"]["tokens_per_step"]:
            raise RuntimeError("TOKEN_BUDGET_CHANGED")
        self.cost["main_updates"] += 1
        self.cost["main_tokens"] += tokens
        self.cost["elapsed_update_seconds"] += time.perf_counter() - begin
        row = {
            "step": self.step,
            "ce": float(torch.stack(losses).mean()),
            "cell_gradient_norms": g.detach().cpu().tolist(),
            "cell_counts": torch.stack(counts).sum((0, 1)).cpu().tolist(),
        }
        self.curve.append(row)
        return row

    def verify_frozen(self):
        if (
            fingerprint(
                {
                    n: p
                    for n, p in self.model.named_parameters()
                    if not n.startswith(("cells.", "connection."))
                }
            )
            != self.frozen_sha
        ):
            raise RuntimeError("FROZEN_WEIGHTS_CHANGED")

    def state(self):
        self.verify_frozen()
        return {
            "model": self.model.state_dict(),
            "cell_optimizer": self.cell_opt.state_dict(),
            "connection_optimizer": None,
            "step": self.step,
            "data_cursor": self.data_cursor,
            "policy_updates": self.policy_updates,
            "connection_version": self.connection_version,
            "proposals": self.proposals,
            "curve": self.curve,
            "cost": self.cost,
            "pending": self.pending,
            "data_rng": copy.deepcopy(self.data_rng.bit_generator.state),
            "route_rng": self.route_rng.get_state(),
            "cf_rng": self.cf_rng.get_state(),
            "global_cpu_rng": torch.get_rng_state(),
            "global_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "optimizer_identity": self.opt_identity,
            "frozen_sha256": self.frozen_sha,
            "work": copy.deepcopy(self.model.work),
        }

    def restore(self, s):
        if s["optimizer_identity"] != self.opt_identity or s["frozen_sha256"] != self.frozen_sha:
            raise RuntimeError("OPTIMIZER_IDENTITY_CHANGED")
        self.model.load_state_dict(s["model"], strict=True)
        self.cell_opt.load_state_dict(s["cell_optimizer"])
        for key in (
            "step",
            "data_cursor",
            "policy_updates",
            "connection_version",
            "proposals",
            "curve",
            "cost",
            "pending",
        ):
            setattr(self, key, copy.deepcopy(s[key]))
        self.data_rng.bit_generator.state = copy.deepcopy(s["data_rng"])
        self.route_rng.set_state(s["route_rng"].cpu())
        self.cf_rng.set_state(s["cf_rng"].cpu())
        torch.set_rng_state(s["global_cpu_rng"].cpu())
        random.setstate(s["python_rng"])
        np.random.set_state(s["numpy_rng"])
        if s["global_cuda_rng"]:
            if (
                not torch.cuda.is_available()
                or len(s["global_cuda_rng"]) != torch.cuda.device_count()
            ):
                raise RuntimeError("RESTORE_CUDA_LAYOUT_CHANGED")
            torch.cuda.set_rng_state_all(s["global_cuda_rng"])
        self.model.work = copy.deepcopy(s["work"])
        self.verify_frozen()
        if self.data_cursor != self.step * self.c["training"]["tokens_per_step"]:
            raise RuntimeError("DATA_CURSOR_CHANGED")

    def clone(self):
        branch = Trainer(copy.deepcopy(self.model), self.c, self.arm)
        branch.restore(copy.deepcopy(self.state()))
        return branch
