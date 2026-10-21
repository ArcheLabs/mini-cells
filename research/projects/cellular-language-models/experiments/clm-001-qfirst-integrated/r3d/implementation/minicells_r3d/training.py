"""Alternating joint training; CF snapshot is captured before the Cell update."""

import copy
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .common import canonical, fingerprint
from .counterfactual import collect, preference_loss


class Trainer:
    def __init__(self, model, c, arm):
        self.model = model
        self.c = c
        self.arm = arm
        model.set_trainable(arm)
        self.step = 0
        self.data_cursor = 0
        self.policy_updates = 0
        self.data_rng = np.random.default_rng(model.seed + c["training"]["seed_data_offset"])
        self.cf_rng = torch.Generator().manual_seed(model.seed + c["training"]["cf_rng_offset"])
        self.route_rng = torch.Generator().manual_seed(
            model.seed + c["training"]["explore_rng_offset"]
        )
        t = c["engine"]["training"]
        q = c["training"]
        self.cell_opt = (
            torch.optim.AdamW(
                model.cells.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"]
            )
            if arm in ("F", "P", "J")
            else None
        )
        self.policy_opt = (
            torch.optim.AdamW(
                model.connection.parameters(),
                lr=q["connection_lr"],
                weight_decay=q["connection_weight_decay"],
            )
            if arm in ("L", "M", "J")
            else None
        )
        self.opt_identity = canonical({"cell": t, "policy": q, "arm": arm})
        self.frozen_sha = fingerprint(
            {n: p for n, p in model.named_parameters() if not p.requires_grad}
        )

    def epsilon(self):
        if self.arm not in self.c["training"]["exploration_arms"]:
            return 0.0
        q = self.c["training"]
        fraction = self.step / max(1, q["b_steps"] - 1)
        return q["epsilon_initial"] + (q["epsilon_final"] - q["epsilon_initial"]) * fraction

    def update(self, batches, *, role="train"):
        if role != "train":
            raise ValueError("only Train may update")
        m = self.model
        t = self.c["engine"]["training"]
        q = self.c["training"]
        if len(batches) != t["effective_batch"] // t["microbatch"]:
            raise ValueError("batch budget mismatch")
        policy_due = self.arm in ("L", "M") or (
            self.arm == "J" and (self.step + 1) % q["policy_every"] == 0
        )
        # Only J needs a parameter snapshot; L/M never change Cell or routing during scoring.
        snapshot = copy.deepcopy(m) if self.arm == "J" and policy_due else m
        if snapshot is not m:
            for p in snapshot.parameters():
                p.requires_grad_(False)
        version = {
            "cell_step": self.step,
            "connection_updates": self.policy_updates,
            "timing": "PRE_CELL_UPDATE",
        }
        m.train()
        visited = []
        losses = []
        counts = []
        changes = []
        if self.cell_opt:
            self.cell_opt.zero_grad(set_to_none=True)
        epsilon = self.epsilon()
        start = time.perf_counter()
        for x, y in batches:
            if x.shape != y.shape or x.shape != (t["microbatch"], t["sequence_length"]):
                raise ValueError("LM shape changed")
            with torch.set_grad_enabled(self.cell_opt is not None):
                out = m(
                    x, route_rng=self.route_rng, epsilon=epsilon, original=self.arm in ("F", "F0")
                )
                loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), y.reshape(-1))
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite CE")
                if self.cell_opt:
                    (loss / len(batches)).backward()
            losses.append(loss.detach())
            counts.append(out.operator_rows.detach())
            for st in out.stages:
                base = m.connection.base_action(st.logits.detach())
                actual = (
                    (st.indices.sort(-1).values[:, None, :] == m.connection.pairs[None, :, :])
                    .all(-1)
                    .long()
                    .argmax(-1)
                )
                changes.append((base != actual).float().mean())
            if policy_due:
                visited.append((out, y))
        grads = []
        for cell in m.cells:
            g = [p.grad.detach().square().sum() for p in cell.parameters() if p.grad is not None]
            grads.append(
                torch.stack(g).sum().sqrt() if g else torch.tensor(0.0, device=loss.device)
            )
        if self.cell_opt:
            finite = [
                torch.isfinite(p.grad).all() for p in m.cells.parameters() if p.grad is not None
            ]
            if not finite or not bool(torch.stack(finite).all()):
                raise FloatingPointError("nonfinite Cell gradient")
            for group in self.cell_opt.param_groups:
                group["lr"] = t["learning_rate"] * min(
                    1, (self.step + 1) / max(1, t["warmup_steps"])
                )
            torch.nn.utils.clip_grad_norm_(m.cells.parameters(), t["clip"])
            self.cell_opt.step()
        work = {}
        metrics = {"scored_pairs": 0}
        policy_loss = None
        update_norm = 0.0
        if policy_due:
            # All records use pre-update Cells/policy; current model has already received its Cell update.
            records, work = collect(
                snapshot,
                visited,
                self.cf_rng,
                q["states_per_policy_update"],
                "short" if self.arm == "M" else "long",
            )
            policy_loss, metrics = preference_loss(m, records, q["preference_temperature"])
            if policy_loss is not None:
                self.policy_opt.zero_grad(set_to_none=True)
                policy_loss.backward()
                finite = [
                    torch.isfinite(p.grad).all()
                    for p in m.connection.parameters()
                    if p.grad is not None
                ]
                if (
                    not torch.isfinite(policy_loss)
                    or not finite
                    or not bool(torch.stack(finite).all())
                ):
                    raise FloatingPointError("nonfinite preference gradient")
                torch.nn.utils.clip_grad_norm_(m.connection.parameters(), q["connection_clip"])
                before = [p.detach().clone() for p in m.connection.parameters()]
                self.policy_opt.step()
                with torch.no_grad():
                    delta = [p - old for p, old in zip(m.connection.parameters(), before)]
                    size = torch.stack([v.square().sum() for v in delta]).sum().sqrt()
                    factor = (q["maximum_parameter_update_norm"] / (size + 1e-30)).clamp(max=1)
                    for p, old, v in zip(m.connection.parameters(), before, delta):
                        p.copy_(old + factor * v)
                    update_norm = float(size * factor)
                self.policy_updates += 1
        self.step += 1
        self.data_cursor += sum(x.numel() for x, y in batches)
        return {
            "cell_optimizer_updated": self.cell_opt is not None,
            "step": self.step,
            "tokens": self.data_cursor,
            "ce": float(torch.stack(losses).mean()),
            "connection_loss": None if policy_loss is None else float(policy_loss.detach()),
            "policy_updates": self.policy_updates,
            "cell_grad_norms": [float(v) for v in grads],
            "stage_cell_counts": torch.stack(counts).sum(0).cpu().tolist(),
            "action_change_fraction": float(torch.stack(changes).mean()),
            "epsilon": epsilon,
            "connection_parameter_update_norm": update_norm,
            "counterfactual_snapshot": version if policy_due else None,
            "counterfactual_work": work,
            "preference": metrics,
            "compute_seconds": time.perf_counter() - start,
        }

    def state(self):
        return {
            "model": self.model.state_dict(),
            "cell_optimizer": None if self.cell_opt is None else self.cell_opt.state_dict(),
            "connection_optimizer": None
            if self.policy_opt is None
            else self.policy_opt.state_dict(),
            "step": self.step,
            "data_cursor": self.data_cursor,
            "policy_updates": self.policy_updates,
            "data_rng": copy.deepcopy(self.data_rng.bit_generator.state),
            "cf_rng": self.cf_rng.get_state(),
            "route_rng": self.route_rng.get_state(),
            "global_cpu_rng": torch.get_rng_state(),
            "global_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "optimizer_identity": self.opt_identity,
            "frozen_sha256": self.frozen_sha,
        }

    def verify_frozen(self):
        current = fingerprint(
            {n: p for n, p in self.model.named_parameters() if not p.requires_grad}
        )
        if current != self.frozen_sha:
            raise RuntimeError("frozen parameters changed")

    def restore(self, s):
        if s["optimizer_identity"] != self.opt_identity or s["frozen_sha256"] != self.frozen_sha:
            raise ValueError("state/optimizer identity mismatch")
        self.model.load_state_dict(s["model"], strict=True)
        for opt, key in [
            (self.cell_opt, "cell_optimizer"),
            (self.policy_opt, "connection_optimizer"),
        ]:
            if opt is not None:
                opt.load_state_dict(s[key])
        self.step = s["step"]
        self.data_cursor = s["data_cursor"]
        self.policy_updates = s["policy_updates"]
        self.data_rng.bit_generator.state = s["data_rng"]
        self.cf_rng.set_state(s["cf_rng"].cpu())
        self.route_rng.set_state(s["route_rng"].cpu())
        torch.set_rng_state(s["global_cpu_rng"].cpu())
        random.setstate(s["python_rng"])
        np.random.set_state(s["numpy_rng"])
        if s["global_cuda_rng"]:
            if (
                not torch.cuda.is_available()
                or len(s["global_cuda_rng"]) != torch.cuda.device_count()
            ):
                raise ValueError("restore same visible GPU layout")
            torch.cuda.set_rng_state_all(s["global_cuda_rng"])
        self.verify_frozen()
