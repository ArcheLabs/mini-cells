"""One coordinated update: live continuous CE plus detached-state Q, or matched J0."""

from copy import deepcopy
import time
import torch
from torch.nn import functional as F
from pd001.losses import joint_loss, terminal_ce
from .oracle import Snapshot, OracleUntrusted
from .sampling import streams, proposals
from .common import canonical_sha256


def optimizer(model):
    c = model.experiment_config["training"]
    theta = [p for n, p in model.named_parameters() if not n.startswith("controller.")]
    return torch.optim.AdamW(
        [
            {"params": theta, "lr": c["learning_rate"]},
            {"params": model.controller.parameters(), "lr": c["connection_learning_rate"]},
        ],
        weight_decay=c["weight_decay"],
    )


def q_loss(model, rows, snapshot):
    terms = []
    weights = []
    critics = []
    for state, target, label in rows:
        label.verify(snapshot, state, target)
        legal = model.registry.legal(
            state.depth,
            model.config["budget"],
            model.config["minimum_actions"],
            state.hidden.device,
        )
        q, v = model.controller(
            state.hidden.detach()[None],
            torch.tensor([state.previous], device=state.hidden.device),
            state.depth,
            legal,
            model.registry.version,
            detach_state=True,
        )
        goal = q.new_tensor(label.value)
        terms.append(F.smooth_l1_loss(q[0, label.action], goal, reduction="sum") / label.inclusion)
        weights.append(1 / label.inclusion)
        critics.append((v[0] - goal).square() / label.inclusion)
    if not terms:
        return next(model.parameters()).new_zeros(()), next(model.parameters()).new_zeros(())
    return torch.stack(terms).sum() / sum(weights), torch.stack(critics).sum() / sum(weights)


class Trainer:
    def __init__(self, model):
        self.model = model
        self.opt = optimizer(model)
        self.rng = streams(model.experiment_config["seed"], next(model.parameters()).device)
        self.step = 0
        self.cache = {}
        self.data_position = 0
        self.journal = []

    def update(self, batches, *, role="train", failure=None):
        if role != "train":
            raise ValueError("only Train can produce gradients or candidate labels")
        model = self.model
        c = model.experiment_config
        t = c["training"]
        device = next(model.parameters()).device
        if len(batches) != t["effective_batch"] // t["microbatch"]:
            raise ValueError("effective batch is fixed")
        begin = time.perf_counter()
        snapshot = Snapshot(model)
        snapshot_s = time.perf_counter() - begin
        self.opt.zero_grad(set_to_none=True)
        model.train()
        selected = {}
        total_positions = sum(x.numel() for x, y in batches)
        # Pick token-step coordinates without consulting targets or trajectories, at most two total.
        picks = torch.randperm(total_positions, generator=self.rng["proposal"], device=device)[
            : min(total_positions, c["oracle"]["probes_per_step"])
        ]
        cursor = 0
        for i, (x, y) in enumerate(batches):
            selected[i] = [
                (
                    int(p) - cursor,
                    int(
                        torch.randint(
                            model.config["budget"],
                            (),
                            generator=self.rng["proposal"],
                            device=device,
                        )
                    ),
                )
                for p in picks
                if cursor <= int(p) < cursor + x.numel()
            ]
            cursor += x.numel()
        rows = []
        ce_sum = 0.0
        cost = 0.0
        stops = 0.0
        forced = 0.0
        oracle_s = 0.0
        live_s = 0.0
        cf_work = 0
        forward_s = 0.0
        backward_s = 0.0
        path_records = []
        actor_sum = 0.0
        critic_sum = 0.0
        mean_steps = 0.0
        ctx_reads = 0
        self_loops = 0
        visits = {str(a): 0 for a in range(model.registry.slots)}
        for i, (x, y) in enumerate(batches):
            tick = time.perf_counter()
            out = model(x, generator=self.rng["live"], probe_coordinates=selected[i])
            ce = terminal_ce(out, y).mean()
            forward_s += time.perf_counter() - tick
            backward_begin = time.perf_counter()
            if model.arm == "J0-FULL":
                loss, metrics = joint_loss([out], y, t, "J0", snapshot.policy["lambda_cost"])
                actor_sum += float(metrics["actor_loss"])
                critic_sum += float(metrics["critic_loss"])
            else:
                loss = ce
            if loss.requires_grad:
                (loss / len(batches)).backward()
            backward_s += time.perf_counter() - backward_begin
            live_s += time.perf_counter() - tick
            ce_sum += float(ce.detach())
            cost += float(out.cost_units.detach().mean())
            stops += float(out.actual_stop.float().mean())
            forced += float(out.forced_end.float().mean())
            traces = out.actions.detach()
            path_records.append(traces.cpu().tolist())
            mean_steps += float(out.lengths.float().mean())
            ctx_reads += int((traces == model.registry.ctx).sum())
            self_loops += int(
                (
                    (traces[..., 1:] == traces[..., :-1])
                    & (traces[..., 1:] >= 0)
                    & (traces[..., 1:] < model.registry.slots)
                ).sum()
            )
            for a in range(model.registry.slots):
                visits[str(a)] += int((traces == a).sum())
            tick = time.perf_counter()
            for state in out.probes:
                legal = model.registry.legal(
                    state.depth, model.config["budget"], model.config["minimum_actions"], device
                )
                candidates = proposals(
                    legal,
                    state.chosen,
                    model.registry.stop,
                    c["oracle"]["extra_actions"],
                    self.rng["proposal"],
                )
                suffix_seed = int(
                    torch.randint(2**31 - 1, (), generator=self.rng["suffix"], device=device)
                )
                suffix_gen = torch.Generator(device=device).manual_seed(suffix_seed)
                # Same underlying uniforms by absolute depth for every alternative at this state.
                uniforms = torch.rand(
                    c["oracle"]["suffix_samples"],
                    model.config["budget"],
                    generator=suffix_gen,
                    device=device,
                )
                target = y[state.token_index]
                for a, inclusion in candidates:
                    label = snapshot.label(
                        state, a, target, uniforms, inclusion=inclusion, suffix_seed=suffix_seed
                    )
                    rows.append((state, target.detach(), label))
                    cf_work += label.token_work
            oracle_s += time.perf_counter() - tick
        snapshot.assert_current(model)
        q_value = None
        q_mae = None
        q_accuracy = None
        if rows:
            errors = []
            grouped = {}
            with torch.no_grad():
                for state, target, label in rows:
                    legal = model.registry.legal(
                        state.depth, model.config["budget"], model.config["minimum_actions"], device
                    )
                    pred, _ = model.controller(
                        state.hidden[None],
                        torch.tensor([state.previous], device=device),
                        state.depth,
                        legal,
                        model.registry.version,
                    )
                    value = float(pred[0, label.action])
                    errors.append(abs(value - label.value))
                    grouped.setdefault(state.fingerprint(), []).append((value, label.value))
            q_mae = sum(errors) / len(errors)
            comparisons = []
            for values in grouped.values():
                for j, (a, b) in enumerate(values):
                    for aa, bb in values[j + 1 :]:
                        if abs(b - bb) > 1e-8:
                            comparisons.append(int((a - aa) * (b - bb) > 0))
            q_accuracy = sum(comparisons) / len(comparisons) if comparisons else None
        if model.arm == "QF-FULL" and rows:
            q, critic = q_loss(model, rows, snapshot)
            (q + t["critic_coefficient"] * critic).backward()
            q_value = float(q.detach())
            critic_sum = float(critic.detach())
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
            self.opt.zero_grad(set_to_none=True)
            raise OracleUntrusted("nonfinite gradient; no update committed")
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip"])
        # Preserve exact AdamW state on injected or actual optimizer failure.
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        opt_backup = deepcopy(self.opt.state_dict())
        try:
            warmup = max(1, t["warmup_steps"])
            scale = min(1.0, (self.step + 1) / warmup)
            for group, lr in zip(
                self.opt.param_groups, (t["learning_rate"], t["connection_learning_rate"])
            ):
                group["lr"] = lr * scale
            self.opt.step()
            if failure:
                failure()
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise OracleUntrusted("nonfinite parameter update")
        except Exception:
            model.load_state_dict(backup)
            self.opt.load_state_dict(opt_backup)
            self.opt.zero_grad(set_to_none=True)
            raise
        self.step += 1
        model.version += 1
        model.registry.model_version += 1
        model.registry.policy_version += 1
        self.data_position += total_positions
        self.cache.clear()
        row = {
            "step": self.step,
            "arm": model.arm,
            "model_version": model.registry.model_version,
            "policy_version": model.registry.policy_version,
            "snapshot_sha": snapshot.sha,
            "policy_sha": snapshot.policy_sha,
            "live_ce": ce_sum / len(batches),
            "path_cost": cost / len(batches),
            "actual_stop_rate": stops / len(batches),
            "forced_end_rate": forced / len(batches),
            "q_loss": q_value,
            "actor_loss": actor_sum / len(batches),
            "critic_loss": critic_sum,
            "candidate_labels": len(rows),
            "q_mae": q_mae,
            "rank_spearman": None,
            "q_pairwise_accuracy": q_accuracy,
            "seed": c["seed"],
            "train_token_count": self.data_position,
            "q_label_count": len(rows),
            "epsilon": c["policy"]["epsilon"],
            "temperature": c["policy"]["temperature"],
            "lambda_cost": snapshot.policy["lambda_cost"],
            "mean_steps": mean_steps / len(batches),
            "ctx_reads": ctx_reads,
            "self_loops": self_loops,
            "cell_visits": visits,
            "train_seconds": live_s,
            "rollout_seconds": forward_s,
            "backward_seconds": backward_s,
            "timer_scope": "host enqueue wall; synchronized total measured only in isolated GPU Pilot",
            "shadow_seconds": 0.0,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device)
            if str(device).startswith("cuda")
            else None,
            "live_tokens": total_positions,
            "counterfactual_operator_calls": cf_work,
            "counterfactual_token_steps": cf_work,
            "shadow_training_tokens": 0,
            "live_seconds": live_s,
            "oracle_seconds": oracle_s,
            "snapshot_seconds": snapshot_s,
            "total_seconds": time.perf_counter() - begin,
            "rng_sha": {k: canonical_sha256(v.get_state().tolist()) for k, v in self.rng.items()},
            "label_record_hashes": [r[2].record_sha for r in rows],
            "checkpoint_sha256": None,
        }
        self.journal.append(row)
        return row, path_records, rows
