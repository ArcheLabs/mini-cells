"""R2 accepted transaction, visited-state supervision, separate gradient budgets."""

from copy import deepcopy
import itertools
import time
import random
import numpy as np
import torch
from torch.nn import functional as F
from pd001.losses import joint_loss, terminal_ce
from clm001.training import optimizer
from .oracle import Snapshot, paired
from .sampling import streams, proposals
from .common import epsilon


class NumericalFailure(RuntimeError):
    def __init__(self, stage, details):
        self.stage = stage
        self.details = details
        super().__init__(stage + ": " + str(details))


class NoLearningSignal(RuntimeError):
    pass


class ContractFailure(RuntimeError):
    pass


def finite_named(model, grad=False):
    values = [p.grad if grad else p for p in model.parameters() if not grad or p.grad is not None]
    # Single normal-path host check. Per-parameter scans run only on a detected fault.
    if not values or bool(
        torch.isfinite(torch.stack(torch._foreach_norm(values, float("inf")))).all()
    ):
        return
    for name, p in model.named_parameters():
        v = p.grad if grad else p
        if v is not None and not torch.isfinite(v).all():
            raise NumericalFailure(
                "gradient" if grad else "parameter",
                {
                    "parameter_name": name,
                    "shape": list(v.shape),
                    "dtype": str(v.dtype),
                    "nan_count": int(v.isnan().sum()),
                    "posinf_count": int(torch.isposinf(v).sum()),
                    "neginf_count": int(torch.isneginf(v).sum()),
                },
            )


def grouped_clip(model, c):
    result = {}
    for group, prefix in [("continuous", False), ("connection", True)]:
        params = [
            p
            for n, p in model.named_parameters()
            if n.startswith("controller.") == prefix and p.grad is not None
        ]
        limit = c["training"]["clip_" + group]
        try:
            norm = (
                torch.nn.utils.clip_grad_norm_(params, limit, error_if_nonfinite=True)
                if params
                else torch.tensor(0.0)
            )
        except RuntimeError as error:
            raise NumericalFailure(
                "gradient_norm", {"group": group, "error": str(error)}
            ) from error
        result["grad_norm_" + group + "_preclip"] = float(norm)
        result["grad_norm_" + group + "_postclip"] = float(
            torch.sqrt(
                sum(
                    (p.grad.square().sum() for p in params),
                    torch.tensor(0.0, device=next(model.parameters()).device),
                )
            )
        )
    return result


def relative_loss(model, rows, snapshot, c):
    groups = {}
    for state, target, lab in rows:
        lab.verify(snapshot, state, target)
        groups.setdefault(state.fingerprint(), []).append((state, target, lab))
    regression = []
    rank = []
    pairs_valid = 0
    pairs_total = 0
    errors = []
    gaps = []
    accuracy = []
    for records in groups.values():
        state = records[0][0]
        legal = model.registry.legal(
            state.depth,
            model.config["budget"],
            model.config["minimum_actions"],
            state.hidden.device,
        )
        pred, _ = model.controller(
            state.hidden[None].detach(),
            torch.tensor([state.previous], device=state.hidden.device),
            state.depth,
            legal,
            model.registry.version,
            detach_state=True,
        )
        errors.extend(abs(float(pred[0, lab.action].detach()) - lab.value) for _, _, lab in records)
        gaps.extend(lab.value for _, _, lab in records)
        terms = [
            F.smooth_l1_loss(pred[0, lab.action], pred.new_tensor(lab.value), reduction="sum")
            / lab.inclusion
            for _, _, lab in records
        ]
        # HT denominator is the population size, never the random sum of inverse weights.
        population = int(legal[:-1].sum())
        regression.append(torch.stack(terms).sum() / population)
        alternatives = [(lab.action, lab.samples) for _, _, lab in records]
        if legal[-1]:
            alternatives.append((model.registry.stop, [0.0] * len(records[0][2].samples)))
        for (a, x), (b, y) in itertools.combinations(alternatives, 2):
            info = paired(x, y, c["oracle"])
            pairs_total += 1
            if not info["identifiable"]:
                continue
            pairs_valid += 1
            accuracy.append(int(float((pred[0, a] - pred[0, b]).detach()) * info["difference"] > 0))
            # d<0: a better; softplus(pred_a-pred_b) penalizes reversing that order.
            direction = 1.0 if info["difference"] < 0 else -1.0
            weight = min(
                1.0,
                abs(info["difference"])
                / (c["oracle"]["pair_tolerance"] + c["oracle"]["pair_z"] * info["se"] + 1e-12),
            )
            rank.append(
                weight
                * F.softplus(
                    direction * (pred[0, a] - pred[0, b]) / c["training"]["rank_temperature"]
                )
            )
    zero = next(model.parameters()).new_zeros(())
    return (
        torch.stack(regression).mean() if regression else zero,
        torch.stack(rank).mean() if rank else zero,
        {
            "q_mae": sum(errors) / len(errors) if errors else None,
            "q_target_delta_mean": sum(gaps) / len(gaps) if gaps else None,
            "q_pairwise_accuracy": sum(accuracy) / len(accuracy) if accuracy else None,
            "rank_pairs_valid": pairs_valid,
            "rank_pairs_total": pairs_total,
            "rank_estimator": "biased_auxiliary_sampled_pairs",
        },
    )


class Trainer:
    def __init__(self, model):
        self.model = model
        self.opt = optimizer(model)
        self.rng = streams(model.experiment_config["seed"], next(model.parameters()).device)
        self.step = 0
        self.data_position = 0
        self.cache = {}
        self.journal = []

    def update(self, batches, *, role="train", failure=None):
        if role != "train":
            raise ValueError("only Train produces gradients")
        # Snapshot all transaction state including RNG before consuming a trajectory.
        backup = {
            "step": self.step,
            "data_position": self.data_position,
            "journal": deepcopy(self.journal),
            "cache": deepcopy(self.cache),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "model": deepcopy(self.model.state_dict()),
            "opt": deepcopy(self.opt.state_dict()),
            "rng": {k: g.get_state().clone() for k, g in self.rng.items()},
            "global": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "registry": deepcopy(self.model.registry),
            "version": self.model.version,
            "epsilon": self.model.experiment_config["policy"]["epsilon"],
        }
        try:
            return self._update(batches, failure)
        except Exception:
            self.step = backup["step"]
            self.data_position = backup["data_position"]
            self.journal = backup["journal"]
            self.cache = backup["cache"]
            np.random.set_state(backup["numpy"])
            random.setstate(backup["python"])
            self.model.load_state_dict(backup["model"])
            self.opt.load_state_dict(backup["opt"])
            self.model.registry = backup["registry"]
            self.model.version = backup["version"]
            self.model.experiment_config["policy"]["epsilon"] = backup["epsilon"]
            for k, g in self.rng.items():
                g.set_state(backup["rng"][k].cpu())
            torch.set_rng_state(backup["global"])
            if backup["cuda"] is not None:
                torch.cuda.set_rng_state_all(backup["cuda"])
            self.opt.zero_grad(set_to_none=True)
            raise

    def _update(self, batches, failure):
        start = time.perf_counter()
        m = self.model
        c = m.experiment_config
        t = c["training"]
        device = next(m.parameters()).device
        if len(batches) != t["effective_batch"] // t["microbatch"]:
            raise ValueError("fixed effective batch")
        c["policy"]["epsilon"] = epsilon(c, self.step + 1)
        finite_named(m)
        snapshot = Snapshot(m)
        snapshot_seconds = time.perf_counter() - start
        self.opt.zero_grad(set_to_none=True)
        m.train()
        pool = {}
        populations = {}
        rows = []
        ce_sum = 0.0
        cost = 0.0
        stops = 0.0
        forced = 0.0
        ctx = 0
        calls = 0
        first_context_counts = {}
        no_context_tokens = 0
        cell_visits = {str(a): 0 for a in range(m.registry.slots)}
        actor = 0.0
        critic = 0.0
        live_begin = time.perf_counter()
        for x, y in batches:
            try:
                out = m(x, generator=self.rng["live"], probe_coordinates="reservoir")
            except ValueError as error:
                if str(error) in (
                    "illegal accepted action/STOP",
                    "no legal actions at a nonterminal decision",
                ):
                    raise ContractFailure(str(error)) from error
                raise
            ce = terminal_ce(out, y).mean()
            ce_sum += float(ce.detach())
            if m.arm == "J0-R2":
                loss, metrics = joint_loss([out], y, t, "J0", c["policy"]["lambda_cost"])
                actor += float(metrics["actor_loss"])
                critic += float(metrics["critic_loss"])
            else:
                loss = ce
            if not torch.isfinite(loss):
                raise NumericalFailure("loss", {"component": "live"})
            if loss.requires_grad:
                (loss / len(batches)).backward()
            cost += float(out.cost_units.mean())
            stops += float(out.actual_stop.float().mean())
            forced += float(out.forced_end.float().mean())
            ctx += int((out.actions == m.registry.ctx).sum())
            mask = out.actions == m.registry.ctx
            no_context_tokens += int((~mask.any(-1)).sum())
            first = mask.to(torch.long).argmax(-1)
            histogram = (
                torch.bincount(first[mask.any(-1)], minlength=m.config["budget"]).cpu().tolist()
            )
            for depth, count in enumerate(histogram):
                first_context_counts[str(depth)] = first_context_counts.get(str(depth), 0) + count
            selected = out.actions[(out.actions >= 0) & (out.actions < m.registry.slots)]
            for a, count in enumerate(
                torch.bincount(selected, minlength=m.registry.slots).cpu().tolist()
            ):
                cell_visits[str(a)] += count
            calls += int(out.lengths.sum())
            for state in out.probes:
                b = state.bucket
                n = state.population
                old = populations.get(b, 0)
                populations[b] = old + n
                pick = int(
                    torch.randint(old + n, (), generator=self.rng["proposal"], device=device)
                )
                if pick >= old:
                    pool[b] = (state, y[state.token_index].detach())
        live_seconds = time.perf_counter() - live_begin
        # Alternating deeper quota, shallow quota, target-free fallback. Distribution is explicitly stratified.
        available = sorted(pool)
        preferred = [
            0,
            1 + (self.step % max(1, (m.config["budget"] - 1) // c["oracle"]["depth_bucket_width"])),
        ]
        chosen = []
        for b in preferred + available:
            if b in pool and b not in chosen:
                chosen.append(b)
            if len(chosen) == c["oracle"]["probes_per_step"]:
                break
        oracle_begin = time.perf_counter()
        work = 0
        sample_info = []
        for b in chosen:
            state, target = pool[b]
            legal = m.registry.legal(
                state.depth, m.config["budget"], m.config["minimum_actions"], device
            )
            candidates = proposals(
                legal,
                state.chosen,
                m.registry.stop,
                c["oracle"]["extra_actions"],
                self.rng["proposal"],
            )
            seed = int(torch.randint(2**31 - 1, (), device=device, generator=self.rng["suffix"]))
            uniforms = torch.rand(
                8,
                m.config["budget"],
                device=device,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
            # Reserve initial samples for every remaining probe before adaptive refinement.
            remaining = len(chosen) - len(sample_info) - 1
            reserve = remaining * 2 * (c["oracle"]["extra_actions"] + 1) * m.config["budget"]
            labels, info = snapshot.estimate(
                state,
                target,
                candidates,
                uniforms,
                c["oracle"],
                c["oracle"]["max_counterfactual_operator_calls_per_step"] - work - reserve,
                seed,
            )
            work += info["operator_calls"]
            sample_info.append(info)
            rows.extend((state, target, lab) for lab in labels)
        snapshot.assert_current(m)
        oracle_seconds = time.perf_counter() - oracle_begin
        q = next(m.parameters()).new_zeros(())
        rank = q
        stats = {"rank_pairs_valid": 0, "rank_pairs_total": 0}
        if m.arm == "QF-R2":
            q, rank, stats = relative_loss(m, rows, snapshot, c)
            qtotal = q + t["rank_loss_weight"] * rank
            if not torch.isfinite(qtotal):
                raise NumericalFailure("loss", {"component": "relative_q"})
            if qtotal.requires_grad:
                qtotal.backward()
        finite_named(m, grad=True)
        if not any(p.grad is not None for p in m.parameters()):
            raise NoLearningSignal("no live or probe supervision; no optimizer update")
        norms = grouped_clip(m, c)
        for group, lr in zip(
            self.opt.param_groups, (t["learning_rate"], t["connection_learning_rate"])
        ):
            group["lr"] = lr * min(1.0, (self.step + 1) / max(1, t["warmup_steps"]))
        self.opt.step()
        if failure:
            failure()
        finite_named(m)
        self.step += 1
        m.version += 1
        m.registry.model_version += 1
        m.registry.policy_version += 1
        self.data_position += sum(x.numel() for x, _ in batches)
        self.cache.clear()
        row = {
            "event": "train_step",
            "step": self.step,
            "arm": m.arm,
            "live_ce": ce_sum / len(batches),
            "path_cost": cost / len(batches),
            "actual_stop_rate": stops / len(batches),
            "forced_end_rate": forced / len(batches),
            "q_loss": float(q.detach()),
            "loss_rel": float(q.detach()),
            "loss_rank": float(rank.detach()),
            "actor_loss": actor / len(batches),
            "critic_loss": critic / len(batches),
            "ctx_reads": ctx,
            "first_context_depth_counts": first_context_counts,
            "no_context_token_fraction": no_context_tokens / sum(x.numel() for x, _ in batches),
            "cell_visits": cell_visits,
            "parameters_without_gradient": [
                name for name, p in m.named_parameters() if p.grad is None
            ],
            "operator_calls": calls,
            "oracle_states": len(chosen),
            "candidate_labels": len(rows),
            "counterfactual_operator_calls": work,
            "counterfactual_token_steps": work,
            "snapshot_sha": snapshot.sha,
            "policy_sha": snapshot.policy_sha,
            "epsilon": c["policy"]["epsilon"],
            "live_tokens": sum(x.numel() for x, _ in batches),
            "train_token_count": self.data_position,
            "probe_bucket_population": populations,
            "probe_buckets_selected": chosen,
            "probe_state_measure": "alternating_stratified_visited_not_natural_event_mean",
            "probe_conditional_inclusion": {str(b): 1 / populations[b] for b in chosen},
            "coverage_missing": [b for b in preferred if b not in pool],
            "oracle_sampling": sample_info,
            "live_seconds": live_seconds,
            "oracle_seconds": oracle_seconds,
            "snapshot_seconds": snapshot_seconds,
            "total_seconds": time.perf_counter() - start,
            "timer_scope": "host wall; CUDA synchronization only in Pilot",
            **norms,
            **stats,
        }
        self.journal.append(row)
        return row, snapshot, rows
