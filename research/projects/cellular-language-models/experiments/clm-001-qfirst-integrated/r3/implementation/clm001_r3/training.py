"""Frozen R2 accepted transaction/Q objective plus router-only local competition."""

from copy import deepcopy
import time
import random
import numpy as np
import torch
from pd001.losses import terminal_ce
from clm001_r2.training import (
    NumericalFailure,
    NoLearningSignal,
    ContractFailure,
    optimizer,
    finite_named,
    grouped_clip,
    relative_loss,
)
from .oracle import Snapshot
from .sampling import streams, proposals
from .common import epsilon
from .probing import EventSampler
from .training_errors import CreditFailure
from .competitive_credit import loss as credit_loss, vector

__all__ = [
    "Trainer",
    "NumericalFailure",
    "NoLearningSignal",
    "ContractFailure",
    "finite_named",
    "grouped_clip",
    "relative_loss",
]


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
        live_begin = time.perf_counter()
        credit_events = []
        exposure = {0: {}, 1: {}}
        micro_index = 0
        for x, y in batches:
            try:
                quota = c["competition"]["events_per_step"] // len(batches) + (
                    micro_index < c["competition"]["events_per_step"] % len(batches)
                )
                sampler = EventSampler(
                    int(quota),
                    self.rng["competition"],
                    c["competition"]["depth_bucket_width"],
                    features=x.reshape(-1) % 2,
                )
                out = m(
                    x,
                    generator=self.rng["live"],
                    probe_coordinates="reservoir",
                    competition_sampler=sampler,
                )
            except ValueError as error:
                if str(error) in (
                    "illegal accepted action/STOP",
                    "no legal actions at a nonterminal decision",
                ):
                    raise ContractFailure(str(error)) from error
                raise
            ce = terminal_ce(out, y).mean()
            ce_sum += float(ce.detach())
            loss = ce
            if not torch.isfinite(loss):
                raise NumericalFailure("loss", {"component": "live"})
            if loss.requires_grad:
                (loss / len(batches)).backward()
            credit_events.extend(sampler.events)
            for group, counts in sampler.exposure.items():
                for action, count in counts.items():
                    exposure[group][action] = exposure[group].get(action, 0) + count
            sampler.close()
            micro_index += 1
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
        if m.arm in c["arms"]:
            q, rank, stats = relative_loss(m, rows, snapshot, c)
            qtotal = q + t["rank_loss_weight"] * rank
            if not torch.isfinite(qtotal):
                raise NumericalFailure("loss", {"component": "relative_q"})
            parameters = [p for p in m.controller.parameters() if p.requires_grad]
            qgrads = (
                torch.autograd.grad(qtotal, parameters, retain_graph=True, allow_unused=True)
                if qtotal.requires_grad
                else [None] * len(parameters)
            )
            qvector = vector(qgrads, parameters).detach()
            competition = next(m.parameters()).new_zeros(())
            credit_records = []
            candidate_calls = 0
            credit_norm = None
            cosine = None
            auxiliary = m.arm in c["competition"]["enabled_arms"]
            if auxiliary:
                competition, credit_records, candidate_calls = credit_loss(
                    m, credit_events, self.rng["competition"]
                )
                if competition.requires_grad:
                    cgrads = torch.autograd.grad(
                        competition, parameters, retain_graph=True, allow_unused=True
                    )
                    cvector = vector(cgrads, parameters).detach()
                    if not torch.isfinite(cvector).all():
                        raise CreditFailure("nonfinite local surrogate gradient")
                    credit_norm = float(cvector.norm())
                    cosine = (
                        float(
                            torch.dot(qvector, cvector) / (qvector.norm() * cvector.norm() + 1e-30)
                        )
                        if credit_norm > 0 and float(qvector.norm()) > 0
                        else None
                    )
                    beta = c["competition"]["beta"]
                    if beta is None:
                        if not c.get("pilot_rehearsal") and not c.get("synthetic"):
                            raise RuntimeError("LCG beta must be frozen by gradient Pilot")
                        beta = c["competition"]["pilot_beta"]
                    # Zero-credit sham computes the same forward/gradient diagnostics, but applies none.
                    if m.arm == "QC-R3":
                        (beta * competition).backward()
                    elif m.arm == "QS-R3":
                        (0.0 * competition).backward()
            if qtotal.requires_grad:
                qtotal.backward()
            self.last_credit = credit_records
            stats.update(
                lcg_loss=float(competition.detach()) if auxiliary else None,
                lcg_events=len(credit_events),
                lcg_candidate_cell_calls=candidate_calls,
                lcg_grad_norm=credit_norm,
                q_grad_norm=float(qvector.norm()),
                lcg_q_cosine=cosine,
                sham=c["competition"]["sham"] if m.arm == "QS-R3" else None,
                lcg_beta=c["competition"]["beta"],
                lcg_depth_bucket_histogram={
                    str(b): sum(
                        e.state.depth // c["competition"]["depth_bucket_width"] == b
                        for e in credit_events
                    )
                    for b in range(4)
                },
                cell_exposure_by_group=exposure,
                cell_exposure_scope="all_executed_cell_decision_events_initial_token_parity_groups",
            )
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
            "actor_loss": None,
            "critic_loss": None,
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
