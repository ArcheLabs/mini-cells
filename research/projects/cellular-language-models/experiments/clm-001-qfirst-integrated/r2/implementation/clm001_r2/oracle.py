"""STOP anchored labels from immutable R1 full-suffix engine; paired CRN refinement."""

from dataclasses import dataclass, asdict
import math
import itertools
import torch
from clm001.oracle import Snapshot as AbsoluteSnapshot, OracleUntrusted
from .common import canonical_sha256
from .sampling import distribution


@dataclass
class RelativeLabel:
    absolute: object
    stop_loss: float
    value: float
    samples: list
    stop_legal: bool
    record_sha: str = ""

    def seal(self):
        d = asdict(self)
        d.pop("record_sha")
        self.record_sha = canonical_sha256(d)
        return self

    def verify(self, snapshot, state, target):
        self.absolute.verify(snapshot, state, target)
        d = asdict(self)
        d.pop("record_sha")
        if self.record_sha != canonical_sha256(d) or not math.isfinite(self.value):
            raise OracleUntrusted("relative label changed or nonfinite")
        if abs(self.value - (self.absolute.value - self.stop_loss)) > 1e-8:
            raise OracleUntrusted("STOP reference inconsistent")

    def __getattr__(self, name):
        # Original provenance, action inclusion, traces and work remain sealed.
        return getattr(self.absolute, name)


def paired(a, b, c):
    if len(a) != len(b):
        raise OracleUntrusted("CRN sample counts differ")
    d = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    mean = float(d.mean())
    se = float(d.std(unbiased=True) / math.sqrt(len(d))) if len(d) > 1 else float("inf")
    reliable = (
        len(d) >= c["pairwise_min_samples"] and abs(mean) > c["pair_tolerance"] + c["pair_z"] * se
    )
    return {"difference": mean, "se": se, "identifiable": reliable, "samples": len(d)}


class Snapshot(AbsoluteSnapshot):
    def probabilities(self, h, previous, depth):
        m = self.model
        legal = m.registry.legal(depth, m.config["budget"], m.config["minimum_actions"], h.device)
        q, _ = m.controller(
            h[None], torch.tensor([previous], device=h.device), depth, legal, m.registry.version
        )
        p = distribution(q, legal, self.policy["epsilon"], self.policy["temperature"], "sampled")[0]
        if m.registry.plan and depth < len(m.registry.plan):
            a = m.registry.plan[depth]
            if not legal[a]:
                raise OracleUntrusted("illegal causal plan")
            p = torch.nn.functional.one_hot(
                torch.tensor(a, device=h.device), m.registry.actions
            ).to(p)
        return p

    @torch.no_grad()
    def relative(
        self, state, action, target, uniforms, *, inclusion=1.0, suffix_seed=0, baseline=None
    ):
        baseline = self.terminal(state.hidden, target) if baseline is None else baseline
        lab = super().label(
            state, action, target, uniforms, inclusion=inclusion, suffix_seed=suffix_seed
        )
        legal = self.model.registry.legal(
            state.depth,
            self.model.config["budget"],
            self.model.config["minimum_actions"],
            state.hidden.device,
        )
        return RelativeLabel(
            lab,
            baseline,
            lab.value - baseline,
            [v - baseline for v in lab.q_sample_values],
            bool(legal[-1]),
        ).seal()

    @torch.no_grad()
    def estimate(self, state, target, candidates, uniforms, config, work_budget, seed=0):
        self.assert_integrity()
        stop = self.model.registry.stop
        baseline = self.terminal(state.hidden, target)
        candidates = [(a, p) for a, p in candidates if a != stop]
        rows = []
        used = 0
        truncated = False
        max_depth = self.model.config["budget"] - state.depth
        count = 0
        sampled = {a: [] for a, _ in candidates}
        traces = {a: [] for a, _ in candidates}
        costs = {a: [] for a, _ in candidates}
        for m in config["suffix_samples_allowed"]:
            # Conservative reservation prevents exceeding cap even when no suffix stops.
            if used + (m - count) * len(candidates) * max_depth > work_budget:
                truncated = True
                break
            for a, _ in candidates:
                for u in uniforms[count:m]:
                    value, trace, work = self.suffix(state, a, target, u)
                    sampled[a].append(value)
                    traces[a].append(trace)
                    costs[a].append(work / self.model.config["budget"])
                    used += work
            count = m
            arrays = [[v - baseline for v in sampled[a]] for a, _ in candidates] + [[0.0] * m]
            uncertainties = [paired(a, b, config) for a, b in itertools.combinations(arrays, 2)]
            ambiguous = any(
                abs(v["difference"]) <= config["pair_tolerance"] + config["pair_z"] * v["se"]
                for v in uncertainties
            )
            if not ambiguous:
                break
        if not count:
            raise OracleUntrusted("initial suffix budget unavailable")
        # Reuse sealed R1 label constructor without rerunning paths.
        from clm001.oracle import Label
        from .common import tensor_hash

        for a, p in candidates:
            values = sampled[a]
            mean = sum(values) / count
            absolute = Label(
                a,
                mean,
                sum((v - mean) ** 2 for v in values) / count,
                p,
                self.sha,
                self.registry_sha,
                state.fingerprint(),
                tensor_hash(target.reshape(())),
                self.policy_sha,
                tensor_hash(uniforms[:count]),
                seed,
                traces[a],
                round(sum(costs[a]) * self.model.config["budget"]),
                q_sample_values=values,
                remaining_cost=costs[a],
                available_actions=[
                    int(v)
                    for v in torch.where(
                        self.model.registry.legal(
                            state.depth,
                            self.model.config["budget"],
                            self.model.config["minimum_actions"],
                            target.device,
                        )
                    )[0]
                ],
                source_action=state.chosen,
            ).seal()
            lab = RelativeLabel(
                absolute,
                baseline,
                mean - baseline,
                [v - baseline for v in values],
                state.depth >= self.model.config["minimum_actions"],
            ).seal()
            lab.verify(self, state, target)
            rows.append(lab)
        self.assert_integrity()
        return rows, {
            "operator_calls": used,
            "samples": count,
            "budget_truncated": truncated,
            "stop_loss": baseline,
        }
