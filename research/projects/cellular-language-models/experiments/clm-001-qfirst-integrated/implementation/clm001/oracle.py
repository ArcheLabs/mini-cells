"""Immutable full-suffix counterfactual Oracle; no cached labels across optimizer steps."""

from copy import deepcopy
from dataclasses import dataclass, asdict
import math
import torch
from torch.nn import functional as F
from .common import hash_accepted_state, canonical_sha256, tensor_hash
from .sampling import distribution, inverse_cdf


class OracleUntrusted(RuntimeError):
    pass


@dataclass
class Label:
    action: int
    value: float
    variance: float
    inclusion: float
    snapshot_sha: str
    registry_sha: str
    state_sha: str
    target_sha: str
    policy_sha: str
    uniforms_sha: str
    suffix_seed: int
    traces: list
    token_work: int
    record_sha: str = ""
    q_sample_values: list = None
    remaining_cost: list = None
    available_actions: list = None
    source_action: int = -1
    suffix_policy_kind: str = "epsilon_boltzmann_frozen_q"
    candidate_status: str = "MEASUREMENT_ONLY"

    def seal(self):
        payload = asdict(self)
        payload.pop("record_sha")
        self.record_sha = canonical_sha256(payload)
        return self

    def verify(self, snapshot, state, target):
        payload = asdict(self)
        payload.pop("record_sha")
        if self.record_sha != canonical_sha256(payload):
            raise OracleUntrusted("label payload was changed")
        if (
            self.snapshot_sha != snapshot.sha
            or self.registry_sha != snapshot.registry_sha
            or self.state_sha != state.fingerprint()
        ):
            raise OracleUntrusted("stale snapshot/state/registry")
        if (
            self.target_sha != tensor_hash(target.reshape(()))
            or self.policy_sha != snapshot.policy_sha
        ):
            raise OracleUntrusted("target/policy identity differs")
        if (
            not math.isfinite(self.value)
            or not math.isfinite(self.variance)
            or not 0 < self.inclusion <= 1
        ):
            raise OracleUntrusted("nonfinite or invalid label")


class Snapshot:
    def __init__(self, model):
        self.model = deepcopy(model).eval()
        self.model.requires_grad_(False)
        if model.experiment_config["policy"]["suffix_kind"] != "epsilon_boltzmann_frozen_q":
            raise OracleUntrusted("unrecognized suffix policy")
        self.config_sha = canonical_sha256(model.experiment_config)
        self.execution_config_sha = canonical_sha256(model.config)
        self.sha = hash_accepted_state(model)
        self.registry_sha = model.registry.fingerprint
        c = model.experiment_config
        self.policy = {
            "epsilon": c["policy"]["epsilon"],
            "temperature": c["policy"]["temperature"],
            "lambda_cost": c["policy"]["lambda_cost"]
            if c["policy"]["lambda_cost"] is not None
            else c["policy"]["lambda_candidate"],
            "suffix_kind": c["policy"]["suffix_kind"],
            "registry": self.registry_sha,
            "weights": self.sha,
            "budget": c["model"]["budget"],
            "minimum_actions": c["model"]["minimum_actions"],
        }
        self.policy_sha = canonical_sha256(self.policy)

    def assert_integrity(self):
        if (
            canonical_sha256(self.policy) != self.policy_sha
            or canonical_sha256(self.model.experiment_config) != self.config_sha
            or canonical_sha256(self.model.config) != self.execution_config_sha
            or hash_accepted_state(self.model) != self.sha
            or self.model.registry.fingerprint != self.registry_sha
        ):
            raise OracleUntrusted("snapshot weights/policy/registry were changed")

    def assert_current(self, model):
        if (
            self.sha != hash_accepted_state(model)
            or self.registry_sha != model.registry.fingerprint
        ):
            raise OracleUntrusted("accepted state mutated during snapshot Oracle")

    def check(self, state, action):
        m = self.model
        c = m.experiment_config
        if state.registry_sha != self.registry_sha:
            raise OracleUntrusted("stale probe registry")
        if not m.registry.legal(
            state.depth, c["model"]["budget"], c["model"]["minimum_actions"], state.hidden.device
        )[action]:
            raise OracleUntrusted("illegal candidate")

    def probabilities(self, h, previous, depth):
        m = self.model
        c = m.experiment_config
        legal = m.registry.legal(
            depth, c["model"]["budget"], c["model"]["minimum_actions"], h.device
        )
        q, _ = m.controller(
            h[None], torch.tensor([previous], device=h.device), depth, legal, m.registry.version
        )
        p = distribution(q, legal, self.policy["epsilon"], self.policy["temperature"], "sampled")[0]
        if m.registry.plan and depth < len(m.registry.plan):
            a = m.registry.plan[depth]
            if not legal[a]:
                raise OracleUntrusted("illegal causal path plan")
            p = F.one_hot(torch.tensor(a, device=h.device), m.registry.actions).to(p)
        return p

    def advance(self, h, action, state):
        if action == self.model.registry.stop:
            return h
        return self.model.execute(
            h[None],
            action,
            state.memory,
            torch.tensor([state.batch], device=h.device),
            torch.tensor([state.position], device=h.device),
        )[0]

    def terminal(self, h, target):
        return F.cross_entropy(self.model.head(h[None]), target.reshape(1)).item()

    @torch.no_grad()
    def suffix(self, state, action, target, uniforms):
        self.check(state, action)
        m = self.model
        budget = m.config["budget"]
        h = state.hidden.clone()
        trace = []
        previous = state.previous
        work = 0
        for depth in range(state.depth, budget):
            if depth > state.depth:
                action = int(
                    inverse_cdf(
                        self.probabilities(h, previous, depth)[None], uniforms[depth].reshape(1)
                    )[0]
                )
            trace.append(action)
            if action == m.registry.stop:
                break
            h = self.advance(h, action, state)
            work += 1
            previous = action
        return self.terminal(h, target) + self.policy["lambda_cost"] * work / budget, trace, work

    @torch.no_grad()
    def label(self, state, action, target, uniforms, *, inclusion=1.0, suffix_seed=0):
        self.assert_integrity()
        values = []
        traces = []
        work = 0
        costs = []
        for sample in uniforms:
            value, trace, count = self.suffix(state, action, target, sample)
            values.append(value)
            traces.append(trace)
            costs.append(count / self.model.config["budget"])
            work += count
        if not values:
            raise OracleUntrusted("no suffix samples")
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        out = Label(
            action,
            mean,
            variance,
            inclusion,
            self.sha,
            self.registry_sha,
            state.fingerprint(),
            tensor_hash(target.reshape(())),
            self.policy_sha,
            tensor_hash(uniforms),
            suffix_seed,
            traces,
            work,
            q_sample_values=values,
            remaining_cost=costs,
            available_actions=torch.where(
                self.model.registry.legal(
                    state.depth,
                    self.model.config["budget"],
                    self.model.config["minimum_actions"],
                    state.hidden.device,
                )
            )[0].tolist(),
            source_action=state.chosen,
        ).seal()
        out.verify(self, state, target)
        return out

    @torch.no_grad()
    def exact(self, state, action, target):
        """Exhaustive expectation for short CPU worlds only; includes STOP and forced end."""
        self.assert_integrity()
        self.check(state, action)
        budget = self.model.config["budget"]
        stop = self.model.registry.stop
        lam = self.policy["lambda_cost"] / budget
        if budget - state.depth > 4:
            raise ValueError("exact enumeration restricted to at most four decisions")

        def branch(h, a, depth):
            if a == stop:
                return self.terminal(h, target)
            h = self.advance(h, a, state)
            if depth + 1 == budget:
                return lam + self.terminal(h, target)
            p = self.probabilities(h, a, depth + 1)
            return lam + sum(
                float(p[i]) * branch(h, i, depth + 1) for i in range(len(p)) if p[i] > 0
            )

        return branch(state.hidden, action, state.depth)
