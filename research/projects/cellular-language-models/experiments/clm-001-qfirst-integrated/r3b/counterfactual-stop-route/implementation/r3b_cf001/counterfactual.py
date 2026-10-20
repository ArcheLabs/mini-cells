"""Two explicitly separate suffix policies on reconstructed real visited states."""

import math
import numpy as np
import torch
from clm001_r3.oracle import Snapshot
from clm001_r3.sampling import distribution
from minicells_runtime.store import canonical, ContinuationError


class GreedySnapshot(Snapshot):
    def probabilities(self, h, previous, depth):
        m = self.model
        legal = m.registry.legal(depth, m.config["budget"], m.config["minimum_actions"], h.device)
        q, _ = m.controller(
            h[None], torch.tensor([previous], device=h.device), depth, legal, m.registry.version
        )
        p = distribution(q, legal, self.policy["epsilon"], self.policy["temperature"], "greedy")[0]
        if m.registry.plan and depth < len(m.registry.plan):
            p = torch.nn.functional.one_hot(
                torch.tensor(m.registry.plan[depth], device=h.device), m.registry.actions
            ).to(p)
        return p


def uniforms(record, seed, samples, budget, device):
    key = [
        seed,
        record["story_id"],
        record["window_id"],
        record["token_position"],
        record["depth_zero_based"],
        record["state_fingerprint"],
    ]
    local = int(canonical(key)[:16], 16) % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(local)
    return torch.rand(samples, budget, generator=generator).to(device), local


@torch.no_grad()
def audit_state(greedy_snapshot, sampled_snapshot, state, target, record, config):
    model = sampled_snapshot.model
    r = model.registry
    if not record["stop_legal"] or state.chosen == r.stop:
        raise ContinuationError("CF_INELIGIBLE_STOP_STATE")
    if r.plan and state.depth < len(r.plan) and r.plan[state.depth] != r.stop:
        raise ContinuationError("CF_FROZEN_PLAN_CONFLICT")
    if abs(record["q_stop"]) > config["q_zero_tolerance"]:
        raise ContinuationError("CF_STOP_ZERO_ANCHOR_MISMATCH")
    samples = config["mc_samples"]
    u, seed = uniforms(
        record, config["mc_seed"], samples, model.config["budget"], state.hidden.device
    )
    stop = sampled_snapshot.terminal(state.hidden, target)
    exact, greedy_trace, greedy_calls = greedy_snapshot.suffix(state, state.chosen, target, u[0])
    # Reuse original STOP-anchored Oracle, with fixed 8 CRN suffixes; no adaptive expansion.
    suffixes = [sampled_snapshot.suffix(state, state.chosen, target, v) for v in u]
    values = [v[0] for v in suffixes]
    true_adv = np.asarray(values, dtype=np.float64) - stop
    se = float(true_adv.std(ddof=1) / math.sqrt(samples)) if samples > 1 else None
    mean = float(true_adv.mean())
    interval = [mean - config["mc_z"] * se, mean + config["mc_z"] * se] if se is not None else None
    pred = record["q_chosen"] - record["q_stop"]
    sampled_status = (
        "UNRESOLVED"
        if interval is None or interval[0] <= 0 <= interval[1]
        else "STOP_BETTER"
        if interval[0] > 0
        else "CONTINUE_BETTER"
    )
    return {
        "story_id": record["story_id"],
        "window_id": record["window_id"],
        "token_position": record["token_position"],
        "depth": state.depth,
        "state_fingerprint": state.fingerprint(),
        "previous_action": state.previous,
        "n_ctx_so_far": record["n_ctx_so_far"],
        "forced_end_path": record["forced_end_path"],
        "design_weight": record["design_weight"],
        "inclusion_probability": record["inclusion_probability"],
        "pred_stop_advantage": pred,
        "stop_terminal_nll": stop,
        "stop_remaining_operator_calls": 0,
        "pred_gap_stop_chosen": -pred,
        "greedy_consistent": {
            "true_stop_advantage": exact - stop,
            "continue_cost_aware_utility": exact,
            "remaining_cost": greedy_calls / model.config["budget"],
            "remaining_operator_calls": greedy_calls,
            "suffix_actions": greedy_trace,
            "true_gap_stop_chosen": stop - exact,
            "stop_better": exact > stop,
            "misranking": pred < 0 and exact > stop,
        },
        "frozen_sampled_oracle": {
            "true_stop_advantage": mean,
            "true_gap_stop_chosen": -mean,
            "samples": samples,
            "mc_se": se,
            "mc_interval": interval,
            "status": sampled_status,
            "misranking": pred < 0 and sampled_status == "STOP_BETTER",
            "sample_values": values,
            "sample_remaining_operator_calls": [v[2] for v in suffixes],
            "sample_suffix_actions": [v[1] for v in suffixes],
            "crn_seed": seed,
            "uniforms_sha256": canonical(u.cpu().tolist()),
        },
        "ctx_auxiliary_truth": "NOT_MEASURED_FIXED_BUDGET",
        "optimizer_updates": 0,
    }
