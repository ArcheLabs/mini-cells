"""Label-free Admission temperature fit; exact prefix-conditioned hard-path probability."""

import torch
from .evaluation import windows
from .sampler import probabilities
from minicells_r3f.stable_connection import baseline_action, hard_action


def summary(x):
    x = torch.cat([v.detach().double().cpu().reshape(-1) for v in x])
    return dict(
        mean=float(x.mean()),
        median=float(x.median()),
        quantiles={str(q): float(torch.quantile(x, q)) for q in [0.05, 0.25, 0.75, 0.95]},
        count=len(x),
    )


@torch.no_grad()
def temperature(m, c, data):
    records = []
    m.training_context = None
    m.observer = None
    for _, x, _ in windows(
        data,
        c,
        "admission",
        next(m.parameters()).device,
        c["alignment"]["reference_windows"],
        labels=False,
    ):
        for r in m(x, keep_trace=True).trace:
            q, _, z = m.scores(r["state"], r["stage"], r["previous"])
            a = hard_action(q, baseline_action(m, z), m.route_switch_margin)
            records.append((q, a))
    rows = []
    chosen = c["alignment"]["temperatures"][-1]
    reached = False
    for t in c["alignment"]["temperatures"]:
        p = torch.cat(
            [(q.double() / t).softmax(-1).gather(-1, a[:, None]).flatten() for q, a in records]
        )
        mean = float(p.mean())
        rows.append(dict(temperature=t, mean_hard_match_without_epsilon=mean))
        if mean >= c["alignment"]["target"]:
            chosen = t
            reached = True
            break
    return dict(
        temperature=chosen,
        status="TEMPERATURE_ALIGNMENT_TARGET_REACHED"
        if reached
        else "TEMPERATURE_ALIGNMENT_TARGET_NOT_REACHED",
        candidates=rows,
        target=c["alignment"]["target"],
        role="admission",
        uses_targets=False,
        frozen_once=True,
    )


@torch.no_grad()
def measure(m, c, data, epsilon):
    m.training_context = None
    m.observer = None
    stages = {}
    paths = []
    for _, x, _ in windows(
        data,
        c,
        "admission",
        next(m.parameters()).device,
        c["alignment"]["reference_windows"],
        labels=False,
    ):
        logs = []
        for r in m(x, keep_trace=True).trace:
            q, _, z = m.scores(r["state"], r["stage"], r["previous"])
            hard = hard_action(q, baseline_action(m, z), m.route_switch_margin)
            p = probabilities(q, hard, m.routing_mode, epsilon, m.temperature, m.rho)
            match = p.gather(-1, hard[:, None]).flatten()
            entropy = -(p * p.clamp_min(1e-300).log()).sum(-1) / torch.log(
                torch.tensor(p.shape[-1], dtype=p.dtype, device=p.device)
            )
            row = stages.setdefault(str(r["stage"]), {"match": [], "entropy": []})
            row["match"].append(match)
            row["entropy"].append(entropy)
            logs.append(match.log())
        paths.append(torch.stack(logs, -1).sum(-1).exp())
    allmatch = [v for r in stages.values() for v in r["match"]]
    allentropy = [v for r in stages.values() for v in r["entropy"]]
    return dict(
        hard_match_probability=summary(allmatch),
        normalized_entropy=summary(allentropy),
        stages={k: {n: summary(v) for n, v in r.items()} for k, r in stages.items()},
        hard_path_match_probability=summary(paths),
        epsilon=epsilon,
        path_probability_scope="PRODUCT_OF_TOKEN_CONDITIONAL_PROBABILITIES_ON_ACTUAL_DETERMINISTIC_PREFIX_STATES",
        local_match_is_not_closed_loop=True,
        independent_keyed_stage_streams=True,
    )
