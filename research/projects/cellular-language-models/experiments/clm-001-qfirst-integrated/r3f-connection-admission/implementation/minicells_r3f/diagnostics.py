"""Calibration-only read-only diagnostics; surrogate scans never masquerade as saved proposals."""

import copy
import torch
from minicells_r3e.policy_proposal import difference, add_work
from .evaluation import reference, states, distribution, trust, admission
from .stable_connection import baseline_action


@torch.no_grad()
def calibrate(model, c, data):
    refs = reference(model, c, data)
    # Before any Connection training, the infinite bootstrap margin enforces the D baseline.
    records = states(model, refs)
    margins = []
    scales = []
    disagreements = []
    for r in records:
        raw, _, z = model.scores(r["state"], r["stage"], r["previous"])
        ordered = z.sort(descending=True, stable=True).values
        margins.append(
            ordered[:, 0] - ordered[:, 1] if model.graph else ordered[:, 1] - ordered[:, 2]
        )
        scales.append(raw.abs().max())
        disagreements.append(raw.argmax(-1) != baseline_action(model, z))
    margin = torch.cat(margins)
    score_scale = max(1.0, float(torch.stack(scales).max()))
    numeric = (
        c["proposal"]["numeric_epsilon_multiplier"] * torch.finfo(margin.dtype).eps * score_scale
    )
    tau = max(
        numeric, c["proposal"]["route_switch_margin_fraction"] * float(torch.quantile(margin, 0.5))
    )
    model.route_switch_margin.fill_(tau)
    model.route_numeric_tolerance.fill_(numeric)
    evidence = {
        "source": "INITIAL_PARENT_D500_CALIBRATION_REFERENCE",
        "seed": model.seed,
        "architecture": "B" if model.graph else "A",
        "route_switch_margin": float(model.route_switch_margin),
        "tau_numeric": float(model.route_numeric_tolerance),
        "route_switch_margin_fraction": c["proposal"]["route_switch_margin_fraction"],
        "gate_margin": distribution(margin),
        "exact_tie_rate": float((margin == 0).float().mean()),
        "near_tie_rate": float((margin <= numeric).float().mean()),
        "raw_argmax_vs_gate_disagreement": float(torch.cat(disagreements).float().mean()),
        "frozen_once": True,
        "data_role": "CALIBRATION_ONLY",
    }
    model.margin_evidence = evidence
    return evidence


@torch.no_grad()
def diagnostic(model, c, data, *, historical=False):
    refs = reference(model, c, data)
    before = copy.deepcopy(model.state_dict())
    work_before = copy.deepcopy(model.work)
    clone = copy.deepcopy(model)
    zero = trust(model, clone, refs)
    if abs(zero["kl"]) > 1e-12 or zero["action_changed_count"] or zero["path_changed_count"]:
        raise RuntimeError("IDENTICAL_MODEL_DETERMINISTIC_ROUTING_MISMATCH")
    records = states(model, refs)
    observed_margins = []
    baseline_mismatches = []
    for rec in records:
        _, _, z = model.scores(rec["state"], rec["stage"], rec["previous"])
        ordered = z.sort(descending=True, stable=True).values
        observed_margins.append(
            ordered[:, 0] - ordered[:, 1] if model.graph else ordered[:, 1] - ordered[:, 2]
        )
        base_scores = z if model.graph else z[:, model.connection.pairs].sum(-1)
        baseline_mismatches.append(base_scores.argmax(-1) != baseline_action(model, z))
    observed = torch.cat(observed_margins)
    observed_stats = {
        **distribution(observed),
        "exact_tie_rate": float((observed == 0).float().mean()),
        "near_tie_rate": float((observed <= model.route_numeric_tolerance).float().mean()),
        "raw_gate_pair_argmax_disagreement": float(torch.cat(baseline_mismatches).float().mean()),
        "near_tie_criterion": "FROZEN_D500_NUMERIC_TOLERANCE",
    }
    expected = []
    scans = []
    rng = torch.Generator().manual_seed(model.seed + c["admission"]["diagnostic_seed_offset"])
    oldparams = {n: p.detach().clone() for n, p in model.connection.named_parameters()}
    direction = {
        n: torch.randn(p.shape, generator=rng).to(p) for n, p in model.connection.named_parameters()
    }
    size = torch.stack([p.square().sum() for p in direction.values()]).sum().sqrt()
    direction = {n: p / size for n, p in direction.items()}
    for n, p in clone.connection.named_parameters():
        p.copy_(oldparams[n] + direction[n])
    for r in records:
        expected.append(clone.scores(r["state"], r["stage"], r["previous"])[1])
    for scale in c["admission"]["diagnostic_scales"]:
        for n, p in clone.connection.named_parameters():
            p.copy_(oldparams[n] + scale * direction[n])
        metrics = trust(
            model,
            clone,
            refs,
            fixed=records,
            expected_delta=expected,
            scale=scale,
            rtol=c["proposal"]["interpolation_rtol"],
            atol=c["proposal"]["interpolation_atol"],
        )
        if scale == 0 and (
            metrics["action_changed_count"]
            or metrics["path_changed_count"]
            or metrics["parameter_delta_norm"]
        ):
            raise RuntimeError("ZERO_SCALE_POLICY_CHANGED")
        scans.append(
            {
                "scale": scale,
                **metrics,
                **admission(metrics, c, float(model.route_numeric_tolerance)),
            }
        )
    if any(not torch.equal(before[n], p) for n, p in model.state_dict().items()):
        raise RuntimeError("READONLY_DIAGNOSTIC_MUTATED_MODEL")
    work = difference(model.work, work_before)
    add_work(work, difference(clone.work, work_before))
    model.work = work_before
    return {
        "zero_correction_identity": zero,
        "observed_checkpoint_gate_margins": observed_stats,
        "margin": model.margin_evidence,
        "scale_scan": scans,
        "direction": c["admission"]["diagnostic_direction"],
        "historical_checkpoint": historical,
        "surrogate_scan_not_historical_proposal": True,
        "training_updates": 0,
        "work": work,
    }
