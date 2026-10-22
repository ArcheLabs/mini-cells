"""R3f admission adapter. Unchanged all-action Q collector remains in frozen R3e."""

import copy
import torch
from minicells_r3e.policy_proposal import collect, add_work, difference
from .evaluation import reference, states, trust, admission
from .common import fingerprint


def optimize(tr, data):
    m = tr.model
    c = tr.c
    start = copy.deepcopy(m.work)
    snapshot = fingerprint(m.state_dict())
    records = collect(tr, data)
    cf_work = difference(m.work, start)
    candidate = copy.deepcopy(m)
    for p in candidate.parameters():
        p.requires_grad_(False)
    for p in candidate.connection.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(
        candidate.connection.parameters(),
        lr=c["training"]["connection_lr"],
        weight_decay=c["training"]["connection_weight_decay"],
    )
    losses = []
    for _ in range(c["training"]["policy_steps"]):
        opt.zero_grad(set_to_none=True)
        terms = []
        for r in records:
            q, _, _ = candidate.scores(r["h"], r["stage"], r["previous"])
            costs = r["values"] - r["values"].mean()
            terms.append((q.softmax(-1) * costs).sum())
        loss = torch.stack(terms).mean()
        loss.backward()
        grads = [p.grad for p in candidate.connection.parameters() if p.grad is not None]
        if not torch.isfinite(loss) or not grads or not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError("INVALID_POLICY_GRADIENT")
        torch.nn.utils.clip_grad_norm_(
            candidate.connection.parameters(), c["training"]["connection_clip"]
        )
        opt.step()
        losses.append(float(loss.detach()))
    work = difference(m.work, start)
    add_work(work, difference(candidate.work, m.work))
    phases = {
        "counterfactual": cf_work,
        "connection_optimization": difference(candidate.work, m.work),
        "trust_scales": {},
    }
    refs = reference(m, c, data)
    old_params = {n: p.detach().clone() for n, p in m.connection.named_parameters()}
    new_params = {n: p.detach().clone() for n, p in candidate.connection.named_parameters()}
    ref_before = copy.deepcopy(m.work)
    fixed = states(m, refs)
    phases["trust_reference"] = difference(m.work, ref_before)
    add_work(work, phases["trust_reference"])
    interpolation_before = copy.deepcopy(candidate.work)
    with torch.no_grad():
        expected = [
            candidate.scores(r["state"], r["stage"], r["previous"])[1].clone() for r in fixed
        ]
    phases["interpolation_reference"] = difference(candidate.work, interpolation_before)
    add_work(work, phases["interpolation_reference"])
    attempts = []
    chosen = None
    chosen_state = None
    for scale in c["proposal"]["backtrack"]:
        with torch.no_grad():
            for n, p in candidate.connection.named_parameters():
                p.copy_(old_params[n] + scale * (new_params[n] - old_params[n]))
        old_work = copy.deepcopy(m.work)
        new_work = copy.deepcopy(candidate.work)
        metrics = trust(
            m,
            candidate,
            refs,
            fixed=fixed,
            expected_delta=expected,
            scale=scale,
            rtol=c["proposal"]["interpolation_rtol"],
            atol=c["proposal"]["interpolation_atol"],
        )
        status = admission(metrics, c, float(m.route_numeric_tolerance))
        old_delta = difference(m.work, old_work)
        new_delta = difference(candidate.work, new_work)
        add_work(work, old_delta)
        add_work(work, new_delta)
        add_work(phases["trust_scales"], old_delta)
        add_work(phases["trust_scales"], new_delta)
        attempts.append({"scale": scale, **metrics, **status})
        # Scan every frozen scale; choose the first admissible one, never assume monotonicity.
        if status["trust_region_passed"] and chosen is None:
            chosen = scale
            chosen_state = copy.deepcopy(candidate.connection.state_dict())
    if chosen_state is not None:
        candidate.connection.load_state_dict(chosen_state, strict=True)
    numerically_stable = [
        a for a in attempts if a["kl_pass"] and a["action_change_pass"] and a["path_change_pass"]
    ]
    rejection = (
        "NO_EFFECTIVE_UPDATE"
        if numerically_stable and all(not a["effective_update"] for a in numerically_stable)
        else "ALL_TRUST_REGION_SCALES_REJECTED"
    )
    for p in candidate.parameters():
        p.requires_grad_(False)
    for p in candidate.cells.parameters():
        p.requires_grad_(True)
    # Probe forwards are additional work, not normal main-stream updates.
    m.work = start
    return candidate if chosen is not None else None, {
        "losses": losses,
        "snapshot_fingerprint": snapshot,
        "snapshot_cell_step": tr.step,
        "snapshot_connection_version": tr.connection_version,
        "work_by_phase": phases,
        "trust_region_passed": chosen is not None,
        "rejection_reason": None if chosen is not None else rejection,
        "numeric_tolerance": float(m.route_numeric_tolerance),
        "route_switch_margin": float(m.route_switch_margin),
        "attempts": attempts,
        "chosen_scale": chosen,
        "counterfactual_states": len(records),
        "actions_per_state": 6 if m.graph else 15,
        "counterfactual_suffix": "CURRENT_POLICY_REROUTED",
        "work": work,
        "policy_optimizer_updates": len(losses),
        "proposal_optimizer_final_state": opt.state_dict(),
    }
