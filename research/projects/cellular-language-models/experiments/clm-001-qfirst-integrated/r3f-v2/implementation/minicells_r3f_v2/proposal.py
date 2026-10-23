"""Same collector, objective and AdamW; durable full candidate BEFORE any trust evaluation."""

import copy
import time
import torch
from minicells_r3e.policy_proposal import collect, difference, add_work
from .common import fingerprint
from .storage import latest, save


def generate(tr, data, root, identity, durable):
    archived = latest(root, identity)
    if archived is not None:
        tr.restore(archived["official"])
        durable()
        return archived
    m = tr.model
    start = copy.deepcopy(m.work)
    rng_before = tr.cf_rng.get_state().clone()
    wall = time.perf_counter()
    records = collect(tr, data)
    cf_seconds = time.perf_counter() - wall
    counterfactual_work = difference(m.work, start)
    proposed = copy.deepcopy(m)
    for p in proposed.parameters():
        p.requires_grad_(False)
    for p in proposed.connection.parameters():
        p.requires_grad_(True)
    q = tr.c["training"]
    opt = torch.optim.AdamW(
        proposed.connection.parameters(),
        lr=q["connection_lr"],
        weight_decay=q["connection_weight_decay"],
    )
    losses = []
    wall = time.perf_counter()
    for _ in range(q["policy_steps"]):
        opt.zero_grad(set_to_none=True)
        terms = []
        for r in records:
            scores, _, _ = proposed.scores(r["h"], r["stage"], r["previous"])
            terms.append((scores.softmax(-1) * (r["values"] - r["values"].mean())).sum())
        loss = torch.stack(terms).mean()
        loss.backward()
        grads = [p.grad for p in proposed.connection.parameters() if p.grad is not None]
        if not torch.isfinite(loss) or not grads or not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError("NONFINITE_POLICY_PROPOSAL")
        torch.nn.utils.clip_grad_norm_(proposed.connection.parameters(), q["connection_clip"])
        opt.step()
        if not all(torch.isfinite(p).all() for p in proposed.connection.parameters()):
            raise FloatingPointError("NONFINITE_POLICY_PROPOSAL")
        losses.append(float(loss.detach()))
    if next(m.parameters()).is_cuda:
        torch.cuda.synchronize()
    opt_seconds = time.perf_counter() - wall
    optimizer_work = difference(proposed.work, m.work)
    total = copy.deepcopy(counterfactual_work)
    add_work(total, optimizer_work)
    m.work = start
    original = copy.deepcopy(m.connection.state_dict())
    candidate = copy.deepcopy(proposed.connection.state_dict())
    package = {
        "phase": "FULL_CANDIDATE_SAVED_BEFORE_TRUST",
        "identity": identity,
        "official": copy.deepcopy(tr.state()),
        "original_connection": original,
        "candidate_connection": candidate,
        "parameter_delta": {k: candidate[k] - original[k] for k in original},
        "candidate_fingerprint": fingerprint(candidate),
        "original_fingerprint": fingerprint(original),
        "losses": losses,
        "optimizer_state": opt.state_dict(),
        "optimizer_configuration": {
            k: q[k]
            for k in ["connection_lr", "connection_weight_decay", "connection_clip", "policy_steps"]
        },
        "sampled_states": records,
        "state_sample_fingerprint": fingerprint(records),
        "cf_rng_before": rng_before,
        "cf_rng_after": tr.cf_rng.get_state(),
        "route_switch_margin": float(m.route_switch_margin),
        "numeric_tolerance": float(m.route_numeric_tolerance),
        "cost": {
            "counterfactual_seconds": cf_seconds,
            "optimization_seconds": opt_seconds,
            "counterfactual_work": counterfactual_work,
            "optimization_work": optimizer_work,
            "total_work": total,
        },
    }
    saved, _ = save(root, identity, package)
    durable()  # Mandatory private persistence before the caller may evaluate ANY scale.
    return saved
