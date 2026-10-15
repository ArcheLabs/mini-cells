"""Each lifecycle unit commits its private state and event cursor in the same checkpoint."""

from .checkpoint import save
from .common import canonical_sha256, write_json, hash_accepted_state
from .evaluation import evaluate, q_diagnostics
from .admission import assess, promote
from .data import batches
import numpy as np


def pending(c, step):
    s = c["shadow"]
    events = []
    if step == s["cell_proposal_step"]:
        events += ["propose-cell"]
    if step == s["connection_path_step"]:
        events += ["propose-connection", "propose-path"]
    if step == s["admission_step"]:
        events += ["pre-admission", "admit-cell", "admit-connection", "admit-path", "promote"]
    return events


def process(c, trainer, nursery, data, data_rng, folder, identity, *, crash=None):
    step = trainer.step
    model = trainer.model
    device = next(model.parameters()).device
    previous = [v for v in nursery.events.values() if v["step"] == step]
    base_registry = previous[0]["base_registry"] if previous else model.registry.fingerprint
    for kind in pending(c, step):
        key = canonical_sha256(
            {"identity": identity, "step": step, "event": kind, "base_registry": base_registry}
        )
        if key in nursery.events:
            continue
        if crash:
            crash(kind, "before")
        detail = {
            "event": kind,
            "step": step,
            "parent_sha": hash_accepted_state(model),
            "base_registry": base_registry,
        }
        if kind.startswith("propose-"):
            nursery.propose(
                model,
                kind.removeprefix("propose-") if hasattr(kind, "removeprefix") else kind[8:],
                trainer.opt,
            )
        elif kind == "pre-admission":
            detail["quality"] = evaluate(model, data, c, "development")
            detail["q_calibration"] = q_diagnostics(model, data, c)
            detail["policies"] = {
                "greedy": detail["quality"],
                **{
                    policy: evaluate(model, data, c, "development", policy)
                    for policy in ("sampled", "uniform", "constant")
                },
            }
            detail["admission"] = evaluate(model, data, c, "admission", policy="sampled")
            write_json(folder / "pre-admission-eval.json", detail)
        elif kind.startswith("admit-"):
            candidate = next(v for v in nursery.candidates if v.kind == kind[6:])
            if candidate.state == "PROPOSED":
                nursery.rebase(candidate, model, trainer.opt)
                seed = (
                    c["seed"] + step * 31 + {"cell": 1, "connection": 2, "path": 3}[candidate.kind]
                )
                private = batches(data, c, np.random.default_rng(seed), device)
                if nursery.probation(
                    candidate,
                    model,
                    private,
                    c["shadow"]["private_train_steps"],
                    accepted_trainer=trainer,
                ):
                    baseline = next(
                        v["admission"]
                        for v in nursery.events.values()
                        if v["event"] == "pre-admission"
                    )
                    challenger = evaluate(candidate.model, data, c, "admission", policy="sampled")
                    assess(candidate, model, baseline, challenger, baseline["groups"], c)
            detail["candidate_state"] = candidate.state
        elif kind == "promote":
            passing = [
                v for v in nursery.candidates if v.state == "EVALUATED" and v.metrics.get("passed")
            ]
            if passing:
                winner = min(passing, key=lambda v: v.metrics["cost_delta_upper95"])
                promote(winner, model, trainer)
                nursery.expire_stale(model)
                # QF optimizer never starts updating its retained critic after a promotion.
                if model.arm in model.experiment_config["arms"]:
                    model.controller.critic.requires_grad_(False)
        nursery.events[key] = detail
        # If interrupted before this atomic latest replacement, restore replays only the uncommitted event.
        save(folder, trainer, nursery, data_rng, identity, event_revision=len(nursery.events))
        if crash:
            crash(kind, "after")
