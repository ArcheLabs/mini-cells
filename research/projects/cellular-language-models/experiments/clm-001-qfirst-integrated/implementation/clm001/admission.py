"""Separate story-paired Admission evidence and a rollback-capable whole-bundle commit."""

from copy import deepcopy
from dataclasses import asdict
import numpy as np
import torch
from .common import hash_accepted_state, canonical_sha256


@torch.no_grad()
def regression_contract(candidate, accepted):
    model = candidate.model
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("nonfinite Shadow")
    old = accepted.state_dict()
    for name, value in model.state_dict().items():
        if name.startswith("controller.") or (
            candidate.kind == "cell" and name.startswith("cells.%d." % model.registry.cells)
        ):
            continue
        if not torch.equal(value, old[name]):
            raise ValueError("Shadow modified frozen accepted computation")
    device = next(model.parameters()).device
    n = min(4, model.config["max_sequence_length"])
    x = torch.arange(n, device=device)[None] % model.config["vocab_size"]
    replay = torch.full((1, n, model.config["budget"]), -1, device=device, dtype=torch.long)
    replay[:, :, 0] = model.registry.ctx
    replay[:, :, 1] = model.registry.stop
    model.eval()
    before = model(x, replay=replay).logits
    changed = x.clone()
    changed[:, -1] = (changed[:, -1] + 1) % model.config["vocab_size"]
    after = model(changed, replay=replay).logits
    if not torch.allclose(before[:, :-1], after[:, :-1], atol=1e-6, rtol=1e-5):
        raise ValueError("Shadow violated causal regression")
    return True


def assess(candidate, accepted, baseline, challenger, groups, config, *, role="admission"):
    if role != "admission" or candidate.state != "PROBATION":
        raise ValueError("Admission role/lifecycle required")
    if (
        candidate.parent_sha != hash_accepted_state(accepted)
        or candidate.parent_registry != accepted.registry.fingerprint
    ):
        candidate.transition("EXPIRED", {"reason": "parent version changed"})
        return False
    regression_contract(candidate, accepted)
    s = config["shadow"]
    margin = s["min_cost_aware_improvement"]
    tolerance = s["nll_regression_tolerance"]
    if margin is None or tolerance is None:
        raise RuntimeError("Admission thresholds are not frozen")
    if baseline["paired_identity"] != challenger["paired_identity"]:
        raise ValueError("Admission windows/RNG/budget differ")
    groups = np.asarray(groups)
    keys = np.unique(groups)
    if len(keys) < s["minimum_admission_stories"]:
        raise ValueError("insufficient independent Admission stories")
    nll = np.asarray(challenger["nll"]) - np.asarray(baseline["nll"])
    lam = config["policy"]["lambda_cost"]
    if lam is None:
        raise RuntimeError("cost coefficient not frozen")
    delta = nll + lam * (np.asarray(challenger["cost"]) - np.asarray(baseline["cost"]))
    if nll.shape != groups.shape or not np.isfinite(delta).all():
        raise ValueError("invalid paired Admission evidence")
    cluster = np.asarray([delta[groups == k].mean() for k in keys])
    reg = np.asarray([nll[groups == k].mean() for k in keys])
    rng = np.random.default_rng(config["evaluation"]["seed"])
    indices = rng.integers(0, len(keys), (s["bootstrap_replicates"], len(keys)))
    upper = float(np.quantile(cluster[indices].mean(1), 0.975))
    reg_upper = float(np.quantile(reg[indices].mean(1), 0.975))
    passed = upper < -margin and reg_upper <= tolerance
    candidate.metrics = {
        "cost_delta_mean": float(cluster.mean()),
        "cost_delta_upper95": upper,
        "nll_delta_upper95": reg_upper,
        "stories": len(keys),
        "passed": passed,
        "regression_contract_pass": True,
        "role": role,
        "paired_identity": baseline["paired_identity"],
    }
    candidate.transition("EVALUATED", candidate.metrics)
    if not passed:
        candidate.transition("REJECTED", {"reason": "paired Admission gate failed"})
    return passed


def promote(candidate, accepted, trainer, *, failure=None):
    if candidate.state != "EVALUATED" or not candidate.metrics.get("passed"):
        raise ValueError("no passing Admission evidence")
    if (
        candidate.parent_sha != hash_accepted_state(accepted)
        or candidate.parent_registry != accepted.registry.fingerprint
    ):
        candidate.transition("EXPIRED", {"reason": "stale promotion"})
        raise ValueError("stale parent")
    backup = {
        "model": deepcopy(accepted.state_dict()),
        "registry": deepcopy(accepted.registry),
        "version": accepted.version,
        "optimizer": deepcopy(trainer.opt.state_dict()),
        "cache": deepcopy(trainer.cache),
        "rng": {k: v.get_state() for k, v in trainer.rng.items()},
        "journal": deepcopy(candidate.journal),
        "state": candidate.state,
    }
    try:
        accepted.load_state_dict(candidate.model.state_dict())
        accepted.registry = deepcopy(candidate.model.registry)
        trainer.opt.load_state_dict(deepcopy(candidate.trainer.opt.state_dict()))
        if failure:
            failure()
        accepted.version = backup["version"] + 1
        accepted.registry.model_version = (
            max(backup["registry"].model_version, candidate.model.registry.model_version) + 1
        )
        accepted.registry.policy_version = (
            max(backup["registry"].policy_version, candidate.model.registry.policy_version) + 1
        )
        accepted.registry.version = (
            max(backup["registry"].version, candidate.model.registry.version) + 1
        )
        trainer.cache.clear()
        candidate.transition(
            "PROMOTED",
            {
                "accepted_sha": hash_accepted_state(accepted),
                "registry_sha": accepted.registry.fingerprint,
                "optimizer_bundle": True,
            },
        )
    except Exception:
        accepted.load_state_dict(backup["model"])
        accepted.registry = backup["registry"]
        accepted.version = backup["version"]
        trainer.opt.load_state_dict(backup["optimizer"])
        trainer.cache = backup["cache"]
        for key, state in backup["rng"].items():
            trainer.rng[key].set_state(state)
        candidate.journal = backup["journal"]
        candidate.state = backup["state"]
        raise
    return canonical_sha256(
        {
            "accepted": hash_accepted_state(accepted),
            "registry": asdict(accepted.registry),
            "event": candidate.journal[-1],
        }
    )
