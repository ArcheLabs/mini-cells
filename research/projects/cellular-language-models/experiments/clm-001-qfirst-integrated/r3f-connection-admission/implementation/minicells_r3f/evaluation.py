"""Same-state raw KL and margin actions; independently rerun complete deterministic paths."""

import math
import torch
from minicells_r3e.evaluation import (
    reference as reference,
    calibration as calibration,
    terminal as terminal,
)


def distribution(values):
    x = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if not len(x):
        return {"count": 0}
    if not torch.isfinite(x).all():
        raise FloatingPointError("NONFINITE_DIAGNOSTIC_VALUES")
    levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {
        "count": len(x),
        "min": float(x.min()),
        "max": float(x.max()),
        "median": float(x.median()),
        "percentiles": {str(q): float(torch.quantile(x, q / 100)) for q in levels},
    }


@torch.no_grad()
def states(model, refs):
    return [rec for x in refs for rec in model(x, keep_trace=True).trace]


@torch.no_grad()
def trust(old, new, refs, *, fixed=None, expected_delta=None, scale=None, rtol=1e-5, atol=1e-7):
    records = states(old, refs) if fixed is None else fixed
    kl = []
    changed = []
    deltas = []
    centered = []
    margins = []
    changed_margins = []
    errors = []
    for i, r in enumerate(records):
        h, t, p = r["state"], r["stage"], r["previous"]
        qo, do, z = old.scores(h, t, p)
        qn, dn, _ = new.scores(h, t, p)
        if not torch.isfinite(qo).all() or not torch.isfinite(qn).all():
            raise FloatingPointError("NONFINITE_ROUTING_LOGITS")
        logp = qo.double().log_softmax(-1)
        logq = qn.double().log_softmax(-1)
        kl.append((logp.exp() * (logp - logq)).sum(-1))
        a, _ = old.select(h, t, p)
        b, _ = new.select(h, t, p)
        different = a != b
        changed.append(different)
        deltas.append((dn - do).reshape(-1))
        centered.append(((dn - do) - (dn - do).mean(-1, keepdim=True)).reshape(-1))
        ordered = z.sort(descending=True, stable=True).values
        margin = ordered[:, 0] - ordered[:, 1] if old.graph else ordered[:, 1] - ordered[:, 2]
        margins.append(margin)
        changed_margins.append(margin[different])
        if expected_delta is not None:
            expected = do + scale * (expected_delta[i] - do)
            error = (dn - expected).abs().max()
            errors.append(error)
            if not torch.allclose(dn, expected, rtol=rtol, atol=atol):
                raise RuntimeError("CONNECTION_LOGIT_INTERPOLATION_MISMATCH")
    # Full path comparison is an independent closed loop, not a proxy using local changes.
    path = [(old(x).actions != new(x).actions).any(-1) for x in refs]
    oldp = dict(old.connection.named_parameters())
    newp = dict(new.connection.named_parameters())
    norm = torch.stack([(newp[k] - oldp[k]).double().square().sum() for k in oldp]).sum().sqrt()
    delta = torch.cat(deltas).double()
    k = torch.cat(kl)
    if not torch.isfinite(k).all() or not torch.isfinite(delta).all():
        raise FloatingPointError("NONFINITE_TRUST_METRIC")
    return {
        "parameter_delta_norm": float(norm),
        "score_delta_rms": float(delta.square().mean().sqrt()),
        "score_delta_max_abs": float(delta.abs().max()),
        "centered_score_delta_max_abs": float(torch.cat(centered).abs().max()),
        "kl": max(0.0, float(k.mean())),
        "action_change": float(torch.cat(changed).float().mean()),
        "path_change": float(torch.cat(path).float().mean()),
        "action_changed_count": int(torch.cat(changed).sum()),
        "path_changed_count": int(torch.cat(path).sum()),
        "changed_action_gate_margins": distribution(torch.cat(changed_margins)),
        "all_gate_margins": distribution(torch.cat(margins)),
        "interpolation_max_abs_error": float(torch.stack(errors).max()) if errors else None,
        "kl_scope": "RAW_SOFTMAX_EXCLUDES_SHARED_EPSILON_UNIFORM_MIXTURE",
    }


def admission(metrics, c, numeric):
    if not all(
        math.isfinite(metrics[k])
        for k in (
            "kl",
            "action_change",
            "path_change",
            "score_delta_max_abs",
            "centered_score_delta_max_abs",
            "parameter_delta_norm",
        )
    ):
        raise FloatingPointError("NONFINITE_TRUST_METRIC")
    keys = [
        ("kl", "kl_max", "kl_pass"),
        ("action_change", "action_change_max", "action_change_pass"),
        ("path_change", "path_change_max", "path_change_pass"),
    ]
    result = {name: metrics[k] <= c["proposal"][limit] for k, limit, name in keys}
    effective = not (
        metrics["centered_score_delta_max_abs"] <= numeric
        and metrics["action_changed_count"] == 0
        and metrics["path_changed_count"] == 0
    )
    reasons = [k.upper() + "_EXCEEDED" for k, _, name in keys if not result[name]]
    if not effective:
        reasons.append("NO_EFFECTIVE_UPDATE")
    result.update(
        effective_update=effective,
        trust_region_passed=all(result.values()) and effective,
        reject_reasons=reasons,
    )
    result["passed"] = result["trust_region_passed"]
    return result
