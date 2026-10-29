"""Finer scales, unchanged thresholds and hard routing, mandatory zero-perturbation check."""

import copy
from types import SimpleNamespace

import torch
from minicells_fast001.engine import Engine as FrozenEngine
from minicells_fast001.parent import make_engine as original_make_engine
from minicells_r3e import policy_proposal as historical
from minicells_r3e.common import fingerprint
from minicells_r3e.evaluation import reference, trust
from minicells_r3k.reuse import bind

collect = historical.collect


@torch.no_grad()
def zero_check(model, refs):
    if getattr(model, "active_step", None) is not None or model.training_context is not None:
        raise RuntimeError("FAST002_TRUST_REQUIRES_DETERMINISTIC_CONTEXT")
    before = fingerprint(model.state_dict())
    metrics = trust(model, copy.deepcopy(model), refs)
    if (
        abs(metrics["kl"]) > 1e-8
        or metrics["action_change"] != 0
        or metrics["path_change"] != 0
        or fingerprint(model.state_dict()) != before
    ):
        raise RuntimeError("FAST002_ZERO_PERTURBATION_TRUST_FAILED:" + str(metrics))
    return metrics


def optimize(tr, data):
    refs = reference(tr.model, tr.c, data)
    baseline = zero_check(tr.model, refs)
    before = fingerprint(tr.model.connection.state_dict())

    def measured(old, new, inputs):
        metrics = trust(old, new, inputs)
        metrics["parameter_delta_l2"] = float(
            torch.stack(
                [
                    (p.detach() - dict(old.connection.named_parameters())[n].detach())
                    .square()
                    .sum()
                    for n, p in new.connection.named_parameters()
                ]
            )
            .sum()
            .sqrt()
        )
        metrics["rejected_by"] = [
            key
            for key in ("kl", "action_change", "path_change")
            if metrics[key] > tr.c["proposal"][key + "_max"]
        ]
        return metrics

    native = bind(
        historical.optimize,
        collect=collect,
        reference=lambda *args: refs,
        trust=measured,
        torch=torch,
    )
    candidate, report = native(tr, data)
    report["zero_perturbation"] = baseline
    report["connection_before_sha256"] = before
    report["connection_after_sha256"] = (
        fingerprint(candidate.connection.state_dict()) if candidate is not None else before
    )
    report["connection_parameters_changed"] = (
        report["connection_before_sha256"] != report["connection_after_sha256"]
    )
    if candidate is not None and not report["connection_parameters_changed"]:
        raise RuntimeError("FAST002_ADMITTED_CANDIDATE_HAS_NO_PARAMETER_CHANGE")
    report["trust_thresholds_unchanged"] = True
    return candidate, report


class Engine(FrozenEngine):
    update = bind(
        FrozenEngine.update, policy_proposal=SimpleNamespace(collect=collect, optimize=optimize)
    )


def make_engine(*args, **kwargs):
    return bind(original_make_engine, Engine=Engine)(*args, **kwargs)
