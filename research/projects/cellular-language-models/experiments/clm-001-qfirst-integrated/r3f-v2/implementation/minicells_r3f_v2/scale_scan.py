"""22 frozen scales, one saved candidate; same-state actions and closed-loop paths."""

import copy
import time
import torch
from minicells_r3f.evaluation import states, trust
from minicells_r3e.policy_proposal import difference, add_work
from .evaluation import admission_windows
from .admission import decide, select
from .storage import latest, save
from .common import fingerprint


@torch.no_grad()
def scan(tr, data, package, root, identity):
    old = tr.model
    if fingerprint(old.connection.state_dict()) != package["original_fingerprint"]:
        raise RuntimeError("SOURCE_OR_CHECKPOINT_MISMATCH")
    candidate = copy.deepcopy(old)
    candidate.connection.load_state_dict(package["candidate_connection"], strict=True)
    progress = latest(root, identity) or {"attempts": [], "work": {}, "seconds": 0.0}
    begin = time.perf_counter()
    old_work = copy.deepcopy(old.work)
    new_work = copy.deepcopy(candidate.work)
    refs = list(admission_windows(old, tr.c, data))
    fixed = states(old, refs)
    expected = [candidate.scores(r["state"], r["stage"], r["previous"])[1].clone() for r in fixed]
    scales = [0.0] + tr.c["proposal"]["backtrack"]
    for i, scale in enumerate(scales):
        if i < len(progress["attempts"]):
            if progress["attempts"][i]["scale"] != scale:
                raise RuntimeError("FROZEN_SCALES_CHANGED")
            continue
        for name, p in candidate.connection.named_parameters():
            p.copy_(
                package["original_connection"][name].to(p)
                + scale * package["parameter_delta"][name].to(p)
            )
        try:
            metrics = trust(
                old,
                candidate,
                refs,
                fixed=fixed,
                expected_delta=expected,
                scale=scale,
                rtol=tr.c["proposal"]["interpolation_rtol"],
                atol=tr.c["proposal"]["interpolation_atol"],
            )
        except RuntimeError as e:
            if str(e) == "CONNECTION_LOGIT_INTERPOLATION_MISMATCH":
                raise RuntimeError("INVALID_INTERPOLATION") from e
            raise
        centered = []
        for r in fixed:
            before = old.scores(r["state"], r["stage"], r["previous"])[1]
            after = candidate.scores(r["state"], r["stage"], r["previous"])[1]
            d = after - before
            centered.append((d - d.mean(-1, keepdim=True)).reshape(-1))
        metrics["centered_logit_delta_rms"] = float(
            torch.cat(centered).double().square().mean().sqrt()
        )
        if scale == 0 and (
            metrics["kl"] != 0 or metrics["action_changed_count"] or metrics["path_changed_count"]
        ):
            raise RuntimeError("ZERO_SCALE_FORWARD_INCONSISTENT")
        progress["attempts"].append(
            {"scale": scale, **metrics, **decide(metrics, tr.c, package["numeric_tolerance"])}
        )
        progress["data_role"] = "admission"
        progress["data_range"] = tr.c["engine"]["input"]["data_config"]["admission_range"]
        work = difference(old.work, old_work)
        add_work(work, difference(candidate.work, new_work))
        add_work(progress["work"], work)
        old_work, new_work = copy.deepcopy(old.work), copy.deepcopy(candidate.work)
        if next(old.parameters()).is_cuda:
            torch.cuda.synchronize()
        progress["seconds"] += time.perf_counter() - begin
        save(root, identity, progress, i)
        begin = time.perf_counter()
    chosen, reason = select(progress["attempts"])
    progress.update(chosen_scale=chosen, rejection_reason=reason)
    # Do not mutate the official model or its accounting during read-only scale evaluation.
    old.work = package["official"]["work"].copy()
    return progress
