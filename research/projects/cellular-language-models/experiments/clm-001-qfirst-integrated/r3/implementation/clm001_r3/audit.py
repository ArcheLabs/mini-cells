"""Frozen per-target fixed-suffix truth and separately labelled resampled-suffix truth."""

from copy import copy
import math
import numpy as np
import torch
from torch.nn import functional as F
from clm001_r2.oracle import paired, OracleUntrusted
from .oracle import Snapshot
from .common import hash_accepted_state


def clustered(values, groups, replicates, seed):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    if not len(values):
        return {"mean": None, "ci95": None, "stories": 0}
    keys = np.unique(groups)
    cluster = np.array([values[groups == g].mean() for g in keys])
    if not np.isfinite(cluster).all():
        raise OracleUntrusted("nonfinite audit statistic")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(keys), (replicates, len(keys)))
    return {
        "mean": float(cluster.mean()),
        "ci95": np.quantile(cluster[indices].mean(1), [0.025, 0.975]).tolist(),
        "stories": len(keys),
    }


def grid(c, data, role="calibration", independent=False):
    if role not in ("calibration", "development"):
        raise ValueError("independent audit roles only")
    meta = data["windows"][role]["evaluation"]
    length = c["training"]["sequence_length"]
    a = c["alignment"]
    for window, (start, group) in enumerate(
        zip(meta["starts"][: a["grid_stories"]], meta["groups"][: a["grid_stories"]])
    ):
        # Fixed positions/depths declared before labels. Do not resample stopped states.
        positions = sorted(
            set(
                min(
                    length - 1,
                    int(f * length) + (a["independent_position_offset"] if independent else 0),
                )
                for f in a["token_fractions"]
            )
        )
        yield {
            "window": window,
            "start": start,
            "group": int(group),
            "positions": positions,
            "depths": [d for d in a["depths"] if d < c["model"]["budget"]],
        }


def capture(model, data, c, role="calibration", independent=False):
    before = hash_accepted_state(model)
    bank = []
    missing = []
    device = next(model.parameters()).device
    length = c["training"]["sequence_length"]
    gen = torch.Generator(device=device).manual_seed(c["evaluation"]["seed"] + 270)
    for row in grid(c, data, role, independent):
        start = row["start"]
        x = torch.tensor(
            np.array(data[role][start : start + length], dtype="int64")[None], device=device
        )
        y = torch.tensor(
            np.array(data[role][start + 1 : start + length + 1], dtype="int64")[None], device=device
        )
        coordinates = [
            (position, depth) for position in row["positions"] for depth in row["depths"]
        ]
        with torch.no_grad():
            out = model(x, policy="uniform", generator=gen, probe_coordinates=coordinates)
        present = {(state.position, state.depth): state for state in out.probes}
        for position, depth in coordinates:
            state = present.get((position, depth))
            meta = {
                "window": row["window"],
                "group": row["group"],
                "position": position,
                "depth": depth,
                "start": start,
                "role": role,
            }
            if state is None or state.chosen >= model.registry.slots:
                missing.append(
                    {
                        **meta,
                        "reason": "STOPPED_BEFORE_STATE" if state is None else "NON_CELL_EVENT",
                    }
                )
                continue
            trace = out.actions[0, position].detach().tolist()
            bank.append(
                {
                    "state": state,
                    "target": y[0, position].detach(),
                    "suffix": trace[depth + 1 :],
                    "ctx_seen": model.registry.ctx in trace[:depth],
                    "group_feature": int(x[0, position]) % 2,
                    "has_past_cache": False,
                    "source_model_sha": before,
                    **meta,
                }
            )
    if hash_accepted_state(model) != before:
        raise OracleUntrusted("capture mutated accepted model")
    return bank, missing


def fixed_suffix(snapshot, state, action, target, suffix, gradient=False):
    snapshot.check(state, action)
    if action >= snapshot.model.registry.slots:
        raise ValueError("Cell-to-Cell substitution only")
    m = snapshot.model
    with torch.no_grad():
        post = snapshot.advance(state.hidden, action, state).detach()
    post.requires_grad_(gradient)
    h = post
    work = 1
    ended = False
    for offset, next_action in enumerate(suffix, 1):
        depth = state.depth + offset
        if next_action < 0:
            if not ended:
                raise OracleUntrusted("padding without STOP")
            continue
        if ended or depth >= m.config["budget"]:
            raise OracleUntrusted("fixed suffix exceeds termination/budget")
        if not m.registry.legal(depth, m.config["budget"], m.config["minimum_actions"], h.device)[
            next_action
        ]:
            raise OracleUntrusted("illegal fixed suffix")
        if next_action == m.registry.stop:
            ended = True
            continue
        h = m.execute(
            h[None],
            next_action,
            state.memory,
            torch.tensor([state.batch], device=h.device),
            torch.tensor([state.position], device=h.device),
        )[0]
        work += 1
    ce = F.cross_entropy(m.head(h[None]), target.reshape(1))
    if not bool(torch.isfinite(ce)):
        raise OracleUntrusted("nonfinite fixed-suffix terminal loss")
    g = torch.autograd.grad(ce, post)[0].detach() if gradient else None
    if g is not None and not bool(torch.isfinite(g).all()):
        raise OracleUntrusted("nonfinite fixed-suffix post-state gradient")
    value = float(ce.detach()) + snapshot.policy["lambda_cost"] * work / m.config["budget"]
    return value, g, post.detach()


def measure(model, bank, c, *, resampled=True):
    snapshot = Snapshot(model)
    records = []
    state_records = []
    pairs = [tuple(pair) for pair in c["alignment"]["cell_pairs"]]
    needed = sorted(set(range(model.registry.cells)) | set(model.registry.activated))
    for index, item in enumerate(bank):
        if index % 32 == 0 and len(bank) > 16:
            print("[R3 audit] frozen states " + str(index + 1) + "/" + str(len(bank)), flush=True)
        state = item["state"]
        target = item["target"]
        suffix = item["suffix"]
        if state.registry_sha != snapshot.registry_sha:
            raise OracleUntrusted("probe registry mismatch")
        _, g, post = fixed_suffix(snapshot, state, state.chosen, target, suffix, True)
        values = {}
        delta = {}
        long = {}
        legal = model.registry.legal(
            state.depth,
            model.config["budget"],
            model.config["minimum_actions"],
            state.hidden.device,
        )
        with torch.no_grad():
            q, _ = model.controller(
                state.hidden[None],
                torch.tensor([state.previous], device=state.hidden.device),
                state.depth,
                legal,
                model.registry.version,
            )
            probabilities = (
                (-q[0, : model.registry.slots] / c["policy"]["temperature"])
                .masked_fill(~legal[: model.registry.slots], float("-inf"))
                .softmax(-1)
                .cpu()
                .tolist()
            )
            uniforms = torch.rand(
                c["alignment"]["mc_samples"],
                model.config["budget"],
                device=target.device,
                generator=torch.Generator(device=target.device).manual_seed(
                    c["evaluation"]["seed"] + index + 490
                ),
            )
            for action in needed:
                values[action] = fixed_suffix(snapshot, state, action, target, suffix)[0]
                delta[action] = snapshot.advance(state.hidden, action, state) - state.hidden
                if resampled:
                    long[action] = snapshot.relative(state, action, target, uniforms)
        shuffled = (
            np.random.default_rng(c["evaluation"]["seed"] + index).permutation(needed).tolist()
        )
        permutation = dict(zip(needed, shuffled))
        for a, b in pairs:
            linear = float(torch.dot(g, delta[a] - delta[b]))
            truth = values[a] - values[b]
            null = float(torch.dot(g, delta[permutation[a]] - delta[permutation[b]]))
            reliable = abs(truth) > c["alignment"]["numeric_tolerance"]
            long_info = paired(long[a].samples, long[b].samples, c["oracle"]) if resampled else None
            records.append(
                {
                    k: item[k]
                    for k in ("group", "group_feature", "depth", "position", "ctx_seen", "window")
                }
            )
            ordered = None
            if (
                suffix
                and suffix[0] >= 0
                and suffix[0] != model.registry.stop
                and state.depth + 1 < model.config["budget"]
            ):
                ordered_values = {}
                for first, second in ((a, b), (b, a), (a, a), (b, b)):
                    next_state = copy(state)
                    with torch.no_grad():
                        next_state.hidden = snapshot.advance(state.hidden, first, state)
                    next_state.previous = first
                    next_state.depth = state.depth + 1
                    ordered_values[(first, second)] = (
                        fixed_suffix(snapshot, next_state, second, target, suffix[1:])[0]
                        + snapshot.policy["lambda_cost"] / model.config["budget"]
                    )
                ordered = {
                    "ordered_transition_value": ordered_values[(b, a)] - ordered_values[(a, b)],
                    "pair_synergy_vs_self_paths": (ordered_values[(a, a)] + ordered_values[(b, b)])
                    / 2
                    - ordered_values[(a, b)],
                    "scope": "fixed_two_nonSTOP_calls_same_remaining_suffix_not_proof_of_useful_routing",
                }
            records[-1].update(
                ordered_pair=ordered,
                cell_i=a,
                cell_j=b,
                linear_gap=linear,
                fixed_suffix_gap=truth,
                identifiable=reliable,
                agreement=bool(linear * truth > 0),
                random_agreement=bool(null * truth > 0),
                long_gap=long_info["difference"] if long_info else None,
                long_identifiable=long_info["identifiable"] if long_info else None,
                long_agreement=bool(linear * long_info["difference"] > 0) if long_info else None,
                delta_distance=float((delta[a] - delta[b]).norm()),
                gradient_norm=float(g.norm()),
                route_log_preference=math.log(max(probabilities[a], 1e-30))
                - math.log(max(probabilities[b], 1e-30)),
                state_sha=state.fingerprint(),
                snapshot_sha=snapshot.sha,
            )
        best = min(values, key=values.get)
        chosen = min(needed, key=lambda a: float(q[0, a]))
        state_records.append(
            {
                "group": item["group"],
                "position": item["position"],
                "depth": item["depth"],
                "oracle_best_cell": best,
                "router_top_cell": chosen,
                "paired_regret": values[chosen] - values[best],
                "router_choice_agreement": best == chosen,
                "cell_probabilities": probabilities,
                "cell_router_entropy": -sum(p * math.log(max(p, 1e-30)) for p in probabilities),
                "snapshot_sha": snapshot.sha,
            }
        )
    snapshot.assert_current(model)
    return {"records": records, "state_records": state_records, "summary": summarize(records, c)}


def summarize(records, c):
    valid = [r for r in records if r["identifiable"]]
    a = c["alignment"]
    local = clustered(
        [r["agreement"] for r in valid], [r["group"] for r in valid], a["replicates"], c["seed"]
    )
    null = clustered(
        [int(r["agreement"]) - int(r["random_agreement"]) for r in valid],
        [r["group"] for r in valid],
        a["replicates"],
        c["seed"],
    )
    long = [r for r in records if r.get("long_identifiable")]
    horizon = clustered(
        [r["long_agreement"] for r in long], [r["group"] for r in long], a["replicates"], c["seed"]
    )
    passed = (
        len(valid) >= a["minimum_pairs"]
        and local["stories"] >= a["minimum_stories"]
        and len(set(r["depth"] // c["competition"]["depth_bucket_width"] for r in valid)) >= 2
        and local["ci95"] is not None
        and local["ci95"][0] > a["local_agreement_lower95"]
        and null["ci95"][0] > 0
    )
    return {
        "status": "LOCAL_ALIGNMENT_PASS"
        if passed
        else (
            "INCONCLUSIVE"
            if len(valid) < a["minimum_pairs"] or local["stories"] < a["minimum_stories"]
            else "LOCAL_ALIGNMENT_FAIL"
        ),
        "pairs": len(records),
        "identifiable_pairs": len(valid),
        "excluded_fraction": 1 - len(valid) / len(records) if records else None,
        "local_agreement": local,
        "local_minus_random": null,
        "long_horizon_agreement": horizon,
        "scope": "single_target_frozen_fixed_suffix_vs_independent_MC_replanned_suffix",
        "scientific_decision": False,
    }
