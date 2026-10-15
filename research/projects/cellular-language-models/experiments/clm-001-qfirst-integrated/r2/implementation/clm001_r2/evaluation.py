"""Independent role evaluation plus finite-MC-aware relative-Q calibration."""

import itertools
import numpy as np
import torch
from clm001.evaluation import evaluate, rank_correlation

__all__ = ["evaluate", "q_diagnostics", "rank_correlation"]
from .oracle import Snapshot, paired
from .sampling import distribution


@torch.no_grad()
def q_diagnostics(model, data, c, role="calibration"):
    if role != "calibration":
        raise ValueError("independent fixed Calibration only")
    snapshot = Snapshot(model)
    device = next(model.parameters()).device
    length = c["training"]["sequence_length"]
    meta = data["windows"][role]["evaluation"]
    records = []
    gen = torch.Generator(device=device).manual_seed(c["evaluation"]["seed"] + 90)
    # Fixed token coordinate; cyclic depth target selected before seeing targets or Q labels.
    for i, start in enumerate(meta["starts"][: c["evaluation"]["oracle_states"]]):
        x = torch.tensor(
            np.array(data[role][start : start + length], dtype="int64")[None], device=device
        )
        y = torch.tensor(
            np.array(data[role][start + 1 : start + length + 1], dtype="int64")[None], device=device
        )
        depth = i % model.config["budget"]
        out = model(x, policy="uniform", generator=gen, probe_coordinates=[(0, depth)])
        if not out.probes:
            records.append({"depth": depth, "coverage_missing": True})
            continue
        state = out.probes[0]
        legal = model.registry.legal(
            depth, model.config["budget"], model.config["minimum_actions"], device
        )
        q, _ = model.controller(
            state.hidden[None],
            torch.tensor([state.previous], device=device),
            depth,
            legal,
            model.registry.version,
        )
        actions = torch.where(legal)[0].tolist()
        u = torch.rand(
            c["evaluation"]["oracle_samples"], model.config["budget"], generator=gen, device=device
        )
        baseline = snapshot.terminal(state.hidden, y[0, 0])
        truth = []
        samples = []
        for a in actions:
            if a == model.registry.stop:
                truth.append(0.0)
                samples.append([0.0] * len(u))
            else:
                lab = snapshot.relative(state, a, y[0, 0], u, baseline=baseline)
                truth.append(lab.value)
                samples.append(lab.samples)
        pred = q[0, actions].cpu().numpy()
        truth = np.asarray(truth)
        chosen = int(distribution(q, legal, 0, 1, "greedy")[0, actions].argmax())
        best = int(truth.argmin())
        ordered = np.argsort(truth)
        ident = all(
            paired(samples[best], samples[j], c["oracle"])["identifiable"]
            for j in range(len(actions))
            if j != best
        )
        pair_info = [
            paired(samples[a], samples[b], c["oracle"])
            for a, b in itertools.combinations(range(len(actions)), 2)
        ]
        reliable = [
            (a, b)
            for a, b in itertools.combinations(range(len(actions)), 2)
            if paired(samples[a], samples[b], c["oracle"])["identifiable"]
        ]
        stop = actions.index(model.registry.stop) if model.registry.stop in actions else None
        records.append(
            {
                "depth": depth,
                "group": meta["groups"][i],
                "coverage_missing": False,
                "identifiable": ident,
                "mae": float(abs(pred - truth).mean()),
                "spearman": rank_correlation(pred, truth),
                "regret": float(truth[chosen] - truth[best]),
                "uniform_regret": float(truth.mean() - truth[best]),
                "stop_reference_legal": stop is not None,
                "stop_error": bool(
                    stop is not None and ident and ((chosen == stop) != (best == stop))
                ),
                "stop_too_early": bool(
                    stop is not None and ident and chosen == stop and best != stop
                ),
                "stop_too_late": bool(
                    stop is not None and ident and chosen != stop and best == stop
                ),
                "pairwise_accuracy": float(
                    np.mean([(pred[a] - pred[b]) * (truth[a] - truth[b]) > 0 for a, b in reliable])
                )
                if reliable
                else None,
                "pair_se_mean": float(np.mean([v["se"] for v in pair_info])) if pair_info else None,
                "best_gap": float(truth[ordered[1]] - truth[best]) if len(truth) > 1 else 0.0,
                "snapshot_sha": snapshot.sha,
                "state_sha": state.fingerprint(),
            }
        )
    valid = [r for r in records if not r["coverage_missing"]]
    ident = [r for r in valid if r["identifiable"]]

    def mean(key, rs=valid):
        vs = [r[key] for r in rs if r.get(key) is not None]
        return float(np.mean(vs)) if vs else None

    return {
        "role": role,
        "states": len(valid),
        "coverage_missing": len(records) - len(valid),
        "identifiable_states": len(ident),
        "oracle_status": "TRUSTED" if ident else "UNIDENTIFIABLE",
        "mae": mean("mae"),
        "spearman": mean("spearman"),
        "regret": mean("regret"),
        "uniform_regret": mean("uniform_regret"),
        "pairwise_accuracy": mean("pairwise_accuracy"),
        "stop_error_rate": mean("stop_error", [r for r in ident if r["stop_reference_legal"]]),
        "records": records,
        "scientific_decision": False,
    }
