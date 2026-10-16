"""Read-only frozen probes; all training RNGs and module modes survive measurement."""

from contextlib import contextmanager
from pathlib import Path
import random
import time
import numpy as np
import torch
from minicells_runtime.store import atomic, canonical, digest, read, ContinuationError
from clm001_r3.audit import capture, fixed_suffix, summarize
from clm001_r3.monitoring import bank as original_bank
from clm001_r3.specialization import rebind, crossover
from clm001_r3.evaluation import evaluate
from clm001_r3.oracle import Snapshot, paired
from clm001_r3.report import seal, verified


@contextmanager
def isolated(model, trainer=None):
    states = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    modes = [(module, module.training) for module in model.modules()]
    generators = {k: v.get_state().clone() for k, v in trainer.rng.items()} if trainer else {}
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if states[3] is not None:
            torch.cuda.set_rng_state_all(states[3])
        for module, mode in modes:
            module.training = mode
        for key, value in generators.items():
            trainer.rng[key].set_state(value.cpu())


def banks(model, data, c, folder, identity, restored=False, *, pilot=False):
    folder = Path(folder)
    if restored and not (folder / "frozen-probe-bank.json").exists():
        raise ContinuationError("FROZEN_PROBE_MISSING")
    with isolated(model):
        first = original_bank(model, data, c, folder, identity)
        name = folder / "development-probe-bank.pt"
        meta = folder / "development-probe-bank.json"
        if meta.exists():
            witness = read(meta)
            if witness["identity"] != identity or digest(name) != witness["sha256"]:
                raise ContinuationError("FROZEN_PROBE_MISMATCH")
            second = torch.load(
                name, map_location=next(model.parameters()).device, weights_only=False
            )
        else:
            if restored:
                raise ContinuationError("FROZEN_PROBE_MISSING", "development")
            second, missing = capture(
                model, data, c, role="calibration" if pilot else "development", independent=True
            )
            torch.save(second, name)
            atomic(
                meta,
                {
                    "sha256": digest(name),
                    "identity": identity,
                    "coverage_missing": missing,
                    "role": "calibration" if pilot else "development",
                    "source": "STEP_ZERO_ONLY",
                },
            )
    return {"calibration": first, "development": second}


def router_rows(model, frozen, c):
    rows = []
    with torch.no_grad():
        for item in rebind(frozen, model):
            state = item["state"]
            legal = model.registry.legal(
                state.depth,
                model.config["budget"],
                model.config["minimum_actions"],
                state.hidden.device,
            )
            q, _ = model.controller(
                state.hidden[None],
                torch.tensor([state.previous], device=state.hidden.device),
                state.depth,
                legal,
                model.registry.version,
            )
            cells = [a for a in range(model.registry.slots) if bool(legal[a])]
            if not cells:
                continue
            best = min(cells, key=lambda a: float(q[0, a]))
            values = q[0, cells].detach().cpu().numpy()
            probs = (
                torch.softmax(-q[0, cells] / c["policy"]["temperature"], -1).detach().cpu().tolist()
            )
            minimum = float(q[0, best])
            rows.append(
                {
                    "depth": state.depth,
                    "ctx_seen": item["ctx_seen"],
                    "group": item["group"],
                    "position": item["position"],
                    "router_top_cell": best,
                    "stop_legal": bool(legal[model.registry.stop]),
                    "stop_minus_best_cell": float(q[0, model.registry.stop]) - minimum
                    if legal[model.registry.stop]
                    else None,
                    "ctx_minus_best_cell": float(q[0, model.registry.ctx]) - minimum
                    if legal[model.registry.ctx]
                    else None,
                    "cell_entropy": -sum(p * np.log(max(p, 1e-30)) for p in probs),
                    "near_tie": int(
                        np.sum(
                            np.abs(values - values.min())
                            <= c["r3b"]["diagnostics"]["near_tie_tolerance"]
                        )
                    )
                    > 1,
                }
            )
    return rows


def action_calibration(model, frozen, c):
    plan = c["r3b"]["diagnostics"]
    snapshot = Snapshot(model)
    records = []
    for i, item in enumerate(rebind(frozen, model)[: plan["action_type_states"]]):
        state = item["state"]
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
        cells = [a for a in range(model.registry.slots) if legal[a]]
        if not cells:
            continue
        best = min(cells, key=lambda a: float(q[0, a]))
        actions = [a for a in (best, model.registry.ctx, model.registry.stop) if legal[a]]
        uniforms = torch.rand(
            plan["action_type_mc_samples"],
            model.config["budget"],
            device=state.hidden.device,
            generator=torch.Generator(device=state.hidden.device).manual_seed(
                c["evaluation"]["seed"] + 911 + i
            ),
        )
        labels = {a: snapshot.relative(state, a, item["target"], uniforms) for a in actions}
        truth = {a: label.value for a, label in labels.items()}
        for action in actions:
            if action == best:
                continue
            identified = paired(labels[action].samples, labels[best].samples, c["oracle"])
            gap = truth[action] - truth[best]
            predicted = float(q[0, action] - q[0, best])
            records.append(
                {
                    "group": item["group"],
                    "depth": state.depth,
                    "ctx_seen": item["ctx_seen"],
                    "type": "STOP" if action == model.registry.stop else "CTX",
                    "q_gap": predicted,
                    "truth_gap": gap,
                    "identifiable": identified["identifiable"],
                    "mc_samples": identified["samples"],
                    "standard_error": identified["se"],
                    "direction_agreement": predicted * gap > 0,
                    "scope": "fixed_state_common_random_sampled_suffix_cost_aware_utility",
                }
            )
    snapshot.assert_current(model)
    return records


def gradient_row(trainer, c):
    last = trainer.journal[-1] if trainer.journal else {}
    result = {
        k: last.get(k)
        for k in (
            "q_grad_norm",
            "lcg_grad_norm",
            "lcg_q_cosine",
            "lcg_events",
            "lcg_candidate_cell_calls",
        )
    }
    q, lcg = result["q_grad_norm"], result["lcg_grad_norm"]
    result["scaled_lcg_q_ratio"] = (
        c["competition"]["beta"] * lcg / q if q and lcg is not None else None
    )
    result["empty_event"] = last.get("lcg_events", 0) == 0
    result["finite"] = all(v is None or np.isfinite(v) for v in result.values())
    for group in ("continuous", "connection"):
        before = last.get("grad_norm_" + group + "_preclip")
        after = last.get("grad_norm_" + group + "_postclip")
        result[group + "_clip_ratio"] = after / before if before and after is not None else None
    result["source"] = "unchanged_R3_Trainer_journal_no_extra_training_backward"
    return result


def snapshot(trainer, data, c, frozen, folder, phase, *, quality_role="development"):
    folder = Path(folder)
    step = trainer.step
    plan = c["r3b"]["diagnostics"]
    if not plan["enabled"] or step not in plan["steps"]:
        return None
    latest = read(folder / "latest.json")
    revision = (
        int(Path(latest["file"]).stem.split("-event-")[-1]) if "-event-" in latest["file"] else 0
    )
    target = folder / "diagnostics" / ("snapshot-%06d-%s-event-%06d.json" % (step, phase, revision))
    if target.exists():
        return verified(target)
    model = trainer.model
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
        gpu_start, gpu_end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        gpu_start.record()
    begin = time.perf_counter()
    with isolated(model, trainer):
        quality = {
            p: evaluate(model, data, c, quality_role, p)
            for p in ("greedy", "sampled", "uniform", "constant")
        }
        common_grid = canonical(
            {
                "windows": data["windows"][quality_role]["evaluation"],
                "seed": c["evaluation"]["seed"],
                "assets": data["asset_sha256"],
                "role": quality_role,
            }
        )
        # A common pairing key verifies the same observations across different
        # policies; the original policy-specific identity is retained separately.
        for row in quality.values():
            row["original_policy_identity"] = row["paired_identity"]
            row["paired_identity"] = common_grid
        routing = router_rows(model, frozen["calibration"], c)
        calibrated = action_calibration(model, frozen["calibration"], c)
        high = step in plan["functional_steps"]
        functional = {}
        if high:
            for role in ("calibration", "development"):
                # Pilot uses Calibration only and never inspects Development to
                # select beta or a scientific model/configuration.
                if quality_role == "calibration" and role == "development":
                    continue
                measured = functional_measure(
                    model, rebind(frozen[role], model), c, resampled=(role == "calibration")
                )
                functional[role] = {**measured, "crossover": crossover(measured["records"], c)}
        if next(model.parameters()).is_cuda:
            gpu_end.record()
            torch.cuda.synchronize()
        result = {
            "schema": "minicells.r3b.snapshot.v1",
            "identity": read(folder / "identity.json"),
            "step": step,
            "phase": phase,
            "event_revision": revision,
            "checkpoint_sha256": latest["sha256"],
            "runtime_config_sha256": digest(folder.parents[1] / "runtime-config.json"),
            "runtime_semantic_config_sha256": canonical(c),
            "probe_sha256": digest(folder / "frozen-probe-bank.pt"),
            "development_probe_sha256": digest(folder / "development-probe-bank.pt"),
            "evaluation_protocol_sha256": canonical(plan),
            "quality": quality,
            "router": routing,
            "action_type_calibration": calibrated,
            "functional": functional,
            "gradient_audit": gradient_row(trainer, c),
            "training_exposure_by_group": {
                str(group): {
                    str(a): sum(
                        r.get("cell_exposure_by_group", {})
                        .get(group, r.get("cell_exposure_by_group", {}).get(str(group), {}))
                        .get(
                            a,
                            r.get("cell_exposure_by_group", {})
                            .get(group, r.get("cell_exposure_by_group", {}).get(str(group), {}))
                            .get(str(a), 0),
                        )
                        for r in trainer.journal
                    )
                    for a in range(model.registry.slots)
                }
                for group in (0, 1)
            },
            "elapsed_seconds": time.perf_counter() - begin,
            "gpu_diagnostic_seconds": gpu_start.elapsed_time(gpu_end) / 1000
            if next(model.parameters()).is_cuda
            else None,
            "expected_probe_grid": {
                role: sum(
                    len(g["positions"]) * len(g["depths"])
                    for g in __import__("clm001_r3.audit", fromlist=["grid"]).grid(
                        c, data, role, independent=(role == "development")
                    )
                )
                for role in ("calibration", "development")
            },
            "missing_probe_rule": "STOPPED_BEFORE_STATE_OR_NON_CELL_EVENT_NO_REFILL",
            "sample_counts": {
                "fixed_states": len(frozen["calibration"]),
                "development_states": len(frozen["development"]),
                "action_type_pairs": len(calibrated),
            },
            "quality_role": quality_role,
            "scientific_decision": False,
            "causal_feedback_status": "CAUSAL_FEEDBACK_NOT_TESTED",
        }
        seal(target, result)
    return result


def spearman(a, b):
    def ranks(values):
        values = np.asarray(values)
        result = np.zeros(len(values))
        for value in np.unique(values):
            lower = np.sum(values < value)
            result[values == value] = lower + (np.sum(values == value) - 1) / 2
        return result

    x, y = ranks(a), ranks(b)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def functional_measure(model, frozen, c, *, resampled):
    """Reuse original fixed_suffix/Snapshot labels; expose full Cell rank once.

    No ordered-pair enumeration: this protocol's primary object is single-Cell
    advantage. Avoid the old optional four-way two-step computations per pair.
    """
    oracle = Snapshot(model)
    records, states = [], []
    needed = sorted(set(range(model.registry.cells)) | set(model.registry.activated))
    for index, item in enumerate(frozen):
        state, target, suffix = item["state"], item["target"], item["suffix"]
        chosen_loss, grad, chosen_post = fixed_suffix(
            oracle, state, state.chosen, target, suffix, True
        )
        values, deltas, long = {}, {}, {}
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
                torch.softmax(-q[0, needed] / c["policy"]["temperature"], -1).cpu().tolist()
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
                if action == state.chosen:
                    values[action], post = chosen_loss, chosen_post
                else:
                    values[action], _, post = fixed_suffix(oracle, state, action, target, suffix)
                deltas[action] = post - state.hidden
                if resampled:
                    long[action] = oracle.relative(state, action, target, uniforms)
        permutation = dict(
            zip(
                needed,
                np.random.default_rng(c["evaluation"]["seed"] + index).permutation(needed).tolist(),
            )
        )
        for a, b in c["alignment"]["cell_pairs"]:
            linear = float(torch.dot(grad, deltas[a] - deltas[b]))
            truth = values[a] - values[b]
            null = float(torch.dot(grad, deltas[permutation[a]] - deltas[permutation[b]]))
            paired_long = (
                paired(long[a].samples, long[b].samples, c["oracle"]) if resampled else None
            )
            records.append(
                {
                    **{
                        k: item[k]
                        for k in (
                            "group",
                            "group_feature",
                            "depth",
                            "position",
                            "ctx_seen",
                            "window",
                        )
                    },
                    "cell_i": a,
                    "cell_j": b,
                    "linear_gap": linear,
                    "fixed_suffix_gap": truth,
                    "identifiable": abs(truth) > c["alignment"]["numeric_tolerance"],
                    "agreement": linear * truth > 0,
                    "random_agreement": null * truth > 0,
                    "long_gap": paired_long["difference"] if paired_long else None,
                    "long_identifiable": paired_long["identifiable"] if paired_long else None,
                    "long_agreement": linear * paired_long["difference"] > 0
                    if paired_long
                    else None,
                    "route_log_preference": float(
                        torch.log_softmax(-q[0, needed] / c["policy"]["temperature"], -1)[
                            needed.index(a)
                        ]
                        - torch.log_softmax(-q[0, needed] / c["policy"]["temperature"], -1)[
                            needed.index(b)
                        ]
                    ),
                    "state_sha": state.fingerprint(),
                    "snapshot_sha": oracle.sha,
                    "ordered_pair": None,
                    "ordered_pair_status": "NOT_PART_OF_R3B_SINGLE_CELL_PROTOCOL",
                }
            )
        best, selected = min(needed, key=values.get), min(needed, key=lambda a: float(q[0, a]))
        states.append(
            {
                "group": item["group"],
                "position": item["position"],
                "depth": item["depth"],
                "oracle_best_cell": best,
                "router_top_cell": selected,
                "paired_regret": values[selected] - values[best],
                "router_choice_agreement": best == selected,
                "rank_correlation": spearman(
                    [float(q[0, a]) for a in needed], [values[a] for a in needed]
                ),
                "cell_probabilities": probabilities,
                "cell_router_entropy": -sum(p * np.log(max(p, 1e-30)) for p in probabilities),
                "snapshot_sha": oracle.sha,
            }
        )
    oracle.assert_current(model)
    return {"records": records, "state_records": states, "summary": summarize(records, c)}
