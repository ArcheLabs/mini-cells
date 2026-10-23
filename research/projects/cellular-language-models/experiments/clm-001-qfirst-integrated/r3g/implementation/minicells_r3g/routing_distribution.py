"""Analytic train distributions on the same Calibration states; no Gumbel samples."""

import copy
import math
import torch
from minicells_r3e.evaluation import reference
from minicells_r3f.evaluation import distribution
from .common import fingerprint


def summarize(p, hard):
    ordered = p.sort(descending=True).values
    h = -(p * p.log()).sum(-1)
    match = p.gather(-1, hard[:, None]).squeeze(-1)
    return {
        "entropy": distribution(h),
        "normalized_entropy": distribution(h / math.log(p.shape[-1])),
        "mean_top1_probability": float(ordered[:, 0].mean()),
        "mean_second_probability": float(ordered[:, 1].mean()),
        "top1_second_probability_gap": distribution(ordered[:, 0] - ordered[:, 1]),
        "hard_action_probability": distribution(match),
        "expected_hard_sample_agreement": float(match.mean()),
    }


@torch.no_grad()
def analyze(c, models, data):
    runtime = c["parent_runtime"]
    base = models["official"]
    before = {name: fingerprint(m.state_dict()) for name, m in models.items()}
    xs = reference(base, runtime, data)[: c["routing_reference_windows"]]
    records = [r for x in xs for r in base(x, keep_trace=True).trace]
    result = {
        "reference_state_origin": "FIXED_STEP100_CELLS_OLD_CONNECTION_CALIBRATION_GREEDY_STATES",
        "reference_state_hash": fingerprint(records),
        "data_role": "calibration",
        "action_match_is_not_closed_loop_path_probability": True,
        "connections": {},
    }
    for version in ("old", "new"):
        m = copy.deepcopy(base)
        m.connection.load_state_dict(models[version].connection.state_dict(), strict=True)
        raw = []
        hard = []
        stages = []
        previous = []
        for r in records:
            q = m.scores(r["state"], r["stage"], r["previous"])[0]
            action = m.select(r["state"], r["stage"], r["previous"], epsilon=0)[0]
            raw.append(q.double().softmax(-1))
            hard.append(action)
            stages.append(torch.full_like(action, r["stage"]))
            previous.append(r["previous"])
        p = torch.cat(raw)
        a = torch.cat(hard)
        t = torch.cat(stages)
        prev = torch.cat(previous)
        entry = {
            "action_space_size": p.shape[-1],
            "uniform_probability": 1 / p.shape[-1],
            "connection_version": version,
            "raw_softmax": summarize(p, a),
            "epsilon_schedule": [],
        }
        for step in range(c["start_step"], c["end_step"]):
            q = runtime["training"]
            epsilon = q["epsilon_initial"] + (
                q["epsilon_final"] - q["epsilon_initial"]
            ) * step / max(1, q["steps"] - 1)
            mixed = (1 - epsilon) * p + epsilon / p.shape[-1]
            stats = {
                "optimizer_step_after_update": step + 1,
                "pre_update_step": step,
                "epsilon": epsilon,
                "mixture": summarize(mixed, a),
                "stage_breakdown": {
                    str(int(v)): summarize(mixed[t == v], a[t == v]) for v in torch.unique(t)
                },
                "predecessor_breakdown": {
                    str(int(v)): summarize(mixed[prev == v], a[prev == v])
                    for v in torch.unique(prev)
                },
            }
            entry["epsilon_schedule"].append(stats)
        mean_entropy = sum(
            row["mixture"]["normalized_entropy"]["percentiles"]["50"]
            for row in entry["epsilon_schedule"]
        ) / len(entry["epsilon_schedule"])
        mean_match = sum(
            row["mixture"]["expected_hard_sample_agreement"] for row in entry["epsilon_schedule"]
        ) / len(entry["epsilon_schedule"])
        thresholds = c["descriptive_routing_thresholds"]
        near_uniform = (
            mean_entropy >= thresholds["normalized_entropy_near_uniform_min"]
            and abs(mean_match / (1 / p.shape[-1]) - 1)
            <= thresholds["hard_match_uniform_relative_deviation_max"]
        )
        entry["train_inference_description"] = {
            "median_normalized_entropy_averaged_over_epsilon": mean_entropy,
            "mean_hard_action_match_probability": mean_match,
            "expected_local_mismatch_probability": 1 - mean_match,
            "high_entropy_near_uniform": near_uniform,
            "descriptive_thresholds": thresholds,
            "status": "HIGH_ENTROPY_TRAIN_DETERMINISTIC_EVAL_GAP"
            if near_uniform
            else "NOT_CLASSIFIED_AS_NEAR_UNIFORM",
            "statistical_confirmation": False,
            "closed_loop_path_probability": False,
        }
        entry["predecessor_semantics"] = (
            "START_ONLY_FOR_A; PREVIOUS_CELL_FOR_B; ID_6_IS_START_SENTINEL"
        )
        result["connections"][version] = entry
    if any(fingerprint(m.state_dict()) != before[name] for name, m in models.items()):
        raise RuntimeError("ROUTING_ANALYSIS_MUTATED_MODEL")
    return result
