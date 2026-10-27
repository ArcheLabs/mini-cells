"""All 22 scales; R3f-v2 same-state KL/action metrics and independent closed-loop paths."""

import copy
import csv

import torch
from minicells_r3e.common import fingerprint
from minicells_r3f.evaluation import states, trust
from minicells_r3f_v2.admission import decide as historical_decide

from .common import atomic, read, sha


@torch.no_grad()
def run(engine, package, refs, journal, directory, c, identity):
    if (
        not getattr(engine, "r1_cpu_fixture", False)
        and not c["diagnostic"]["gate1_execution_authorized"]
    ):
        raise RuntimeError("R1_REAL_ADMISSION_SCAN_REQUIRES_EXPLICIT_GATE1_APPROVAL")
    if refs.get("role") != "admission" or refs.get("manifest_sha256") != c["admission"].get(
        "manifest_sha256"
    ):
        raise RuntimeError("R1_ONLY_FROZEN_ADMISSION_INPUTS_ALLOWED")
    refs = refs["inputs"]
    if [sha(x.squeeze(0).cpu().tolist()) for x in refs] != c["admission"].get(
        "ordered_input_hashes"
    ):
        raise RuntimeError("R1_ADMISSION_INPUT_OR_ORDER_CHANGED")
    if package["identity"] != identity:
        raise RuntimeError("R1_CANDIDATE_CONTEXT_IDENTITY_CHANGED")
    old = engine.model
    if getattr(old, "active_step", None) is not None or old.training_context is not None:
        raise RuntimeError("R1_ADMISSION_REQUIRES_DETERMINISTIC_ROUTING")
    if fingerprint(old.state_dict()) != package["source_model_sha256"]:
        raise RuntimeError("R1_CANDIDATE_DOES_NOT_BELONG_TO_PAUSED_CONTEXT")
    source = fingerprint(old.state_dict())
    work = copy.deepcopy(old.work)
    candidate = copy.deepcopy(old)
    candidate.connection.load_state_dict(package["candidate_connection"], strict=True)
    for p in candidate.parameters():
        p.requires_grad_(False)
    scales = [0.0] + c["connection"]["backtrack"]
    path = directory / "SCAN_PROGRESS.json"
    progress = read(path) if path.exists() else {"identity": identity, "attempts": []}
    if progress["identity"] != identity:
        raise RuntimeError("R1_SCAN_IDENTITY_CHANGED")
    if [r["scale"] for r in progress["attempts"]] != scales[: len(progress["attempts"])]:
        raise RuntimeError("R1_FROZEN_SCAN_ORDER_CHANGED")
    try:
        with journal.phase("admission_fixed_state_preparation"):
            fixed = states(old, refs)
            expected = [
                candidate.scores(r["state"], r["stage"], r["previous"])[1].clone() for r in fixed
            ]
        for index, scale in enumerate(scales):
            if index < len(progress["attempts"]):
                continue
            with journal.phase("admission_scale_" + str(index)):
                for name, p in candidate.connection.named_parameters():
                    original = package["original_connection"][name].to(p)
                    proposed = package["candidate_connection"][name].to(p)
                    p.copy_(original + scale * (proposed - original))
                metrics = trust(
                    old,
                    candidate,
                    refs,
                    fixed=fixed,
                    expected_delta=expected,
                    scale=scale,
                    rtol=c["diagnostic"]["interpolation_rtol"],
                    atol=c["diagnostic"]["interpolation_atol"],
                )
                nonzero = fingerprint(candidate.connection.state_dict()) != fingerprint(
                    package["original_connection"]
                )
                reasons = [
                    name
                    for name in ("kl", "action_change", "path_change")
                    if metrics[name] > c["connection"]["trust"][name + "_max"]
                ]
                if not nonzero:
                    reasons.append("ZERO_PARAMETER_CHANGE")
                if scale == 0 and (
                    metrics["kl"] != 0
                    or metrics["action_changed_count"]
                    or metrics["path_changed_count"]
                ):
                    raise RuntimeError("R1_ZERO_BASELINE_INCONSISTENT")
                historical = historical_decide(
                    metrics, engine.c, float(old.route_numeric_tolerance)
                )
                row = {
                    "scale": scale,
                    **metrics,
                    "scaled_connection_sha256": fingerprint(candidate.connection.state_dict()),
                    "original_connection_sha256": fingerprint(package["original_connection"]),
                    "full_candidate_connection_sha256": fingerprint(
                        package["candidate_connection"]
                    ),
                    "nonzero_parameter_change": nonzero,
                    "passed": scale > 0 and not reasons,
                    "rejected_by": reasons,
                    "r3fv2_nonempty_path_pass": historical["passed"],
                }
            progress["attempts"].append(row)
            atomic(path, progress)
            if not getattr(engine, "r1_cpu_fixture", False):
                print(
                    "[R1 Gate1]",
                    identity["seed"],
                    identity["arm"],
                    "scale",
                    scale,
                    "passed",
                    row["passed"],
                    flush=True,
                )
    finally:
        old.work = work
    if fingerprint(old.state_dict()) != source or len(progress["attempts"]) != 22:
        raise RuntimeError("R1_INCOMPLETE_SCAN_OR_OFFICIAL_MODEL_MUTATION")
    passing = [r for r in progress["attempts"] if r["passed"]]
    progress.update(
        status="FULL_SCALE_SCAN_COMPLETE",
        chosen_scale=max(r["scale"] for r in passing) if passing else None,
        metric_definition="FROZEN_R3F_V2_SAME_STATE_RAW_DOUBLE_SOFTMAX_AND_INDEPENDENT_CLOSED_LOOP",
        formal_connection_commits=0,
        confirm_read=False,
        admission_policy_review_required=True,
    )
    atomic(directory / "ADMISSION_DECISION.json", progress)
    fields = [
        "scale",
        "kl",
        "action_change",
        "path_change",
        "parameter_delta_norm",
        "nonzero_parameter_change",
        "passed",
        "r3fv2_nonempty_path_pass",
        "scaled_connection_sha256",
    ]
    with (directory / "CONNECTION_SCALE_SCAN.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(progress["attempts"])
    return progress


def aggregate(c, rows):
    required = {
        (seed, arm + "-" + condition)
        for seed in c["seeds"]
        for arm in ("B01", "B11")
        for condition in ("S", "H")
    }
    actual = {(r["seed"], r["arm"]) for r in rows}
    complete = (
        len(rows) == 20 and actual == required and all(len(r["attempts"]) == 22 for r in rows)
    )
    seeds = [
        seed
        for seed in c["seeds"]
        if all(
            any(
                r["seed"] == seed
                and r["arm"] == arm + "-" + condition
                and r["chosen_scale"] is not None
                for r in rows
            )
            for arm in ("B01", "B11")
            for condition in ("S", "H")
        )
    ]
    return {
        "complete_contexts": complete,
        "paired_admitted_seeds": seeds,
        "proposed_four_seed_prerequisite": complete and len(seeds) >= 4,
        "gate1_approved": False,
        "gate3_authorized": False,
        "status": "INDEPENDENT_REVIEW_REQUIRED" if complete else "INCOMPLETE_BLOCKED",
        "scientific_joint_benefit_tested": False,
    }
