"""Frozen input and native baseline checks, with explicit historical evidence limits."""

import numpy as np
import torch
from clm001_r3.actions import Registry
from clm001_r3.model import Model
from clm001_r3b.common import ENGINE
from minicells_runtime.store import read, digest, MountedInputStore, ContinuationError
from .common import RUN_ID, TRAINING_SHA


def load_model(root, config, device):
    root = __import__("pathlib").Path(root)
    meta = read(root / "continuation.json")
    if meta["run_id"] != RUN_ID or meta["source_sha"] != TRAINING_SHA:
        raise ContinuationError("CF_INPUT_IDENTITY_MISMATCH")
    for arm, expected in config["checkpoint_sha256"].items():
        selected = MountedInputStore(root).select(arm, 500, 0)
        if selected["checkpoint_sha256"] != expected:
            raise ContinuationError("CF_CHECKPOINT_HASH_MISMATCH")
    selected = MountedInputStore(root).select("QC-R3b", 500, 0)
    file = root / selected["checkpoint"]
    if digest(file) != config["checkpoint_sha256"]["QC-R3b"]:
        raise ContinuationError("CF_CHECKPOINT_HASH_MISMATCH")
    payload = torch.load(file, map_location="cpu", weights_only=False)
    if payload["step"] != 500:
        raise ContinuationError("CF_CHECKPOINT_STEP_MISMATCH")
    c = read(root / "runtime-config.json")
    if (
        c["model"]["budget"] != 16
        or c["model"]["minimum_actions"] != 1
        or c["policy"]["lambda_cost"] != 0.01
        or c["training"]["sequence_length"] != 64
    ):
        raise ContinuationError("CF_EXECUTION_CONTRACT_CHANGED")
    model = Model(c, ENGINE["QC-R3b"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.registry = Registry(**payload["extra"]["registry"])
    model.version = payload["extra"]["version"]
    model.experiment_config["policy"]["epsilon"] = payload["extra"]["execution_epsilon"]
    model.eval().requires_grad_(False)
    return (
        model,
        c,
        {
            "status": "PASS",
            "checkpoint_records_verified": 57,
            "qc_step500_sha256": digest(file),
            "step": payload["step"],
            "registry_plan": model.registry.plan,
            "execution_epsilon": model.experiment_config["policy"]["epsilon"],
            "optimizer_updates": 0,
        },
    )


def baseline_parity(base, noop, reference, config):
    if set(base) != set(noop):
        raise ContinuationError("CF_BASELINE_STORY_MISMATCH")
    for key in base:
        a, b = base[key], noop[key]
        if (
            a["actions"] != b["actions"]
            or a["state_sha256"] != b["state_sha256"]
            or a["actual_stop"] != b["actual_stop"]
            or a["lengths"] != b["lengths"]
        ):
            raise ContinuationError("CF_NOOP_ACTION_STATE_PARITY_FAILED")
        if not np.allclose(a["nll"], b["nll"], atol=config["nll_atol"], rtol=config["nll_rtol"]):
            raise ContinuationError("CF_NOOP_NLL_PARITY_FAILED")
    keys = sorted(base, key=lambda k: base[k]["window_index"])
    old = reference["result"]["quality"]["greedy"]
    means = [float(np.mean(base[k]["nll"])) for k in keys]
    if not np.allclose(means, old["nll"], atol=config["nll_atol"], rtol=config["nll_rtol"]):
        raise ContinuationError("CF_HISTORICAL_STORY_NLL_PARITY_FAILED")
    rows = [base[k] for k in keys]
    tokens = sum(len(r["nll"]) for r in rows)
    total_length = sum(sum(r["lengths"]) for r in rows)
    ctx = sum(sum(r["ctx_calls"]) for r in rows)
    edges = sum(r["transition_count"] for r in rows)
    loops = sum(sum(r["self_loops"]) for r in rows)
    actual = {
        "mean_nll": float(np.mean(means)),
        "mean_path_length": total_length / tokens,
        "actual_stop_rate": sum(sum(r["actual_stop"]) for r in rows) / tokens,
        "forced_end_rate": sum(sum(r["forced_end"]) for r in rows) / tokens,
        "context_call_fraction": ctx / total_length,
        "self_loop_rate": loops / max(1, edges),
        "path_count": tokens,
    }
    expected = {
        "mean_nll": 4.699343863874674,
        "mean_path_length": 14.7386474609375,
        "actual_stop_rate": 0.258056640625,
        "forced_end_rate": 0.741943359375,
        "context_call_fraction": 0.27624048567571374,
        "self_loop_rate": 0.20667809892755915,
        "path_count": 8192,
    }
    errors = {k: actual[k] - expected[k] for k in expected}
    for k in expected:
        if not np.isclose(
            actual[k],
            expected[k],
            atol=config["nll_atol"] if k == "mean_nll" else 0,
            rtol=config["nll_rtol"] if k == "mean_nll" else 0,
        ):
            raise ContinuationError("CF_HISTORICAL_AGGREGATE_PARITY_FAILED", k)
    transitions = {}
    for r in rows:
        for key, n in r["transition_matrix"].items():
            transitions[key] = transitions.get(key, 0) + n
    if transitions != old["transition_matrix"]:
        raise ContinuationError("CF_HISTORICAL_TRANSITION_PARITY_FAILED")
    return {
        "status": "BASELINE_PARITY_PASS",
        "actual": actual,
        "expected": expected,
        "errors": errors,
        "native_vs_noop": "EXACT_ACTIONS_STATES_STOP_LENGTH",
        "historical_story_means": "WITHIN_PREDECLARED_FLOAT_TOLERANCE",
        "historical_raw_action_fingerprint": "NOT_RECORDED_BY_OLD_EVALUATOR",
        "historical_trace_evidence": "EXACT_AGGREGATE_TRANSITIONS_AND_TERMINATIONS_PLUS_NATIVE_EXECUTOR_REPLAY",
        "window_count": len(rows),
        "training_updates": 0,
    }
