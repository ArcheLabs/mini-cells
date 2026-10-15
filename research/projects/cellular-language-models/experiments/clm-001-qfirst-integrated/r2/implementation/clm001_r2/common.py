"""R2 identity and strict configuration; historical helpers remain read only."""

import json
from copy import deepcopy
from pathlib import Path
from clm001.common import (
    ROOT,
    tensor_hash,
    source_sha,
    canonical_sha256,
    write_json,
    file_sha256,
    hash_accepted_state,
    AcceptedModelSnapshot,
    validate as validate_r1,
)

__all__ = [
    "ROOT",
    "tensor_hash",
    "source_sha",
    "canonical_sha256",
    "write_json",
    "file_sha256",
    "hash_accepted_state",
    "AcceptedModelSnapshot",
    "BASE",
    "ARMS",
    "load",
    "validate",
    "authorize",
    "epsilon",
]

BASE = Path(__file__).resolve().parents[2]
ARMS = ("QF-R2", "J0-R2")


def load(path=None):
    c = json.loads(Path(path or BASE / "config/proposed.json").read_text())
    validate(c)
    return c


def validate(c):
    if c["experiment_id"] != "CLM-001-R2" or tuple(c["arms"]) != ARMS:
        raise ValueError("independent R2 identity required")
    old = deepcopy(c)
    old.update(experiment_id="CLM-001-QFIRST", arms=["QF-FULL", "J0-FULL"])
    validate_r1(old)
    if (
        c.get("formal_confirmation")
        or c.get("scientific_decision")
        or c["gates"]["confirmation"]
        or c["authorization"]["confirmation"]
    ):
        raise ValueError("confirmation remains sealed")
    if (
        c["policy"]["lambda_cost"] != 0.01
        or c["policy"]["stop_tie_rule"] != "prefer_STOP_when_legal"
    ):
        raise ValueError("cost/tie contract changed")
    schedule = c["policy"]["epsilon_schedule"]
    if (
        not schedule
        or schedule[0][0] != 1
        or any(not 0 < v <= 1 for _, v in schedule)
        or any(a[0] >= b[0] for a, b in zip(schedule, schedule[1:]))
    ):
        raise ValueError("invalid shared exploration schedule")
    if c["training"]["qf_auxiliary_critic_training"] or c["oracle"]["suffix_samples_allowed"] != [
        2,
        4,
        8,
    ]:
        raise ValueError("relative-Q/MC contract")
    if (
        c["oracle"]["pairwise_min_samples"] < 4
        or c["oracle"]["pair_z"] <= 0
        or c["oracle"]["pair_tolerance"] < 0
    ):
        raise ValueError("invalid pair confidence")
    if (
        c["oracle"]["max_counterfactual_operator_calls_per_step"]
        < 2
        * c["oracle"]["probes_per_step"]
        * (c["oracle"]["extra_actions"] + 1)
        * c["model"]["budget"]
    ):
        raise ValueError("budget must cover initial samples")
    if c["shadow"]["nll_regression_tolerance"] != c["shadow"]["max_upper95_nll_regression"]:
        raise ValueError("Admission upper95 regression field mismatch")
    if not c.get("synthetic") and (
        c["model"]["budget"] != 16
        or c["training"]["steps"] != 500
        or c["training"]["target_tokens"] != 256000
    ):
        raise ValueError("R2 development budget changed")
    if c["model"].get("connection_extra_context_summary", False) or c["model"].get(
        "cell_direct_memory_access", False
    ):
        raise ValueError("information channel change forbidden")


def epsilon(c, step):
    return next(v for start, v in reversed(c["policy"]["epsilon_schedule"]) if step >= start)


def authorize(c, phase, device="cpu"):
    validate(c)
    if c.get("synthetic"):
        if str(device) != "cpu":
            raise RuntimeError("synthetic fixtures CPU only")
        return
    if phase in ("train-dev", "evaluate") and not c["authorization"]["development_training"]:
        raise RuntimeError("500-step GPU development not authorized in this implementation task")
    if phase not in ("gpu-pilot", "train-dev", "evaluate"):
        raise RuntimeError("unknown execution phase")
