"""R3 identity and gates. Source protocol remains immutable; runtime gates are automatic."""

from copy import deepcopy
import json
from pathlib import Path
from clm001_r2.common import (
    ROOT,
    source_sha,
    canonical_sha256,
    write_json,
    file_sha256,
    tensor_hash,
    hash_accepted_state,
    AcceptedModelSnapshot,
    epsilon,
    validate as validate_r2,
)

__all__ = [
    "ROOT",
    "BASE",
    "ARMS",
    "source_sha",
    "canonical_sha256",
    "write_json",
    "file_sha256",
    "tensor_hash",
    "hash_accepted_state",
    "AcceptedModelSnapshot",
    "epsilon",
    "load",
    "validate",
    "authorize",
]
BASE = Path(__file__).resolve().parents[2]
ARMS = ("Q-R3", "QC-R3", "QS-R3")


def load(path=None):
    c = json.loads(Path(path or BASE / "config/proposed.json").read_text())
    validate(c)
    return c


def validate(c):
    if c["experiment_id"] != "CLM-001-R3" or tuple(c["arms"]) != ARMS:
        raise ValueError("isolated three-arm R3 identity required")
    old = deepcopy(c)
    old.update(experiment_id="CLM-001-R2", arms=["QF-R2", "J0-R2"])
    validate_r2(old)
    if not c.get("synthetic") and (c["seed"] not in c["seeds"] or c["seed"] != 151001301):
        raise ValueError("only seed1 development authorized")
    x = c["competition"]
    if (
        x["enabled_arms"] != ["QC-R3", "QS-R3"]
        or x["candidate_sampling"] != "uniform_without_replacement"
    ):
        raise ValueError("frozen treatment/sampling contract")
    if (
        x["method"] != "post_hard_backward_router_surrogate"
        or not x["cell_only"]
        or x["counterfactual_gradients"]
        or x["balance_weight"] != 0
    ):
        raise ValueError("LCG information/gradient contract")
    if x["sham"] != "zero_competition_gradient_same_auxiliary_forward" or x["temperature"] <= 0:
        raise ValueError("frozen sham/temperature contract")
    if (
        not 2 <= x["candidate_cells"] <= c["model"]["accepted_cells"]
        or not 0 < x["events_per_step"] <= 32
    ):
        raise ValueError("candidate/event budget")
    if x["beta"] is not None and not 0 < x["beta"] <= x["beta_max"]:
        raise ValueError("invalid frozen beta")
    if c["alignment"]["minimum_pairs"] < 1 or c["alignment"]["minimum_stories"] < 2:
        raise ValueError("invalid alignment gate")
    if not c.get("synthetic") and (
        c["alignment"]["minimum_pairs"] < 100 or c["alignment"]["minimum_stories"] < 32
    ):
        raise ValueError("do not weaken real-data alignment gate")
    if c["authorization"]["confirmation"]:
        raise ValueError("confirmation sealed")


def authorize(c, phase, device="cpu"):
    validate(c)
    if c.get("synthetic"):
        if str(device) != "cpu":
            raise RuntimeError("synthetic CPU only")
        return
    if phase == "gpu-pilot":
        return
    if phase not in ("train-dev", "evaluate"):
        raise RuntimeError("unknown phase")
    e = c.get("execution", {})
    if not c["authorization"]["development_training"] or c["competition"]["beta"] is None:
        raise RuntimeError("successful automatic gradient/alignment preflight required")
    if (
        not all(
            e.get(k) is True for k in ("cpu_pass", "gradient_pilot_pass", "local_alignment_pass")
        )
        or e.get("source_sha") != source_sha()
    ):
        raise RuntimeError("R3 automatic preflight identity mismatch")
    if e.get("configuration_sha256") != canonical_sha256(
        {k: v for k, v in c.items() if k != "execution"}
    ):
        raise RuntimeError("runtime configuration changed after beta freeze")
