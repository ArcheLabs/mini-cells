"""Reuse historical atomic writes, checkpoints and byte-accurate accepted-state witnesses."""

import hashlib
import json
from pathlib import Path
import subprocess
from exp006.reuse import canonical_sha256, write_json as _write
from exp006.reuse import file_sha256 as file_sha256
from minicells.shadow_maturation import hash_accepted_state as hash_accepted_state
from minicells.shadow_maturation import AcceptedModelSnapshot as AcceptedModelSnapshot

BASE = Path(__file__).resolve().parents[2]
ROOT = next(p for p in BASE.parents if (p / "pyproject.toml").exists())
ARMS = ("QF-FULL", "J0-FULL")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, value)


def tensor_hash(tensor):
    h = hashlib.sha256()
    t = tensor.detach().cpu().contiguous()
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def source_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load(path=None):
    c = json.loads(Path(path or BASE / "config/proposed.json").read_text())
    validate(c)
    return c


def validate(c):
    if c["experiment_id"] != "CLM-001-QFIRST" or tuple(c["arms"]) != ARMS:
        raise ValueError("exact two-arm Q-first identity required")
    m, t = c["model"], c["training"]
    if c["policy"]["suffix_kind"] != "epsilon_boltzmann_frozen_q":
        raise ValueError("only the fixed replanning suffix policy is registered")
    if m["budget"] < 2 or m["minimum_actions"] >= m["budget"] or m["width"] % m["heads"]:
        raise ValueError("invalid model/STOP capacity")
    if c["oracle"]["suffix_samples"] < 2:
        raise ValueError("MC labels need at least two suffix samples")
    if (
        m["reserve_slots"] != 1
        or m["memory_mode"] != "static_input_readonly"
        or m["minimum_actions"] < 1
    ):
        raise ValueError("reserve/context/minimum STOP contract")
    if t["target_tokens"] != t["steps"] * t["effective_batch"] * t["sequence_length"]:
        raise ValueError("formal live target token budget mismatch")
    if t["steps"] not in (50, 500) and not c.get("synthetic"):
        raise ValueError("only smoke50/dev500")
    if t["effective_batch"] % t["microbatch"] or t["precision"] != "float32":
        raise ValueError("batch/precision mismatch")
    if (
        c["oracle"]["probes_per_step"] > 2
        or c["oracle"]["extra_actions"] > 2
        or c["oracle"]["cross_version_replay"]
    ):
        raise ValueError("bounded same-version oracle required")
    if (
        c["policy"]["q_convention"] != "lower_is_better"
        or not 0 < c["policy"]["epsilon"] <= 1
        or c["policy"]["temperature"] <= 0
    ):
        raise ValueError("Q cost/policy convention")
    if c["gates"]["confirmation"]:
        raise ValueError("confirmation is sealed")
    ranges = [
        c["data"][key]
        for key in ("train_range", "calibration_range", "admission_range", "development_range")
    ]
    if c["data"]["split"] != "train" or any(
        a < 0 or b <= a or b > min(c["data"]["reserved_boundary"], 60000) for a, b in ranges
    ):
        raise ValueError("reserved data range")
    if any(max(a, x) < min(b, y) for i, (a, b) in enumerate(ranges) for x, y in ranges[i + 1 :]):
        raise ValueError("story-role overlap")


def authorize(c, phase, device="cpu"):
    validate(c)
    if c.get("synthetic"):
        if str(device) != "cpu":
            raise RuntimeError("synthetic engineering is CPU only")
        return
    if not c["gates"]["oracle_cpu_pass"]:
        raise RuntimeError("ORACLE_UNTRUSTED: pass the same-source CPU oracle gate")
    if phase == "gpu-pilot" and not c["gates"]["gpu_pilot_approved"]:
        raise RuntimeError("GPU Oracle Pilot requires separate explicit user approval")
    if phase in ("train-smoke", "train-dev", "evaluate"):
        required = ("gpu_pilot_pass", "development_approved", "thresholds_frozen")
        if phase != "train-smoke":
            required = required + ("portable_pass",)
        if not all(c["gates"][k] for k in required):
            raise RuntimeError("development/portable/threshold review remains pending")
        if c["policy"]["lambda_cost"] is None or any(v is None for v in c["thresholds"].values()):
            raise RuntimeError("freeze cost and numerical research thresholds first")
        if (
            c["shadow"]["min_cost_aware_improvement"] is None
            or c["shadow"]["nll_regression_tolerance"] is None
        ):
            raise RuntimeError("freeze admission thresholds first")
    if c.get("approved_source") != source_sha():
        raise RuntimeError("approved source identity mismatch")
    expected = c.get("approved_config_sha256")
    if expected != canonical_sha256({k: v for k, v in c.items() if k != "approved_config_sha256"}):
        raise RuntimeError("approved configuration identity mismatch")
