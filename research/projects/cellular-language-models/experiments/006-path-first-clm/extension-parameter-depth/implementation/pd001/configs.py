"""Actual instantiated parameter search, proposal validation and fail-closed gates."""

import copy
import json
import re
from pathlib import Path
import torch
from exp006.reuse import canonical_sha256, file_sha256
from .models import make_model

BASE = Path(__file__).resolve().parents[2]
PARENT = BASE.parent
ROOT = next(p for p in BASE.parents if (p / "pyproject.toml").exists())
ARMS = (
    "M0",
    "P100-F16",
    "P50-F16",
    "P50-F32",
    "P50-A32",
    "P25-F32",
    "P25-A64",
    "P50-SELF32",
    "P50-STATE32",
    "P50-A32-J1",
)


def load(path=None):
    c = json.loads(Path(path or BASE / "config/proposed.json").read_text())
    p = json.loads((BASE / "protocol.json").read_text())
    if c.get("execution", {}).get("mode") == "automatic_development":
        p = c["execution"]["protocol"]
    validate(c)
    return c, p


def validate(c):
    if c["experiment_id"] != "006-PD1" or c["model"]["memory_mode"] != "static_input_readonly":
        raise ValueError("PD identity or memory mismatch")
    from .performance import settings

    settings(c)
    t = c["training"]
    if t["tokens"] != t["steps"] * t["effective_batch"] * t["sequence_length"]:
        raise ValueError("effective training token budget mismatch")
    if t["effective_batch"] % t["microbatch"] or t["microbatch"] < 1:
        raise ValueError("invalid accumulation")
    if c["model"]["step_capacity"] < 64 or set(c["arms"]) != set(ARMS):
        raise ValueError("complete declared arms and long-path capacity required")
    if set(c["seeds"]["development"]) & set(c["seeds"]["confirmation"]):
        raise ValueError("seed overlap")
    ranges = [c["data"][key] for key in ("train_range", "development_range", "confirmation_range")]
    if c["data"]["split"] != "train" or any(lo < 0 or hi <= lo for lo, hi in ranges):
        raise ValueError("invalid raw-row ranges")
    if any(max(a, d) < min(b, e) for i, (a, b) in enumerate(ranges) for d, e in ranges[i + 1 :]):
        raise ValueError("data ranges overlap")


def arm_config(c, arm):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    model = copy.deepcopy(c["model"])
    if arm == "M0":
        model["steps"] = 6
        return model
    spec = c["arms"][arm]
    model.update(c["tiers"][spec["tier"]])
    model.update(
        budget=spec["budget"],
        adaptive=spec["adaptive"],
        self_only=spec.get("self_only", False),
        state_only=spec.get("state_only", False),
    )
    return model


def count(c):
    result = {}
    for arm in ARMS:
        # Meta tensors construct the true architecture without allocating candidate weights.
        with torch.device("meta"):
            model = make_model(arm_config(c, arm), arm)
        groups = {}
        for name, p in model.named_parameters():
            group = (
                "critic"
                if ".critic." in name
                else "connection"
                if name.startswith("controller.")
                else "embedding_head"
                if name.startswith(("embedding.", "position."))
                else "cells"
                if name.startswith("cells.")
                else "context"
                if name.startswith(("memory.", "context."))
                else "shared"
                if arm != "M0"
                else "moe_backbone"
            )
            groups[group] = groups.get(group, 0) + p.numel()
        result[arm] = {
            "total": sum(p.numel() for p in model.parameters()),
            "groups": groups,
            "width": model.config["width"],
            "dtype": "float32",
            "tied_head": True,
        }
    baseline = result["M0"]["total"]
    for arm, entry in result.items():
        ratio = entry["total"] / baseline
        goal = 1.0 if arm == "M0" else c["arms"][arm]["ratio"]
        entry.update(
            ratio_to_moe=ratio,
            target_ratio=goal,
            ratio_pass=abs(ratio / goal - 1) <= 0.05,
            resident_weight_bytes=entry["total"] * 4,
        )
    return result


def search(c):
    c = copy.deepcopy(c)
    baseline = count(c)["M0"]["total"]
    # Cell hidden width controls real used operator capacity; never add dummy parameters.
    for tier, target in (("100", 1.0), ("50", 0.5), ("25", 0.25)):
        arm = {"100": "P100-F16", "50": "P50-A32", "25": "P25-A64"}[tier]
        candidates = []
        for hidden in range(32, 2049, 16):
            c["tiers"][tier]["cell_hidden"] = hidden
            with torch.device("meta"):
                model = make_model(arm_config(c, arm), arm)
            n = sum(p.numel() for p in model.parameters())
            candidates.append((abs(n - baseline * target), hidden))
        c["tiers"][tier]["cell_hidden"] = min(candidates)[1]
    return c, count(c)


def blockers(c, p):
    issues = []
    parent = json.loads((PARENT / "protocol.json").read_text())
    if parent.get("status") != "REGISTERED":
        issues.append("parent 006 remains unregistered; independent approval of the PD extension is required")
    if p.get("status") != "REGISTERED":
        issues.append("PD Gate 1 not registered")
    for key in ("source_sha", "approval_record_sha256", "evidence_sha256"):
        if not re.fullmatch("[0-9a-f]{40}" if key == "source_sha" else "[0-9a-f]{64}", p.get(key) or ""):
            issues.append(key + " not frozen")
    evidence = BASE / "registration/evidence.json"
    if not evidence.exists() or file_sha256(evidence) != p.get("evidence_sha256"):
        issues.append("review evidence absent or mismatched")
    else:
        e = json.loads(evidence.read_text())
        if e.get("config_sha256") != canonical_sha256(c) or e.get("user_approved") is not True:
            issues.append("review does not bind config and user approval")
        for key in (
            "seed_collision_audit",
            "asset_conflict_audit",
            "baseline_power_audit",
            "parent_gate_review",
        ):
            if e.get(key) is not True:
                issues.append(key + " missing")
    for key in ("manifest_sha256", "source_sha256", "tokenizer_sha256"):
        if not re.fullmatch("[0-9a-f]{64}", c["data"].get(key) or ""):
            issues.append(key + " not frozen")
    if not c["seeds"]["development"] or len(set(c["seeds"]["confirmation"])) < 3:
        issues.append("independent seed matrix not frozen")
    if not all(x["ratio_pass"] for x in count(c).values()):
        issues.append("actual parameter ratios fail")
    if p.get("thresholds", {}).get("baseline_stability") is None:
        issues.append("baseline stability criterion not frozen")
    return issues


def authorize(c, p, arm, seed, role, mode, source=None, device="cpu"):
    if mode not in ("train", "evaluate", "profile", "publish", "confirmation"):
        raise ValueError("not a scientific mode")
    automatic = c.get("execution", {}).get("mode") == "automatic_development"
    if automatic:
        if role != "development" or mode == "confirmation" or c["authorization"]["confirmation"]:
            raise RuntimeError("automatic development never grants confirmation access")
        if not p.get("automatic_development") or p.get("status") != "AUTOMATIC_DEVELOPMENT":
            raise RuntimeError("automatic development identity missing")
        for key in ("manifest_sha256", "source_sha256", "tokenizer_sha256"):
            if not re.fullmatch("[0-9a-f]{64}", c["data"].get(key) or ""):
                raise RuntimeError("automatic asset identity missing: " + key)
    else:
        issues = blockers(c, p)
        if issues:
            raise RuntimeError("PROPOSED_NOT_REGISTERED: " + "; ".join(issues))
    if arm not in ARMS or role not in ("development", "confirmation") or seed not in c["seeds"][role]:
        raise RuntimeError("unregistered arm/seed/role")
    if source != p["source_sha"]:
        raise RuntimeError("source SHA differs from approved registration")
    import subprocess

    protected = [
        str(BASE.relative_to(ROOT) / "implementation"),
        str(BASE.relative_to(ROOT) / "run_pd001.py"),
        str((PARENT / "implementation").relative_to(ROOT)),
        "shared/python",
        "src/minicells/reproducibility.py",
    ]
    if subprocess.run(["git", "diff", "--quiet", source, "--", *protected], cwd=ROOT).returncode:
        raise RuntimeError("runtime source differs from frozen implementation")
    auth = c["authorization"]
    if not auth[mode if mode == "publish" else "confirmation" if role == "confirmation" else "development"]:
        raise RuntimeError("operation not authorized")
    if str(device).startswith("cuda") and not auth["gpu"]:
        raise RuntimeError("GPU not authorized")
    if mode == "train" and c["training"]["steps"] > 100:
        from .performance import retrain_review

        retrain_review(c, source)
    if role == "confirmation" and (
        not p.get("candidate_locked") or not c["data"].get("confirmation_manifest_sha256")
    ):
        raise RuntimeError("confirmation remains sealed")


def cost_lambda(c, step):
    t = c["training"]
    return t["cost_lambda"] * min(1.0, max(0.0, (step - t["cost_start"]) / max(1, t["cost_ramp"])))
