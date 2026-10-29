"""Immutable D1 identity and checkpoint-resume validation."""

import hashlib
import json


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def identity(config, source_sha, data_sha, arm):
    return {
        "experiment_id": config["experiment_id"],
        "source_sha": source_sha,
        "config_sha256": sha256_bytes(canonical(config).encode("utf-8")),
        "data_sha256": data_sha,
        "seed": config["seed"],
        "anchor_step": config["anchor_step"],
        "arm": arm,
    }


def validate_resume(saved, expected):
    if saved.get("identity") != expected:
        raise ValueError("D1_RESUME_IDENTITY_MISMATCH")
    relative = saved.get("relative_step", -1)
    if relative < 0 or relative > expected.get("max_steps", 200):
        raise ValueError("D1_RESUME_STEP_INVALID")
    if saved.get("step") != expected.get("anchor_step", 300) + relative:
        raise ValueError("D1_RESUME_STEP_INVALID")
    required = (
        "trainer_state_sha256",
        "optimizer_state_sha256",
        "data_rng_state",
        "route_rng_state_sha256",
        "frozen_parameter_sha256",
    )
    if any(key not in saved for key in required):
        raise ValueError("D1_RESUME_STATE_INCOMPLETE")
    return True
