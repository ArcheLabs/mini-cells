"""Frozen fact allocation, task budgets and continuous inherited optimizer schedule."""

import copy
from pathlib import Path
from minicells_r3me.common import (
    atomic as atomic,
    canonical,
    digest as digest,
    fingerprint as fingerprint,
    read,
    source_sha as source_sha,
)
from minicells_r3me.protocol import schedule as schedule, identity as identity

BASE = Path(__file__).resolve().parents[2]
CONFIG = BASE / "config/exploratory.frozen.json"
ARMS = ("F0-STORY", "F1-COPY", "F2-A1")


def load(path=None):
    c = read(path or CONFIG)
    validate(c)
    return c


def validate(c):
    if canonical(c) != (BASE / "config/exploratory.frozen.sha256").read_text().strip():
        raise ValueError("R3MF_FROZEN_CONFIG_CHANGED")
    if tuple(c["arms"]) != ARMS or c["authorization"]["full_training"]:
        raise ValueError("R3MF_PILOT_REQUIRES_INDEPENDENT_REVIEW_AND_RELEASE")
    if c["formal_confirmation"] != "NOT_AUTHORIZED" or c["data"]["confirmation"] != "SEALED":
        raise ValueError("R3MF_CONFIRMATION_FORBIDDEN")
    return True


def execution_config(c, phase):
    q = copy.deepcopy(c)
    q["execution_phase"] = phase
    q["frozen_config_sha256"] = canonical(c)
    if phase == "tiny":
        q.update(
            seeds=[c["tiny"]["seed"]],
            arms=["TINY-A1"],
            continuation_steps=c["tiny"]["steps"],
            barriers=c["tiny"]["barriers"],
        )
        q["training"].update(
            continuation_steps=c["tiny"]["steps"],
            barrier_relative_steps=c["tiny"]["barriers"],
            logical_updates=c["tiny"]["steps"],
            logical_tokens=c["tiny"]["steps"] * c["tokens_per_step"],
        )
    elif phase != "pilot":
        raise ValueError("R3MF_PHASE_INVALID")
    return q


def task_for(step, arm):
    if arm == "TINY-A1":
        return "tiny"
    if arm == "F0-STORY" or step % 2:
        return "story"
    return {"F1-COPY": "copy", "F2-A1": "a1"}[arm]


def cpu_gate(c):
    validate(c)
    counts = {
        a: {t: sum(task_for(s, a) == t for s in range(900)) for t in ("story", "copy", "a1")}
        for a in ARMS
    }
    if counts["F1-COPY"]["story"] != 450 or counts["F2-A1"]["a1"] != 450:
        raise ValueError("R3MF_TOKEN_ALLOCATION_CHANGED")
    return dict(
        status="CPU_PROTOCOL_PASS",
        scope="SYNTHETIC_CPU_ONLY",
        task_steps=counts,
        pilot_updates=8100,
        pilot_tokens=4147200,
        tiny_updates=200,
        tiny_tokens=102400,
        gpu_pilot_completed=False,
        scientific_decision=False,
        full_training_authorized=False,
    )
