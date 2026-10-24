"""Immutable R3m-E matrix, schedule and admission rules."""

import hashlib
import json
import math
from pathlib import Path

from minicells_r3e.common import canonical


BASE = Path(__file__).resolve().parents[2]
CONFIG = BASE / "config/exploratory.frozen.json"
CONFIG_LOCK = BASE / "config/exploratory.frozen.sha256"
ARMS = ("E0-STORY-STABLE", "E1-MIX-STABLE", "E2-MIX-REORG", "E3-MIX-SHAM")
SEEDS = (151001301, 151001302, 151001303)


def load_config():
    config = json.loads(CONFIG.read_text())
    validate(config)
    return config


def validate(c):
    locks = (CONFIG_LOCK, BASE / "config/exploratory.released.sha256")
    if canonical(c) not in {p.read_text().strip() for p in locks if p.is_file()}:
        raise ValueError("R3ME_FROZEN_CONFIG_HASH_MISMATCH")
    if c["authorization"]["full_training"]:
        original = json.loads(CONFIG.read_text())
        excluded = {"status", "authorization", "release", "execution"}
        if {k: v for k, v in c.items() if k not in excluded} != {k: v for k, v in original.items() if k not in excluded}:
            raise ValueError("R3ME_RELEASE_CHANGED_SCIENTIFIC_PROTOCOL")
        if c["release"]["parent_config_sha256"] != canonical(original):
            raise ValueError("R3ME_RELEASE_PARENT_CONFIG_CHANGED")
    if c.get("experiment_id") != "CLM-001-R3m-E-Capability-Differentiation-001":
        raise ValueError("R3ME_EXPERIMENT_ID_CHANGED")
    if tuple(c.get("arms", ())) != ARMS or tuple(c.get("seeds", ())) != SEEDS:
        raise ValueError("R3ME_MATRIX_CHANGED")
    if c.get("formal_confirmation") != "NOT_AUTHORIZED" or c["authorization"]["formal_confirmation"]:
        raise ValueError("R3ME_CONFIRMATION_MUST_REMAIN_SEALED")
    if not isinstance(c["authorization"]["full_training"], bool):
        raise ValueError("R3ME_GPU_AUTHORIZATION_MUST_BE_BOOLEAN")
    if c["model"]["cells"] != 6 or c["model"]["calls_per_token"] != 8:
        raise ValueError("R3ME_MODEL_CAPACITY_CHANGED")
    if (
        c["model"]["sequence_length"] != 64
        or c["model"]["vocab_size"] != 4096
        or c["model"]["tokenizer_sha256"]
        != "612c2aaa09e7f35c89c41c1e6270bdca1ef861a283a4de859cd808c8e6bfe6ff"
    ):
        raise ValueError("R3ME_TOKENIZER_OR_SEQUENCE_IDENTITY_CHANGED")
    if c["anchor_step"] != 300 or c["continuation_steps"] != 600:
        raise ValueError("R3ME_STEP_BUDGET_CHANGED")
    if c["tokens_per_step"] != 512 or c["calls_per_token"] != 8:
        raise ValueError("R3ME_TOKEN_OR_PATH_BUDGET_CHANGED")
    if (
        c["training"]["effective_batch"] != 8
        or c["training"]["microbatch"] != 1
        or c["training"]["sequence_length"] != 64
    ):
        raise ValueError("R3ME_BATCHING_IDENTITY_CHANGED")
    if c["proposal_at"] != 100 or c["shadow_steps"] != 10:
        raise ValueError("R3ME_INTERVENTION_SCHEDULE_CHANGED")
    if c["barriers"] != [0, 100, 110, 200, 400, 600]:
        raise ValueError("R3ME_EVALUATION_BARRIERS_CHANGED")
    if c["baseline"]["source_sha"] != "01f226e259356bb06a8ef275e8314649062446e4":
        raise ValueError("R3ME_PARENT_SOURCE_CHANGED")
    if c["baseline"]["hf_reference"]["hf_revision"] != "c3d182f7684e61b3d1d8fa44e6c7b41a7a5a42df":
        raise ValueError("R3ME_PARENT_REVISION_CHANGED")
    if c["data"]["revision"] != "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64":
        raise ValueError("R3ME_DATA_REVISION_CHANGED")
    if c["data"]["confirmation"] != "SEALED" or not c["data"]["main_candidates_forbidden"]:
        raise ValueError("R3ME_FORBIDDEN_DATA_ROLE_ENABLED")
    if c["intervention"]["trust_region"] != {
        "kl_max": 0.01,
        "hard_action_change_max": 0.1,
        "full_path_change_max": 0.2,
    }:
        raise ValueError("R3ME_TRUST_REGION_RELAXED")
    if c["training"]["logical_updates"] != 7200 or c["training"]["logical_tokens"] != 3686400:
        raise ValueError("R3ME_LOGICAL_BUDGET_CHANGED")
    expected_schedule = [
        {"relative_steps": [0, 100], "story_tokens_fraction": 0.9, "arithmetic_tokens_fraction": 0.1},
        {"relative_steps": [100, 200], "story_tokens_fraction": 0.7, "arithmetic_tokens_fraction": 0.3},
        {"relative_steps": [200, 600], "story_tokens_fraction": 0.5, "arithmetic_tokens_fraction": 0.5},
    ]
    if c["training"]["task_schedule"] != expected_schedule:
        raise ValueError("R3ME_TASK_SCHEDULE_CHANGED")
    if c["arithmetic"]["operand_range_inclusive"] != [10, 99]:
        raise ValueError("R3ME_ARITHMETIC_DOMAIN_CHANGED")
    if c["evaluation"]["relative_steps"] != [0, 100, 110, 200, 400, 600]:
        raise ValueError("R3ME_EVALUATION_BARRIERS_CHANGED")
    if c["data"]["story_dev"] != "REUSED_HISTORICAL_DEVELOPMENT":
        raise ValueError("R3ME_STORY_DEV_IDENTITY_CHANGED")
    return True


def task_for(relative_step):
    if not 0 <= relative_step < 600:
        raise ValueError("R3ME_RELATIVE_STEP_INVALID")
    arithmetic_per_ten = 1 if relative_step < 100 else 3 if relative_step < 200 else 5
    return "arithmetic" if relative_step % 10 < arithmetic_per_ten else "story"


def schedule(anchor_lr, anchor_epsilon, relative_step, c):
    if not math.isfinite(anchor_lr) or anchor_lr <= 0 or not math.isfinite(anchor_epsilon):
        raise ValueError("R3ME_ANCHOR_SCHEDULE_INVALID")
    if not 0 <= relative_step <= c["continuation_steps"]:
        raise ValueError("R3ME_RELATIVE_STEP_INVALID")
    f = min(relative_step, c["continuation_steps"] - 1) / max(1, c["continuation_steps"] - 1)
    return (
        anchor_lr * (1 + (c["lr_terminal_fraction"] - 1) * f),
        anchor_epsilon + (c["epsilon_terminal"] - anchor_epsilon) * f,
    )


def identity(c, source_sha):
    row = {
        "experiment_id": c["experiment_id"],
        "source_sha": source_sha,
        "config_sha256": canonical(c),
        "phase": "EXPLORATORY_DEVELOPMENT",
    }
    row["run_id"] = canonical(row)
    return row


def arithmetic_assignment_sha(c):
    return hashlib.sha256(canonical(c["arithmetic"]).encode()).hexdigest()

