"""Versioned execution schedule; no model forwards or event decisions."""

from pathlib import Path
import time
from minicells_runtime.store import read, atomic, canonical, ContinuationError
from .common import BASE


def load_plan(path=None):
    plan = read(path or BASE / "config/diagnostic-execution-v1.json")
    validate(plan)
    return plan


def validate(plan):
    schema = read(BASE / "config/diagnostic-execution-v1.schema.json")

    def shape(value, spec):
        if "const" in spec and value != spec["const"]:
            raise ContinuationError("ANALYSIS_PROTOCOL_CHANGED")
        if spec.get("type") == "object":
            if not isinstance(value, dict) or set(value) != set(spec["required"]):
                raise ContinuationError("ANALYSIS_PLAN_INVALID")
            for key, child in spec["properties"].items():
                shape(value[key], child)
        elif spec.get("type") == "array":
            if not isinstance(value, list):
                raise ContinuationError("ANALYSIS_PLAN_INVALID")

    for key in ("quality_event_compare_at_400", "functional_event_compare_at_400"):
        if plan.get("posthoc", {}).get(key) != ["pre_event", "post_event"]:
            raise ContinuationError("EVENT_COMPARE_REQUIRED")
    shape(plan, schema)
    if (
        plan.get("schema") != "minicells.r3b.diagnostic-execution.v1"
        or plan.get("mode") != "DEFER_EXPENSIVE_ANALYSIS"
    ):
        raise ContinuationError("ANALYSIS_PLAN_INVALID")
    post = plan["posthoc"]
    if (
        post["primary_policies"] != ["greedy", "uniform"]
        or post["secondary_policies"] != ["sampled", "constant"]
        or post["required_roles"] != ["calibration", "development"]
        or post["confirmation"] != "SEALED"
        or post["sampling"] != "INHERIT_FROZEN_R3B_PROTOCOL"
        or plan["training"]["mandatory_semantic_event_evaluation"] != "UNCHANGED"
        or plan["training"]["preserve_every_original_event_revision"] is not True
        or plan["training"]["save_frozen_probe_banks_at_step_zero"] is not True
        or plan["analysis_publication"]["separate_from_training_latest"] is not True
    ):
        raise ContinuationError("ANALYSIS_PROTOCOL_CHANGED")
    for key in ("quality_primary_steps", "quality_secondary_steps", "functional_steps"):
        values = post[key]
        if values != sorted(set(values)) or any(
            type(v) is not int or not 0 <= v <= 500 for v in values
        ):
            raise ContinuationError("ANALYSIS_PLAN_INVALID")
    for key in ("quality_event_compare_at_400", "functional_event_compare_at_400"):
        if post[key] != ["pre_event", "post_event"]:
            raise ContinuationError("EVENT_COMPARE_REQUIRED")


def deferred(c):
    return c["r3b"].get("diagnostic_execution", {}).get("mode") == "DEFER_EXPENSIVE_ANALYSIS"


def stages(plan, step):
    post = plan["posthoc"]
    result = []
    if step in post["quality_primary_steps"]:
        result.append(("quality", ["greedy", "uniform"], "development"))
    if step in post["quality_secondary_steps"]:
        result.append(("quality", ["sampled", "constant"], "development"))
    if step in post["functional_steps"]:
        for role in post["required_roles"]:
            result.append(("functional", [], role))
        result.append(("action", [], "calibration"))
    return result


def telemetry(folder, trainer, c, phase):
    from .instrumentation import gradient_row

    begin = time.perf_counter()
    folder = Path(folder)
    latest = read(folder / "latest.json")
    key = canonical({"checkpoint": latest["sha256"], "phase": phase})
    path = folder / "telemetry" / (key + ".json")
    if path.exists():
        return read(path)
    plan = c["r3b"]["diagnostic_execution"]
    value = {
        "schema": "minicells.r3b.telemetry.v1",
        "category": "TRAINING_TELEMETRY",
        "identity": read(folder / "identity.json"),
        "step": trainer.step,
        "phase": phase,
        "checkpoint_sha256": latest["sha256"],
        "checkpoint_file": latest["file"],
        "registry_version": trainer.model.version,
        "registry": __import__("dataclasses").asdict(trainer.model.registry),
        "gradient_audit": gradient_row(trainer, c),
        "journal_rows": len(trainer.journal),
        "extra_model_forwards": 0,
        "analysis_status": "DEFERRED",
        "execution_protocol_sha256": canonical(plan),
        "pending_stages": [
            {"stage": s, "policies": p, "role": r} for s, p, r in stages(plan, trainer.step)
        ],
        "telemetry_seconds": time.perf_counter() - begin,
    }
    atomic(path, value)
    return value
