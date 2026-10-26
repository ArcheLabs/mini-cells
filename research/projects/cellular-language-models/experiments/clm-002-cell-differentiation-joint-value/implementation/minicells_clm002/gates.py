"""No Gate 0 result is inferred from CPU success or the earlier JCT authorization."""

from .common import sha


def status(c):
    return {
        "scientific_training": "BLOCKED",
        "stage_a_development": "NOT_RUN",
        "stage_a_confirmation": "NOT_RUN",
        "stage_b": "NOT_RUN",
        "confirmation": "SEALED",
        "registration": "BLOCKED",
        "blockers": [
            "REAL_PARENT_BYTES_AND_RESTORED_VALUES_NOT_VERIFIED_IN_THIS_EXPERIMENT",
            "GLOBAL_STORY_CONTENT_FRAGMENT_AND_WINDOW_COVERAGE_NOT_CLOSED",
            "ARITHMETIC_TRAIN_DEV_CONFIRM_FAMILY_OPERATOR_RANGE_ALLOCATION_UNREGISTERED",
            "CURRENT_R3H_TRAINER_FORBIDS_CONNECTION_UPDATES; RULE_AND_CADENCE_REQUIRE_REGISTRATION",
            "DEVELOPMENT_THRESHOLDS_SCIENTIFIC_BUDGET_QUALITY_CAP_AND_ANALYSIS_NOT_FROZEN",
        ],
        "config_sha256": sha(c),
        "cpu_engineering_is_not_scientific_pass": True,
        "stage_a_native_connection": "FROZEN_AS_INHERITED_NOT_AN_ENGINEERING_ERROR",
        "stage_b_additional_blocker": "PRODUCTION_CONNECTION_CADENCE_AND_GOVERNANCE_REQUIRE_REVIEW",
        "evaluation_weighting_across_S_H": "NOT_FROZEN",
    }


def require_scientific(c, mode):
    # There is no execution authorization or independent data registration in this draft.
    # A future reviewed release must provide actual evidence, not a Boolean --force bypass.
    raise RuntimeError(
        "CLM002_GATE0_BLOCKED_"
        + mode.upper().replace("-", "_")
        + ": "
        + ",".join(status(c)["blockers"])
    )
