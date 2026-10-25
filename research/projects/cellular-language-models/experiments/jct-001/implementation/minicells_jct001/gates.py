"""Scientific admission is distinct from engineering smoke success."""

from .common import BASE, read, sha


def status(c, local_restore=False, update_audit=False, cpu_cost=False):
    return dict(
        schema="minicells.jct001.gates.v1",
        scientific_execution="BLOCKED",
        gates={
            "0.1": dict(
                status="BLOCKED",
                synthetic_fixture_identity="CHECKED_AT_ENGINEERING_RUNTIME",
                real_data_registered=c["real_model"]["registered"],
                reason="R3M_GLOBAL_CONTENT_ISOLATION_UNRESOLVED; NO_NEW_STORY_RANGE_SELECTED",
            ),
            "0.2": dict(
                status="BLOCKED", reason="REAL_ARITHMETIC_TASK_LEARNABILITY_NOT_ESTABLISHED"
            ),
            "0.3": dict(
                status="BLOCKED",
                local_cpu_restore_verified=local_restore,
                cross_session_verified=False,
                reason="LOCAL_RESTORE_IS_NOT_KAGGLE_CROSS_SESSION_RECOVERY",
            ),
            "0.4": dict(
                status="BLOCKED",
                cpu_parameter_update_audit=update_audit,
                reason="SYNTHETIC_TASK_ADAPTER_AND_EXACT_PRODUCTION_MECHANISM_REQUIRE_REVIEW",
            ),
            "0.5": dict(
                status="BLOCKED",
                cpu_fixture_cost_reconciled=cpu_cost,
                gpu_runtime_measured=False,
                reason="CPU_WORK_NOT_GPU_EVIDENCE; SCIENTIFIC_RETRY_BUDGET_UNREGISTERED",
            ),
            "0.6": dict(
                status="BLOCKED",
                reason="SCIENTIFIC_SOURCE_CONFIG_DATA_SEEDS_METRICS_THRESHOLDS_AND_CONFIRMATION_PLAN_NOT_FROZEN",
            ),
        },
        synthetic_pilot="BLOCKED_PENDING_OWN_REGISTRATION_AND_REVIEW",
        real_model="BLOCKED",
        confirmation="SEALED",
        implementation_scientific_pass=False,
        engineering_results_are_not_pilot_results=True,
    )


def require_scientific(stage, c, release=None):
    release = read(BASE / "config/RELEASE.json") if release is None else release
    # A label or launcher switch cannot fill missing source/config/review/data evidence.
    required = ("approved_source_sha", "approved_config_sha256", "review_evidence_sha256")
    if not all(release.get(name) for name in required) or release.get(
        "approved_config_sha256"
    ) != sha(c):
        raise RuntimeError("JCT_SCIENTIFIC_RELEASE_BLOCKED_NO_FROZEN_EVIDENCE")
    if stage == "synthetic-pilot":
        s = c["synthetic_pilot"]
        if not release.get("synthetic_pilot_authorized") or not all(
            s.get(k)
            for k in (
                "registered",
                "seeds",
                "updates",
                "acceptance_thresholds",
                "generator_reviewed",
                "connection_adapter_reviewed",
            )
        ):
            raise RuntimeError("JCT_SYNTHETIC_PILOT_PREREQUISITES_BLOCKED")
    elif stage == "real-model":
        r = c["real_model"]
        if (
            not release.get("real_model_authorized")
            or not all(
                r.get(k)
                for k in (
                    "registered",
                    "checkpoint_refs",
                    "cross_session_verified",
                    "task_learnability_passed",
                    "curriculum_frozen",
                )
            )
            or not all(r["data_roles"].values())
        ):
            raise RuntimeError("JCT_REAL_MODEL_PREREQUISITES_BLOCKED")
    elif stage == "confirmation":
        raise RuntimeError("JCT_CONFIRMATION_SEALED_NO_POWERED_ANALYSIS_RELEASE")
    else:
        raise RuntimeError("JCT_UNKNOWN_SCIENTIFIC_STAGE")
    # v0.1 engineering intentionally has no authorized scientific executor to fall through to.
    raise RuntimeError("JCT_SCIENTIFIC_EXECUTOR_REQUIRES_REVIEWED_PROTOCOL_REVISION")
