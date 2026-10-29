"""Adapt generic event counts to actual language/GPU provenance, without rewriting the journal."""


def summary(engine, *, device):
    value = dict(engine.journal.summary())
    proposals = [r["proposal"] for r in engine.curve if r.get("proposal")]
    committed = [p for p in proposals if p["chosen_scale"] is not None]
    value.update(
        input_unit="LANGUAGE_TARGET_TOKENS",
        training_tokens=engine.tokens,
        main_target_tokens=engine.tokens,
        gpu_used=str(device).startswith("cuda"),
        gpu_seconds=None,
        gpu_time_scope="KERNEL_TIME_NOT_MEASURED; SYNCHRONIZED_TRAINING_WALL_SEPARATE",
        synchronized_training_wall_seconds=engine.training_seconds,
        connection_proposals=len(proposals),
        admitted_proposals=len(committed),
        committed_connection_updates=len(committed),
        candidate_optimizer_calls=sum(p["policy_optimizer_updates"] for p in proposals),
        rejected_candidate_optimizer_calls=sum(
            p["policy_optimizer_updates"] for p in proposals if p["chosen_scale"] is None
        ),
        accounting_status="UNCERTAIN_INFLIGHT_WORK"
        if value["uncertain_optimizer_calls"] or value["pending_attempts"]
        else "RECONCILED_LANGUAGE_EVENT_LEDGER",
        claim_scope="HASH_CHAINED_ACTUAL_OPTIMIZER_EVENTS_AND_LANGUAGE_TARGETS",
    )
    value["wall_seconds_by_phase"] = value.pop("cpu_seconds_by_phase")
    for key in ("analytic_cell_forward_mac", "analytic_router_forward_mac"):
        value.pop(key, None)  # Fixture formulas are not native-model MAC estimates.
    return value
