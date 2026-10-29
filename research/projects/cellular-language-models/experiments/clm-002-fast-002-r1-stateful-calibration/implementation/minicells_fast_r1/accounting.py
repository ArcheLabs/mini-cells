"""Replay's expected paused attempt is distinct from an uncertain optimizer mutation."""


def context(replay_engine, candidate_journal, scan_journal, pause_proof):
    a, b, c = (
        journal.summary() for journal in (replay_engine.journal, candidate_journal, scan_journal)
    )
    if any(s["uncertain_optimizer_calls"] for s in (a, b, c)):
        status = "UNCERTAIN_OPTIMIZER_WORK_BLOCKED"
    elif a["pending_attempts"] != 1 or b["pending_attempts"] or c["pending_attempts"]:
        status = "UNEXPECTED_TRANSACTION_STATE_BLOCKED"
    else:
        status = "REPLAY_PAUSE_AND_CANDIDATE_EVENTS_RECONCILED"
    return {
        "status": status,
        "main_replay_updates": pause_proof["executed_main_updates"],
        "main_target_tokens": pause_proof["effective_replay_target_tokens"],
        "cell_optimizer_calls": a["optimizer_calls_by_group"].get("cell", 0),
        "candidate_optimizer_calls": b["optimizer_calls_by_group"].get("connection", 0),
        "formal_connection_commits": 0,
        "counterfactual_target_evaluations": b["counterfactual_target_evaluations"],
        "uncertain_optimizer_calls": sum(s["uncertain_optimizer_calls"] for s in (a, b, c)),
        "expected_pending_preproposal_attempts": a["pending_attempts"],
        "operator_work_by_stage": {
            "replay": a["executed_operator_work"],
            "candidate": b["executed_operator_work"],
            "admission": c["executed_operator_work"],
        },
        "observed_wall_seconds_by_stage": {
            name: sum(summary["cpu_seconds_by_phase"].values())
            for name, summary in (("replay", a), ("candidate", b), ("admission", c))
        },
        "checkpoint_write_seconds": pause_proof.get("checkpoint_write_seconds"),
        "checkpoint_read_seconds": pause_proof.get("checkpoint_read_seconds"),
        "gpu_kernel_seconds": None,
        "gpu_time_scope": "KERNEL_TIME_NOT_MEASURED_PHASE_WALL_RECORDED",
    }
