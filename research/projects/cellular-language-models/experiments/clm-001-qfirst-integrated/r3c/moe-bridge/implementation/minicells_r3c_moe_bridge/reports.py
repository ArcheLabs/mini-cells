from .common import atomic
from .publication import public_summary


def write_reports(output, report):
    directory = output / "reports"
    summary = public_summary(report)
    atomic(directory / "metrics-summary.json", summary)
    atomic(
        directory / "manifest.json",
        {
            "schema": "minicells.r3c.manifest.v1",
            "provenance": report["provenance"],
            "scope": report["scope"],
            "scientific_decision": False,
        },
    )
    atomic(
        directory / "coverage-ledger.json",
        {
            "arms": [r["arm"] for r in report["arms"]],
            "synthetic_updates_per_arm": 3,
            "real_train_tokens": 0,
            "gpu_pilot": "NOT_MEASURED",
            "confirmation": "SEALED",
        },
    )
    atomic(
        directory / "parity-ledger.json",
        {"cpu_local_restore": report["resume"], "actual_kaggle_persistence_verified": False},
    )
    atomic(
        directory / "gate-gradient-summary.json",
        {
            "arms": [
                {
                    "arm": r["arm"],
                    "task": r["updates"][-1]["gate_task_grad_norm"],
                    "balance": r["updates"][-1]["gate_balance_grad_norm"],
                }
                for r in report["arms"]
            ]
        },
    )
    atomic(
        directory / "routing-summary.json",
        {
            "arms": [
                {"arm": r["arm"], "microbatches": r["updates"][-1]["routing_microbatches"]}
                for r in report["arms"]
            ]
        },
    )
    atomic(
        directory / "performance-summary.json",
        {
            "arms": [{"arm": r["arm"], "operator_cost": r["cost"]} for r in report["arms"]],
            "gpu_time": "NOT_MEASURED",
            "peak_gpu_bytes": "NOT_MEASURED",
            "formal_runtime_projection": "NOT_MEASURED",
        },
    )
    atomic(
        directory / "specialization-summary.json",
        {
            "status": "NOT_MEASURED",
            "reason": "no real training or independent Development evaluation authorized",
            "scientific_decision": False,
        },
    )
    (directory / "SCIENTIFIC-FINDINGS.md").write_text(
        "# Engineering only\n\nSynthetic CPU correctness is not evidence for H-F1–H-F6, specialization, dynamic STOP, or self-reinforcement. No GPU training or Confirmation was run.\n"
    )
    return summary
