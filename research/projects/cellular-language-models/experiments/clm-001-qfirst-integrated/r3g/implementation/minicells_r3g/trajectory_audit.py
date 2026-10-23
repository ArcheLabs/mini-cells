"""Read archived observations; never regenerate training routes."""

from .common import fingerprint


def review(c, arm, restored):
    rows = restored["routes"]
    if len(rows) != 10 or [r["step"] for r in rows] != list(range(101, 111)):
        raise RuntimeError("TRAINING_ROUTE_OBSERVATIONS_INCOMPLETE")
    if fingerprint(rows) != fingerprint(restored["parent_result"]["training_routes"]):
        raise RuntimeError("ORIGINAL_TRAINING_REPORT_MISMATCH")
    tokens = c["parent_runtime"]["training"]["tokens_per_step"]
    difference_steps = []
    reviewed = []
    for row in rows:
        if row["path_change"] > 0:
            difference_steps.append(row["step"])
        counts_old = row["old_cell_counts"]
        counts_new = row["new_cell_counts"]
        delta = [[b - a for a, b in zip(old, new)] for old, new in zip(counts_old, counts_new)]
        grads = [
            b - a for a, b in zip(row["old_cell_gradient_norms"], row["new_cell_gradient_norms"])
        ]
        if (
            sum(sum(stage) for stage in counts_old) != 8 * tokens
            or sum(sum(stage) for stage in counts_new) != 8 * tokens
        ):
            raise RuntimeError("TRAINING_CELL_CALL_TOKEN_COUNT_MISMATCH")
        reviewed.append(
            {
                **row,
                "cell_call_count_difference": delta,
                "cell_gradient_norm_difference": grads,
                "tokens_per_branch": tokens,
            }
        )
    expected = [108] if arm == "A-C" else []
    return {
        "rows": reviewed,
        "any_training_path_changed": bool(difference_steps),
        "path_difference_steps": difference_steps,
        "matches_protocol_described_steps": difference_steps == expected,
        "tokens_per_branch": 10 * tokens,
        "optimizer_updates_performed_by_audit": 0,
        "source": "VERIFIED_PRIVATE_PARENT_RESULT_AND_ADAPTING10_OBSERVATION_ARCHIVE",
        "public_report_comparison": "PENDING",
        "not_inferred_from_gradient_norm_equality": True,
    }
