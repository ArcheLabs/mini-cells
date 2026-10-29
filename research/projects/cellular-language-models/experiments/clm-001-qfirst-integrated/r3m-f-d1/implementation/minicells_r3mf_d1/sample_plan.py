"""Verify each source-row draw against the frozen common plan before model use."""

from .data import build_update
from .identity import canonical, sha256_bytes


def verify_plan_identity(plan, config_sha, data_sha, anchor_sha):
    expected = (config_sha, data_sha, anchor_sha)
    actual = (
        plan.get("config_sha256"),
        plan.get("data_sha256"),
        plan.get("anchor_checkpoint_sha256"),
    )
    if actual != expected:
        raise ValueError("D1_SAMPLE_PLAN_IDENTITY_MISMATCH")
    return True


def verify_start_rng(rng, expected_state):
    if canonical(rng.bit_generator.state) != canonical(expected_state):
        raise ValueError("D1_SAMPLE_PLAN_START_RNG_MISMATCH")
    return True


def replay_planned_update(rng, records, step_plan):
    """Advance the frozen parent sampler and validate all draws before forward."""
    rows = build_update(rng, records)
    expected_rows = step_plan.get("rows", [])
    if len(rows) != len(expected_rows):
        raise ValueError("D1_SAMPLE_PLAN_ROW_COUNT_MISMATCH")
    for row, expected in zip(rows, expected_rows):
        if row.microbatch != expected["source_microbatch"]:
            raise ValueError("D1_SAMPLE_PLAN_MICROBATCH_MISMATCH")
        if list(row.sampled_indices) != expected["sampled_record_indices"]:
            raise ValueError("D1_SAMPLE_PLAN_RECORD_DRAWS_MISMATCH")
        signature = sha256_bytes(canonical(row.tokens).encode("utf-8"))
        if signature != expected["source_token_sha256"]:
            raise ValueError("D1_SAMPLE_PLAN_TOKEN_STREAM_MISMATCH")
    expected_state = step_plan.get("rng_state_after")
    if expected_state is None or canonical(rng.bit_generator.state) != canonical(expected_state):
        raise ValueError("D1_SAMPLE_PLAN_RNG_BOUNDARY_MISMATCH")
    return rows
