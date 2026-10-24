"""R3m-F-D1 read-only engineering and later bounded diagnostic primitives."""

from .data import build_update, split_independent, target_masks
from .loss import selected_target_count, selected_positions, normalized_loss_sum, selected_loss
from .identity import identity, validate_resume
from .sample_plan import verify_plan_identity, verify_start_rng, replay_planned_update

__all__ = [
    "build_update",
    "split_independent",
    "target_masks",
    "selected_target_count",
    "selected_positions",
    "normalized_loss_sum",
    "selected_loss",
    "identity",
    "validate_resume",
    "verify_plan_identity",
    "verify_start_rng",
    "replay_planned_update",
]
