"""Thin bindings from KT001 to the already-validated Native CLM mechanisms.

No algorithm is duplicated here.  The purpose of this module is to make the
integration points explicit and auditable before the five-arm runner exists.
"""

from __future__ import annotations

from typing import Any

from .integrated_replay_free_clm_kt001 import KT001ArmConfig
from .native_clm_m2r0 import (
    measure_realized_update_invariant,
    project_realized_updates_,
    snapshot_cell_weights,
)
from .native_clm_m3l2 import M3L2AddressConfig, OnlineAddressNativeCLM
from .native_clm_m3r import LineageNativeCLM
from .native_clm_v0 import NativeCLM


CANONICAL_ADDRESS_CONFIG = M3L2AddressConfig()
CANONICAL_ADDRESS_CONFIG.validate()


def capture_pre_step_cell_weights(model: NativeCLM):
    """Capture the canonical R0b pre-``optimizer.step`` Cell transaction state."""

    return snapshot_cell_weights(model)


def finalize_realized_adamw_transaction_(
    model: NativeCLM,
    before,
    *,
    arm: KT001ArmConfig,
    step: int,
) -> dict[str, Any]:
    """Install and audit the realized Cell update for one optimizer transaction.

    The caller must invoke this immediately after ``optimizer.step()``.  Protected
    arms call the canonical R0b final-update projector; unprotected arms leave the
    realized AdamW delta untouched.  Both paths are audited using the same R0b
    invariant measurement so the causal difference remains observable.
    """

    retained_ratios: list[float] | None = None
    if arm.realized_update_write_safety:
        retained_ratios = project_realized_updates_(model, before)

    rows = measure_realized_update_invariant(
        model,
        before,
        arm=arm.name,
        step=step,
    )
    return {
        "arm": arm.name,
        "step": int(step),
        "realized_update_projection_applied": bool(arm.realized_update_write_safety),
        "retained_update_norm_ratios": retained_ratios,
        "invariant_rows": rows,
    }


def require_canonical_online_address_model(model: NativeCLM) -> OnlineAddressNativeCLM:
    """Validate that an address-enabled arm is using the exact M3L-2 substrate."""

    if not isinstance(model, OnlineAddressNativeCLM):
        raise TypeError("KT001 historical-address arms require OnlineAddressNativeCLM")
    model.address_config.validate()
    if model.address_config.rank != 32:
        raise RuntimeError("KT001 historical address state must retain M3L-2 rank 32")
    return model


def require_canonical_lineage_model(model: NativeCLM) -> LineageNativeCLM:
    """Validate that a lineage-enabled arm is using the exact M3R model family."""

    if not isinstance(model, LineageNativeCLM):
        raise TypeError("KT001 lineage/Shadow arms require LineageNativeCLM")
    return model


def structural_state_metadata(model: NativeCLM, *, arm: KT001ArmConfig) -> dict[str, Any]:
    """Return the mechanism state that KT001 checkpoints must make auditable."""

    payload: dict[str, Any] = {
        "cell_count": int(model.cell_count),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "historical_address_read": bool(arm.historical_address_read),
        "lineage_shadow_isolation": bool(arm.lineage_shadow_isolation),
    }

    if arm.lineage_shadow_isolation:
        lineage = require_canonical_lineage_model(model)
        payload.update(
            {
                "lineage_root_count": int(lineage.lineage_root_count),
                "lineage_direct_children": {
                    str(parent): int(child)
                    for parent, child in sorted(lineage._direct_children().items())
                },
            }
        )

    if arm.historical_address_read:
        address_model = require_canonical_online_address_model(model)
        payload["address_state"] = address_model.address_state_metrics()

    return payload
