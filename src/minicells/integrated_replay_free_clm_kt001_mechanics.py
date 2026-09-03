"""Thin bindings from KT001 to the already-validated Native CLM mechanisms.

No algorithm is duplicated here.  The purpose of this module is to make the
integration points explicit and auditable before the five-arm runner exists.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .integrated_replay_free_clm_kt001 import KT001ArmConfig
from .native_clm_m2r0 import (
    measure_realized_update_invariant,
    project_realized_updates_,
    snapshot_cell_weights,
)
from .native_clm_m3 import GrowthWindow, NativeCLMM3GrowthConfig
from .native_clm_m3l2 import (
    M3L2AddressConfig,
    OnlineAddressNativeCLM,
    maybe_spawn_online_address,
    observe_online_queries,
)
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


def observe_historical_address_queries(model: NativeCLM, info: dict[str, Any]) -> None:
    """Feed learner-visible queries into the canonical M3L-2 online address state."""

    observe_online_queries(require_canonical_online_address_model(model), info)


def force_shadow_expansion_(
    model: NativeCLM,
    optimizer,
    *,
    growth_config: NativeCLMM3GrowthConfig,
    global_step: int,
    probe_tokens,
) -> dict[str, Any]:
    """Force one phase-boundary Shadow birth through the canonical M3L-2 birth path.

    KT001 removes the *whether-to-grow* lifecycle decision from the experiment.  It
    does not replace M3L-2's birth/read mechanics.  We therefore construct a
    deterministic eligibility window, then call ``maybe_spawn_online_address`` to
    perform the actual gate derivation, parent->child spawn, optimizer param-group
    update, address-state transition, and birth-drift audit.

    Parent choice is learner-only and deterministic: among lineage leaves with both
    historical state and at least two current queries, choose the largest current
    query count, breaking ties by the lowest Cell id.  Evaluation metrics never
    participate in this choice.
    """

    address_model = require_canonical_online_address_model(model)
    growth_config.validate()
    candidates: list[tuple[int, int]] = []
    for cell_id, accumulator in sorted(address_model.current_moments.items()):
        if (
            accumulator.count >= 2
            and address_model.is_lineage_leaf(cell_id)
            and cell_id in address_model.historical_sketches
        ):
            candidates.append((int(accumulator.count), int(cell_id)))
    if not candidates:
        raise RuntimeError("KT001 forced Shadow expansion has no learner-visible address candidate")

    _, parent_id = max(candidates, key=lambda item: (item[0], -item[1]))
    window = GrowthWindow(address_model.config.d_model, address_model.cell_count)
    window.steps = 1
    window.loss_sum = 1.0
    window.route_hits[parent_id] = max(1, address_model.current_moments[parent_id].count)
    window.ratio_weighted_sum[parent_id] = 0.0

    forced_growth = replace(
        growth_config,
        growth_cooldown_steps=0,
        min_parent_route_hits_per_window=1,
        min_parent_certificate_rank=0,
        max_projected_to_raw_gradient_ratio=1.0,
        min_window_train_loss=0.0,
    )
    forced_growth.validate()
    spawned_before = address_model.cell_count - address_model.lineage_root_count
    event = maybe_spawn_online_address(
        address_model,
        optimizer,
        window,
        forced_growth,
        global_step=global_step,
        last_growth_step=None,
        spawned_count=spawned_before,
        probe_tokens=probe_tokens,
    )
    if event is None:
        raise RuntimeError("canonical M3L-2 birth path rejected a protocol-forced Shadow expansion")
    if int(event["parent_id"]) != parent_id:
        raise RuntimeError("KT001 forced Shadow parent selection drifted")
    event = dict(event)
    event.update(
        {
            "forced_by_protocol": True,
            "trigger_uses_evaluation_metrics": False,
            "parent_selection": "max_current_address_queries_then_lowest_cell_id",
        }
    )
    return event


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
