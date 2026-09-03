"""Frozen causal-arm configuration for Integrated Replay-Free CLM Kill Test 001.

This module intentionally contains no training loop.  It makes the five-arm
causal matrix machine-readable before the integrated runner is implemented.
The actual mechanisms remain the canonical repository implementations named by
``PROTOCOL.md``; this file must not reimplement or approximate them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


EXPERIMENT_ID = "integrated-replay-free-clm-kill-test-001"
PROTOCOL_PATH = (
    "research/experiments/04-continual-learning-core/"
    "integrated-replay-free-clm-kill-test-001/PROTOCOL.md"
)
SEED_REGISTRY_PATH = (
    "research/experiments/04-continual-learning-core/"
    "integrated-replay-free-clm-kill-test-001/SEEDS.json"
)

R0B_MECHANISM = "native_clm_m2r0.adamw_final_update_projection"
M3L2_MECHANISM = "native_clm_m3l2.rank32_persistent_online_address_state"
M3R_MECHANISM = "native_clm_m3r.lineage_preserving_read_growth"
SHADOW_POLICY = "forced_phase_boundary_expansion"

ArmName = Literal[
    "unsafe",
    "write_transaction_only",
    "read_history_only",
    "full_no_replay",
    "matched_replay_oracle",
]


@dataclass(frozen=True)
class KT001ArmConfig:
    """One row of the frozen KT001 five-arm causal matrix."""

    name: ArmName
    realized_update_write_safety: bool
    historical_address_read: bool
    lineage_shadow_isolation: bool
    raw_replay: bool

    @property
    def mechanisms(self) -> tuple[str, ...]:
        enabled: list[str] = []
        if self.realized_update_write_safety:
            enabled.append(R0B_MECHANISM)
        if self.historical_address_read:
            enabled.append(M3L2_MECHANISM)
        if self.lineage_shadow_isolation:
            enabled.extend((M3R_MECHANISM, SHADOW_POLICY))
        if self.raw_replay:
            enabled.append("matched_raw_replay_oracle")
        return tuple(enabled)

    def metadata_switches(self) -> dict[str, object]:
        """Return explicit switches that every run artifact must record."""

        payload = asdict(self)
        payload["mechanisms"] = list(self.mechanisms)
        return payload


CANONICAL_ARMS: tuple[KT001ArmConfig, ...] = (
    KT001ArmConfig(
        name="unsafe",
        realized_update_write_safety=False,
        historical_address_read=False,
        lineage_shadow_isolation=False,
        raw_replay=False,
    ),
    KT001ArmConfig(
        name="write_transaction_only",
        realized_update_write_safety=True,
        historical_address_read=False,
        lineage_shadow_isolation=False,
        raw_replay=False,
    ),
    KT001ArmConfig(
        name="read_history_only",
        realized_update_write_safety=False,
        historical_address_read=True,
        lineage_shadow_isolation=True,
        raw_replay=False,
    ),
    KT001ArmConfig(
        name="full_no_replay",
        realized_update_write_safety=True,
        historical_address_read=True,
        lineage_shadow_isolation=True,
        raw_replay=False,
    ),
    KT001ArmConfig(
        name="matched_replay_oracle",
        realized_update_write_safety=True,
        historical_address_read=True,
        lineage_shadow_isolation=True,
        raw_replay=True,
    ),
)


def canonical_arm_map() -> dict[str, KT001ArmConfig]:
    return {arm.name: arm for arm in CANONICAL_ARMS}


def validate_causal_matrix() -> None:
    """Fail closed if implementation-time edits drift from the frozen protocol."""

    arms = canonical_arm_map()
    expected_names = {
        "unsafe",
        "write_transaction_only",
        "read_history_only",
        "full_no_replay",
        "matched_replay_oracle",
    }
    if set(arms) != expected_names or len(CANONICAL_ARMS) != 5:
        raise ValueError("KT001 arm set drifted from the frozen five-arm protocol")

    expected = {
        "unsafe": (False, False, False, False),
        "write_transaction_only": (True, False, False, False),
        "read_history_only": (False, True, True, False),
        "full_no_replay": (True, True, True, False),
        "matched_replay_oracle": (True, True, True, True),
    }
    for name, switches in expected.items():
        arm = arms[name]
        actual = (
            arm.realized_update_write_safety,
            arm.historical_address_read,
            arm.lineage_shadow_isolation,
            arm.raw_replay,
        )
        if actual != switches:
            raise ValueError(f"KT001 causal switches drifted for arm {name}")

    if M3L2_MECHANISM.split(".", 1)[0] != "native_clm_m3l2":
        raise ValueError("KT001 must reuse canonical M3L-2 address state")
    if M3R_MECHANISM.split(".", 1)[0] != "native_clm_m3r":
        raise ValueError("KT001 must reuse canonical M3R lineage routing")
    if R0B_MECHANISM != "native_clm_m2r0.adamw_final_update_projection":
        raise ValueError("KT001 must reuse canonical R0b final-update projection")


validate_causal_matrix()
