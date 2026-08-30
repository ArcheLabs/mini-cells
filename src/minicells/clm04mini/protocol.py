"""Frozen CLM-0.4-mini protocol helpers.

The original M1-v1 protocol remains valid. M1-v2 is an explicit protocol
revision created after development seed 90401 exposed base-task misalignment.
Formal seeds remain shared and unopened.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .model import MiniCLMConfig


FORMAL_EXPERIMENT_ID = "clm-0.4-mini-language-validation"
V2_EXPERIMENT_ID = "clm-0.4-mini-m1-v2-language-validation"
M1_INFRA_SEED = 90400


class ProtocolError(ValueError):
    """Executable configuration drifted from a registered protocol."""


@dataclass(frozen=True)
class CandidateOptimizerConfig:
    optimizer: str
    batch_size: int
    learning_rate: float
    steps: int
    weight_decay: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "steps": self.steps,
            "weight_decay": self.weight_decay,
        }


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_json_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_generation(protocol: Mapping[str, Any]) -> int:
    experiment_id = protocol.get("experiment_id")
    if experiment_id == FORMAL_EXPERIMENT_ID:
        return 1
    if experiment_id == V2_EXPERIMENT_ID:
        return 2
    raise ProtocolError(f"unexpected experiment_id: {experiment_id!r}")


def load_protocol(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_protocol(payload)
    return payload


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    generation = protocol_generation(protocol)
    model = protocol["model"]
    expected = {
        "vocab_size": 8192,
        "sequence_length": 256,
        "layers": 4,
        "model_dim": 256,
        "attention_heads": 8,
        "tie_embedding_lm_head": True,
    }
    for key, value in expected.items():
        if model.get(key) != value:
            raise ProtocolError(
                f"frozen model field drift: {key}={model.get(key)!r}, expected {value!r}"
            )
    dense = model["shared_dense_layers"]
    cells = model["cell_layers"]
    if dense["layers"] != [1, 2] or int(dense["ffn_hidden"]) != 768:
        raise ProtocolError("frozen dense-layer configuration drift")
    if cells["layers"] != [3, 4]:
        raise ProtocolError("frozen Cell-layer configuration drift")
    if int(cells["base_cells_per_layer"]) != 32:
        raise ProtocolError("frozen base Cell count drift")
    if int(cells["base_cell_hidden"]) != 32 or int(cells["topk_base"]) != 2:
        raise ProtocolError("frozen Cell width/top-k drift")

    replication = protocol["replication"]
    expected_dev = 90402 if generation == 2 else 90401
    if int(replication["development_seed"]) != expected_dev:
        raise ProtocolError(
            f"development seed drift for protocol generation {generation}: "
            f"{replication['development_seed']!r}, expected {expected_dev}"
        )
    if [int(x) for x in replication["formal_model_seeds"]] != [90411, 90412, 90413]:
        raise ProtocolError("formal seed drift")
    if int(protocol["continual_curriculum"]["total_transactions"]) != 192:
        raise ProtocolError("M1 transaction-count drift")

    if generation == 2:
        alignment = protocol.get("base_alignment", {})
        if alignment.get("primary_gate") != "teacher-forced-answer-exact":
            raise ProtocolError("M1-v2 primary base gate drift")
        if alignment.get("evaluation_routes") != "held-out-route-balanced":
            raise ProtocolError("M1-v2 evaluation-route policy drift")
        baselines = protocol.get("dense_baselines", {})
        equal_parameter = baselines.get("equal_parameter", {})
        equal_compute = baselines.get("equal_active_compute", {})
        if int(equal_parameter.get("ffn_hidden", -1)) != 904:
            raise ProtocolError("M1-v2 equal-parameter dense baseline drift")
        if int(equal_compute.get("ffn_hidden", -1)) != 416:
            raise ProtocolError("M1-v2 equal-compute dense baseline drift")
        if baselines.get("controls_clm_calibration") is not False:
            raise ProtocolError("dense baselines must remain diagnostic-only")


def formal_model_config(
    protocol: Mapping[str, Any], *, routing_salt: str
) -> MiniCLMConfig:
    validate_protocol(protocol)
    model = protocol["model"]
    return MiniCLMConfig(
        vocab_size=int(model["vocab_size"]),
        max_seq_len=int(model["sequence_length"]),
        num_layers=int(model["layers"]),
        d_model=int(model["model_dim"]),
        n_heads=int(model["attention_heads"]),
        dense_ff_hidden=int(model["shared_dense_layers"]["ffn_hidden"]),
        base_cells=int(model["cell_layers"]["base_cells_per_layer"]),
        cell_hidden=int(model["cell_layers"]["base_cell_hidden"]),
        routing_salt=str(routing_salt),
    )


def smoke_model_config(
    protocol: Mapping[str, Any], *, routing_salt: str = "clm-0.4-mini-m1-smoke"
) -> MiniCLMConfig:
    validate_protocol(protocol)
    return MiniCLMConfig(
        vocab_size=512,
        max_seq_len=48,
        num_layers=4,
        d_model=48,
        n_heads=4,
        dense_ff_hidden=96,
        base_cells=8,
        cell_hidden=12,
        routing_salt=str(routing_salt),
    )


def candidate_grid(
    protocol: Mapping[str, Any], kind: str
) -> list[CandidateOptimizerConfig]:
    validate_protocol(protocol)
    if kind not in {"direct", "growth"}:
        raise ValueError("kind must be direct or growth")
    raw = protocol["calibration"][
        "direct_grid" if kind == "direct" else "growth_private_grid"
    ]
    configs = [
        CandidateOptimizerConfig(
            optimizer=str(raw["optimizer"]),
            batch_size=int(raw["batch_size"]),
            learning_rate=float(lr),
            steps=int(steps),
            weight_decay=float(raw["weight_decay"]),
        )
        for steps in raw["steps"]
        for lr in raw["learning_rates"]
    ]
    return sorted(configs, key=lambda item: (item.steps, item.learning_rate))


def assert_seed_allowed(
    protocol: Mapping[str, Any], *, mode: str, seed: int
) -> None:
    validate_protocol(protocol)
    seed = int(seed)
    development = int(protocol["replication"]["development_seed"])
    formal = [int(x) for x in protocol["replication"]["formal_model_seeds"]]
    if mode == "infrastructure-smoke":
        if seed != M1_INFRA_SEED:
            raise ProtocolError(
                f"M1 infrastructure smoke must use seed {M1_INFRA_SEED}"
            )
        if seed == development or seed in formal:
            raise ProtocolError(
                "infrastructure smoke cannot use development/formal seeds"
            )
        return
    if mode == "calibration":
        if seed != development:
            raise ProtocolError(
                f"calibration must use development seed {development}"
            )
        return
    if mode == "formal":
        if seed not in formal:
            raise ProtocolError(
                f"formal mode requires one of frozen seeds {formal}"
            )
        return
    raise ValueError(f"unknown mode: {mode}")


def m1_thresholds(protocol: Mapping[str, Any]) -> dict[str, float]:
    validate_protocol(protocol)
    metrics = protocol["metrics"]
    return {
        "minimum_new_gain": float(metrics["minimum_new_gain_for_local_pass"]),
        "maximum_local_old_regression": float(
            metrics["maximum_local_old_regression"]
        ),
        "maximum_global_old_regression": float(
            metrics["maximum_global_old_regression_for_oracle_pass"]
        ),
        "structural_logit_tolerance": float(
            metrics["structural_logit_tolerance"]
        ),
    }
