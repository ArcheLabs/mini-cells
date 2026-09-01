"""Resumable/checkpoint infrastructure for Core Validation 007 confirmation.

This module contains only persistence and protocol-integrity helpers. It does
not change the frozen functional-boundary mechanism, data, model, or gates.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


CHECKPOINT_FORMAT = "minicells.core-validation.functional-boundary-seed-checkpoint.v1"
FAILURE_FORMAT = "minicells.core-validation.functional-boundary-seed-failure.v1"
AMENDMENT_FORMAT = "minicells.core-validation.functional-boundary-confirmation-amendment.v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def load_confirmation_amendment(
    path: Path,
    *,
    protocol_sha256: str,
    winner_lock: dict[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing frozen confirmation amendment: {path}")
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if amendment.get("format") != AMENDMENT_FORMAT:
        raise RuntimeError("invalid Core 007 confirmation amendment format")
    if amendment.get("base_discovery_protocol_sha256") != protocol_sha256:
        raise RuntimeError("confirmation amendment references a different discovery protocol")
    if amendment.get("winner") != winner_lock.get("winner"):
        raise RuntimeError("confirmation amendment winner differs from committed winner lock")
    seeds = [int(x) for x in amendment.get("confirmation_seeds", [])]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise RuntimeError("confirmation amendment must freeze exactly three unique seeds")
    retired = {int(x) for x in amendment.get("retired_confirmation_seeds", [])}
    if retired.intersection(seeds):
        raise RuntimeError("recovery confirmation seeds overlap retired seeds")
    invariants = amendment.get("scientific_invariants", {})
    forbidden_true = {
        "winner_changed",
        "boundary_mechanism_changed",
        "model_or_data_changed",
        "gate_thresholds_changed",
        "core006_baselines_changed",
    }
    if any(bool(invariants.get(key)) for key in forbidden_true):
        raise RuntimeError("confirmation amendment contains a scientific mechanism change")
    return amendment


def seed_checkpoint_path(phase_out: Path, seed: int) -> Path:
    return phase_out / "seeds" / f"seed-{seed}.json"


def seed_failure_path(phase_out: Path, seed: int) -> Path:
    return phase_out / "seeds" / f"seed-{seed}.failure.json"


def checkpoint_identity(
    *,
    seed: int,
    protocol_sha256: str,
    amendment_sha256: str,
    data_manifest_sha256: str,
    winner: str,
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "protocol_sha256": protocol_sha256,
        "confirmation_amendment_sha256": amendment_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "winner": winner,
    }


def validate_seed_checkpoint(
    payload: dict[str, Any],
    *,
    seed: int,
    protocol_sha256: str,
    amendment_sha256: str,
    data_manifest_sha256: str,
    winner: str,
) -> dict[str, Any]:
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(f"seed {seed} checkpoint has an unknown format")
    if payload.get("status") != "completed":
        raise RuntimeError(f"seed {seed} checkpoint is not completed")
    expected = checkpoint_identity(
        seed=seed,
        protocol_sha256=protocol_sha256,
        amendment_sha256=amendment_sha256,
        data_manifest_sha256=data_manifest_sha256,
        winner=winner,
    )
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"seed {seed} checkpoint identity mismatch for {key}: "
                f"{payload.get(key)!r} != {value!r}"
            )
    run = payload.get("run")
    if not isinstance(run, dict) or int(run.get("seed", -1)) != int(seed):
        raise RuntimeError(f"seed {seed} checkpoint is missing its run payload")
    return run


def load_completed_seed_runs(
    phase_out: Path,
    *,
    seeds: list[int],
    protocol_sha256: str,
    amendment_sha256: str,
    data_manifest_sha256: str,
    winner: str,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        path = seed_checkpoint_path(phase_out, seed)
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs.append(
            validate_seed_checkpoint(
                payload,
                seed=seed,
                protocol_sha256=protocol_sha256,
                amendment_sha256=amendment_sha256,
                data_manifest_sha256=data_manifest_sha256,
                winner=winner,
            )
        )
    return runs


def incomplete_confirmation_decision(
    *,
    expected_seeds: list[int],
    completed_runs: list[dict[str, Any]],
    winner: str,
    failed_seed: int | None = None,
) -> dict[str, Any]:
    completed = [int(run["seed"]) for run in completed_runs]
    remaining = [seed for seed in expected_seeds if seed not in completed]
    return {
        "status": "CONFIRMATION_INCOMPLETE",
        "scientific_decision": False,
        "pass": None,
        "winner": winner,
        "completed_seeds": completed,
        "remaining_seeds": remaining,
        "failed_seed": failed_seed,
        "passed_seeds_so_far": sum(bool(run.get("pass")) for run in completed_runs),
        "total_seeds": len(expected_seeds),
        "reason": (
            "No scientific Core 007 confirmation decision is permitted until all "
            "amended confirmation seed checkpoints are complete and identity-matched."
        ),
    }
