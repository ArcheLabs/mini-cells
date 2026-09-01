"""Compatibility helpers for the Core Validation 007 discovery runner.

Formal confirmation is intentionally not implemented through the generic
runner anymore. The only canonical confirmation entrypoint is
`scripts/research/orchestrate_core_validation_007_confirmation.py`, which uses
its dedicated per-seed checkpoint format.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

CHECKPOINT_FORMAT = "legacy-generic-confirmation-disabled"
FAILURE_FORMAT = "legacy-generic-confirmation-disabled"


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


def load_confirmation_amendment(*_args, **_kwargs):
    raise RuntimeError(
        "generic Core 007 formal confirmation is retired; run "
        "scripts/research/orchestrate_core_validation_007_confirmation.py instead"
    )


def sha256_file(*_args, **_kwargs):
    raise RuntimeError("legacy generic confirmation helper is disabled")


def checkpoint_identity(*_args, **_kwargs):
    raise RuntimeError("legacy generic confirmation helper is disabled")


def seed_checkpoint_path(*_args, **_kwargs):
    raise RuntimeError("legacy generic confirmation helper is disabled")


def seed_failure_path(*_args, **_kwargs):
    raise RuntimeError("legacy generic confirmation helper is disabled")


def validate_seed_checkpoint(*_args, **_kwargs):
    raise RuntimeError("legacy generic confirmation helper is disabled")


def load_completed_seed_runs(*_args, **_kwargs):
    raise RuntimeError("legacy generic confirmation helper is disabled")


def incomplete_confirmation_decision(*_args, **_kwargs):
    raise RuntimeError("legacy generic confirmation helper is disabled")
