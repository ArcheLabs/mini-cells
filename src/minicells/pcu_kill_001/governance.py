"""Formal-run identity, seed discipline, provenance, and protocol hashes."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


EXPERIMENT_ID = "PCU-KILL-001"
DEVELOPMENT_SEED = 26090501
FORMAL_SEEDS = (26090511, 26090512, 26090513)
SEED_REGISTRY_SCHEMA = "minicells.formal-seed-registry.v1"


class ProtocolMismatch(RuntimeError):
    """Raised whenever formal inputs are not the frozen inputs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _is_ancestor(root: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    try:
        return subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
    except OSError:
        return False


def _status_path(line: str) -> str:
    """Extract the path portion from one porcelain-v1 status line."""
    value = line[3:] if len(line) >= 3 else line
    if " -> " in value:
        value = value.split(" -> ", 1)[1]
    return value.strip().strip('"')


def _split_source_and_generated_status(status: str | None) -> tuple[str, str]:
    """Do not let generated research evidence make the source tree dirty.

    Provenance is intended to answer whether the code/protocol source changed.
    Files under ``artifacts/research/`` are run outputs, not source. We retain
    their porcelain rows separately so the distinction remains auditable.
    """
    source_rows: list[str] = []
    generated_rows: list[str] = []
    for line in (status or "").splitlines():
        if not line:
            continue
        path = _status_path(line)
        if path == "artifacts/research" or path.startswith("artifacts/research/"):
            generated_rows.append(line)
        else:
            source_rows.append(line)
    return "\n".join(source_rows), "\n".join(generated_rows)


def git_provenance(root: Path) -> dict[str, Any]:
    raw_status = _git(root, "status", "--porcelain=v1")
    source_status, generated_status = _split_source_and_generated_status(raw_status)
    return {
        "source_ref": _git(root, "symbolic-ref", "--short", "-q", "HEAD")
        or _git(root, "rev-parse", "--short", "HEAD"),
        "source_commit": _git(root, "rev-parse", "HEAD"),
        "source_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "source_dirty": bool(source_status),
        "status_porcelain": source_status,
        "generated_artifact_status_porcelain": generated_status,
    }


def runtime_provenance(device: str = "cpu") -> dict[str, Any]:
    cuda = None
    gpu = None
    try:
        import torch

        cuda = torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else None
        torch_version = torch.__version__
        deterministic = bool(torch.are_deterministic_algorithms_enabled())
    except Exception:
        torch_version = None
        deterministic = None
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch_version,
        "transformers": transformers_version,
        "cuda": cuda,
        "gpu": gpu,
        "device": device,
        "dtype": "float32",
        "deterministic_algorithms": deterministic,
    }


def set_deterministic_seeds(seed: int) -> dict[str, int]:
    """Set and return every RNG seed used by the engineering/formal runner."""
    import random

    random.seed(int(seed))
    values = {"python": int(seed), "numpy": int(seed), "torch_cpu": int(seed), "torch_cuda": int(seed)}
    try:
        import numpy as np
        import torch

        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass
    return values


def load_seed_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolMismatch(f"missing formal seed registry: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SEED_REGISTRY_SCHEMA:
        raise ProtocolMismatch("formal seed registry schema mismatch")
    return payload


def assert_seed_registry(path: Path, formal_seeds: tuple[int, ...] = FORMAL_SEEDS) -> None:
    payload = load_seed_registry(path)
    entries = payload.get("seeds", [])
    by_seed = {int(item["seed"]): item for item in entries}
    for seed in formal_seeds:
        entry = by_seed.get(int(seed))
        if entry is None or entry.get("experiment") != EXPERIMENT_ID:
            raise ProtocolMismatch(f"formal seed {seed} is not registered for {EXPERIMENT_ID}")
        if entry.get("state") != "RESERVED_UNTOUCHED":
            raise ProtocolMismatch(f"formal seed {seed} has already been touched")


def assert_engineering_seed(seed: int) -> None:
    if int(seed) in FORMAL_SEEDS:
        raise ValueError(f"engineering path refuses formal seed {seed}")
    if int(seed) != DEVELOPMENT_SEED:
        raise ValueError(f"engineering path requires registered seed {DEVELOPMENT_SEED}")


def mark_formal_seed(path: Path, seed: int, valid: bool) -> None:
    """Burn a formal seed after execution; there is intentionally no reset path."""
    payload = load_seed_registry(path)
    for entry in payload.get("seeds", []):
        if int(entry.get("seed", -1)) == int(seed):
            if entry.get("state") != "RUNNING":
                raise ProtocolMismatch(f"formal seed {seed} is already touched")
            entry["state"] = "TOUCHED_VALID" if valid else "TOUCHED_INVALID"
            entry["touched_at"] = datetime.now(timezone.utc).isoformat()
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return
    raise ProtocolMismatch(f"formal seed {seed} is not registered")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_protocol_hash(protocol_path: Path, hash_path: Path) -> str:
    digest = sha256_file(protocol_path)
    hash_path.write_text(digest + "\n", encoding="utf-8")
    return digest


def verify_protocol_hash(protocol_path: Path, hash_path: Path) -> str:
    if not protocol_path.is_file() or not hash_path.is_file():
        raise ProtocolMismatch("frozen protocol or SHA-256 sidecar is missing")
    expected = hash_path.read_text(encoding="utf-8").strip()
    actual = sha256_file(protocol_path)
    if expected != actual:
        raise ProtocolMismatch("PROTOCOL.sha256 does not match PROTOCOL.json")
    return actual


def assert_formal_preflight(
    root: Path,
    protocol_path: Path,
    hash_path: Path,
    seed_registry_path: Path,
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    """Validate formal prerequisites without loading data or generating samples."""
    provenance = git_provenance(root)
    if provenance["source_dirty"]:
        raise ProtocolMismatch("dirty source tree")
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_BEFORE_FORMAL":
        raise ProtocolMismatch("protocol is not frozen")
    if payload.get("formal_execution", {}).get("formal_execution_not_started") is not True:
        raise ProtocolMismatch("frozen protocol does not prove that formal execution has not started")
    model = payload.get("model", {})
    for key in ("model_repo", "model_revision", "config_sha256", "foundation_tensor_sha256", "weight_file_sha256", "tokenizer_sha256", "target_path"):
        if model.get(key) in (None, "", []):
            raise ProtocolMismatch(f"frozen protocol is missing immutable model field {key}")
    training = payload.get("training", {})
    for key in ("optimizer", "learning_rate", "max_optimizer_steps", "max_training_tokens", "selected_k", "lora_rank"):
        if training.get(key) in (None, "", 0):
            raise ProtocolMismatch(f"frozen protocol is missing selected training field {key}")
    allocation = payload.get("allocation", {})
    for key in ("method", "calibration_split", "calibration_sample_rule", "tie_break", "selected_k"):
        if allocation.get(key) in (None, "", []):
            raise ProtocolMismatch(f"frozen protocol is missing allocation field {key}")
    evaluation = payload.get("evaluation", {}).get("generation", {})
    if evaluation.get("do_sample") is not False or evaluation.get("max_new_tokens") in (None, 0):
        raise ProtocolMismatch("frozen protocol has no deterministic generation policy")
    try:
        from .experiment import run_formal_execution
        if not callable(run_formal_execution):
            raise ProtocolMismatch("formal worker is not callable")
    except ImportError as exc:
        raise ProtocolMismatch("formal worker is not importable") from exc
    protocol_sha = verify_protocol_hash(protocol_path, hash_path)
    recorded_commit = payload.get("source_commit")
    if not recorded_commit:
        raise ProtocolMismatch("protocol has no frozen source commit")
    if expected_source_commit is not None and recorded_commit != expected_source_commit:
        raise ProtocolMismatch("source commit does not match requested frozen commit")
    # The protocol is committed after freeze metadata is written. Accept a
    # descendant containing that protocol, but never accept a dirty source tree.
    if provenance.get("source_commit") and not _is_ancestor(root, str(recorded_commit), "HEAD"):
        raise ProtocolMismatch("current source is not descended from frozen source commit")
    assert_seed_registry(seed_registry_path)
    return {
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "source_commit": provenance.get("source_commit"),
        "source_tree": provenance.get("source_tree"),
        "formal_seeds": list(FORMAL_SEEDS),
        "formal_data_generated": False,
    }


def mark_formal_seed_running(path: Path, seed: int) -> None:
    """Transition a reserved seed immediately before actual execution starts."""
    payload = load_seed_registry(path)
    for entry in payload.get("seeds", []):
        if int(entry.get("seed", -1)) == int(seed):
            if entry.get("state") != "RESERVED_UNTOUCHED":
                raise ProtocolMismatch(f"formal seed {seed} is already touched")
            entry["state"] = "RUNNING"
            entry["started_at"] = datetime.now(timezone.utc).isoformat()
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return
    raise ProtocolMismatch(f"formal seed {seed} is not registered")
