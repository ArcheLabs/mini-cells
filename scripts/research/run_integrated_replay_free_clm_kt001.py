"""Run Integrated Replay-Free CLM Kill Test 001 without leaking formal seeds.

Formal execution is fail-closed until IMPLEMENTATION_LOCK.json exists and declares
the implementation sealed. The script never auto-iterates formal seeds.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
from typing import Any

from minicells.integrated_replay_free_clm_kt001 import (
    PROTOCOL_PATH,
    SEED_REGISTRY_PATH,
    canonical_arm_map,
)
from minicells.integrated_replay_free_clm_kt001_replay import run_replay_oracle_stream
from minicells.integrated_replay_free_clm_kt001_runner import (
    KT001RunnerConfig,
    run_non_oracle_stream,
)
from minicells.native_clm_m2 import NativeCLMM2Config
from minicells.native_clm_m3 import NativeCLMM3GrowthConfig
from minicells.native_clm_m3l2 import M3L2AddressConfig


CANONICAL_M3L2_PROTOCOL = Path(
    "research/validations/native-clm-v0-m3l2-online-address-state/protocol.json"
)
EXPERIMENT_DIR = Path(
    "research/experiments/04-continual-learning-core/"
    "integrated-replay-free-clm-kill-test-001"
)
IMPLEMENTATION_LOCK = EXPERIMENT_DIR / "IMPLEMENTATION_LOCK.json"
DEFAULT_OUTPUT = Path("artifacts/experiments/integrated-replay-free-clm-kill-test-001")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_registry() -> dict[str, Any]:
    registry = _load_json(Path(SEED_REGISTRY_PATH))
    if registry.get("experiment_id") != "integrated-replay-free-clm-kill-test-001":
        raise RuntimeError("unexpected KT001 seed registry")
    development = {int(seed) for seed in registry.get("development", [])}
    formal = {int(seed) for seed in registry.get("formal", [])}
    if not development or not formal or development & formal:
        raise RuntimeError("invalid KT001 seed partition")
    return registry


def _require_seed_mode(seed: int, *, formal: bool, registry: dict[str, Any]) -> None:
    development = {int(value) for value in registry["development"]}
    formal_seeds = {int(value) for value in registry["formal"]}
    if seed in formal_seeds:
        if not formal:
            raise RuntimeError("formal KT001 seed requires explicit --formal")
        if not IMPLEMENTATION_LOCK.exists():
            raise RuntimeError("KT001 formal execution is blocked until IMPLEMENTATION_LOCK.json exists")
        lock = _load_json(IMPLEMENTATION_LOCK)
        if lock.get("status") != "SEALED_FOR_FORMAL_EXECUTION":
            raise RuntimeError("KT001 implementation lock is not sealed for formal execution")
        if lock.get("formal_seed_registry_sha256") != _sha256(Path(SEED_REGISTRY_PATH)):
            raise RuntimeError("KT001 implementation lock seed-registry hash drift")
        return
    if formal:
        raise RuntimeError("--formal may only be used with a registered formal seed")
    if seed not in development:
        raise RuntimeError("non-formal KT001 execution requires a registered development seed")


def _canonical_protocol() -> dict[str, Any]:
    protocol = _load_json(CANONICAL_M3L2_PROTOCOL)
    if protocol.get("format") != "minicells.native-clm-v0.m3l2-online-address-state.protocol.v1":
        raise RuntimeError("unexpected canonical M3L-2 protocol")
    if int(protocol["address_state"]["rank"]) != 32:
        raise RuntimeError("KT001 canonical source no longer registers rank-32 address state")
    return protocol


def _dataclass_subset(cls, source: dict[str, Any]):
    names = {field.name for field in fields(cls)}
    return cls(**{name: source[name] for name in names if name in source})


def _train_config(protocol: dict[str, Any]) -> NativeCLMM2Config:
    config = _dataclass_subset(NativeCLMM2Config, protocol["training"])
    config.validate()
    return config


def _growth_config(protocol: dict[str, Any]) -> NativeCLMM3GrowthConfig:
    config = _dataclass_subset(NativeCLMM3GrowthConfig, protocol["growth"])
    config.validate()
    return config


def _address_config(protocol: dict[str, Any]) -> M3L2AddressConfig:
    source = protocol["address_state"]
    config = M3L2AddressConfig(
        rank=int(source["rank"]),
        diagonal_regularization=float(source["diagonal_regularization"]),
        target_old_fpr=float(source["target_old_fpr"]),
        maximum_persistent_bytes_per_cell=int(source["maximum_persistent_bytes_per_cell"]),
        bootstrap_batches=int(protocol["bootstrap"]["batches"]),
        max_queries_per_cell_per_batch=int(source["max_queries_per_cell_per_batch"]),
    )
    config.validate()
    return config


def _validate_data(data_dir: Path, canonical: dict[str, Any]) -> tuple[dict[str, Any], str]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = _load_json(manifest_path)
    if manifest.get("format") not in {
        "minicells.native-clm-v0.m3l2-data-manifest.v1",
        "minicells.kt001-data-manifest.v1",
    }:
        raise RuntimeError("unexpected KT001 data manifest format")
    required = {
        "A_bootstrap",
        "A_eval",
        "B_train",
        "B_eval",
        "C_train",
        "C_eval",
        "D_train",
        "D_eval",
    }
    records = manifest.get("files", {})
    if set(records) != required:
        raise RuntimeError("KT001 data manifest file set mismatch")
    for name, record in records.items():
        path = data_dir / record["path"]
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"KT001 data identity mismatch for {name}")
    revisions = manifest.get("dataset_revisions", {})
    if set(revisions) != {"A", "B", "C_train", "C_eval", "D"}:
        raise RuntimeError("KT001 data manifest lacks exact dataset revisions")
    if revisions["A"]["resolved_revision"] != canonical["bootstrap"]["resolved_revision"]:
        raise RuntimeError("KT001 TinyStories revision drift")
    return manifest, _sha256(manifest_path)


def _paths(data_dir: Path, manifest: dict[str, Any]):
    records = manifest["files"]
    train = {
        "B": data_dir / records["B_train"]["path"],
        "C": data_dir / records["C_train"]["path"],
        "D": data_dir / records["D_train"]["path"],
    }
    evaluation = {
        "A": data_dir / records["A_eval"]["path"],
        "B": data_dir / records["B_eval"]["path"],
        "C": data_dir / records["C_eval"]["path"],
        "D": data_dir / records["D_eval"]["path"],
    }
    bootstrap = data_dir / records["A_bootstrap"]["path"]
    return train, evaluation, bootstrap


def _write_summary(
    summary: dict[str, Any],
    *,
    output_dir: Path,
    seed: int,
    arm: str,
    data_manifest_sha256: str,
) -> None:
    summary = dict(summary)
    summary["provenance"] = {
        "experiment_id": "integrated-replay-free-clm-kill-test-001",
        "git_commit_sha": _git_sha(),
        "protocol_sha256": _sha256(Path(PROTOCOL_PATH)),
        "seed_registry_sha256": _sha256(Path(SEED_REGISTRY_PATH)),
        "canonical_m3l2_protocol_sha256": _sha256(CANONICAL_M3L2_PROTOCOL),
        "data_manifest_sha256": data_manifest_sha256,
        "seed": int(seed),
        "arm": arm,
    }
    target = output_dir / f"seed-{seed}" / arm
    target.mkdir(parents=True, exist_ok=True)
    (target / "arm-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_failure(
    args: argparse.Namespace,
    *,
    seed: int,
    arm: str,
    exc: BaseException,
) -> Path:
    target = args.output_dir / f"seed-{seed}" / arm
    target.mkdir(parents=True, exist_ok=True)
    manifest = args.data_dir / "manifest.json"
    payload = {
        "format": "minicells.kt001-failure.v1",
        "experiment_id": "integrated-replay-free-clm-kill-test-001",
        "seed": int(seed),
        "arm": arm,
        "formal_requested": bool(args.formal),
        "git_commit_sha": _git_sha(),
        "protocol_sha256": _sha256(Path(PROTOCOL_PATH)) if Path(PROTOCOL_PATH).exists() else None,
        "seed_registry_sha256": (
            _sha256(Path(SEED_REGISTRY_PATH)) if Path(SEED_REGISTRY_PATH).exists() else None
        ),
        "data_manifest_sha256": _sha256(manifest) if manifest.exists() else None,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint) if args.checkpoint.exists() else None,
        "data_dir": str(args.data_dir),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
    }
    path = target / "failure.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _run_one(args: argparse.Namespace, *, arm_name: str, seed: int) -> dict[str, Any]:
    registry = _seed_registry()
    _require_seed_mode(seed, formal=args.formal, registry=registry)
    canonical = _canonical_protocol()
    manifest, manifest_sha = _validate_data(args.data_dir, canonical)
    train_paths, eval_paths, bootstrap_path = _paths(args.data_dir, manifest)
    arms = canonical_arm_map()
    arm = arms[arm_name]
    target = args.output_dir / f"seed-{seed}" / arm_name
    common = {
        "checkpoint_path": args.checkpoint,
        "expected_checkpoint_sha256": canonical["parent_checkpoint"]["sha256"],
        "bootstrap_path": bootstrap_path,
        "train_paths": train_paths,
        "eval_paths": eval_paths,
        "output_dir": target,
        "arm": arm,
        "seed": seed,
        "train_config": _train_config(canonical),
        "growth_config": _growth_config(canonical),
        "address_config": _address_config(canonical),
        "runner_config": KT001RunnerConfig(),
        "device": args.device,
    }
    if arm.raw_replay:
        summary = run_replay_oracle_stream(**common)
    else:
        summary = run_non_oracle_stream(**common)
    _write_summary(
        summary,
        output_dir=args.output_dir,
        seed=seed,
        arm=arm_name,
        data_manifest_sha256=manifest_sha,
    )
    print(
        json.dumps(
            {
                "experiment": "KT001",
                "seed": seed,
                "arm": arm_name,
                "final_checkpoint_sha256": summary["final_checkpoint_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--arm",
        choices=[*canonical_arm_map().keys(), "all"],
        default="all",
    )
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()

    registry = _seed_registry()
    if args.seed is None:
        if args.formal:
            raise RuntimeError("formal KT001 execution always requires an explicit --seed")
        args.seed = int(registry["development"][0])

    arm_names = list(canonical_arm_map()) if args.arm == "all" else [args.arm]
    for arm_name in arm_names:
        try:
            _run_one(args, arm_name=arm_name, seed=int(args.seed))
        except BaseException as exc:
            failure_path = _write_failure(
                args,
                seed=int(args.seed),
                arm=arm_name,
                exc=exc,
            )
            print(f"KT001 failure evidence written to {failure_path}", flush=True)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
