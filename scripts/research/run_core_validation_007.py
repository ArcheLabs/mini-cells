#!/usr/bin/env python3
"""Run Core Validation 007 discovery or resumable confirmation."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from minicells.real_representation_006_io import (
    extract_frozen_sequences,
    load_foundation,
    load_frozen_cache,
    save_frozen_cache,
    select_real_sequences,
    write_data_manifest,
)
from minicells.real_representation_007_config import CoreValidation007Config, smoke_config
from minicells.real_representation_007_experiment import (
    run_confirmation_seed,
    run_discovery_seed,
    summarize_confirmation,
    summarize_discovery,
)
from minicells.real_representation_007_runtime import (
    CHECKPOINT_FORMAT,
    FAILURE_FORMAT,
    checkpoint_identity,
    incomplete_confirmation_decision,
    load_completed_seed_runs,
    load_confirmation_amendment,
    seed_checkpoint_path,
    seed_failure_path,
    sha256_file,
    validate_seed_checkpoint,
    write_json_atomic,
)

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery"
DEFAULT_PROTOCOL = VALIDATION / "protocol.json"
DEFAULT_CONFIRMATION_AMENDMENT = VALIDATION / "confirmation-protocol-v1.1.json"
DEFAULT_WINNER_LOCK = VALIDATION / "winner-lock.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-007-functional-boundary-discovery"


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *command], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _tracked_tree_dirty() -> bool | None:
    try:
        a = subprocess.run(["git", "diff", "--quiet"], cwd=ROOT, check=False).returncode
        b = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False).returncode
        return bool(a or b)
    except OSError:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _hf_provenance(cfg: CoreValidation007Config) -> dict[str, str | None]:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        model = api.model_info(cfg.base.model_id, revision=cfg.base.model_revision)
        dataset = api.dataset_info(cfg.base.dataset_id, revision=cfg.base.dataset_revision)
        return {
            "resolved_model_sha": getattr(model, "sha", None),
            "resolved_dataset_sha": getattr(dataset, "sha", None),
        }
    except Exception:
        return {"resolved_model_sha": None, "resolved_dataset_sha": None}


def _provenance(cfg: CoreValidation007Config, device: torch.device) -> dict[str, Any]:
    return {
        "code_commit": _git(["rev-parse", "HEAD"]),
        "code_tree": _git(["rev-parse", "HEAD^{tree}"]),
        "tracked_tree_dirty": _tracked_tree_dirty(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        **_hf_provenance(cfg),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    p.add_argument(
        "--confirmation-amendment", type=Path, default=DEFAULT_CONFIRMATION_AMENDMENT
    )
    p.add_argument("--winner-lock", type=Path, default=DEFAULT_WINNER_LOCK)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, action="append", dest="seeds")
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def _load_winner_lock(path: Path, protocol_sha: str, cfg: CoreValidation007Config) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            "Core 007 confirmation is locked: discovery winner-lock.json must be committed first."
        )
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("format") != "minicells.core-validation.functional-boundary-winner-lock.v1":
        raise RuntimeError("invalid Core 007 winner-lock format")
    if lock.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("winner lock was created for a different frozen protocol")
    if lock.get("winner") not in cfg.boundary_candidates:
        raise RuntimeError("winner lock names an unknown boundary candidate")
    if lock.get("confirmation_opened") is not False:
        raise RuntimeError("winner lock must preserve the pre-confirmation discovery record")
    return lock


def _write_confirmation_aggregate(
    *,
    protocol: dict[str, Any],
    protocol_sha: str,
    amendment: dict[str, Any],
    amendment_sha: str,
    winner_lock: dict[str, Any],
    winner: str,
    phase_out: Path,
    data_manifest_sha256: str,
    cfg: CoreValidation007Config,
    device: torch.device,
    started: float,
    failed_seed: int | None = None,
) -> dict[str, Any]:
    expected_seeds = [int(x) for x in amendment["confirmation_seeds"]]
    runs = load_completed_seed_runs(
        phase_out,
        seeds=expected_seeds,
        protocol_sha256=protocol_sha,
        amendment_sha256=amendment_sha,
        data_manifest_sha256=data_manifest_sha256,
        winner=winner,
    )
    order = {seed: i for i, seed in enumerate(expected_seeds)}
    runs.sort(key=lambda run: order[int(run["seed"])])
    complete = len(runs) == len(expected_seeds)
    if complete:
        decision = summarize_confirmation(
            runs,
            winner=winner,
            positive_status=str(amendment["positive_status"]),
            negative_status=str(amendment["negative_status"]),
        )
    else:
        decision = incomplete_confirmation_decision(
            expected_seeds=expected_seeds,
            completed_runs=runs,
            winner=winner,
            failed_seed=failed_seed,
        )
    payload = {
        "format": protocol["format"],
        "experiment_id": protocol["experiment_id"],
        "protocol_version": amendment["protocol_version"],
        "base_protocol_version": protocol["protocol_version"],
        "phase": "confirmation",
        "protocol_sha256": protocol_sha,
        "confirmation_amendment_sha256": amendment_sha,
        "parent_experiment": protocol["parent_experiment"],
        "data_manifest_sha256": data_manifest_sha256,
        "winner_lock": winner_lock,
        "confirmation_amendment": amendment,
        "expected_seeds": expected_seeds,
        "completed_seeds": [int(run["seed"]) for run in runs],
        "provenance": _provenance(cfg, device),
        "runs": runs,
        "decision": decision,
        "elapsed_seconds": time.time() - started,
    }
    raw = phase_out / "raw.json"
    write_json_atomic(raw, payload)
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {raw}")
    return payload


def _existing_confirmation_checkpoint(
    *,
    phase_out: Path,
    seed: int,
    protocol_sha: str,
    amendment_sha: str,
    expected_manifest: str,
    winner: str,
) -> dict[str, Any] | None:
    path = seed_checkpoint_path(phase_out, seed)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_seed_checkpoint(
        payload,
        seed=seed,
        protocol_sha256=protocol_sha,
        amendment_sha256=amendment_sha,
        data_manifest_sha256=expected_manifest,
        winner=winner,
    )


def main() -> int:
    args = parse_args()
    protocol: dict[str, Any] = json.loads(args.protocol.read_text(encoding="utf-8"))
    formal_cfg = CoreValidation007Config.from_protocol(args.protocol)
    cfg = smoke_config(formal_cfg) if args.smoke else formal_cfg
    protocol_sha = _sha256(args.protocol)
    device = _device(args.device)
    if not args.smoke and bool(protocol["hardware"]["gpu_required_for_real_run"]):
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("formal Core Validation 007 requires CUDA")

    winner_lock: dict[str, Any] | None = None
    amendment: dict[str, Any] | None = None
    amendment_sha: str | None = None
    winner: str | None = None

    if args.smoke:
        seeds = args.seeds or [cfg.smoke_seed]
    elif args.phase == "discovery":
        frozen = list(formal_cfg.discovery_seeds)
        seeds = args.seeds or frozen
        if seeds != frozen:
            raise RuntimeError(f"discovery must run exactly frozen seeds {frozen}")
    else:
        winner_lock = _load_winner_lock(args.winner_lock, protocol_sha, formal_cfg)
        winner = str(winner_lock["winner"])
        amendment = load_confirmation_amendment(
            args.confirmation_amendment,
            protocol_sha256=protocol_sha,
            winner_lock=winner_lock,
        )
        amendment_sha = sha256_file(args.confirmation_amendment)
        allowed = [int(x) for x in amendment["confirmation_seeds"]]
        if not args.seeds or len(args.seeds) != 1:
            raise RuntimeError(
                "resumable formal confirmation requires exactly one --seed per Python/CUDA process; "
                f"allowed amended seeds are {allowed}. Use run_core_validation_007_kaggle.py to orchestrate them."
            )
        if int(args.seeds[0]) not in allowed:
            raise RuntimeError(f"confirmation seed must be one of amended frozen seeds {allowed}")
        seeds = [int(args.seeds[0])]

    phase_out = args.out / ("smoke" if args.smoke else args.phase)
    phase_out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / ("frozen-hidden-smoke.pt" if args.smoke else "frozen-hidden.pt")
    manifest_path = args.out / "data-manifest.json"
    started = time.time()

    # Confirmation resume can skip a completed seed without reloading Pythia or touching CUDA.
    if args.phase == "confirmation" and not args.smoke:
        assert amendment is not None and amendment_sha is not None and winner is not None
        expected_manifest = str(amendment["expected_data_manifest_sha256"])
        existing = _existing_confirmation_checkpoint(
            phase_out=phase_out,
            seed=seeds[0],
            protocol_sha=protocol_sha,
            amendment_sha=amendment_sha,
            expected_manifest=expected_manifest,
            winner=winner,
        )
        if existing is not None:
            print(f"[core-007] seed={seeds[0]} checkpoint already complete; skipping GPU work")
            _write_confirmation_aggregate(
                protocol=protocol,
                protocol_sha=protocol_sha,
                amendment=amendment,
                amendment_sha=amendment_sha,
                winner_lock=winner_lock,
                winner=winner,
                phase_out=phase_out,
                data_manifest_sha256=expected_manifest,
                cfg=cfg,
                device=device,
                started=started,
            )
            return 0

    print(
        f"[core-007] phase={args.phase} loading {cfg.base.model_id}@{cfg.base.model_revision} on {device}",
        flush=True,
    )
    tokenizer, model = load_foundation(cfg.base, device=device)
    records, manifest = select_real_sequences(cfg.base, tokenizer)
    write_data_manifest(manifest, manifest_path)
    print(
        f"[core-007] selected {len(records)} sequences manifest={manifest['manifest_sha256'][:16]}",
        flush=True,
    )

    if args.phase == "confirmation" and not args.smoke:
        assert amendment is not None
        expected_manifest = str(amendment["expected_data_manifest_sha256"])
        if manifest["manifest_sha256"] != expected_manifest:
            raise RuntimeError(
                "confirmation data manifest differs from frozen amendment: "
                f"{manifest['manifest_sha256']} != {expected_manifest}"
            )

    frozen_sequences = None if args.no_cache else load_frozen_cache(manifest, cache_path)
    if frozen_sequences is None:
        frozen_sequences = extract_frozen_sequences(records, model, device=device)
        if not args.no_cache:
            save_frozen_cache(frozen_sequences, manifest, cache_path)
    else:
        print(f"[core-007] reused hidden cache {cache_path}", flush=True)

    hidden_dim = int(frozen_sequences[0].hidden.shape[-1])
    expected_hidden = int(protocol["foundation"]["expected_hidden_dim"])
    if not args.smoke and hidden_dim != expected_hidden:
        raise RuntimeError(f"foundation hidden dim changed: expected {expected_hidden}, got {hidden_dim}")
    lm_head_weight = model.embed_out.weight.detach().clone()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    runs: list[dict[str, Any]] = []
    if args.phase == "confirmation" and not args.smoke:
        assert amendment is not None and amendment_sha is not None
        assert winner_lock is not None and winner is not None
        seed = seeds[0]
        try:
            print(f"[core-007] phase=confirmation seed={seed}", flush=True)
            run = run_confirmation_seed(
                frozen_sequences,
                cfg,
                seed=seed,
                winner=winner,
                lm_head_weight=lm_head_weight,
                device=device,
            )
            c = run["candidate"]
            print(
                f"[core-007] seed={seed} pass={run['pass']} winner={winner} "
                f"split={c['median_split_conflict_reduction']:.3f} "
                f"spawn={c['spawned_fraction_of_addresses']:.3f} "
                f"reg/unsafe={c['regression_ratio_vs_unsafe']:.3f} "
                f"gain/replay={c['gain_ratio_vs_replay']:.3f} "
                f"route={c['routing_agreement']:.3f}",
                flush=True,
            )
            checkpoint = {
                "format": CHECKPOINT_FORMAT,
                "status": "completed",
                **checkpoint_identity(
                    seed=seed,
                    protocol_sha256=protocol_sha,
                    amendment_sha256=amendment_sha,
                    data_manifest_sha256=manifest["manifest_sha256"],
                    winner=winner,
                ),
                "provenance": _provenance(cfg, device),
                "run": run,
            }
            write_json_atomic(seed_checkpoint_path(phase_out, seed), checkpoint)
            seed_failure_path(phase_out, seed).unlink(missing_ok=True)
        except Exception as exc:
            failure = {
                "format": FAILURE_FORMAT,
                "status": "failed",
                **checkpoint_identity(
                    seed=seed,
                    protocol_sha256=protocol_sha,
                    amendment_sha256=amendment_sha,
                    data_manifest_sha256=manifest["manifest_sha256"],
                    winner=winner,
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "provenance": _provenance(cfg, device),
            }
            write_json_atomic(seed_failure_path(phase_out, seed), failure)
            _write_confirmation_aggregate(
                protocol=protocol,
                protocol_sha=protocol_sha,
                amendment=amendment,
                amendment_sha=amendment_sha,
                winner_lock=winner_lock,
                winner=winner,
                phase_out=phase_out,
                data_manifest_sha256=manifest["manifest_sha256"],
                cfg=cfg,
                device=device,
                started=started,
                failed_seed=seed,
            )
            print(failure["traceback"], file=sys.stderr)
            return 2

        _write_confirmation_aggregate(
            protocol=protocol,
            protocol_sha=protocol_sha,
            amendment=amendment,
            amendment_sha=amendment_sha,
            winner_lock=winner_lock,
            winner=winner,
            phase_out=phase_out,
            data_manifest_sha256=manifest["manifest_sha256"],
            cfg=cfg,
            device=device,
            started=started,
        )
        return 0

    for seed in seeds:
        print(f"[core-007] phase={args.phase} seed={seed}", flush=True)
        run = run_discovery_seed(
            frozen_sequences,
            cfg,
            seed=seed,
            lm_head_weight=lm_head_weight,
            device=device,
        )
        print(
            f"[core-007] seed={seed} modes={run['mode_count']} "
            f"route={run['routing_agreement']:.3f} top2={run['soft_top2_coverage']:.3f}",
            flush=True,
        )
        runs.append(run)

    if args.smoke:
        decision = {
            "status": "SMOKE_ONLY",
            "scientific_decision": False,
            "pass": None,
            "reason": "Reduced real-data smoke cannot select or confirm a Core 007 mechanism.",
        }
    else:
        decision = summarize_discovery(runs, formal_cfg)

    payload = {
        "format": protocol["format"],
        "experiment_id": protocol["experiment_id"],
        "protocol_version": protocol["protocol_version"],
        "phase": "smoke" if args.smoke else args.phase,
        "protocol_sha256": protocol_sha,
        "parent_experiment": protocol["parent_experiment"],
        "data_manifest_sha256": manifest["manifest_sha256"],
        "winner_lock": None,
        "provenance": _provenance(cfg, device),
        "runs": runs,
        "decision": decision,
        "elapsed_seconds": time.time() - started,
    }
    raw = phase_out / "raw.json"
    write_json_atomic(raw, payload)
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
