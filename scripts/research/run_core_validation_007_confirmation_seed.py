#!/usr/bin/env python3
"""Run one amended Core Validation 007 confirmation seed atomically."""
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
from minicells.real_representation_007_config import CoreValidation007Config
from minicells.real_representation_007_experiment import run_confirmation_seed

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery"
BASE_PROTOCOL = VALIDATION / "protocol.json"
CONFIRMATION_PROTOCOL = VALIDATION / "confirmation-protocol-v1.1.json"
WINNER_LOCK = VALIDATION / "winner-lock.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-007-functional-boundary-discovery"

SCIENTIFIC_CODE_PATHS = (
    "src/minicells/real_representation_006_config.py",
    "src/minicells/real_representation_006_core.py",
    "src/minicells/real_representation_006_experiment.py",
    "src/minicells/real_representation_006_io.py",
    "src/minicells/real_representation_007_config.py",
    "src/minicells/real_representation_007_core.py",
    "src/minicells/real_representation_007_experiment.py",
    "research/validations/core-007-functional-boundary-discovery/protocol.json",
    "research/validations/core-007-functional-boundary-discovery/confirmation-protocol-v1.1.json",
    "research/validations/core-007-functional-boundary-discovery/winner-lock.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scientific_code_sha256() -> str:
    h = hashlib.sha256()
    for rel in SCIENTIFIC_CODE_PATHS:
        path = ROOT / rel
        h.update(rel.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _tracked_tree_dirty() -> bool | None:
    try:
        unstaged = subprocess.run(
            ["git", "diff", "--quiet"], cwd=ROOT, check=False
        ).returncode
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False
        ).returncode
        return bool(unstaged or staged)
    except OSError:
        return None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


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


def _load_frozen_state() -> tuple[dict[str, Any], dict[str, Any], CoreValidation007Config]:
    base_sha = _sha256(BASE_PROTOCOL)
    amendment = json.loads(CONFIRMATION_PROTOCOL.read_text(encoding="utf-8"))
    if amendment.get("base_discovery_protocol_sha256") != base_sha:
        raise RuntimeError("Core 007 base discovery protocol hash no longer matches amendment")
    lock = json.loads(WINNER_LOCK.read_text(encoding="utf-8"))
    if lock.get("protocol_sha256") != base_sha:
        raise RuntimeError("Core 007 winner lock does not match frozen discovery protocol")
    if lock.get("winner") != amendment.get("winner"):
        raise RuntimeError("Core 007 amendment changed the frozen discovery winner")
    cfg = CoreValidation007Config.from_protocol(BASE_PROTOCOL)
    return amendment, lock, cfg


def _checkpoint_matches(
    path: Path,
    *,
    seed: int,
    amendment_sha: str,
    scientific_sha: str,
    amendment: dict[str, Any],
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(
        payload.get("complete") is True
        and payload.get("seed") == seed
        and payload.get("phase") == "confirmation"
        and payload.get("confirmation_protocol_sha256") == amendment_sha
        and payload.get("base_protocol_sha256") == amendment["base_discovery_protocol_sha256"]
        and payload.get("data_manifest_sha256") == amendment["expected_data_manifest_sha256"]
        and payload.get("winner") == amendment["winner"]
        and payload.get("scientific_code_sha256") == scientific_sha
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    amendment, lock, cfg = _load_frozen_state()
    seeds = [int(x) for x in amendment["confirmation_seeds"]]
    if args.seed not in seeds:
        raise RuntimeError(f"seed {args.seed} is not in amended confirmation seeds {seeds}")

    amendment_sha = _sha256(CONFIRMATION_PROTOCOL)
    scientific_sha = _scientific_code_sha256()
    seed_dir = args.out / "confirmation" / "seeds"
    failure_dir = args.out / "confirmation" / "failures"
    checkpoint = seed_dir / f"seed-{args.seed}.json"
    failure = failure_dir / f"seed-{args.seed}.json"

    if checkpoint.is_file() and not args.force:
        if _checkpoint_matches(
            checkpoint,
            seed=args.seed,
            amendment_sha=amendment_sha,
            scientific_sha=scientific_sha,
            amendment=amendment,
        ):
            print(f"[core-007] seed={args.seed} checkpoint already complete; skipping")
            return 0
        raise RuntimeError(
            f"existing checkpoint {checkpoint} does not match current frozen confirmation identity; "
            "refusing to overwrite it"
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Core 007 amended confirmation requires CUDA")

    started = time.time()
    try:
        print(
            f"[core-007] confirmation-v2 seed={args.seed} loading "
            f"{cfg.base.model_id}@{cfg.base.model_revision} on {device}",
            flush=True,
        )
        tokenizer, model = load_foundation(cfg.base, device=device)
        records, manifest = select_real_sequences(cfg.base, tokenizer)
        manifest_sha = str(manifest["manifest_sha256"])
        if manifest_sha != amendment["expected_data_manifest_sha256"]:
            raise RuntimeError(
                f"data manifest mismatch: expected {amendment['expected_data_manifest_sha256']}, "
                f"got {manifest_sha}"
            )
        args.out.mkdir(parents=True, exist_ok=True)
        write_data_manifest(manifest, args.out / "data-manifest.json")
        cache_path = args.out / "frozen-hidden.pt"
        frozen = None if args.no_cache else load_frozen_cache(manifest, cache_path)
        if frozen is None:
            frozen = extract_frozen_sequences(records, model, device=device)
            if not args.no_cache:
                save_frozen_cache(frozen, manifest, cache_path)
        else:
            print(f"[core-007] reused hidden cache {cache_path}", flush=True)

        expected_hidden = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))["foundation"][
            "expected_hidden_dim"
        ]
        if int(frozen[0].hidden.shape[-1]) != int(expected_hidden):
            raise RuntimeError("foundation hidden dimension changed")

        lm_head_weight = model.embed_out.weight.detach().clone()
        del model
        torch.cuda.empty_cache()

        run = run_confirmation_seed(
            frozen,
            cfg,
            seed=args.seed,
            winner=str(lock["winner"]),
            lm_head_weight=lm_head_weight,
            device=device,
        )
        c = run["candidate"]
        payload = {
            "format": "minicells.core-validation.functional-boundary-confirmation-seed.v1",
            "experiment_id": "core-validation-007",
            "phase": "confirmation",
            "complete": True,
            "seed": args.seed,
            "winner": str(lock["winner"]),
            "base_protocol_sha256": amendment["base_discovery_protocol_sha256"],
            "confirmation_protocol_sha256": amendment_sha,
            "data_manifest_sha256": manifest_sha,
            "scientific_code_sha256": scientific_sha,
            "run": run,
            "provenance": {
                "code_commit": _git(["rev-parse", "HEAD"]),
                "tracked_tree_dirty": _tracked_tree_dirty(),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(0),
                **_hf_provenance(cfg),
            },
            "elapsed_seconds": time.time() - started,
        }
        _atomic_json(checkpoint, payload)
        if failure.exists():
            failure.unlink()
        print(
            f"[core-007] seed={args.seed} pass={run['pass']} "
            f"split={c['median_split_conflict_reduction']:.3f} "
            f"spawn={c['spawned_fraction_of_addresses']:.3f} "
            f"reg/unsafe={c['regression_ratio_vs_unsafe']:.3f} "
            f"gain/replay={c['gain_ratio_vs_replay']:.3f} "
            f"route={c['routing_agreement']:.3f}",
            flush=True,
        )
        print(f"[core-007] wrote atomic checkpoint {checkpoint}")
        return 0
    except Exception as exc:
        _atomic_json(
            failure,
            {
                "format": "minicells.core-validation.functional-boundary-confirmation-failure.v1",
                "experiment_id": "core-validation-007",
                "phase": "confirmation",
                "complete": False,
                "seed": args.seed,
                "winner": amendment.get("winner"),
                "base_protocol_sha256": amendment.get("base_discovery_protocol_sha256"),
                "confirmation_protocol_sha256": amendment_sha,
                "scientific_code_sha256": scientific_sha,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.time() - started,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
