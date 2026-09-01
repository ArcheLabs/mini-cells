#!/usr/bin/env python3
"""Run one Core Validation 009B-1 discovery or confirmation seed."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from minicells.real_representation_006_experiment import prepare_seed
from minicells.real_representation_006_io import (
    extract_frozen_sequences,
    load_foundation,
    load_frozen_cache,
    save_frozen_cache,
    select_real_sequences,
    write_data_manifest,
)
from minicells.real_representation_007_config import CoreValidation007Config
from minicells.real_representation_009b1_experiment import (
    analysis_sequences,
    extract_causal_sequences,
    run_confirmation,
    run_discovery,
)

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-009b1-carrier-causal-sufficiency"
PROTOCOL = VALIDATION / "protocol.json"
SCALE_LOCK = VALIDATION / "scale-lock.json"
CORE007_PROTOCOL = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery" / "protocol.json"
BRIDGE_ARTIFACTS = ROOT / "artifacts" / "experiments" / "core-validation-009a-right-collapse-bridge"
BRIDGE_DECISION = BRIDGE_ARTIFACTS / "decision.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-009b1-carrier-causal-sufficiency"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_parent(protocol: dict[str, Any]) -> None:
    if not BRIDGE_DECISION.is_file():
        raise RuntimeError("009B-1 requires the published 009A right-collapse bridge decision")
    decision = json.loads(BRIDGE_DECISION.read_text(encoding="utf-8"))
    parent = protocol["parent_evidence"]
    if decision.get("status") != parent["right_collapse_bridge_status"]:
        raise RuntimeError("unexpected 009A bridge status")
    if decision.get("protocol_sha256") != parent["bridge_protocol_sha256"]:
        raise RuntimeError("009A bridge protocol hash does not match 009B-1 pin")
    if decision.get("source_009a_status") != parent["core009a_status"]:
        raise RuntimeError("source 009A positive status is not preserved")
    if decision.get("source_009a_status_changed") is not False:
        raise RuntimeError("source 009A status changed unexpectedly")


def _load_scale_lock(protocol: dict[str, Any]) -> dict[str, Any]:
    if not SCALE_LOCK.is_file():
        raise RuntimeError("confirmation forbidden: scale-lock.json has not been committed")
    lock = json.loads(SCALE_LOCK.read_text(encoding="utf-8"))
    if lock.get("format") != "minicells.core-validation.carrier-causal-sufficiency-scale-lock.v1":
        raise RuntimeError("invalid 009B-1 scale lock format")
    if lock.get("protocol_sha256") != _sha256(PROTOCOL):
        raise RuntimeError("009B-1 scale lock protocol hash mismatch")
    rho = float(lock.get("locked_rho", -1))
    allowed = {float(x) for x in protocol["discovery"]["perturbation_ratio_grid"]}
    if rho not in allowed:
        raise RuntimeError(f"locked rho {rho} is not in the frozen discovery grid")
    if lock.get("confirmation_allowed") is not True:
        raise RuntimeError("confirmation forbidden by scale lock")
    return lock


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    _validate_parent(protocol)
    allowed = tuple(int(x) for x in protocol[args.phase]["seeds"])
    if args.seed not in allowed:
        raise ValueError(f"seed {args.seed} is not frozen for {args.phase}; expected {allowed}")
    lock = _load_scale_lock(protocol) if args.phase == "confirmation" else None

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    cfg = CoreValidation007Config.from_protocol(CORE007_PROTOCOL)
    started = time.time()
    phase_out = args.out / args.phase
    seed_dir = phase_out / "seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_foundation(cfg.base, device=device)
    records, manifest = select_real_sequences(cfg.base, tokenizer)
    got = str(manifest["manifest_sha256"])
    expected = str(protocol["data"]["expected_manifest_sha256"])
    if got != expected:
        raise RuntimeError(f"data manifest mismatch: expected {expected}, got {got}")
    write_data_manifest(manifest, args.out / "data-manifest.json")

    cache_path = args.out / "frozen-hidden.pt"
    frozen = None if args.no_cache else load_frozen_cache(manifest, cache_path)
    if frozen is None:
        frozen = extract_frozen_sequences(records, model, device=device)
        if not args.no_cache:
            save_frozen_cache(frozen, manifest, cache_path)

    lm_head_weight = model.embed_out.weight.detach().clone()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    u, _, _, projected = prepare_seed(frozen, cfg.base, seed=args.seed)
    extracted = extract_causal_sequences(projected, u, lm_head_weight, device=device)
    causal = analysis_sequences(extracted)
    if not causal or any(s.partition not in {"train", "eval"} for s in causal):
        raise RuntimeError("009B-1 train/eval filtering failed")

    if args.phase == "discovery":
        payload = run_discovery(
            causal, protocol, seed=args.seed, u=u,
            lm_head_weight=lm_head_weight, device=device
        )
    else:
        assert lock is not None
        payload = run_confirmation(
            causal, protocol, seed=args.seed, rho=float(lock["locked_rho"]), u=u,
            lm_head_weight=lm_head_weight, device=device
        )
        payload["scale_lock"] = lock

    payload.update(
        {
            "phase": args.phase,
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": _sha256(PROTOCOL),
            "data_manifest_sha256": got,
            "device": str(device),
            "elapsed_seconds": time.time() - started,
        }
    )
    path = seed_dir / f"seed-{args.seed}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "phase": args.phase,
        "seed": args.seed,
        "elapsed_seconds": payload["elapsed_seconds"],
        "output": str(path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
