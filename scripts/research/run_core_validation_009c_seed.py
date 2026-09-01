#!/usr/bin/env python3
"""Run one frozen Core Validation 009C discovery or confirmation seed."""
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
from minicells.real_representation_009b1_experiment import analysis_sequences, extract_causal_sequences
from minicells.real_representation_009c_experiment import run_geometry

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-009c-sparse-local-effect-geometry"
PROTOCOL = VALIDATION / "protocol.json"
LOCK = VALIDATION / "representation-lock.json"
CORE007_PROTOCOL = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery" / "protocol.json"
PARENT_DECISION = ROOT / "artifacts" / "experiments" / "core-validation-009b2-persistent-effect-geometry" / "discovery" / "decision.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-009c-sparse-local-effect-geometry"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _heartbeat(seed: int, message: str) -> None:
    print(f"[core009c seed={seed}] {message}", flush=True)


def _validate_parent(protocol: dict[str, Any]) -> None:
    if not PARENT_DECISION.is_file():
        raise RuntimeError("009C requires the published 009B-2 discovery decision")
    decision = json.loads(PARENT_DECISION.read_text(encoding="utf-8"))
    parent = protocol["parent_evidence"]
    if decision.get("status") != parent["core009b2_status"]:
        raise RuntimeError("009B-2 parent status does not match the frozen 009C protocol")
    if decision.get("protocol_sha256") != parent["core009b2_protocol_sha256"]:
        raise RuntimeError("009B-2 parent protocol hash mismatch")
    if decision.get("confirmation_allowed") is not False:
        raise RuntimeError("009C is only valid after the frozen 009B-2 compact-global failure")


def _load_lock() -> dict[str, Any]:
    if not LOCK.is_file():
        raise RuntimeError("confirmation requires committed representation-lock.json; publish discovery then refresh checkout")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("format") != "minicells.core-validation.sparse-local-effect-geometry-lock.v1":
        raise RuntimeError("invalid 009C representation lock format")
    if lock.get("protocol_sha256") != _sha256(PROTOCOL):
        raise RuntimeError("009C lock protocol hash mismatch")
    if lock.get("confirmation_allowed") is not True:
        raise RuntimeError("009C confirmation forbidden by representation lock")
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
    lock = _load_lock() if args.phase == "confirmation" else None

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

    _heartbeat(args.seed, f"loading foundation on {device}")
    tokenizer, model = load_foundation(cfg.base, device=device)
    _heartbeat(args.seed, "foundation loaded; selecting pinned data")
    records, manifest = select_real_sequences(cfg.base, tokenizer)
    got = str(manifest["manifest_sha256"])
    expected = str(protocol["data"]["expected_manifest_sha256"])
    if got != expected:
        raise RuntimeError(f"data manifest mismatch: expected {expected}, got {got}")
    write_data_manifest(manifest, args.out / "data-manifest.json")

    cache_path = args.out / "frozen-hidden.pt"
    frozen = None if args.no_cache else load_frozen_cache(manifest, cache_path)
    if frozen is None:
        _heartbeat(args.seed, "frozen hidden cache miss; extracting hidden states")
        frozen = extract_frozen_sequences(records, model, device=device)
        if not args.no_cache:
            save_frozen_cache(frozen, manifest, cache_path)
    else:
        _heartbeat(args.seed, "reused frozen hidden cache")

    lm_head_weight = model.embed_out.weight.detach().clone()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    _heartbeat(args.seed, "preparing seeded projection")
    u, _, _, projected = prepare_seed(frozen, cfg.base, seed=args.seed)
    _heartbeat(args.seed, "extracting frozen write signatures")
    extracted = extract_causal_sequences(projected, u, lm_head_weight, device=device)
    causal = analysis_sequences(extracted)
    if not causal or any(s.partition not in {"train", "eval"} for s in causal):
        raise RuntimeError("009C train/eval filtering failed")
    _heartbeat(args.seed, f"carrier effects ready: train={sum(s.partition == 'train' for s in causal)} eval={sum(s.partition == 'eval' for s in causal)}")

    locked_config = None if lock is None else dict(lock["configuration"])
    payload = run_geometry(causal, protocol, seed=args.seed, phase=args.phase, locked=locked_config)
    if lock is not None:
        payload["representation_lock"] = lock
    payload.update({
        "phase": args.phase,
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": _sha256(PROTOCOL),
        "data_manifest_sha256": got,
        "device": str(device),
        "elapsed_seconds": time.time() - started,
    })
    path = seed_dir / f"seed-{args.seed}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _heartbeat(args.seed, f"complete in {payload['elapsed_seconds']:.1f}s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
