#!/usr/bin/env python3
"""Run one Core Validation 009A discovery or confirmation seed."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

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
from minicells.real_representation_007_experiment import _signature_batches
from minicells.real_representation_009a_experiment import make_rows, run_geometry

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-009a-factorized-functional-coordinates"
PROTOCOL = VALIDATION / "protocol.json"
WINNER_LOCK = VALIDATION / "winner-lock.json"
CORE007_PROTOCOL = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery" / "protocol.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-009a-factorized-functional-coordinates"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_winner_lock(protocol: dict) -> dict:
    if not WINNER_LOCK.is_file():
        raise RuntimeError("confirmation forbidden: winner-lock.json has not been committed")
    lock = json.loads(WINNER_LOCK.read_text(encoding="utf-8"))
    if lock.get("format") != "minicells.core-validation.factorized-functional-coordinates-winner-lock.v1":
        raise RuntimeError("invalid Core 009A winner lock format")
    if lock.get("protocol_sha256") != _sha256(PROTOCOL):
        raise RuntimeError("winner lock protocol hash does not match current frozen protocol")
    if lock.get("winner_meets_viability") is not True:
        raise RuntimeError("confirmation forbidden: discovery winner did not meet viability reference")
    split = lock.get("locked_split", {})
    allowed = {tuple(map(int, x)) for x in protocol["write_geometry"]["budget_matched_splits"]}
    pair = (int(split.get("left_dim", -1)), int(split.get("right_dim", -1)))
    if pair not in allowed:
        raise RuntimeError(f"winner lock split {pair} is not a frozen budget split")
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
    allowed = tuple(int(x) for x in protocol[args.phase]["seeds"])
    if args.seed not in allowed:
        raise ValueError(f"seed {args.seed} is not frozen for {args.phase}; expected {allowed}")
    lock = _load_winner_lock(protocol) if args.phase == "confirmation" else None

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
    signatures = _signature_batches(projected, u, lm_head_weight, device=device)
    payload = run_geometry(make_rows(projected, signatures), protocol, seed=args.seed)
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
    if lock is not None:
        payload["winner_lock"] = lock

    path = seed_dir / f"seed-{args.seed}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"phase": args.phase, "seed": args.seed, "elapsed_seconds": payload["elapsed_seconds"], "output": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
