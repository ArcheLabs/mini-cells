#!/usr/bin/env python3
"""Run one Core 008 postmortem functional-capacity diagnostic seed."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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
from minicells.real_representation_007_experiment import _signature_batches
from minicells.real_representation_006_experiment import prepare_seed
from minicells.real_representation_008_postmortem import run_capacity_diagnostics

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "research" / "validations" / "core-008-postmortem-functional-capacity" / "protocol.json"
CORE007_PROTOCOL = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery" / "protocol.json"
CORE008_ARTIFACTS = ROOT / "artifacts" / "experiments" / "core-validation-008-certified-functional-atoms"
DEFAULT_OUT = ROOT / "results" / "core-008-postmortem-functional-capacity"
SEEDS = (80821, 80822, 80823)


def _reference(seed: int) -> dict:
    path = CORE008_ARTIFACTS / "seeds" / f"seed-{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(f"published Core 008 seed artifact required: {path}")
    src = json.loads(path.read_text(encoding="utf-8"))
    adaptive = src["variant_results"]["adaptive_atoms"]
    return {
        "status": "CERTIFIED_ADAPTIVE_FUNCTIONAL_ATOMS_NOT_SUPPORTED",
        "adaptive_oracle_local_action_residual": adaptive["median_eval_oracle_local_action_residual"],
        "adaptive_deploy_local_action_residual": adaptive["median_eval_deploy_local_action_residual"],
        "adaptive_unresolved_write_fraction": adaptive["unresolved_write_fraction"],
        "adaptive_total_rank_units": adaptive["total_rank_units"],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True, choices=SEEDS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cfg = CoreValidation007Config.from_protocol(CORE007_PROTOCOL)
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    seed_dir = args.out / "seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_foundation(cfg.base, device=device)
    records, manifest = select_real_sequences(cfg.base, tokenizer)
    got = str(manifest["manifest_sha256"])
    expected = str(protocol["data_identity"]["expected_manifest_sha256"])
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
    payload = run_capacity_diagnostics(
        projected,
        signatures,
        protocol,
        seed=args.seed,
        core008_reference=_reference(args.seed),
    )
    payload.update({
        "protocol_format": protocol["format"],
        "data_manifest_sha256": got,
        "device": str(device),
        "elapsed_seconds": time.time() - started,
    })
    path = seed_dir / f"seed-{args.seed}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "seed": args.seed,
        "classification": payload["classification"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "output": str(path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
