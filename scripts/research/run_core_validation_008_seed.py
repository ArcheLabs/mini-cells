#!/usr/bin/env python3
"""Run one frozen Core Validation 008 seed."""
from __future__ import annotations

import argparse
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
from minicells.real_representation_008_experiment import run_seed

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-008-certified-functional-atoms"
PROTOCOL_PATH = VALIDATION / "protocol.json"
CORE007_PROTOCOL = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery" / "protocol.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-008-certified-functional-atoms"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--signature-batch-size", type=int, default=8)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    allowed = {int(protocol["replication"]["smoke_seed"]), *map(int, protocol["replication"]["formal_seeds"])}
    if args.seed not in allowed:
        raise ValueError(f"seed {args.seed} is not frozen in Core 008 protocol")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    started = time.time()
    base_cfg = CoreValidation007Config.from_protocol(CORE007_PROTOCOL)
    print(f"[core008] seed={args.seed} rehydrating pinned model/data", flush=True)
    tokenizer, model = load_foundation(base_cfg.base, device=device)
    records, manifest = select_real_sequences(base_cfg.base, tokenizer)
    expected = str(protocol["data"]["expected_manifest_sha256"])
    actual = str(manifest["manifest_sha256"])
    if actual != expected:
        raise RuntimeError(f"data manifest mismatch: expected {expected}, got {actual}")

    args.out.mkdir(parents=True, exist_ok=True)
    write_data_manifest(manifest, args.out / "data-manifest.json")
    cache = args.out / "frozen-hidden.pt"
    frozen = None if args.no_cache else load_frozen_cache(manifest, cache)
    if frozen is None:
        frozen = extract_frozen_sequences(records, model, device=device)
        if not args.no_cache:
            save_frozen_cache(frozen, manifest, cache)
    lm_head_weight = model.embed_out.weight.detach().clone()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    u, _centroids, _base_assignment, projected = prepare_seed(frozen, base_cfg.base, seed=args.seed)
    signatures = _signature_batches(
        projected,
        u,
        lm_head_weight,
        device=device,
        batch_size=args.signature_batch_size,
    )
    result = run_seed(projected, signatures, protocol, seed=args.seed)
    result["data_manifest_sha256"] = actual
    result["elapsed_seconds"] = time.time() - started
    result["device"] = str(device)
    result["protocol_version"] = protocol["protocol_version"]

    seed_dir = args.out / "seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    path = seed_dir / f"seed-{args.seed}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    for name, row in result["variant_results"].items():
        print(
            f"  {name}: atoms={row['atom_count']} rank={row['total_rank_units']} "
            f"reuse={row['online_reuse_fraction']:.3f} unresolved={row['unresolved_write_fraction']:.3f} "
            f"eval-deploy-action={row['median_eval_deploy_local_action_residual']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
