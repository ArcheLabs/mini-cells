#!/usr/bin/env python3
"""Run one Core Validation 009A right-collapse diagnostic seed."""
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
from minicells.real_representation_009a_bridge import extract_bridge_sequences, run_bridge

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-009a-right-collapse-bridge"
PROTOCOL = VALIDATION / "protocol.json"
CORE007_PROTOCOL = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery" / "protocol.json"
SOURCE_ARTIFACTS = ROOT / "artifacts" / "experiments" / "core-validation-009a-factorized-functional-coordinates" / "confirmation"
SOURCE_DECISION = SOURCE_ARTIFACTS / "decision.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-009a-right-collapse-bridge"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_009a_seed(seed: int, protocol: dict[str, Any]) -> tuple[dict[str, Any], float]:
    source = protocol["source_009a"]
    if not SOURCE_DECISION.is_file():
        raise RuntimeError("source Core 009A confirmation decision artifact is missing")
    decision = json.loads(SOURCE_DECISION.read_text(encoding="utf-8"))
    if decision.get("status") != source["status"] or decision.get("scientific_decision") is not True:
        raise RuntimeError("source Core 009A is not the frozen positive confirmation expected by this bridge")
    if decision.get("protocol_sha256") != source["protocol_sha256"]:
        raise RuntimeError("source Core 009A protocol hash does not match bridge pin")
    path = SOURCE_ARTIFACTS / "seeds" / f"seed-{seed}.json"
    if not path.is_file():
        raise RuntimeError(f"source Core 009A confirmation seed artifact missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = source["locked_split"]
    row = next(
        r
        for r in payload["budget_splits"]
        if r["partition"] == "eval"
        and int(r["left_dim"]) == int(split["left_dim"])
        and int(r["right_dim"]) == int(split["right_dim"])
    )
    return payload, float(row["median_local_action_residual"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    allowed = tuple(int(x) for x in protocol["replication"]["diagnostic_seeds"])
    if args.seed not in allowed:
        raise ValueError(f"seed {args.seed} is not frozen for this bridge; expected {allowed}")
    source_seed, source_expected = _source_009a_seed(args.seed, protocol)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    cfg = CoreValidation007Config.from_protocol(CORE007_PROTOCOL)
    started = time.time()
    seed_dir = args.out / "seeds"
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
    bridge_sequences = extract_bridge_sequences(
        projected, u, lm_head_weight, device=device
    )
    payload = run_bridge(bridge_sequences, protocol, seed=args.seed)

    source_observed = float(payload["raw_source_reference"]["eval_median_local_action_residual"])
    delta = abs(source_observed - source_expected)
    tolerance = float(
        protocol["replication"]["maximum_source_009a_raw_56x8_action_residual_delta"]
    )
    if delta > tolerance:
        raise RuntimeError(
            "raw bridge path failed to reproduce frozen 009A (56,8): "
            f"expected={source_expected:.17g}, observed={source_observed:.17g}, delta={delta:.3g}, tolerance={tolerance:.3g}"
        )

    payload.update(
        {
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": _sha256(PROTOCOL),
            "data_manifest_sha256": got,
            "device": str(device),
            "elapsed_seconds": time.time() - started,
            "source_009a_reproduction": {
                "source_seed_format": source_seed.get("format"),
                "expected_eval_median_local_action_residual_56x8": source_expected,
                "observed_eval_median_local_action_residual_56x8": source_observed,
                "absolute_delta": delta,
                "tolerance": tolerance,
                "pass": True,
            },
        }
    )

    path = seed_dir / f"seed-{args.seed}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "seed": args.seed,
                "elapsed_seconds": payload["elapsed_seconds"],
                "source_reproduction_delta": delta,
                "output": str(path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
