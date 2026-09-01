#!/usr/bin/env python3
"""Run frozen Core Validation 007 discovery/smoke.

The original monolithic confirmation entrypoint is deliberately retired after
it exposed 80711 and terminated during 80712 without a durable per-seed
checkpoint. Formal confirmation v1.1 must use the isolated resumable
orchestrator instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
from minicells.real_representation_007_config import CoreValidation007Config, smoke_config
from minicells.real_representation_007_experiment import run_discovery_seed, summarize_discovery

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery"
DEFAULT_PROTOCOL = VALIDATION / "protocol.json"
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, action="append", dest="seeds")
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase == "confirmation" and not args.smoke:
        raise RuntimeError(
            "The original Core 007 monolithic confirmation entrypoint is retired. "
            "Use scripts/research/orchestrate_core_validation_007_confirmation.py; "
            "it runs amended seeds 80721/80722/80723 in isolated resumable processes."
        )

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    formal_cfg = CoreValidation007Config.from_protocol(args.protocol)
    cfg = smoke_config(formal_cfg) if args.smoke else formal_cfg
    protocol_sha = _sha256(args.protocol)
    device = _device(args.device)
    if not args.smoke and bool(protocol["hardware"]["gpu_required_for_real_run"]):
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("formal Core Validation 007 discovery requires CUDA")

    if args.smoke:
        seeds = args.seeds or [cfg.smoke_seed]
    else:
        frozen = list(formal_cfg.discovery_seeds)
        seeds = args.seeds or frozen
        if seeds != frozen:
            raise RuntimeError(f"discovery must run exactly frozen seeds {frozen}")

    phase_out = args.out / ("smoke" if args.smoke else "discovery")
    phase_out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / ("frozen-hidden-smoke.pt" if args.smoke else "frozen-hidden.pt")
    manifest_path = args.out / "data-manifest.json"
    started = time.time()

    print(
        f"[core-007] phase=discovery loading {cfg.base.model_id}@{cfg.base.model_revision} on {device}",
        flush=True,
    )
    tokenizer, model = load_foundation(cfg.base, device=device)
    records, manifest = select_real_sequences(cfg.base, tokenizer)
    write_data_manifest(manifest, manifest_path)
    print(
        f"[core-007] selected {len(records)} sequences manifest={manifest['manifest_sha256'][:16]}",
        flush=True,
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

    runs = []
    for seed in seeds:
        print(f"[core-007] phase=discovery seed={seed}", flush=True)
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
        "phase": "smoke" if args.smoke else "discovery",
        "protocol_sha256": protocol_sha,
        "parent_experiment": protocol["parent_experiment"],
        "data_manifest_sha256": manifest["manifest_sha256"],
        "winner_lock": None,
        "provenance": {
            "code_commit": _git(["rev-parse", "HEAD"]),
            "code_tree": _git(["rev-parse", "HEAD^{tree}"]),
            "tracked_tree_dirty": _tracked_tree_dirty(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            **_hf_provenance(cfg),
        },
        "runs": runs,
        "decision": decision,
        "elapsed_seconds": time.time() - started,
    }
    raw = phase_out / "raw.json"
    raw.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
