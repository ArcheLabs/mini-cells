#!/usr/bin/env python3
"""Run Core Validation 006 — Real-Representation Continual Plasticity."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from minicells.real_representation_006_config import CoreValidation006Config, smoke_config
from minicells.real_representation_006_experiment import run_seed, summarize_experiment
from minicells.real_representation_006_io import (
    extract_frozen_sequences,
    load_foundation,
    load_frozen_cache,
    save_frozen_cache,
    select_real_sequences,
    write_data_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    ROOT
    / "research"
    / "validations"
    / "core-006-real-representation-continual-plasticity"
    / "protocol.json"
)
DEFAULT_OUT = ROOT / "results" / "core-validation-006-real-representation-continual-plasticity"


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *command], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _tracked_tree_dirty() -> bool | None:
    try:
        a = subprocess.run(
            ["git", "diff", "--quiet"], cwd=ROOT, check=False, stderr=subprocess.DEVNULL
        ).returncode
        b = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=ROOT,
            check=False,
            stderr=subprocess.DEVNULL,
        ).returncode
        return bool(a or b)
    except OSError:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _hf_provenance(cfg: CoreValidation006Config) -> dict[str, str | None]:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        model = api.model_info(cfg.model_id, revision=cfg.model_revision)
        dataset = api.dataset_info(cfg.dataset_id, revision=cfg.dataset_revision)
        return {
            "resolved_model_sha": getattr(model, "sha", None),
            "resolved_dataset_sha": getattr(dataset, "sha", None),
        }
    except Exception:
        return {"resolved_model_sha": None, "resolved_dataset_sha": None}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, action="append", dest="seeds")
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    protocol: dict[str, Any] = json.loads(args.protocol.read_text(encoding="utf-8"))
    cfg = CoreValidation006Config.from_protocol(args.protocol)
    formal_cfg = cfg
    if args.smoke:
        cfg = smoke_config(cfg)

    device = _device(args.device)
    if not args.smoke and bool(protocol["hardware"]["gpu_required_for_formal_run"]):
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("formal Core Validation 006 requires CUDA")

    protocol_seeds = list(formal_cfg.formal_seeds)
    if args.smoke:
        seeds = args.seeds or [cfg.smoke_seed]
    else:
        seeds = args.seeds or protocol_seeds
        if seeds != protocol_seeds:
            raise RuntimeError(
                f"formal Core Validation 006 must run exactly frozen seeds {protocol_seeds}"
            )

    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / ("frozen-hidden-smoke.pt" if args.smoke else "frozen-hidden.pt")
    manifest_path = args.out / "data-manifest.json"
    started = time.time()

    print(f"[core-006] loading {cfg.model_id}@{cfg.model_revision} on {device}", flush=True)
    tokenizer, model = load_foundation(cfg, device=device)
    records, manifest = select_real_sequences(cfg, tokenizer)
    write_data_manifest(manifest, manifest_path)
    print(
        f"[core-006] selected {len(records)} sequences "
        f"manifest={manifest['manifest_sha256'][:16]}",
        flush=True,
    )

    frozen = None if args.no_cache else load_frozen_cache(manifest, cache_path)
    if frozen is None:
        frozen = extract_frozen_sequences(records, model, device=device)
        if not args.no_cache:
            save_frozen_cache(frozen, manifest, cache_path)
    else:
        print(f"[core-006] reused hidden cache {cache_path}", flush=True)

    expected_hidden = int(protocol["foundation"]["expected_hidden_dim"])
    hidden_dim = int(frozen[0].hidden.shape[-1])
    if not args.smoke and hidden_dim != expected_hidden:
        raise RuntimeError(
            f"foundation hidden dim changed: expected {expected_hidden}, got {hidden_dim}"
        )

    lm_head_weight = model.embed_out.weight.detach().clone()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    runs = []
    for seed in seeds:
        print(f"[core-006] seed={seed} device={device}", flush=True)
        run = run_seed(
            frozen,
            cfg,
            seed=seed,
            lm_head_weight=lm_head_weight,
            device=device,
        )
        runs.append(run)
        g = run["gate_summary"]
        growth = g["variant_summaries"]["certificate_mitosis"]
        print(
            "[core-006] "
            f"seed={seed} pass={g['pass']} "
            f"rank_mid={g['midstream_energy_rank_fraction']:.3f} "
            f"reuse={g['midstream_reuse_ratio']:.3f} "
            f"reg/unsafe={g['registered_regression_ratio_vs_unsafe']:.3f} "
            f"gain/replay={g['gain_ratio_vs_replay']:.3f} "
            f"spawn={growth['spawned_cells']} "
            f"child_reuse={growth['child_reuse_transactions']}",
            flush=True,
        )

    if args.smoke:
        decision = {
            "status": "SMOKE_ONLY",
            "pass": None,
            "scientific_decision": False,
            "passed_seeds": None,
            "total_seeds": len(runs),
            "reason": "Smoke mode uses reduced real-data quotas and cannot emit a scientific decision.",
        }
    else:
        decision = summarize_experiment(
            runs,
            positive_status=str(protocol["gates"]["positive_status"]),
            negative_status=str(protocol["gates"]["negative_status"]),
        )

    payload = {
        "format": protocol["format"],
        "experiment_id": protocol["experiment_id"],
        "protocol_version": protocol["protocol_version"],
        "mode": "smoke" if args.smoke else "formal",
        "protocol_sha256": _sha256(args.protocol),
        "parent_experiment": protocol["parent_experiment"],
        "data_manifest_sha256": manifest["manifest_sha256"],
        "foundation": protocol["foundation"],
        "data": {
            "dataset_id": cfg.dataset_id,
            "dataset_revision": cfg.dataset_revision,
            "sources": list(cfg.sources),
            "sequence_length": cfg.sequence_length,
        },
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
    raw = args.out / "raw.json"
    raw.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"wrote {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
