#!/usr/bin/env python3
"""Run one Core 008 preflight functional-equivalence bridge seed.

This is a diagnostic bridge, not a Core 007 rerun and not a Core 008 scientific
decision. It deterministically rehydrates the frozen Core 007 model/data identity,
reconstructs the interference_cut candidate for an already-observed completed
seed, and measures counterfactual functional regret that was not persisted by
Core 007.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from minicells.real_representation_006_experiment import build_transactions, prepare_seed
from minicells.real_representation_006_io import (
    extract_frozen_sequences,
    load_foundation,
    load_frozen_cache,
    save_frozen_cache,
    select_real_sequences,
    write_data_manifest,
)
from minicells.real_representation_007_config import CoreValidation007Config
from minicells.real_representation_007_experiment import (
    CandidateState,
    FunctionalSystem,
    _adjusted_batch,
    _eval_mode_ids,
    _signature_batches,
    _train_candidate_transaction,
)

ROOT = Path(__file__).resolve().parents[2]
CORE007_VALIDATION = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery"
CORE007_PROTOCOL = CORE007_VALIDATION / "protocol.json"
CORE007_AMENDMENT = CORE007_VALIDATION / "confirmation-protocol-v1.1.json"
CORE007_GATES = (
    ROOT
    / "artifacts"
    / "experiments"
    / "core-validation-007-functional-boundary-discovery"
    / "confirmation"
    / "gate-summary.csv"
)
DEFAULT_OUT = ROOT / "results" / "core-008-preflight-functional-equivalence"
OBSERVED_SEEDS = (80721, 80722)
_EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True, choices=OBSERVED_SEEDS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def _canonical_reference(seed: int) -> dict[str, float]:
    with CORE007_GATES.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if int(r["seed"]) == seed)
    return {
        "oracle_eval_nll": float(row["oracle_eval_nll"]),
        "deploy_eval_nll": float(row["deploy_eval_nll"]),
        "eval_routing_agreement": float(row["eval_routing_agreement"]),
        "train_routing_agreement": float(row["train_routing_agreement"]),
    }


def _per_sequence_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).reshape(labels.shape)
    return loss.mean(dim=1)


def _rms(x: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    return torch.sqrt(torch.mean(x * x, dim=dims).clamp_min(0.0))


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _aggregate(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    subset = [r for r in rows if predicate(r)]
    keys = (
        "absolute_nll_regret",
        "normalized_nll_regret",
        "hidden_route_difference_rms",
        "normalized_hidden_route_difference",
        "logit_route_difference_rms",
        "normalized_logit_route_difference",
        "symmetric_logit_kl",
        "oracle_cell_effect_abs_nll",
        "deploy_cell_effect_abs_nll",
    )
    out: dict[str, Any] = {"count": len(subset)}
    for key in keys:
        vals = [float(r[key]) for r in subset if math.isfinite(float(r[key]))]
        out[f"{key}_mean"] = _mean(vals)
        out[f"{key}_median"] = _median(vals)
    if subset:
        out["same_owner_fraction"] = sum(bool(r["owner_match"]) for r in subset) / len(subset)
    else:
        out["same_owner_fraction"] = 0.0
    return out


def _diagnose(
    eval_sequences: list[Any],
    oracle_modes: list[int],
    deploy_modes: list[int],
    state: CandidateState,
    u: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    weight = lm_head_weight.to(device=device, dtype=torch.float32)
    for start in range(0, len(eval_sequences), batch_size):
        stop = min(start + batch_size, len(eval_sequences))
        seqs = eval_sequences[start:stop]
        om = oracle_modes[start:stop]
        dm = deploy_modes[start:stop]
        hidden = torch.stack([s.hidden for s in seqs]).to(device=device, dtype=torch.float32)
        labels = torch.stack([s.labels for s in seqs]).to(device=device, dtype=torch.long)
        oracle_adjusted, _, _ = _adjusted_batch(seqs, om, state.system, u, device=device)
        deploy_adjusted, _, _ = _adjusted_batch(seqs, dm, state.system, u, device=device)
        oracle_delta = oracle_adjusted - hidden
        deploy_delta = deploy_adjusted - hidden

        with torch.no_grad():
            foundation_logits = F.linear(hidden, weight)
            oracle_logits = F.linear(oracle_adjusted, weight)
            deploy_logits = F.linear(deploy_adjusted, weight)
            foundation_nll = _per_sequence_nll(foundation_logits, labels)
            oracle_nll = _per_sequence_nll(oracle_logits, labels)
            deploy_nll = _per_sequence_nll(deploy_logits, labels)

            oracle_logp = F.log_softmax(oracle_logits, dim=-1)
            deploy_logp = F.log_softmax(deploy_logits, dim=-1)
            oracle_p = oracle_logp.exp()
            deploy_p = deploy_logp.exp()
            kl_od = torch.sum(oracle_p * (oracle_logp - deploy_logp), dim=-1).mean(dim=1)
            kl_do = torch.sum(deploy_p * (deploy_logp - oracle_logp), dim=-1).mean(dim=1)
            sym_kl = 0.5 * (kl_od + kl_do)

            oracle_hidden_rms = _rms(oracle_delta)
            deploy_hidden_rms = _rms(deploy_delta)
            hidden_diff_rms = _rms(deploy_delta - oracle_delta)
            oracle_logit_delta = oracle_logits - foundation_logits
            deploy_logit_delta = deploy_logits - foundation_logits
            oracle_logit_rms = _rms(oracle_logit_delta)
            deploy_logit_rms = _rms(deploy_logit_delta)
            logit_diff_rms = _rms(deploy_logits - oracle_logits)

        for j, seq in enumerate(seqs):
            oi = int(om[j])
            di = int(dm[j])
            oo = int(state.system.mode_owner[oi])
            do = int(state.system.mode_owner[di])
            onll = float(oracle_nll[j].cpu().item())
            dnll = float(deploy_nll[j].cpu().item())
            fnll = float(foundation_nll[j].cpu().item())
            oracle_effect = abs(onll - fnll)
            deploy_effect = abs(dnll - fnll)
            nll_regret = abs(dnll - onll)
            hidden_scale = 0.5 * (
                float(oracle_hidden_rms[j].cpu().item())
                + float(deploy_hidden_rms[j].cpu().item())
            )
            logit_scale = 0.5 * (
                float(oracle_logit_rms[j].cpu().item())
                + float(deploy_logit_rms[j].cpu().item())
            )
            rows.append(
                {
                    "index": start + j,
                    "token_sha256": seq.token_sha256,
                    "address": int(seq.address),
                    "oracle_mode": oi,
                    "deploy_mode": di,
                    "mode_match": oi == di,
                    "oracle_owner": oo,
                    "deploy_owner": do,
                    "owner_match": oo == do,
                    "foundation_nll": fnll,
                    "oracle_nll": onll,
                    "deploy_nll": dnll,
                    "signed_nll_regret_deploy_minus_oracle": dnll - onll,
                    "absolute_nll_regret": nll_regret,
                    "oracle_cell_effect_abs_nll": oracle_effect,
                    "deploy_cell_effect_abs_nll": deploy_effect,
                    "normalized_nll_regret": nll_regret / max(oracle_effect, deploy_effect, 1e-8),
                    "oracle_hidden_delta_rms": float(oracle_hidden_rms[j].cpu().item()),
                    "deploy_hidden_delta_rms": float(deploy_hidden_rms[j].cpu().item()),
                    "hidden_route_difference_rms": float(hidden_diff_rms[j].cpu().item()),
                    "normalized_hidden_route_difference": float(hidden_diff_rms[j].cpu().item()) / max(hidden_scale, 1e-12),
                    "oracle_logit_delta_rms": float(oracle_logit_rms[j].cpu().item()),
                    "deploy_logit_delta_rms": float(deploy_logit_rms[j].cpu().item()),
                    "logit_route_difference_rms": float(logit_diff_rms[j].cpu().item()),
                    "normalized_logit_route_difference": float(logit_diff_rms[j].cpu().item()) / max(logit_scale, 1e-12),
                    "symmetric_logit_kl": float(sym_kl[j].cpu().item()),
                }
            )
        del foundation_logits, oracle_logits, deploy_logits, oracle_logp, deploy_logp, oracle_p, deploy_p
    return rows


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    amendment = json.loads(CORE007_AMENDMENT.read_text(encoding="utf-8"))
    cfg = CoreValidation007Config.from_protocol(CORE007_PROTOCOL)
    winner = str(amendment["winner"])
    started = time.time()

    print(f"[core008-preflight] seed={args.seed} rehydrating frozen Core007 identity", flush=True)
    tokenizer, model = load_foundation(cfg.base, device=device)
    records, manifest = select_real_sequences(cfg.base, tokenizer)
    manifest_sha = str(manifest["manifest_sha256"])
    expected_manifest = str(amendment["expected_data_manifest_sha256"])
    if manifest_sha != expected_manifest:
        raise RuntimeError(f"data manifest mismatch: expected {expected_manifest}, got {manifest_sha}")

    args.out.mkdir(parents=True, exist_ok=True)
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

    u, centroids, base_assignment, projected = prepare_seed(frozen, cfg.base, seed=args.seed)
    signatures = _signature_batches(projected, u, lm_head_weight, device=device)
    transactions = build_transactions(projected, cfg.base)
    eval_sequences = [s for s in projected if s.partition == "eval"]
    system = FunctionalSystem.initialize(
        dim=cfg.base.cell_dim,
        base_address_owner=base_assignment,
        base_cells=cfg.base.base_cells,
        certificate_energy=cfg.base.certificate_energy,
        maximum_modes_per_address=cfg.maximum_modes_per_address,
        maximum_write_rank=cfg.maximum_write_rank,
        mode_creation_cosine_threshold=cfg.mode_creation_cosine_threshold,
    )
    state = CandidateState(candidate=winner, system=system)
    for tx, current in enumerate(transactions):
        _train_candidate_transaction(
            state,
            current,
            signatures,
            u,
            lm_head_weight,
            cfg,
            transaction=tx,
            device=device,
        )

    oracle_modes, eval_agreement = _eval_mode_ids(eval_sequences, signatures, state.system, deploy=False)
    deploy_modes, _ = _eval_mode_ids(eval_sequences, signatures, state.system, deploy=True)
    train_agreement = sum(bool(r["agreement"]) for r in state.routing_records) / max(len(state.routing_records), 1)
    diagnostics = _diagnose(
        eval_sequences,
        oracle_modes,
        deploy_modes,
        state,
        u,
        lm_head_weight,
        device=device,
        batch_size=args.batch_size,
    )

    oracle_eval_nll = _mean([float(r["oracle_nll"]) for r in diagnostics])
    deploy_eval_nll = _mean([float(r["deploy_nll"]) for r in diagnostics])
    reference = _canonical_reference(args.seed)
    reproduction = {
        "canonical_reference": reference,
        "rehydrated": {
            "oracle_eval_nll": oracle_eval_nll,
            "deploy_eval_nll": deploy_eval_nll,
            "eval_routing_agreement": float(eval_agreement),
            "train_routing_agreement": float(train_agreement),
        },
        "absolute_differences": {
            "oracle_eval_nll": abs(oracle_eval_nll - reference["oracle_eval_nll"]),
            "deploy_eval_nll": abs(deploy_eval_nll - reference["deploy_eval_nll"]),
            "eval_routing_agreement": abs(float(eval_agreement) - reference["eval_routing_agreement"]),
            "train_routing_agreement": abs(float(train_agreement) - reference["train_routing_agreement"]),
        },
    }
    reproduction["matches_reference"] = bool(
        reproduction["absolute_differences"]["oracle_eval_nll"] <= 1e-4
        and reproduction["absolute_differences"]["deploy_eval_nll"] <= 1e-4
        and reproduction["absolute_differences"]["eval_routing_agreement"] <= 1e-12
        and reproduction["absolute_differences"]["train_routing_agreement"] <= 1e-12
    )

    summary = {
        "all_eval": _aggregate(diagnostics, lambda r: True),
        "mode_mismatch": _aggregate(diagnostics, lambda r: not bool(r["mode_match"])),
        "mode_mismatch_same_owner": _aggregate(
            diagnostics, lambda r: (not bool(r["mode_match"])) and bool(r["owner_match"])
        ),
        "owner_mismatch": _aggregate(diagnostics, lambda r: not bool(r["owner_match"])),
    }
    mode_mismatch = [r for r in diagnostics if not bool(r["mode_match"])]
    summary["mode_mismatch_same_owner_fraction"] = (
        sum(bool(r["owner_match"]) for r in mode_mismatch) / len(mode_mismatch)
        if mode_mismatch
        else 0.0
    )

    payload = {
        "format": "minicells.core008-preflight.functional-equivalence-seed.v1",
        "scientific_decision": False,
        "seed": args.seed,
        "winner": winner,
        "data_manifest_sha256": manifest_sha,
        "source_core007_status_unchanged": True,
        "reproduction": reproduction,
        "summary": summary,
        "records": diagnostics,
        "provenance": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "elapsed_seconds": time.time() - started,
        },
    }
    seed_dir = args.out / "seeds"
    seed_dir.mkdir(parents=True, exist_ok=True)
    path = seed_dir / f"seed-{args.seed}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[core008-preflight] wrote {path}", flush=True)
    print(json.dumps({"seed": args.seed, "reproduction": reproduction, "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
