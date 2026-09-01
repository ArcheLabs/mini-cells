#!/usr/bin/env python3
"""Aggregate the two completed Core 007 seeds for the Core 008 preflight bridge."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "results" / "core-008-preflight-functional-equivalence"
DEFAULT_OUT = ROOT / "results" / "core-008-preflight-functional-equivalence" / "RESULTS.md"
SEEDS = (80721, 80722)


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _load(indir: Path) -> list[dict[str, Any]]:
    runs = []
    for seed in SEEDS:
        path = indir / "seeds" / f"seed-{seed}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    return runs


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="indir", type=Path, default=DEFAULT_IN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    runs = _load(args.indir)

    reproduction_ok = all(bool(r["reproduction"]["matches_reference"]) for r in runs)
    if not reproduction_ok:
        status = "REHYDRATION_MISMATCH"
    else:
        per_seed_weak = []
        per_seed_near = []
        any_same_owner_redundancy = False
        for run in runs:
            records = run["records"]
            ratios = [
                max(float(r["oracle_cell_effect_abs_nll"]), float(r["deploy_cell_effect_abs_nll"]))
                / max(abs(float(r["foundation_nll"])), 1e-12)
                for r in records
            ]
            per_seed_weak.append(_median(ratios) <= 1e-4)
            owner = run["summary"]["owner_mismatch"]
            per_seed_near.append(
                int(owner["count"]) > 0
                and float(owner["normalized_logit_route_difference_median"]) <= 0.25
                and float(owner["normalized_nll_regret_median"]) <= 0.25
            )
            any_same_owner_redundancy = any_same_owner_redundancy or (
                int(run["summary"]["mode_mismatch_same_owner"]["count"]) > 0
            )
        weak = all(per_seed_weak)
        near = all(per_seed_near)
        if weak and not near:
            status = "WEAK_EFFECT_CONFOUND_DOMINATES"
        elif near or any_same_owner_redundancy:
            status = "FUNCTIONAL_REDUNDANCY_EVIDENCE"
        else:
            status = "MIXED_BRIDGE_EVIDENCE"

    lines = [
        "# Core 008 Preflight — Functional Equivalence Bridge Results",
        "",
        f"- Status: `{status}`",
        "- Scientific decision: `False`",
        "- Source Core 007 status changed: `False`",
        "",
        "This bridge uses the already-observed completed Core 007 seeds only to decide what Core 008 should test. It is not a new confirmation result.",
        "",
        "## Seed diagnostics",
        "",
        "| seed | Core007 reproduced | eval mode agreement | mismatch same-owner fraction | owner-mismatch normalized NLL regret (median) | owner-mismatch normalized logit difference (median) | owner-mismatch symmetric KL (median) |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        s = run["summary"]
        lines.append(
            "| {seed} | {rep} | {agree:.6f} | {same:.6f} | {nll:.6g} | {logit:.6g} | {kl:.6g} |".format(
                seed=run["seed"],
                rep=str(bool(run["reproduction"]["matches_reference"])),
                agree=float(run["reproduction"]["rehydrated"]["eval_routing_agreement"]),
                same=float(s["mode_mismatch_same_owner_fraction"]),
                nll=float(s["owner_mismatch"]["normalized_nll_regret_median"]),
                logit=float(s["owner_mismatch"]["normalized_logit_route_difference_median"]),
                kl=float(s["owner_mismatch"]["symmetric_logit_kl_median"]),
            )
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "The decisive quantity is not raw mode-label agreement. A mode mismatch that retains the same final Cell owner is exactly functionally equivalent under the current architecture because both routes use the same matrix A. For true owner mismatches, route regret is normalized by the magnitude of the Cell's own intervention relative to the frozen foundation path.",
        "",
        "`WEAK_EFFECT_CONFOUND_DOMINATES` means the tiny whole-model NLL gap cannot be used as evidence of functional equivalence: Cell interventions are themselves too small, while normalized owner-mismatch regret is not near-equivalent. `FUNCTIONAL_REDUNDANCY_EVIDENCE` means either mode labels substantially over-split identical owners or true owner mismatches remain behaviorally near-equivalent after normalization.",
        "",
        "The resulting status is a bridge diagnostic only. Core 008 requires a fresh protocol and fresh formal seeds.",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
