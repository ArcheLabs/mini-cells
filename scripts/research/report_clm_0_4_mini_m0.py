#!/usr/bin/env python3
"""Generate a human-readable report for the CLM-0.4-mini M0 smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "clm-0.4-mini-m0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    decision = json.loads((args.out / "decision.json").read_text(encoding="utf-8"))
    summary = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
    replay = json.loads((args.out / "replay.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (args.out / "transactions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]

    if decision.get("status") != "SMOKE_ONLY" or decision.get("scientific_decision") is not False:
        raise RuntimeError("M0 report refuses to describe a scientific decision")

    rows = []
    for record in records:
        attempts = " → ".join(attempt["candidate_kind"] for attempt in record["attempts"])
        max_escape = max(attempt["structural_escape_count"] for attempt in record["attempts"])
        rows.append(
            f"| {record['transaction_id']} | `{record['address_id']}` | "
            f"{attempts} | `{record['final_decision']}` | {max_escape} |"
        )

    text = f"""# CLM-0.4-mini M0 Execution Smoke

> Status: **SMOKE_ONLY**
>
> This report has no scientific meaning and does not use development or formal seeds.

## Purpose

M0 validates the token-level execution pipeline required before development-seed calibration:

`route → candidate → dependency validation → rollback → growth → atomic commit → reuse → checkpoint → journal replay`

## Result

- seed: `{decision['seed']}`
- transactions: {summary['transaction_count']}
- replay valid: `{str(replay['valid']).lower()}`
- private addresses after smoke: {summary['private_addresses']}
- dependency probes after smoke: {summary['dependency_probe_count']}
- final state hash: `{summary['final_state_hash']}`

## Paths exercised

| Tx | Address | Attempts | Final decision | Structural escapes |
|---:|---|---|---|---:|
{chr(10).join(rows)}

The first transaction explicitly exercises direct-candidate rollback followed by a zero-output
probationary growth bundle. The second reuses the committed private bundle. The third exercises
a direct base-Cell commit.

## Artifacts

- `decision.json`
- `summary.json`
- `transactions.jsonl`
- `cell-registry.json`
- `replay.json`
- `checkpoints/tx-*.pt`

## Scientific boundary

M0 may only emit `SMOKE_ONLY`. It does **not** calibrate the candidate optimizer, does not use
development seed `90401`, and does not run formal seeds `90411`, `90412`, or `90413`.
"""
    (args.out / "RESULTS.md").write_text(text, encoding="utf-8")
    print(f"wrote {args.out / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
