#!/usr/bin/env python3
"""Freeze the Core 007 discovery winner before confirmation seeds may open."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery"
DEFAULT_PROTOCOL = VALIDATION / "protocol.json"
DEFAULT_DISCOVERY = ROOT / "results" / "core-validation-007-functional-boundary-discovery" / "discovery" / "raw.json"
DEFAULT_LOCK = VALIDATION / "winner-lock.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    p.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    p.add_argument("--out", type=Path, default=DEFAULT_LOCK)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    payload = json.loads(args.discovery.read_text(encoding="utf-8"))
    decision = payload.get("decision", {})
    if payload.get("phase") != "discovery":
        raise RuntimeError("winner selection requires a discovery raw.json")
    if decision.get("status") != "FUNCTIONAL_BOUNDARY_DISCOVERY_COMPLETED":
        raise RuntimeError("discovery did not complete under the frozen protocol")
    expected = list(protocol["replication"]["discovery_seeds"])
    actual = [int(r["seed"]) for r in payload.get("runs", [])]
    if actual != expected:
        raise RuntimeError(f"discovery seed identity mismatch: expected {expected}, got {actual}")
    winner = str(decision["provisional_winner"])
    candidates = list(protocol["functional_modes"]["boundary_candidates"])
    if winner not in candidates:
        raise RuntimeError("discovery winner is not a frozen candidate")
    summary = next(
        r for r in decision["candidate_summary"] if r["candidate"] == winner
    )
    lock = {
        "format": "minicells.core-validation.functional-boundary-winner-lock.v1",
        "experiment_id": "core-validation-007",
        "protocol_sha256": _sha256(args.protocol),
        "discovery_data_manifest_sha256": payload["data_manifest_sha256"],
        "discovery_seeds": expected,
        "winner": winner,
        "winner_summary": summary,
        "winner_meets_routing_floor": bool(decision["winner_meets_routing_floor"]),
        "selection_rule": protocol["discovery"]["selection_rule"],
        "selection_weights": protocol["discovery"]["selection_weights"],
        "discovery_code_commit": payload.get("provenance", {}).get("code_commit"),
        "confirmation_opened": False,
    }
    rendered = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    if args.out.exists():
        existing = args.out.read_text(encoding="utf-8")
        if existing != rendered:
            raise RuntimeError(
                "winner-lock.json already exists with different content; confirmation lock is immutable"
            )
        print(args.out)
        return 0
    args.out.write_text(rendered, encoding="utf-8")
    print(json.dumps(lock, indent=2, sort_keys=True))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
