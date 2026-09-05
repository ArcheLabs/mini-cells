#!/usr/bin/env python3
"""Run PCU-KILL-001 with strict separation between engineering and formal phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minicells.pcu_kill_001.experiment import run_engineering, run_formal_execution  # noqa: E402
from minicells.pcu_kill_001.governance import (  # noqa: E402
    DEVELOPMENT_SEED,
    FORMAL_SEEDS,
    ProtocolMismatch,
    assert_formal_preflight,
    assert_seed_registry,
    mark_formal_seed,
)


PROTOCOL = ROOT / "artifacts/research/pcu-kill-001/frozen/PROTOCOL.json"
PROTOCOL_SHA = ROOT / "artifacts/research/pcu-kill-001/frozen/PROTOCOL.sha256"
SEED_REGISTRY = ROOT / "research/formal_seed_registry.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("engineering", "formal"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backend", choices=("granite", "toy"), default="granite")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--protocol-sha", type=Path, default=PROTOCOL_SHA)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute-formal", action="store_true", help="explicitly authorize formal seed execution")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase == "engineering":
        if args.preflight_only:
            raise SystemExit("--preflight-only is only valid in formal phase")
        seed = DEVELOPMENT_SEED if args.seed is None else args.seed
        if args.out is None:
            args.out = ROOT / "artifacts/research/pcu-kill-001/engineering" / str(seed)
        result = run_engineering(seed=seed, backend=args.backend, output=args.out, device=args.device)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.preflight_only:
        try:
            result = assert_formal_preflight(ROOT, args.protocol, args.protocol_sha, SEED_REGISTRY)
        except (OSError, ValueError, ProtocolMismatch) as exc:
            payload = {"status": "FAIL", "scientific_decision": False, "error": f"{type(exc).__name__}: {exc}", "formal_execution_not_started": True}
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 1
        result["formal_execution_not_started"] = True
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.seed not in FORMAL_SEEDS:
        raise SystemExit(f"formal execution requires one of {FORMAL_SEEDS}")
    if not args.execute_formal:
        raise SystemExit("formal seed execution requires explicit --execute-formal authorization")
    assert_formal_preflight(ROOT, args.protocol, args.protocol_sha, SEED_REGISTRY)
    assert_seed_registry(SEED_REGISTRY)
    output = args.out or ROOT / "artifacts/research/pcu-kill-001/formal" / str(args.seed)
    valid = False
    try:
        result = run_formal_execution(args.seed, args.protocol, output, args.device if args.device != "auto" else "cpu")
        valid = bool(result.get("scientific_evidence") and result.get("g0") and result.get("cache") and result.get("dataset_audit"))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if valid else 1
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        payload = {"status": "FORMAL_EXECUTION_FAILED", "scientific_evidence": False, "formal_execution_not_started": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    finally:
        mark_formal_seed(SEED_REGISTRY, args.seed, valid=valid)


if __name__ == "__main__":
    raise SystemExit(main())
