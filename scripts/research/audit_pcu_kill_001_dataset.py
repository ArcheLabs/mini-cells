#!/usr/bin/env python3
"""Audit a PCU-KILL-001 dataset manifest without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minicells.pcu_kill_001.synthetic import audit_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = audit_dataset(json.loads(args.manifest.read_text(encoding="utf-8")))
    destination = args.output or args.manifest.with_name("DATASET_AUDIT.json")
    destination.write_text(json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("DATASET_LEAKAGE_AUDIT=PASS" if audit.passed else "DATASET_LEAKAGE_AUDIT=FAIL")
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
