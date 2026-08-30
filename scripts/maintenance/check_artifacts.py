#!/usr/bin/env python3
"""Compare the current artifact tree with a previously captured SHA-256 manifest."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_artifacts.py <sha256-manifest>")
    expected = {}
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
        digest, path = line.split(None, 1)
        expected[path.lstrip("*")] = digest
    current = {}
    for path in sorted((ROOT / "artifacts/experiments").rglob("*")):
        if path.is_file():
            current[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != current:
        raise SystemExit("artifact integrity: ERROR: manifest differs")
    print(f"artifact integrity: OK ({len(current)} files)")


if __name__ == "__main__":
    main()
