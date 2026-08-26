#!/usr/bin/env python3
"""Create a structurally valid MCT1 payload for guest diagnostics only."""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

PARAMETERS = 4476


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.length <= 32:
        raise SystemExit("length must be in [1, 32]")
    ids = bytearray(256 * 64)
    for row in range(256):
        for cell in range(args.length):
            ids[row * 64 + cell] = (cell % 43) + 1
    lengths = bytes([args.length] * 256)
    payload = b"MCT1" + struct.pack("<Q", 0)
    payload += bytes(PARAMETERS * 4 * 3) + ids + lengths
    args.output.write_bytes(payload)
    print(f"wrote {len(payload)} bytes to {args.output}")


if __name__ == "__main__":
    main()
