#!/usr/bin/env python3
"""Pack one exported fixture step into the dedicated guest's MCT1 payload."""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

PARAMETERS = 4476
STATE_BYTES = 4 + 8 + PARAMETERS * 4 * 3 + 256 * 64 + 256


def read_exact(path: Path, size: int) -> bytes:
    value = path.read_bytes()
    if len(value) != size:
        raise SystemExit(f"{path} has {len(value)} bytes; expected {size}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=Path("fixtures/training-fidelity-v1"))
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.step < 1:
        raise SystemExit("step must be positive")
    batch = args.fixture / f"batch-{args.step:06d}.bin"
    raw = read_exact(batch, 12 + 256 * 64 + 256)
    if raw[:4] != b"MCB1" or struct.unpack_from("<II", raw, 4) != (256, 64):
        raise SystemExit(f"invalid batch header: {batch}")
    ids = raw[12 : 12 + 256 * 64]
    lengths = raw[12 + 256 * 64 :]
    payload = bytearray(b"MCT1" + struct.pack("<Q", args.step - 1))
    if args.step == 1:
        payload += read_exact(args.fixture / "initial-weights-f32.bin", PARAMETERS * 4)
        payload += bytes(PARAMETERS * 4 * 2)
    else:
        expected = args.fixture / "expected"
        payload += read_exact(expected / f"step-{args.step - 1:06d}-weights-f32.bin", PARAMETERS * 4)
        payload += read_exact(expected / f"step-{args.step - 1:06d}-adam-m-f32.bin", PARAMETERS * 4)
        payload += read_exact(expected / f"step-{args.step - 1:06d}-adam-v-f32.bin", PARAMETERS * 4)
    payload += ids + lengths
    if len(payload) != STATE_BYTES:
        raise SystemExit(f"internal payload size error: {len(payload)} != {STATE_BYTES}")
    args.output.write_bytes(payload)
    print(f"wrote {len(payload)} bytes to {args.output}")


if __name__ == "__main__":
    main()
