#!/usr/bin/env python3
"""Pack a canonical subset into the training guest MCP1 payload."""
import argparse, struct
from pathlib import Path

def main():
    p = argparse.ArgumentParser(); p.add_argument("--fixture", type=Path, default=Path("fixtures/training-fidelity-v1")); p.add_argument("--samples", type=int, default=16); p.add_argument("--format", choices=("subset", "full"), default="subset"); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    if not 1 <= a.samples <= 256: raise SystemExit("samples must be 1..=256")
    initial = (a.fixture / "initial-weights-f32.bin").read_bytes()
    batch = (a.fixture / "batch-000001.bin").read_bytes()
    if batch[:4] != b"MCB1": raise SystemExit("invalid batch")
    rows, width = struct.unpack_from("<II", batch, 4)
    ids = batch[12:12 + rows * width]; lengths = batch[12 + rows * width:]
    if a.format == "full":
        a.samples = 256
        out = bytearray(b"MCT1" + struct.pack("<Q", 0) + initial + bytes(len(initial)) + bytes(len(initial)))
    else:
        out = bytearray(b"MCP1" + struct.pack("<Q", 0) + initial + bytes(len(initial)) + bytes(len(initial)) + struct.pack("<H", a.samples))
    for i in range(a.samples): out.extend(ids[i * 64:(i + 1) * 64])
    out.extend(lengths[:a.samples]); a.output.write_bytes(out)
if __name__ == "__main__": main()
