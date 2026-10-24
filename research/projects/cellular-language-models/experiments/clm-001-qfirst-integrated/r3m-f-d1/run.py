"""R3m-F-D1 engineering gate. Scientific training is deliberately unavailable here."""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "implementation"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpu-gate",), default="cpu-gate")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from minicells_r3mf_d1.cpu_gate import run

    result = run(args.output)
    print(json.dumps(result["proof"], sort_keys=True))


if __name__ == "__main__":
    main()
