"""Run All entry for the three-Seed screening; no full benchmark."""

import argparse
from pathlib import Path

from run import bootstrap

if __name__ == "__main__":
    bootstrap()
    from screening import run

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--publish-existing", action="store_true")
    run(p.parse_args())
