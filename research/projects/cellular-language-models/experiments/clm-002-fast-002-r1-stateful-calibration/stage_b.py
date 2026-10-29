"""Separate formal Stage B entry; the historical Gate0/Gate1 CLI is unchanged."""

import argparse
from pathlib import Path

from run import bootstrap


def main():
    bootstrap()
    from minicells_fast_r1.stageb import main as execute

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--publish-github", action="store_true")
    parser.add_argument("--publish-existing", action="store_true")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--workers", type=int, default=1)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
