"""Run All entry for the three-Seed screening; no full benchmark."""

import argparse
from pathlib import Path

from run import bootstrap


def main():
    bootstrap()
    from screening import run

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--publish-existing", action="store_true")
    parser.add_argument(
        "--publish-github", action="store_true", help="Reports always use verified GitHub return"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
