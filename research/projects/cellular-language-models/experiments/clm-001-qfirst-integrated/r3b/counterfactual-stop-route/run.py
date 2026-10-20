"""Bootstrap CF orchestration after binding scientific imports to the immutable engine."""

from pathlib import Path
import argparse
import runpy
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--orchestrator-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/kaggle/temp/cf001-input"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=6900,
        help="Package after a completed chunk; never interrupt or terminate a process",
    )
    parser.add_argument("--publish-github", action="store_true")
    args = parser.parse_args()
    base = Path(
        "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3b"
    )
    sys.path.insert(0, str(args.orchestrator_root / base / "implementation"))
    from kaggle_posthoc.source_guard import verify_engine

    engine_attestation = verify_engine(args.engine_root)
    runpy.run_path(
        str(args.engine_root / base / "analyze.py"), run_name="frozen_cf_engine_bootstrap"
    )
    sys.path.insert(0, str(Path(__file__).resolve().parent / "implementation"))
    from clm001_r3b import runner, full_analysis

    for module in (runner, full_analysis):
        if args.engine_root.resolve() not in Path(module.__file__).resolve().parents:
            raise RuntimeError("CF scientific import origin mismatch")
    from r3b_cf001.driver import run

    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.orchestrator_root, text=True
    ).strip()
    source_attestation = verify_engine(args.orchestrator_root, expected=source)
    from minicells_runtime.store import atomic

    atomic(
        args.output / "source-attestation.json",
        {"engine": engine_attestation, "orchestrator": source_attestation},
    )
    run(args, source)


if __name__ == "__main__":
    main()
