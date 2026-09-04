from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
LOCAL_ROOT = ROOT / "scripts" / "research" / "granite_hybrid_clm_v01"
RUNNER = LOCAL_ROOT / "run_milestone.py"
RELOAD = LOCAL_ROOT / "verify_reload.py"
VALIDATOR = LOCAL_ROOT / "validate_result.py"

_BOOTSTRAP = r"""
import importlib.util
import runpy
import sys
from pathlib import Path

script = Path(sys.argv[1]).resolve()
forwarded = sys.argv[2:]
local_dataset = script.parent / "dataset.py"
if local_dataset.exists():
    spec = importlib.util.spec_from_file_location("dataset", local_dataset)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Hybrid CLM dataset from {local_dataset}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dataset"] = module
    spec.loader.exec_module(module)
sys.argv = [str(script), *forwarded]
runpy.run_path(str(script), run_name="__main__")
"""


def _run(script: Path, *args: str) -> None:
    command = [sys.executable, "-c", _BOOTSTRAP, str(script), *args]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SRC_ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(SRC_ROOT)
    )
    print("+", sys.executable, script, *args, flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Granite Hybrid CLM v0.1 on Kaggle")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=26090471)
    args = parser.parse_args()

    if args.mode == "smoke":
        facts = 3
        address_steps = 80
        transform_steps = 12
        output_dir = ROOT / "results" / "granite-hybrid-clm-v0.1-smoke"
    else:
        facts = 50
        address_steps = 240
        transform_steps = 40
        output_dir = ROOT / "results" / "granite-hybrid-clm-v0.1"

    _run(
        RUNNER,
        "--device",
        args.device,
        "--facts",
        str(facts),
        "--seed",
        str(args.seed),
        "--address-steps",
        str(address_steps),
        "--transform-steps",
        str(transform_steps),
        "--output-dir",
        str(output_dir),
    )
    _run(RELOAD, "--device", args.device, "--result-dir", str(output_dir))
    if args.mode == "full":
        _run(VALIDATOR, "--result-dir", str(output_dir))

    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    reload_report = json.loads(
        (output_dir / "reload_verification.json").read_text(encoding="utf-8")
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "result_status": result["status"],
                "committed_facts": result["committed_facts"],
                "retention_choice_accuracy": result["retention_choice_accuracy"],
                "reload_status": reload_report["status"],
                "output_dir": str(output_dir.relative_to(ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
