"""One-launch D1 flow: frozen CPU Gate first, optional explicitly authorized B/C/D run."""

from pathlib import Path
import argparse
import json
import os
import runpy
import subprocess
import sys
import time


BASE = Path(__file__).resolve().parent
REPO = BASE.parents[5]


def _bootstrap():
    # Reuse the repository's dependency path; these wrappers only add sys.path entries
    # when run with a non-main run_name and never execute their historical studies.
    runpy.run_path(str(BASE.parent / "r3m-e/run.py"), run_name="r3mf_d1_dependencies")
    sys.path.insert(0, str(BASE.parent / "r3m-e/implementation"))
    sys.path.insert(0, str(BASE / "implementation"))


def _source_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
    ).strip()


def _record_run_failure(output, source_sha):
    """Mark this source's durable run manifest failed without exposing exception details."""
    for path in (Path(output) / "runs").glob("*/RUN_MANIFEST.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("run_status") != "RUNNING" or value.get("source_sha") != source_sha:
                continue
            value["run_status"] = "FAILED_RESUMABLE"
            value["exit_code"] = 1
            value["failure_type"] = "EXECUTION_EXCEPTION"
            value["finished_at_unix"] = time.time()
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, path)
        except Exception:
            # Preserve the original failure; an invalid manifest remains inspectable in Kaggle Output.
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpu-gate", "auto"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorize-d1-run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    sys.path.insert(0, str(BASE / "implementation"))

    from minicells_r3mf_d1.cpu_gate import run as run_cpu_gate

    preflight_root = args.output / "engineering" / "preflight"
    gate = run_cpu_gate(preflight_root)
    preflight_dir = preflight_root / gate["manifest"]["run_id"]
    print(
        json.dumps(
            {
                "experiment_id": gate["manifest"]["experiment_id"],
                "source_sha": _source_sha(),
                "cpu_gate": gate["proof"]["status"],
                "optimizer_updates": 0,
                "gpu_used": False,
                "preflight_output": str(preflight_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.mode == "cpu-gate" or not args.authorize_d1_run:
        print(
            json.dumps(
                {
                    "status": "CPU_GATE_PASS_AWAITING_INDEPENDENT_REVIEW_AND_EXPLICIT_RUN_AUTHORIZATION",
                    "training_started": False,
                    "arms": ["B", "C", "D"],
                    "per_arm_update_cap": 200,
                    "next_action": "After independent review and separate authorization, set RUN_BCD=True in the Kaggle Notebook and Run All.",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    _bootstrap()
    config = json.loads((BASE / "config/frozen.json").read_text(encoding="utf-8"))
    from minicells_r3mf_d1.execution import run_bcd

    try:
        result = run_bcd(
            config,
            preflight_dir,
            args.output,
            _source_sha(),
            device=args.device,
        )
    except Exception:
        _record_run_failure(args.output, _source_sha())
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
