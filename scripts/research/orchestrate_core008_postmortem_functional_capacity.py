#!/usr/bin/env python3
"""Run all Core 008 postmortem capacity seeds in fresh processes and optionally publish."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "core-008-postmortem-functional-capacity"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "core-008-postmortem-functional-capacity"
SEEDS = (80821, 80822, 80823)
FORMAT = "minicells.core008-postmortem.functional-capacity-seed.v1"


def _run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def _valid(path: Path, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("format") == FORMAT and int(payload.get("seed", -1)) == seed and payload.get("scientific_decision") is False


def _hydrate(seed: int) -> bool:
    src = ARTIFACTS / "seeds" / f"seed-{seed}.json"
    dst = RESULTS / "seeds" / f"seed-{seed}.json"
    if not _valid(src, seed):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[core008-postmortem] hydrated published seed {seed}", flush=True)
    return True


def _report() -> None:
    _run([sys.executable, "scripts/research/report_core008_postmortem_functional_capacity.py"])


def _publish(branch: str, secret_name: str) -> None:
    _run([
        sys.executable,
        "scripts/research/publish_core008_postmortem_functional_capacity.py",
        "--push-results",
        "--branch", branch,
        "--secret-name", secret_name,
    ])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--branch", default="codex/core-008-postmortem-functional-capacity")
    p.add_argument("--secret-name", default="GITHUB_TOKEN")
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--push-results", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.push_results:
        _run([
            sys.executable,
            "scripts/research/publish_core008_postmortem_functional_capacity.py",
            "--preflight-only",
            "--branch", args.branch,
            "--secret-name", args.secret_name,
        ])

    RESULTS.mkdir(parents=True, exist_ok=True)
    changed = False
    for seed in SEEDS:
        dst = RESULTS / "seeds" / f"seed-{seed}.json"
        if not args.force and (_valid(dst, seed) or _hydrate(seed)):
            print(f"[core008-postmortem] seed={seed} complete; skipping", flush=True)
            continue
        _run([
            sys.executable,
            "scripts/research/run_core008_postmortem_functional_capacity_seed.py",
            "--seed", str(seed),
            "--device", args.device,
        ])
        changed = True
        # Persist each completed diagnostic seed before attempting the next one.
        if args.push_results:
            _report()
            _publish(args.branch, args.secret_name)

    _report()
    if args.push_results and (changed or not (ARTIFACTS / "decision.json").is_file()):
        _publish(args.branch, args.secret_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
