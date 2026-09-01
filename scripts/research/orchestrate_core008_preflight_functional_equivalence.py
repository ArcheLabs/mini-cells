#!/usr/bin/env python3
"""Run the Core 008 preflight bridge in fresh per-seed processes and publish it."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "core-008-preflight-functional-equivalence"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "core-008-preflight-functional-equivalence"
SEEDS = (80721, 80722)


def _run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def _hydrate_seed(seed: int) -> bool:
    src = ARTIFACTS / "seeds" / f"seed-{seed}.json"
    dst = RESULTS / "seeds" / f"seed-{seed}.json"
    if not src.is_file():
        return False
    payload = json.loads(src.read_text(encoding="utf-8"))
    if payload.get("format") != "minicells.core008-preflight.functional-equivalence-seed.v1":
        return False
    if int(payload.get("seed", -1)) != seed:
        return False
    if not bool(payload.get("reproduction", {}).get("matches_reference")):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[core008-preflight] hydrated published seed {seed}", flush=True)
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--branch", default="codex/core-008-preflight-functional-equivalence")
    p.add_argument("--secret-name", default="GITHUB_TOKEN")
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--push-results", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.push_results:
        _run([
            sys.executable,
            "scripts/research/publish_core008_preflight_functional_equivalence.py",
            "--preflight-only",
            "--branch",
            args.branch,
            "--secret-name",
            args.secret_name,
        ])

    RESULTS.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        dst = RESULTS / "seeds" / f"seed-{seed}.json"
        if not args.force:
            if dst.is_file():
                payload = json.loads(dst.read_text(encoding="utf-8"))
                if bool(payload.get("reproduction", {}).get("matches_reference")):
                    print(f"[core008-preflight] seed={seed} already complete; skipping", flush=True)
                    continue
            if _hydrate_seed(seed):
                continue
        _run([
            sys.executable,
            "scripts/research/run_core008_preflight_functional_equivalence_seed.py",
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
        ])

    _run([sys.executable, "scripts/research/report_core008_preflight_functional_equivalence.py"])
    if args.push_results:
        _run([
            sys.executable,
            "scripts/research/publish_core008_preflight_functional_equivalence.py",
            "--push-results",
            "--branch",
            args.branch,
            "--secret-name",
            args.secret_name,
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
