#!/usr/bin/env python3
"""Run the frozen Core Validation 009A right-collapse bridge with checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "core-validation-009a-right-collapse-bridge"
ARTIFACTS = ROOT / "artifacts" / "experiments" / "core-validation-009a-right-collapse-bridge"
VALIDATION = ROOT / "research" / "validations" / "core-009a-right-collapse-bridge"
PROTOCOL = VALIDATION / "protocol.json"
FORMAT = "minicells.core-validation.009a-right-collapse-bridge-seed.v1"
DEFAULT_BRANCH = "codex/core-validation-009a-right-collapse-bridge"


def _run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def _protocol_sha() -> str:
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def _valid(path: Path, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        payload.get("format") == FORMAT
        and int(payload.get("seed", -1)) == seed
        and payload.get("protocol_sha256") == _protocol_sha()
        and payload.get("scientific_decision") is False
        and payload.get("source_009a_status_changed") is False
        and payload.get("source_009a_reproduction", {}).get("pass") is True
    )


def _hydrate(seed: int) -> bool:
    src = ARTIFACTS / "seeds" / f"seed-{seed}.json"
    dst = RESULTS / "seeds" / f"seed-{seed}.json"
    if not _valid(src, seed):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[core009a-bridge] hydrated published seed {seed}", flush=True)
    return True


def _report() -> dict:
    _run([sys.executable, "scripts/research/report_core_validation_009a_bridge.py"])
    return json.loads((RESULTS / "decision.json").read_text(encoding="utf-8"))


def _publish(branch: str, secret_name: str) -> None:
    _run(
        [
            sys.executable,
            "scripts/research/publish_core_validation_009a_bridge.py",
            "--push-results",
            "--branch",
            branch,
            "--secret-name",
            secret_name,
        ]
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--branch", default=DEFAULT_BRANCH)
    p.add_argument("--secret-name", default="GITHUB_TOKEN")
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--push-results", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    protocol = _protocol()
    seeds = [int(x) for x in protocol["replication"]["diagnostic_seeds"]]
    if args.push_results:
        _run(
            [
                sys.executable,
                "scripts/research/publish_core_validation_009a_bridge.py",
                "--preflight-only",
                "--branch",
                args.branch,
                "--secret-name",
                args.secret_name,
            ]
        )

    for seed in seeds:
        dst = RESULTS / "seeds" / f"seed-{seed}.json"
        complete = False
        if not args.force:
            complete = _valid(dst, seed) or _hydrate(seed)
        if complete:
            print(f"[core009a-bridge] seed={seed} complete; skipping", flush=True)
        else:
            _run(
                [
                    sys.executable,
                    "scripts/research/run_core_validation_009a_bridge_seed.py",
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                ]
            )
        decision = _report()
        if args.push_results:
            _publish(args.branch, args.secret_name)

    decision = json.loads((RESULTS / "decision.json").read_text(encoding="utf-8"))
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
