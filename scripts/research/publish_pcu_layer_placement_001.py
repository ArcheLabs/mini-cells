#!/usr/bin/env python3
"""Publish immutable PCU-LAYER-PLACEMENT-001 engineering evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

BRANCH = "codex/pcu-composability-kill-001"
RUN_ID = "26090501-layer-only"
OUTPUT = Path("artifacts/research/pcu-layer-placement-001/engineering") / RUN_ID
SEED_REGISTRY = Path("research/formal_seed_registry.json")
FORMAL_SEEDS = (26090511, 26090512, 26090513)
SAFE_SUFFIXES = {".json", ".md", ".txt", ".csv"}


def run(command: list[str], *, env: dict[str, str] | None = None, capture: bool = False) -> str:
    result = subprocess.run(command, check=True, env=env, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def assert_formal_seeds_untouched() -> None:
    payload = json.loads(SEED_REGISTRY.read_text(encoding="utf-8"))
    states = {int(row["seed"]): str(row["state"]) for row in payload.get("seeds", [])}
    expected = {seed: "RESERVED_UNTOUCHED" for seed in FORMAL_SEEDS}
    if states != expected:
        raise RuntimeError(f"formal seed registry changed: {states}")


def validate() -> dict:
    required = ("RUN_IDENTITY.json", "DESIGN.json", "DECISION.json")
    for name in required:
        if not (OUTPUT / name).is_file():
            raise RuntimeError(f"missing layer-placement artifact: {name}")
    layers = sorted(OUTPUT.glob("LAYER_*.json"))
    if len(layers) != 2:
        raise RuntimeError(f"expected exactly two new layer results, found {len(layers)}")
    identity = json.loads((OUTPUT / "RUN_IDENTITY.json").read_text(encoding="utf-8"))
    decision = json.loads((OUTPUT / "DECISION.json").read_text(encoding="utf-8"))
    if identity.get("experiment") != "PCU-LAYER-PLACEMENT-001":
        raise RuntimeError("wrong experiment identity")
    if identity.get("phase") != "engineering_diagnostic":
        raise RuntimeError("layer-placement evidence is not engineering diagnostic")
    if int(identity.get("seed", -1)) != 26090501:
        raise RuntimeError("unexpected engineering seed")
    if identity.get("formal_execution_not_started") is not True:
        raise RuntimeError("formal execution boundary was crossed")
    source = identity.get("source", {})
    if source.get("source_dirty") is not False or not source.get("source_commit") or not source.get("source_tree"):
        raise RuntimeError("layer-placement evidence lacks clean immutable source provenance")
    if decision.get("valid_run") is not True or decision.get("scientific_evidence") is not False:
        raise RuntimeError("layer-placement decision is not valid engineering evidence")
    if decision.get("formal_execution_not_started") is not True:
        raise RuntimeError("decision crossed formal execution boundary")
    if decision.get("status") not in {
        "LAYER_PLACEMENT_RESCUES_LOCAL_CELL_MUTATION",
        "LAYER_PLACEMENT_DID_NOT_RESCUE",
    }:
        raise RuntimeError(f"unexpected layer-placement status: {decision.get('status')}")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()
    if run(["git", "branch", "--show-current"], capture=True) != args.branch:
        raise RuntimeError(f"expected branch {args.branch}")
    assert_formal_seeds_untouched()
    decision = validate()
    run(["git", "fetch", "origin"])
    remote_probe = subprocess.run(
        ["git", "show", f"origin/{args.branch}:{OUTPUT}/DECISION.json"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if remote_probe.returncode == 0:
        raise RuntimeError(f"{OUTPUT} already exists remotely; refusing to overwrite")

    paths = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.suffix.lower() in SAFE_SUFFIXES)
    if not paths:
        raise RuntimeError("no lightweight layer-placement evidence found")
    run(["git", "reset"])
    run(["git", "add", "-f", "--", *[str(path) for path in paths]])
    staged = run(["git", "diff", "--cached", "--name-only"], capture=True).splitlines()
    prefix = str(OUTPUT) + "/"
    if any(not path.startswith(prefix) for path in staged):
        raise RuntimeError(f"unexpected staged files: {staged}")
    if str(SEED_REGISTRY) in staged:
        raise RuntimeError("formal seed registry must never be staged")
    changed = run(["git", "diff", "HEAD", "--name-only"], capture=True).splitlines()
    if any(not path.startswith(prefix) for path in changed):
        raise RuntimeError(f"tracked source changes exist during publication: {changed}")

    run(["git", "config", "user.name", "MiniCells Research"])
    run(["git", "config", "user.email", "research@minicells.local"])
    run(["git", "commit", "-m", f"artifacts: publish PCU-LAYER-PLACEMENT-001 {RUN_ID}"])
    run(["git", "rebase", f"origin/{args.branch}"])
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")
    askpass = Path("/tmp/minicells-pcu-layer-placement-askpass.sh")
    askpass.write_text(
        '#!/bin/sh\ncase "$1" in\n  *Username*) echo "x-access-token" ;;\n  *) echo "$GITHUB_TOKEN" ;;\nesac\n',
        encoding="utf-8",
    )
    askpass.chmod(0o700)
    env = os.environ.copy()
    env["GITHUB_TOKEN"] = token
    env["GIT_ASKPASS"] = str(askpass)
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        run(["git", "push", "https://github.com/ArcheLabs/mini-cells.git", f"HEAD:{args.branch}"], env=env)
    finally:
        askpass.unlink(missing_ok=True)
    assert_formal_seeds_untouched()
    print(json.dumps({
        "published": True,
        "status": decision["status"],
        "commit": run(["git", "rev-parse", "HEAD"], capture=True),
        "formal_seeds": "RESERVED_UNTOUCHED",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
