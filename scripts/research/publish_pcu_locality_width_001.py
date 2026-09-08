#!/usr/bin/env python3
"""Publish immutable PCU-LOCALITY-WIDTH-001 engineering evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

BRANCH = "codex/pcu-composability-kill-001"
RUN_ID = "26090501-l7-width"
OUTPUT = Path("artifacts/research/pcu-locality-width-001/engineering") / RUN_ID
SEED_REGISTRY = Path("research/formal_seed_registry.json")
FORMAL_SEEDS = (26090511, 26090512, 26090513)
SAFE_SUFFIXES = {".json", ".md", ".txt", ".csv"}
VALID_STATUSES = {
    "LOCALITY_WIDTH_RESCUES_LOCAL_CELL_MUTATION",
    "LOCALITY_WIDTH_IMPROVES_BUT_DOES_NOT_RESCUE",
    "LOCALITY_WIDTH_DID_NOT_IMPROVE",
}


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
    for name in ("RUN_IDENTITY.json", "DESIGN.json", "DECISION.json"):
        if not (OUTPUT / name).is_file():
            raise RuntimeError(f"missing locality-width artifact: {name}")
    identity = json.loads((OUTPUT / "RUN_IDENTITY.json").read_text(encoding="utf-8"))
    design = json.loads((OUTPUT / "DESIGN.json").read_text(encoding="utf-8"))
    decision = json.loads((OUTPUT / "DECISION.json").read_text(encoding="utf-8"))
    if identity.get("experiment") != "PCU-LOCALITY-WIDTH-001":
        raise RuntimeError("wrong locality-width experiment identity")
    if identity.get("phase") != "engineering_diagnostic" or int(identity.get("seed", -1)) != 26090501:
        raise RuntimeError("locality-width evidence is not the registered engineering diagnostic")
    if identity.get("formal_execution_not_started") is not True:
        raise RuntimeError("locality-width identity crossed formal execution boundary")
    source = identity.get("source", {})
    if source.get("source_dirty") is not False or not source.get("source_commit") or not source.get("source_tree"):
        raise RuntimeError("locality-width evidence lacks clean immutable source provenance")
    if design.get("causal_variable") != "selected_cell_width_k_only":
        raise RuntimeError("locality-width design changed more than K")
    if int(design.get("fixed", {}).get("target_layer", -1)) != 7:
        raise RuntimeError("locality-width target layer is not frozen at L7")
    if decision.get("valid_run") is not True or decision.get("scientific_evidence") is not False:
        raise RuntimeError("locality-width decision is not valid engineering evidence")
    if decision.get("formal_execution_not_started") is not True:
        raise RuntimeError("locality-width decision crossed formal execution boundary")
    if decision.get("status") not in VALID_STATUSES:
        raise RuntimeError(f"unexpected locality-width status: {decision.get('status')}")

    widths = sorted(OUTPUT.glob("WIDTH_*.json"))
    expected_count = 3 if decision.get("fallback_k64_required") is True else 2
    if len(widths) != expected_count:
        raise RuntimeError(f"expected {expected_count} new width results, found {len(widths)}")
    expected_names = {"WIDTH_016.json", "WIDTH_032.json"}
    if expected_count == 3:
        expected_names.add("WIDTH_064.json")
    if {path.name for path in widths} != expected_names:
        raise RuntimeError(f"unexpected width evidence set: {[path.name for path in widths]}")
    for path in widths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "minicells.pcu-locality-width-001.width-result.v1":
            raise RuntimeError(f"unexpected width schema: {path}")
        if payload.get("formal_execution_not_started") is not True or payload.get("scientific_evidence") is not False:
            raise RuntimeError(f"invalid engineering boundary in {path}")
        if payload.get("allocation", {}).get("baseline_prefix_match") is not True:
            raise RuntimeError(f"allocation prefix drift in {path}")
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
        raise RuntimeError("no lightweight locality-width evidence found")
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
    run(["git", "commit", "-m", f"artifacts: publish PCU-LOCALITY-WIDTH-001 {RUN_ID}"])
    run(["git", "rebase", f"origin/{args.branch}"])
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")
    askpass = Path("/tmp/minicells-pcu-locality-width-askpass.sh")
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
