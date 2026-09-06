#!/usr/bin/env python3
"""Publish one immutable PCU-KILL-001 engineering run.

Engineering reruns may reuse the registered development seed after a testbed or
implementation repair, but every published run gets a distinct run-id. Existing
engineering evidence is never overwritten. Formal/frozen and binary runtime
artifacts are never staged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

BRANCH = "codex/pcu-composability-kill-001"
ENGINEERING_SEED = 26090501
DEFAULT_RUN_ID = "26090501-oracle-v2"
POSITIVE_CONTROL_VERSION = "pcu-kill-001-context-oracle-v2"
FORMAL_SEEDS = (26090511, 26090512, 26090513)
OUTPUT_ROOT = Path("artifacts/research/pcu-kill-001/engineering")
SEED_REGISTRY = Path("research/formal_seed_registry.json")
SAFE_SUFFIXES = {".json", ".csv", ".md", ".txt"}
FORBIDDEN_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".npy", ".npz"}


def run(command: list[str], *, env: dict[str, str] | None = None, capture: bool = False) -> str:
    result = subprocess.run(command, check=True, env=env, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def output_for_run(run_id: str) -> Path:
    if not re.fullmatch(r"26090501-[a-z0-9][a-z0-9-]*", run_id):
        raise ValueError("engineering run-id must be 26090501-<lowercase-tag>")
    return OUTPUT_ROOT / run_id


def assert_formal_seeds_untouched() -> None:
    payload = json.loads(SEED_REGISTRY.read_text(encoding="utf-8"))
    rows = {int(row["seed"]): str(row["state"]) for row in payload.get("seeds", [])}
    expected = {seed: "RESERVED_UNTOUCHED" for seed in FORMAL_SEEDS}
    if rows != expected:
        raise RuntimeError(f"formal seed registry changed: expected {expected}, got {rows}")


def validate_engineering_evidence(output: Path) -> dict:
    decision_path = output / "ENGINEERING_DECISION.json"
    identity_path = output / "RUN_IDENTITY.json"
    run_manifest_path = output / "RUN_MANIFEST.json"
    if not decision_path.is_file():
        raise RuntimeError("ENGINEERING_DECISION.json is required before publication")
    if not identity_path.is_file():
        raise RuntimeError("RUN_IDENTITY.json is required before publication")

    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("phase") != "engineering" or int(identity.get("seed", -1)) != ENGINEERING_SEED:
        raise RuntimeError("run identity is not the registered engineering seed")
    if identity.get("run_id") != output.name:
        raise RuntimeError("run identity does not match output directory")
    if identity.get("positive_control_version") != POSITIVE_CONTROL_VERSION:
        raise RuntimeError("run does not use the registered context-oracle v2")
    if identity.get("formal_execution_not_started") is not True:
        raise RuntimeError("run identity crossed the formal execution boundary")
    source = identity.get("source")
    if not isinstance(source, dict) or source.get("source_dirty") is not False:
        raise RuntimeError("refusing to publish engineering evidence produced from a dirty source tree")
    if not source.get("source_commit") or not source.get("source_tree"):
        raise RuntimeError("run identity is missing immutable source commit/tree provenance")

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("phase") != "engineering":
        raise RuntimeError("refusing to publish a non-engineering decision")
    if decision.get("scientific_evidence") is not False:
        raise RuntimeError("engineering evidence must not be labelled scientific/formal evidence")
    if decision.get("formal_execution_not_started") is not True:
        raise RuntimeError("formal execution boundary was not preserved")
    if decision.get("valid_run") is not True:
        raise RuntimeError("refusing to publish an invalid engineering run")
    if decision.get("status") in {None, "REAL_GRANITE_E0_NOT_RUN", "FORMAL_EXECUTION_FAILED"}:
        raise RuntimeError(f"engineering run is not interpretable: {decision.get('status')}")

    # Every v2 run must preserve the raw gates even if it killed before training.
    for required in (
        "MODEL_MANIFEST.json",
        "DATASET_MANIFEST.json",
        "DATASET_AUDIT.json",
        "EQUIVALENCE.json",
        "CACHE_EQUIVALENCE.json",
        "CONTEXT_ORACLE.json",
    ):
        if not (output / required).is_file():
            raise RuntimeError(f"missing fail-fast audit artifact: {required}")

    if run_manifest_path.is_file():
        manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("phase") != "engineering":
            raise RuntimeError("refusing to publish a non-engineering run manifest")
        if int(manifest.get("seed", -1)) != ENGINEERING_SEED:
            raise RuntimeError("refusing to publish an unexpected engineering seed")
        if manifest.get("backend") != "granite":
            raise RuntimeError("refusing to publish a non-Granite PCU E0 run")
    else:
        foundation = decision.get("foundation", {})
        if foundation.get("model_repo") != "ibm-granite/granite-3.1-1b-a400m-base":
            raise RuntimeError("early-kill decision is not bound to the registered Granite foundation")
    return decision


def stage_engineering_evidence(output: Path) -> list[str]:
    assert_formal_seeds_untouched()
    decision = validate_engineering_evidence(output)
    paths = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.suffix.lower() in SAFE_SUFFIXES
    )
    if not paths:
        raise RuntimeError("no lightweight PCU engineering evidence found")

    forbidden_present = sorted(
        str(path) for path in output.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    )
    if forbidden_present:
        print("Binary runtime artifacts remain Kaggle-only and will not be staged:\n  " + "\n  ".join(forbidden_present))

    run(["git", "reset"])
    run(["git", "add", "-f", "--", *[str(path) for path in paths]])
    staged = run(["git", "diff", "--cached", "--name-only"], capture=True).splitlines()
    prefix = str(output) + "/"
    unexpected = [path for path in staged if not path.startswith(prefix)]
    if unexpected:
        raise RuntimeError("unexpected staged files: " + ", ".join(unexpected))
    forbidden = [path for path in staged if Path(path).suffix.lower() in FORBIDDEN_SUFFIXES]
    if forbidden:
        raise RuntimeError("binary PCU artifacts must not enter Git: " + ", ".join(forbidden))
    if str(SEED_REGISTRY) in staged:
        raise RuntimeError("formal seed registry must never be staged by engineering publisher")

    changed = run(["git", "diff", "HEAD", "--name-only"], capture=True).splitlines()
    unexpected_changed = [path for path in changed if not path.startswith(prefix)]
    if unexpected_changed:
        raise RuntimeError(
            "working tree contains tracked changes outside the engineering evidence prefix: "
            + ", ".join(unexpected_changed)
        )
    print(json.dumps({
        "run_id": output.name,
        "engineering_status": decision.get("status"),
        "staged_files": staged,
        "formal_seeds": "RESERVED_UNTOUCHED",
    }, indent=2))
    return staged


def remote_already_has_run(branch: str, output: Path) -> bool:
    remote_ref = f"origin/{branch}:{output}/ENGINEERING_DECISION.json"
    probe = subprocess.run(
        ["git", "show", remote_ref],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return probe.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()
    output = output_for_run(args.run_id)

    current = run(["git", "branch", "--show-current"], capture=True)
    if current != args.branch:
        raise RuntimeError(f"expected branch {args.branch}, got {current}")
    run(["git", "fetch", "origin"])
    if remote_already_has_run(args.branch, output):
        raise RuntimeError(f"engineering evidence {args.run_id} already exists remotely; refusing to overwrite it")

    staged = stage_engineering_evidence(output)
    if not staged:
        raise RuntimeError("nothing staged for PCU engineering publication")
    run(["git", "config", "user.name", "MiniCells Research"])
    run(["git", "config", "user.email", "research@minicells.local"])
    run(["git", "commit", "-m", f"artifacts: publish PCU-KILL-001 {args.run_id} evidence"])
    run(["git", "rebase", f"origin/{args.branch}"])

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")
    askpass = Path("/tmp/minicells-pcu-kill-001-askpass.sh")
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

    print(json.dumps({
        "published": True,
        "branch": args.branch,
        "commit": run(["git", "rev-parse", "HEAD"], capture=True),
        "engineering_seed": ENGINEERING_SEED,
        "run_id": args.run_id,
        "formal_execution_not_started": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
