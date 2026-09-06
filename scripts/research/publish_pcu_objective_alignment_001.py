#!/usr/bin/env python3
"""Publish immutable PCU-OBJECTIVE-ALIGNMENT-001 engineering evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

BRANCH = "codex/pcu-composability-kill-001"
RUN_ID = "26090501-l7-k64-ranking"
OUTPUT = Path("artifacts/research/pcu-objective-alignment-001/engineering") / RUN_ID
LOCALITY = Path("artifacts/research/pcu-locality-width-001/engineering/26090501-l7-width")
SEED_REGISTRY = Path("research/formal_seed_registry.json")
FORMAL_SEEDS = (26090511, 26090512, 26090513)
FORMAL_REGISTRY_SHA = "71a3015a7d54e795538b3aa6750860f0b9168cb3"
SAFE_SUFFIXES = {".json", ".md", ".txt", ".csv"}
VALID_STATUSES = {
    "OBJECTIVE_ALIGNMENT_RESCUES_LOCAL_CELL_MUTATION",
    "ASSOCIATION_LEARNED_GENERATION_UNRESOLVED",
    "OBJECTIVE_ALIGNMENT_IMPROVES_BUT_DOES_NOT_RESCUE",
    "OBJECTIVE_ALIGNMENT_DID_NOT_RESCUE",
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
    blob = run(["git", "hash-object", str(SEED_REGISTRY)], capture=True)
    if blob != FORMAL_REGISTRY_SHA:
        raise RuntimeError(f"formal seed registry blob changed: {blob}")


def validate_locality_baseline() -> dict:
    for name in ("RUN_IDENTITY.json", "DESIGN.json", "DECISION.json", "WIDTH_064.json"):
        if not (LOCALITY / name).is_file():
            raise RuntimeError(f"missing locality-width prerequisite: {name}")
    decision = json.loads((LOCALITY / "DECISION.json").read_text(encoding="utf-8"))
    width = json.loads((LOCALITY / "WIDTH_064.json").read_text(encoding="utf-8"))
    if decision.get("status") != "LOCALITY_WIDTH_IMPROVES_BUT_DOES_NOT_RESCUE":
        raise RuntimeError("final objective test requires the completed locality-width non-rescue")
    if decision.get("valid_run") is not True or decision.get("formal_execution_not_started") is not True:
        raise RuntimeError("locality-width prerequisite is not valid pre-formal evidence")
    selected = list(width.get("allocation", {}).get("selected", []))
    if len(selected) != 64 or int(width.get("identity", {}).get("selected_k", -1)) != 64:
        raise RuntimeError("locality-width prerequisite is not exact K64")
    if width.get("allocation", {}).get("baseline_prefix_match") is not True:
        raise RuntimeError("locality-width K64 allocation prefix drifted")
    return {
        "selected": selected,
        "direct_accuracy": float(width["direct_accuracy"]),
        "dataset_manifest_sha256": str(width["identity"]["dataset_manifest_sha256"]),
    }


def validate_final(baseline: dict) -> dict:
    for name in ("RUN_IDENTITY.json", "DESIGN.json", "RESULT.json", "DECISION.json"):
        if not (OUTPUT / name).is_file():
            raise RuntimeError(f"missing objective-alignment artifact: {name}")
    identity = json.loads((OUTPUT / "RUN_IDENTITY.json").read_text(encoding="utf-8"))
    design = json.loads((OUTPUT / "DESIGN.json").read_text(encoding="utf-8"))
    result = json.loads((OUTPUT / "RESULT.json").read_text(encoding="utf-8"))
    decision = json.loads((OUTPUT / "DECISION.json").read_text(encoding="utf-8"))

    if identity.get("experiment") != "PCU-OBJECTIVE-ALIGNMENT-001":
        raise RuntimeError("wrong final objective experiment identity")
    if identity.get("phase") != "engineering_diagnostic" or int(identity.get("seed", -1)) != 26090501:
        raise RuntimeError("final objective evidence is not the registered engineering diagnostic")
    if identity.get("formal_execution_not_started") is not True:
        raise RuntimeError("final objective identity crossed formal execution boundary")
    source = identity.get("source", {})
    if source.get("source_dirty") is not False or not source.get("source_commit") or not source.get("source_tree"):
        raise RuntimeError("final objective evidence lacks clean immutable provenance")

    if design.get("causal_variable") != "training_objective_only":
        raise RuntimeError("final objective design changed more than the objective")
    fixed = design.get("fixed", {})
    if int(fixed.get("target_layer", -1)) != 7 or int(fixed.get("selected_k", -1)) != 64:
        raise RuntimeError("final objective design moved layer or K")
    if list(fixed.get("selected_cells", [])) != baseline["selected"]:
        raise RuntimeError("final objective design changed selected Cells")
    if str(fixed.get("dataset_manifest_sha256")) != baseline["dataset_manifest_sha256"]:
        raise RuntimeError("final objective design changed dataset")
    changed = design.get("changed", {})
    if changed.get("from") != "answer-token-causal-cross-entropy":
        raise RuntimeError("final objective baseline loss is not CE")
    if changed.get("to") != "16-way-candidate-ranking-cross-entropy-over-mean-completion-loglikelihood":
        raise RuntimeError("unexpected final objective")
    if int(changed.get("candidate_pool_size", -1)) != 16:
        raise RuntimeError("final objective candidate pool changed")

    if result.get("valid_run") is not True or result.get("scientific_evidence") is not False:
        raise RuntimeError("final objective result is not valid engineering evidence")
    if result.get("formal_execution_not_started") is not True:
        raise RuntimeError("final objective result crossed formal boundary")
    if result.get("status") not in VALID_STATUSES or decision.get("status") != result.get("status"):
        raise RuntimeError("invalid or inconsistent final objective status")
    if list(result.get("selected_cells", [])) != baseline["selected"]:
        raise RuntimeError("final objective runtime allocation differs from K64 baseline")
    if int(result.get("selected_k", -1)) != 64:
        raise RuntimeError("final objective runtime K changed")
    if str(result.get("dataset_manifest_sha256")) != baseline["dataset_manifest_sha256"]:
        raise RuntimeError("final objective runtime dataset changed")
    if abs(float(result.get("baseline", {}).get("ce_direct_accuracy", -1.0)) - baseline["direct_accuracy"]) > 1e-12:
        raise RuntimeError("final objective CE comparison baseline changed")
    if result.get("training", {}).get("objective") != "16-way-candidate-ranking-cross-entropy-over-mean-completion-loglikelihood":
        raise RuntimeError("final objective training artifact reports wrong loss")
    if int(result.get("training", {}).get("training_steps", -1)) != 128:
        raise RuntimeError("final objective did not complete 128 optimizer steps")
    if int(result.get("training", {}).get("batch_size", -1)) != 8:
        raise RuntimeError("final objective effective batch changed")
    if list(result.get("training", {}).get("selected_cells", [])) != baseline["selected"]:
        raise RuntimeError("final objective training allocation drifted")
    if decision.get("selected_cells_exact_baseline_match") is not True:
        raise RuntimeError("final decision did not certify selected-Cell identity")
    return decision


def assert_locality_published(branch: str) -> None:
    run(["git", "fetch", "origin"])
    probe = subprocess.run(
        ["git", "show", f"origin/{branch}:{LOCALITY}/DECISION.json"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "PCU-LOCALITY-WIDTH-001 evidence is not published remotely; publish that prerequisite first"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()
    if run(["git", "branch", "--show-current"], capture=True) != args.branch:
        raise RuntimeError(f"expected branch {args.branch}")
    assert_formal_seeds_untouched()
    baseline = validate_locality_baseline()
    decision = validate_final(baseline)
    assert_locality_published(args.branch)

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
        raise RuntimeError("no lightweight final objective evidence found")
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
    run(["git", "commit", "-m", f"artifacts: publish PCU-OBJECTIVE-ALIGNMENT-001 {RUN_ID}"])
    run(["git", "rebase", f"origin/{args.branch}"])
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")
    askpass = Path("/tmp/minicells-pcu-objective-askpass.sh")
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
