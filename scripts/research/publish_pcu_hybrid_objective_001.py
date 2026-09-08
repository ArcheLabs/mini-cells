#!/usr/bin/env python3
"""Publish immutable PCU-HYBRID-OBJECTIVE-001 engineering evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

BRANCH = "codex/pcu-composability-kill-001"
RUN_ID = "26090501-l7-k64-rank-plus-ce025"
OUTPUT = Path("artifacts/research/pcu-hybrid-objective-001/engineering") / RUN_ID
BASELINE = Path("artifacts/research/pcu-objective-alignment-001/engineering/26090501-l7-k64-ranking")
SEED_REGISTRY = Path("research/formal_seed_registry.json")
FORMAL_REGISTRY_SHA = "71a3015a7d54e795538b3aa6750860f0b9168cb3"
FORMAL_SEEDS = (26090511, 26090512, 26090513)
VALID_STATUSES = {
    "HYBRID_OBJECTIVE_RESCUES_ASSOCIATION_AND_GENERATION",
    "HYBRID_OBJECTIVE_PRESERVES_ASSOCIATION_GENERATION_UNRESOLVED",
    "HYBRID_OBJECTIVE_RESCUES_GENERATION_ASSOCIATION_REGRESSED",
    "HYBRID_OBJECTIVE_DID_NOT_JOINTLY_RESCUE",
}
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
    if run(["git", "hash-object", str(SEED_REGISTRY)], capture=True) != FORMAL_REGISTRY_SHA:
        raise RuntimeError("formal seed registry blob changed")


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_baseline() -> dict:
    decision = load_json(BASELINE / "DECISION.json")
    result = load_json(BASELINE / "RESULT.json")
    ce = load_json(BASELINE / "PAIRED_CE_K64.json")
    if decision.get("status") != "ASSOCIATION_LEARNED_GENERATION_UNRESOLVED":
        raise RuntimeError("hybrid publisher requires objective-alignment association-learned baseline")
    if abs(float(decision.get("ranking_eval_accuracy", -1)) - 0.8203125) > 1e-12:
        raise RuntimeError("objective-alignment ranking eval changed")
    if abs(float(decision.get("direct_accuracy", -1)) - 0.0) > 1e-12:
        raise RuntimeError("objective-alignment greedy baseline changed")
    if abs(float(ce.get("direct_accuracy", -1)) - 0.265625) > 1e-12:
        raise RuntimeError("paired CE direct baseline changed")
    selected = list(ce.get("selected_cells", []))
    if len(selected) != 64 or list(result.get("selected_cells", [])) != selected:
        raise RuntimeError("objective baselines do not share exact K64 Cells")
    return {
        "selected": selected,
        "dataset_manifest_sha256": str(ce["dataset_manifest_sha256"]),
    }


def validate_final(baseline: dict) -> dict:
    identity = load_json(OUTPUT / "RUN_IDENTITY.json")
    design = load_json(OUTPUT / "DESIGN.json")
    result = load_json(OUTPUT / "RESULT.json")
    decision = load_json(OUTPUT / "DECISION.json")
    if identity.get("experiment") != "PCU-HYBRID-OBJECTIVE-001" or int(identity.get("seed", -1)) != 26090501:
        raise RuntimeError("wrong hybrid experiment identity")
    if identity.get("formal_execution_not_started") is not True:
        raise RuntimeError("hybrid identity crossed formal boundary")
    source = identity.get("source", {})
    if source.get("source_dirty") is not False or not source.get("source_commit") or not source.get("source_tree"):
        raise RuntimeError("hybrid evidence lacks clean immutable source provenance")
    if design.get("causal_variable") != "ce_readout_regularizer_weight_only":
        raise RuntimeError("hybrid design changed more than the CE regularizer")
    fixed = design.get("fixed", {})
    changed = design.get("changed", {})
    if int(fixed.get("target_layer", -1)) != 7 or int(fixed.get("selected_k", -1)) != 64:
        raise RuntimeError("hybrid design moved layer or K")
    if list(fixed.get("selected_cells", [])) != baseline["selected"]:
        raise RuntimeError("hybrid design changed selected Cells")
    if str(fixed.get("dataset_manifest_sha256")) != baseline["dataset_manifest_sha256"]:
        raise RuntimeError("hybrid design changed dataset")
    if changed.get("from") != "ranking_only" or changed.get("to") != "ranking_plus_original_answer_token_ce":
        raise RuntimeError("hybrid objective identity changed")
    if abs(float(changed.get("ce_weight", -1)) - 0.25) > 1e-12:
        raise RuntimeError("hybrid CE weight changed")
    if changed.get("ce_encoding") != "original_task_sequence_encoding":
        raise RuntimeError("hybrid CE no longer matches the original CE control")

    if result.get("valid_run") is not True or result.get("formal_execution_not_started") is not True:
        raise RuntimeError("hybrid result is not valid pre-formal evidence")
    if result.get("scientific_evidence") is not False:
        raise RuntimeError("hybrid engineering diagnostic mislabeled as formal evidence")
    if result.get("status") not in VALID_STATUSES or decision.get("status") != result.get("status"):
        raise RuntimeError("hybrid decision/result status mismatch")
    if list(result.get("selected_cells", [])) != baseline["selected"]:
        raise RuntimeError("hybrid runtime allocation drifted")
    if str(result.get("dataset_manifest_sha256")) != baseline["dataset_manifest_sha256"]:
        raise RuntimeError("hybrid runtime dataset changed")
    training = result.get("training", {})
    if training.get("objective") != "ranking-plus-original-answer-token-ce":
        raise RuntimeError("hybrid training objective artifact changed")
    if abs(float(training.get("ce_weight", -1)) - 0.25) > 1e-12:
        raise RuntimeError("hybrid training CE weight changed")
    if int(training.get("training_steps", -1)) != 128 or int(training.get("batch_size", -1)) != 8:
        raise RuntimeError("hybrid training budget changed")
    if list(training.get("selected_cells", [])) != baseline["selected"]:
        raise RuntimeError("hybrid training allocation drifted")
    if decision.get("selected_cells_exact_baseline_match") is not True:
        raise RuntimeError("hybrid decision did not certify K64 identity")
    return decision


def assert_baseline_published(branch: str) -> None:
    run(["git", "fetch", "origin"])
    remote = f"origin/{branch}:{BASELINE}/DECISION.json"
    probe = subprocess.run(["git", "show", remote], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode != 0:
        raise RuntimeError("objective-alignment prerequisite is not published remotely")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()
    if run(["git", "branch", "--show-current"], capture=True) != args.branch:
        raise RuntimeError(f"expected branch {args.branch}")
    assert_formal_seeds_untouched()
    baseline = validate_baseline()
    decision = validate_final(baseline)
    assert_baseline_published(args.branch)

    remote_probe = subprocess.run(
        ["git", "show", f"origin/{args.branch}:{OUTPUT}/DECISION.json"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if remote_probe.returncode == 0:
        raise RuntimeError(f"{OUTPUT} already exists remotely; refusing to overwrite")

    paths = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.suffix.lower() in SAFE_SUFFIXES)
    if not paths:
        raise RuntimeError("no hybrid evidence found")
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
    run(["git", "commit", "-m", f"artifacts: publish PCU-HYBRID-OBJECTIVE-001 {RUN_ID}"])
    run(["git", "rebase", f"origin/{args.branch}"])
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")
    askpass = Path("/tmp/minicells-pcu-hybrid-askpass.sh")
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
