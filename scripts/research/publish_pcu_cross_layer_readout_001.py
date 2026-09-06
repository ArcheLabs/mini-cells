#!/usr/bin/env python3
"""Publish immutable PCU-CROSS-LAYER-READOUT-001 engineering evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

BRANCH = "codex/pcu-composability-kill-001"
RUN_ID = "26090501-l7k64-plus-l23k16"
OUTPUT = Path("artifacts/research/pcu-cross-layer-readout-001/engineering") / RUN_ID
HYBRID = Path("artifacts/research/pcu-hybrid-objective-001/engineering/26090501-l7-k64-rank-plus-ce025")
READOUT = Path("artifacts/research/pcu-readout-localization-001/engineering/26090501-l7-k64-hybrid-readout")
SEED_REGISTRY = Path("research/formal_seed_registry.json")
FORMAL_REGISTRY_SHA = "71a3015a7d54e795538b3aa6750860f0b9168cb3"
FORMAL_SEEDS = (26090511, 26090512, 26090513)
VALID_STATUSES = {
    "SPARSE_CROSS_LAYER_READOUT_RESCUE_SUPPORTED",
    "L23_ONLY_READOUT_SUFFICIENT_CROSS_LAYER_NOT_REQUIRED",
    "CROSS_LAYER_GENERATION_RESCUE_ASSOCIATION_REGRESSED",
    "CROSS_LAYER_READOUT_IMPROVES_BUT_DOES_NOT_RESCUE",
    "MINIMAL_L23_READOUT_DID_NOT_HELP",
}
SAFE_SUFFIXES = {".json", ".md", ".txt", ".csv"}


def run(command: list[str], *, env: dict[str, str] | None = None, capture: bool = False) -> str:
    result = subprocess.run(command, check=True, env=env, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_formal_seeds_untouched() -> None:
    payload = load_json(SEED_REGISTRY)
    states = {int(row["seed"]): str(row["state"]) for row in payload.get("seeds", [])}
    expected = {seed: "RESERVED_UNTOUCHED" for seed in FORMAL_SEEDS}
    if states != expected:
        raise RuntimeError(f"formal seed registry changed: {states}")
    if run(["git", "hash-object", str(SEED_REGISTRY)], capture=True) != FORMAL_REGISTRY_SHA:
        raise RuntimeError("formal seed registry blob changed")


def validate_prerequisites() -> dict:
    hybrid_result = load_json(HYBRID / "RESULT.json")
    hybrid_decision = load_json(HYBRID / "DECISION.json")
    readout_result = load_json(READOUT / "RESULT.json")
    readout_decision = load_json(READOUT / "DECISION.json")
    if hybrid_decision.get("status") != "HYBRID_OBJECTIVE_PRESERVES_ASSOCIATION_GENERATION_UNRESOLVED":
        raise RuntimeError("cross-layer publisher requires published hybrid readout failure")
    if readout_decision.get("status") != "SINGLE_LAYER_GOLD_PREFIX_READOUT_INADEQUATE":
        raise RuntimeError("cross-layer publisher requires published single-layer readout inadequacy")
    if abs(float(hybrid_decision.get("ranking_eval_accuracy", -1)) - 0.8359375) > 1e-12:
        raise RuntimeError("published L7 ranking baseline changed")
    if abs(float(hybrid_decision.get("direct_accuracy", -1)) - 0.03125) > 1e-12:
        raise RuntimeError("published L7 direct baseline changed")
    if abs(float(readout_decision.get("later_token_top1_accuracy", -1)) - 0.535031847133758) > 1e-12:
        raise RuntimeError("published L7 gold-prefix readout baseline changed")
    if readout_decision.get("hybrid_reproduction_exact") is not True:
        raise RuntimeError("published readout diagnostic did not exactly reproduce hybrid")
    selected_l7 = list(hybrid_result.get("selected_cells", []))
    if len(selected_l7) != 64 or list(readout_result.get("selected_cells", [])) != selected_l7:
        raise RuntimeError("published hybrid/readout L7 Cell identity changed")
    dataset_sha = str(hybrid_result["dataset_manifest_sha256"])
    if str(readout_result.get("dataset_manifest_sha256")) != dataset_sha:
        raise RuntimeError("published hybrid/readout datasets differ")
    return {"selected_l7": selected_l7, "dataset_manifest_sha256": dataset_sha}


def validate_final(prereq: dict) -> dict:
    identity = load_json(OUTPUT / "RUN_IDENTITY.json")
    design = load_json(OUTPUT / "DESIGN.json")
    result = load_json(OUTPUT / "RESULT.json")
    decision = load_json(OUTPUT / "DECISION.json")
    if identity.get("experiment") != "PCU-CROSS-LAYER-READOUT-001" or int(identity.get("seed", -1)) != 26090501:
        raise RuntimeError("wrong cross-layer experiment identity")
    if identity.get("formal_execution_not_started") is not True:
        raise RuntimeError("cross-layer identity crossed formal boundary")
    source = identity.get("source", {})
    if source.get("source_dirty") is not False or not source.get("source_commit") or not source.get("source_tree"):
        raise RuntimeError("cross-layer evidence lacks clean immutable source provenance")

    if design.get("causal_question") != "does_a_minimal_late_readout_footprint_rescue_frozen_L7_association":
        raise RuntimeError("cross-layer causal question changed")
    association = design.get("association_state", {})
    readout = design.get("readout_state", {})
    if int(association.get("layer", -1)) != 7 or int(association.get("selected_k", -1)) != 64:
        raise RuntimeError("cross-layer association footprint changed")
    if list(association.get("selected_cells", [])) != prereq["selected_l7"]:
        raise RuntimeError("cross-layer L7 Cell identity changed")
    if int(readout.get("layer", -1)) != 23 or int(readout.get("selected_k", -1)) != 16:
        raise RuntimeError("cross-layer L23 footprint changed")
    if readout.get("objective") != "answer-token-causal-cross-entropy":
        raise RuntimeError("cross-layer L23 readout objective changed")
    if readout.get("allocation") != "first64_A_train_answer_CE_gradient_under_frozen_L7_state":
        raise RuntimeError("cross-layer L23 allocation semantics changed")
    if int(readout.get("max_optimizer_steps", -1)) != 128 or int(readout.get("effective_batch_size", -1)) != 8:
        raise RuntimeError("cross-layer readout training budget changed")

    if result.get("valid_run") is not True or result.get("formal_execution_not_started") is not True:
        raise RuntimeError("cross-layer result is not valid pre-formal evidence")
    if result.get("scientific_evidence") is not False:
        raise RuntimeError("cross-layer engineering diagnostic mislabeled as formal evidence")
    if str(result.get("dataset_manifest_sha256")) != prereq["dataset_manifest_sha256"]:
        raise RuntimeError("cross-layer dataset changed")
    if result.get("status") not in VALID_STATUSES or decision.get("status") != result.get("status"):
        raise RuntimeError("cross-layer decision/result status mismatch")
    if result.get("l7_reproduction", {}).get("exact") is not True or decision.get("l7_reproduction_exact") is not True:
        raise RuntimeError("cross-layer run did not exactly reproduce L7 baseline")

    allocation = result.get("l23_allocation", {})
    selected_l23 = list(allocation.get("selected", []))
    if int(allocation.get("selected_k", -1)) != 16 or len(selected_l23) != 16:
        raise RuntimeError("cross-layer L23 allocation is not exact K16")
    if allocation.get("state") != "frozen_L7_hybrid":
        raise RuntimeError("L23 allocation was not measured under frozen L7 state")
    cross = result.get("cross_layer_arm", {})
    control = result.get("l23_only_control", {})
    if list(cross.get("selected_l23", [])) != selected_l23:
        raise RuntimeError("cross-layer arm did not use registered L23 cells")
    if list(control.get("selected_l23", [])) != selected_l23:
        raise RuntimeError("L23-only matched-footprint control changed cells")
    if decision.get("l23_selected_once_and_reused") is not True:
        raise RuntimeError("decision did not certify matched L23 footprint")
    if list(decision.get("selected_l23", [])) != selected_l23:
        raise RuntimeError("decision L23 Cell identity differs")

    for arm, label in ((cross, "cross-layer"), (control, "L23-only")):
        training = arm.get("training", {})
        if int(training.get("training_steps", -1)) != 128 or int(training.get("batch_size", -1)) != 8:
            raise RuntimeError(f"{label} training budget changed")
        if list(training.get("selected_cells", [])) != selected_l23:
            raise RuntimeError(f"{label} training Cell identity drifted")
    comparison = result.get("comparison", {})
    if abs(float(comparison.get("l7_only_direct", -1)) - 0.03125) > 1e-12:
        raise RuntimeError("comparison L7 baseline changed")
    if abs(float(comparison.get("direct_synergy_floor", -1)) - 0.30) > 1e-12:
        raise RuntimeError("cross-layer synergy floor changed")
    return decision


def assert_prerequisites_published(branch: str) -> None:
    run(["git", "fetch", "origin"])
    for path in (HYBRID / "DECISION.json", READOUT / "DECISION.json"):
        probe = subprocess.run(
            ["git", "show", f"origin/{branch}:{path}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode != 0:
            raise RuntimeError(f"required prerequisite is not published remotely: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()
    if run(["git", "branch", "--show-current"], capture=True) != args.branch:
        raise RuntimeError(f"expected branch {args.branch}")
    assert_formal_seeds_untouched()
    prereq = validate_prerequisites()
    decision = validate_final(prereq)
    assert_prerequisites_published(args.branch)

    remote_probe = subprocess.run(
        ["git", "show", f"origin/{args.branch}:{OUTPUT}/DECISION.json"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if remote_probe.returncode == 0:
        raise RuntimeError(f"{OUTPUT} already exists remotely; refusing to overwrite")

    paths = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.suffix.lower() in SAFE_SUFFIXES)
    if not paths:
        raise RuntimeError("no cross-layer evidence found")
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
    run(["git", "commit", "-m", f"artifacts: publish PCU-CROSS-LAYER-READOUT-001 {RUN_ID}"])
    run(["git", "rebase", f"origin/{args.branch}"])
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")
    askpass = Path("/tmp/minicells-pcu-cross-layer-askpass.sh")
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
