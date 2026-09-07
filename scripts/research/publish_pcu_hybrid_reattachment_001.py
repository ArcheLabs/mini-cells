#!/usr/bin/env python3
"""Validate and publish immutable PCU-HYBRID-REATTACHMENT-001 v3 evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


BRANCH = "codex/pcu-hybrid-reattachment-001"
RUN_ID = "26090501-l7-k64-ranking-causal-reattach-v3"
OUTPUT = Path("artifacts/research/pcu-hybrid-reattachment-001/engineering") / RUN_ID
SEED_REGISTRY = Path("research/formal_seed_registry.json")
FORMAL_REGISTRY_SHA = "71a3015a7d54e795538b3aa6750860f0b9168cb3"
FORMAL_SEEDS = (26090511, 26090512, 26090513)
SAFE_SUFFIXES = {".json", ".md", ".txt", ".csv", ".png"}
REQUIRED_FILES = {
    "DESIGN.json",
    "RUN_IDENTITY.json",
    "RESULT.json",
    "DECISION.json",
    "PRIMARY_RESULT.json",
    "PRIMARY_DECISION.json",
    "SWEEP_RESULT.json",
    "SWEEP_DECISION.json",
    "AMPLITUDE_SWEEP.csv",
    "REPORT.md",
    "equivalence_diffs.png",
    "causal_ranking_on_off.png",
    "alpha_sweep_tradeoff.png",
    "association_locality_pareto.png",
}
VALID_STATUSES = {
    "HYBRID_REATTACHMENT_SUPPORTED_AT_ALPHA_1",
    "HYBRID_REATTACHMENT_SUPPORTED_WITH_BOUNDED_AMPLITUDE",
    "HYBRID_CAUSAL_CONSUMPTION_SUPPORTED_LOCALITY_UNRESOLVED",
    "SAME_GRAPH_ZERO_STATE_EQUIVALENCE_FAILED",
    "REVERSIBILITY_FAILED",
    "CAUSAL_EXPRESSION_PRESENT_GATES_UNRESOLVED",
    "NO_CAUSAL_EXPRESSION_ENGINEERING",
}


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


def validate_final() -> dict:
    missing = sorted(name for name in REQUIRED_FILES if not (OUTPUT / name).is_file())
    if missing:
        raise RuntimeError(f"missing v3 evidence files: {missing}")
    identity = load_json(OUTPUT / "RUN_IDENTITY.json")
    design = load_json(OUTPUT / "DESIGN.json")
    result = load_json(OUTPUT / "RESULT.json")
    decision = load_json(OUTPUT / "DECISION.json")
    primary = load_json(OUTPUT / "PRIMARY_RESULT.json")
    primary_decision = load_json(OUTPUT / "PRIMARY_DECISION.json")
    sweep = load_json(OUTPUT / "SWEEP_RESULT.json")
    sweep_decision = load_json(OUTPUT / "SWEEP_DECISION.json")

    for payload, label in (
        (identity, "identity"),
        (design, "design"),
        (result, "result"),
        (decision, "decision"),
        (primary, "primary"),
        (primary_decision, "primary_decision"),
        (sweep, "sweep"),
        (sweep_decision, "sweep_decision"),
    ):
        if payload.get("experiment") != "PCU-HYBRID-REATTACHMENT-001":
            raise RuntimeError(f"{label} has wrong experiment identity")
        if int(payload.get("protocol_version", -1)) != 3:
            raise RuntimeError(f"{label} is not protocol v3")
        if payload.get("formal_execution_not_started") is not True:
            raise RuntimeError(f"{label} crossed formal boundary")
        if payload.get("scientific_evidence") is not False:
            raise RuntimeError(f"{label} incorrectly claims formal scientific evidence")

    source = identity.get("source", {})
    if source.get("source_dirty") is not False or not source.get("source_commit") or not source.get("source_tree"):
        raise RuntimeError("run identity lacks clean immutable provenance")
    if identity.get("dual_gpu_execution_required") is not True:
        raise RuntimeError("v3 identity lost dual-GPU requirement")
    if identity.get("worker_devices") != {"primary": "cuda:0", "sweep": "cuda:1"}:
        raise RuntimeError("v3 worker-device mapping changed")
    for payload, label in ((primary, "primary"), (sweep, "sweep")):
        worker_source = payload.get("source", {})
        if worker_source.get("source_commit") != source.get("source_commit") or worker_source.get("source_tree") != source.get("source_tree"):
            raise RuntimeError(f"{label} worker provenance differs from orchestrator")

    amendment = design.get("protocol_amendment", {})
    if amendment.get("thresholds_changed") is not False:
        raise RuntimeError("protocol-v3 may not relax thresholds after v2 observation")
    if amendment.get("strict_zero_state_gate") != "PARENT_ZERO_DELTA_vs_CELL_OFF_same_cellular_graph":
        raise RuntimeError("protocol-v3 same-graph equivalence gate changed")
    thresholds = design.get("thresholds", {})
    expected_thresholds = {
        "same_graph_equivalence_max_abs_logit_diff": 1e-5,
        "restoration_max_abs_logit_diff": 1e-5,
        "association_floor": 0.8,
        "minimum_causal_ranking_gain": 0.5,
        "maximum_B_control_answer_nll_increase": 0.1,
    }
    for key, expected in expected_thresholds.items():
        if abs(float(thresholds.get(key, -999.0)) - expected) > 1e-12:
            raise RuntimeError(f"predeclared threshold changed: {key}")
    dual = design.get("dual_gpu_execution", {})
    if dual.get("required") is not True or dual.get("process_isolation") is not True:
        raise RuntimeError("dual-GPU process isolation is not certified")

    if result.get("valid_run") is not True or primary.get("valid_run") is not True or sweep.get("valid_run") is not True:
        raise RuntimeError("publisher refuses invalid/replay-mismatched v3 evidence")
    if result.get("status") not in VALID_STATUSES or decision.get("status") != result.get("status"):
        raise RuntimeError("v3 result/decision status mismatch or unknown status")
    if decision.get("formal_decision") != "RESERVED_UNRUN":
        raise RuntimeError("engineering publisher may not emit a formal decision")
    if decision.get("dual_gpu_execution_required") is not True:
        raise RuntimeError("decision lost dual-GPU requirement")
    if decision.get("new_bridge_used") is not False or decision.get("new_router_used") is not False:
        raise RuntimeError("v3 unexpectedly introduced a bridge/router")
    if decision.get("additional_training_for_amplitude_sweep") is not False:
        raise RuntimeError("amplitude sweep must not add training")
    if primary.get("selected_cells") != sweep.get("selected_cells"):
        raise RuntimeError("primary/sweep Cell identity differs")
    if primary.get("dataset_manifest_sha256") != sweep.get("dataset_manifest_sha256"):
        raise RuntimeError("primary/sweep dataset identity differs")
    if sweep.get("alpha_grid") != [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]:
        raise RuntimeError("predeclared amplitude grid changed")
    if sweep.get("additional_training_after_replay") is not False:
        raise RuntimeError("sweep added optimizer steps after replay")
    return decision


def assert_prerequisite_published(branch: str) -> None:
    run(["git", "fetch", "origin"])
    prerequisite = Path(
        "artifacts/research/pcu-objective-alignment-001/engineering/26090501-l7-k64-ranking/DECISION.json"
    )
    probe = subprocess.run(
        ["git", "show", f"origin/{branch}:{prerequisite}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"required objective-alignment prerequisite not published: {prerequisite}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()
    if run(["git", "branch", "--show-current"], capture=True) != args.branch:
        raise RuntimeError(f"expected branch {args.branch}")
    assert_formal_seeds_untouched()
    decision = validate_final()
    assert_prerequisite_published(args.branch)

    remote_probe = subprocess.run(
        ["git", "show", f"origin/{args.branch}:{OUTPUT}/DECISION.json"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if remote_probe.returncode == 0:
        raise RuntimeError(f"{OUTPUT} already exists remotely; refusing to overwrite")

    paths = sorted(
        path for path in OUTPUT.rglob("*")
        if path.is_file() and path.suffix.lower() in SAFE_SUFFIXES
    )
    if not paths:
        raise RuntimeError("no v3 evidence found")
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
    run(["git", "commit", "-m", f"artifacts: publish PCU-HYBRID-REATTACHMENT-001 {RUN_ID}"])
    run(["git", "rebase", f"origin/{args.branch}"])
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")
    askpass = Path("/tmp/minicells-pcu-hybrid-reattach-v3-askpass.sh")
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
