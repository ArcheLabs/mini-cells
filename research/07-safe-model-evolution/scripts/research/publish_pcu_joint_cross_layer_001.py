#!/usr/bin/env python3
"""Publish immutable PCU-JOINT-CROSS-LAYER-001 engineering evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

BRANCH = "codex/pcu-composability-kill-001"
RUN_ID = "26090501-l15k16-l23k16-joint"
OUTPUT = Path("artifacts/research/pcu-joint-cross-layer-001/engineering") / RUN_ID
DEPTH3_ROOT = Path("artifacts/research/pcu-sparse-path-depth-001/engineering/26090501-depth3-4-5")
SEED_REGISTRY = Path("research/formal_seed_registry.json")
FORMAL_REGISTRY_SHA = "71a3015a7d54e795538b3aa6750860f0b9168cb3"
FORMAL_SEEDS = (26090511, 26090512, 26090513)
VALID_STATUSES = {
    "JOINT_COORDINATION_RESCUES_NATIVE_GENERATION",
    "JOINT_COORDINATION_IMPROVES_BUT_DOES_NOT_RESCUE",
    "EXTRA_JOINT_UPDATES_RESCUE_COORDINATION_ALONE_UNPROVEN",
    "EXTRA_JOINT_UPDATES_IMPROVE_COORDINATION_ALONE_UNPROVEN",
    "JOINT_GENERATION_IMPROVES_ASSOCIATION_REGRESSED",
    "JOINT_COORDINATION_DID_NOT_IMPROVE",
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


def load_depth3_identity() -> dict:
    worker = load_json(DEPTH3_ROOT / "DEPTH_3.json")
    decision = load_json(DEPTH3_ROOT / "DECISION.json")
    if decision.get("status") != "DEEPER_SPARSE_PATH_DID_NOT_IMPROVE":
        raise RuntimeError("joint publisher requires published sparse-path depth decision")
    if worker.get("experiment") != "PCU-SPARSE-PATH-DEPTH-001" or int(worker.get("depth", -1)) != 3:
        raise RuntimeError("wrong sparse-path depth3 identity")
    topology = worker.get("topology", {})
    if topology.get("layers") != [7, 15, 23] or topology.get("transport_k") != [16] or int(topology.get("readout_k", -1)) != 16:
        raise RuntimeError("published depth3 topology changed")
    metrics = worker.get("metrics", {})
    if abs(float(metrics.get("direct_accuracy", -1)) - 0.140625) > 1e-12:
        raise RuntimeError("published depth3 direct baseline changed")
    if abs(float(metrics.get("ranking_eval_accuracy", -1)) - 0.7890625) > 1e-12:
        raise RuntimeError("published depth3 ranking baseline changed")
    stages = worker.get("stages", [])
    if len(stages) != 2:
        raise RuntimeError("published depth3 stages changed")
    stage15, stage23 = stages
    if int(stage15.get("layer", -1)) != 15 or int(stage15.get("selected_k", -1)) != 16:
        raise RuntimeError("published depth3 L15 stage changed")
    if int(stage23.get("layer", -1)) != 23 or int(stage23.get("selected_k", -1)) != 16:
        raise RuntimeError("published depth3 L23 stage changed")
    selected15 = list(stage15.get("selected_cells", []))
    selected23 = list(stage23.get("selected_cells", []))
    if len(selected15) != 16 or len(selected23) != 16:
        raise RuntimeError("published depth3 selected Cell count changed")
    return {
        "selected_l15": selected15,
        "selected_l23": selected23,
        "dataset_manifest_sha256": worker["dataset_manifest_sha256"],
    }


def validate_worker(payload: dict, *, expected_steps: int, expected_arm: str, depth3: dict, source: dict) -> None:
    if payload.get("experiment") != "PCU-JOINT-CROSS-LAYER-001":
        raise RuntimeError(f"{expected_arm} wrong experiment identity")
    if payload.get("arm") != expected_arm or int(payload.get("optimizer_steps", -1)) != expected_steps:
        raise RuntimeError(f"{expected_arm} step/arm identity mismatch")
    if payload.get("valid_run") is not True or payload.get("formal_execution_not_started") is not True:
        raise RuntimeError(f"{expected_arm} invalid/formal")
    if payload.get("scientific_evidence") is not False:
        raise RuntimeError(f"{expected_arm} mislabeled as formal evidence")
    if payload.get("source") != source:
        raise RuntimeError(f"{expected_arm} source provenance differs from run identity")
    if str(payload.get("dataset_manifest_sha256")) != str(depth3["dataset_manifest_sha256"]):
        raise RuntimeError(f"{expected_arm} dataset identity changed")
    if payload.get("l7_reproduction", {}).get("exact") is not True:
        raise RuntimeError(f"{expected_arm} did not exactly reproduce L7")
    topology = payload.get("joint_topology", {})
    if topology.get("layers") != [7, 15, 23] or topology.get("joint_layers") != [15, 23]:
        raise RuntimeError(f"{expected_arm} topology changed")
    if topology.get("selection") != "reuse_exact_published_depth3_cells_no_reallocation":
        raise RuntimeError(f"{expected_arm} reallocation semantics changed")
    if list(topology.get("selected_l15", [])) != depth3["selected_l15"]:
        raise RuntimeError(f"{expected_arm} L15 Cell identity changed")
    if list(topology.get("selected_l23", [])) != depth3["selected_l23"]:
        raise RuntimeError(f"{expected_arm} L23 Cell identity changed")
    training = payload.get("training", {})
    if int(training.get("optimizer_steps", -1)) != expected_steps:
        raise RuntimeError(f"{expected_arm} optimizer steps changed")
    if training.get("jointly_trainable_layers") != [15, 23]:
        raise RuntimeError(f"{expected_arm} was not jointly trainable at L15/L23")
    if list(training.get("selected_l15", [])) != depth3["selected_l15"]:
        raise RuntimeError(f"{expected_arm} training L15 identity changed")
    if list(training.get("selected_l23", [])) != depth3["selected_l23"]:
        raise RuntimeError(f"{expected_arm} training L23 identity changed")


def validate_final(depth3: dict) -> dict:
    identity = load_json(OUTPUT / "RUN_IDENTITY.json")
    design = load_json(OUTPUT / "DESIGN.json")
    result = load_json(OUTPUT / "RESULT.json")
    decision = load_json(OUTPUT / "DECISION.json")
    primary = load_json(OUTPUT / "JOINT_128.json")
    secondary = load_json(OUTPUT / "JOINT_256.json")

    if identity.get("experiment") != "PCU-JOINT-CROSS-LAYER-001" or int(identity.get("seed", -1)) != 26090501:
        raise RuntimeError("wrong joint run identity")
    source = identity.get("source", {})
    if source.get("source_dirty") is not False or not source.get("source_commit") or not source.get("source_tree"):
        raise RuntimeError("joint run lacks clean immutable source provenance")
    if identity.get("dual_gpu_execution_required") is not True:
        raise RuntimeError("joint run did not require dual GPU")
    if identity.get("worker_devices") != {"joint128": "cuda:0", "joint256": "cuda:1"}:
        raise RuntimeError("joint dual-GPU worker mapping changed")

    if design.get("causal_variable") != "joint_vs_sequential_optimization_of_exact_same_L15_L23_cells":
        raise RuntimeError("joint causal variable changed")
    if design.get("primary_scientific_decision_arm") != "joint128_primary":
        raise RuntimeError("Joint-256 was incorrectly promoted to primary")
    dual = design.get("dual_gpu_execution", {})
    if dual.get("required") is not True or dual.get("process_isolation") is not True:
        raise RuntimeError("joint design lost dual-GPU process isolation")
    fixed = design.get("fixed", {})
    if int(fixed.get("transport_layer", -1)) != 15 or int(fixed.get("readout_layer", -1)) != 23:
        raise RuntimeError("joint layer topology changed")
    if int(fixed.get("transport_k", -1)) != 16 or int(fixed.get("readout_k", -1)) != 16:
        raise RuntimeError("joint K budget changed")
    if fixed.get("cell_identity") != "exact_published_depth3_cells_no_reallocation":
        raise RuntimeError("joint design no longer reuses exact published Cells")

    validate_worker(primary, expected_steps=128, expected_arm="joint_128", depth3=depth3, source=source)
    validate_worker(secondary, expected_steps=256, expected_arm="joint_256", depth3=depth3, source=source)

    if result.get("valid_run") is not True or result.get("formal_execution_not_started") is not True:
        raise RuntimeError("joint result invalid/formal")
    if result.get("status") not in VALID_STATUSES or decision.get("status") != result.get("status"):
        raise RuntimeError("joint result/decision status mismatch")
    if result.get("primary_arm") != "joint128_per_parameter_update_matched":
        raise RuntimeError("joint primary arm semantics changed")
    if result.get("secondary_arm") != "joint256_extra_joint_updates_diagnostic":
        raise RuntimeError("joint secondary arm semantics changed")
    if decision.get("exact_published_depth3_cells_reused") is not True or decision.get("no_reallocation") is not True:
        raise RuntimeError("joint decision did not certify exact Cell reuse")
    if decision.get("l7_frozen_before_joint_training") is not True:
        raise RuntimeError("joint decision did not certify frozen L7")
    if decision.get("dual_gpu_execution_required") is not True:
        raise RuntimeError("joint decision lost dual-GPU requirement")
    if int(decision.get("primary_coordination_steps", -1)) != 128:
        raise RuntimeError("joint primary step budget changed")
    if int(decision.get("secondary_extra_joint_steps", -1)) != 256:
        raise RuntimeError("joint secondary step budget changed")
    if abs(float(decision.get("sequential_direct_accuracy", -1)) - 0.140625) > 1e-12:
        raise RuntimeError("joint sequential direct baseline changed")
    if abs(float(decision.get("sequential_ranking_accuracy", -1)) - 0.7890625) > 1e-12:
        raise RuntimeError("joint sequential ranking baseline changed")
    return decision


def assert_prerequisite_published(branch: str) -> None:
    run(["git", "fetch", "origin"])
    for path in (DEPTH3_ROOT / "DEPTH_3.json", DEPTH3_ROOT / "DECISION.json"):
        probe = subprocess.run(
            ["git", "show", f"origin/{branch}:{path}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode != 0:
            raise RuntimeError(f"required depth3 prerequisite is not published remotely: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()
    if run(["git", "branch", "--show-current"], capture=True) != args.branch:
        raise RuntimeError(f"expected branch {args.branch}")
    assert_formal_seeds_untouched()
    depth3 = load_depth3_identity()
    decision = validate_final(depth3)
    assert_prerequisite_published(args.branch)

    remote_probe = subprocess.run(
        ["git", "show", f"origin/{args.branch}:{OUTPUT}/DECISION.json"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if remote_probe.returncode == 0:
        raise RuntimeError(f"{OUTPUT} already exists remotely; refusing to overwrite")

    paths = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.suffix.lower() in SAFE_SUFFIXES)
    if not paths:
        raise RuntimeError("no joint cross-layer evidence found")
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
    run(["git", "commit", "-m", f"artifacts: publish PCU-JOINT-CROSS-LAYER-001 {RUN_ID}"])
    run(["git", "rebase", f"origin/{args.branch}"])
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")
    askpass = Path("/tmp/minicells-pcu-joint-cross-layer-askpass.sh")
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
