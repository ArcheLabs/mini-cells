#!/usr/bin/env python3
"""Publish immutable PCU-SPARSE-PATH-DEPTH-001 engineering evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


BRANCH = "codex/pcu-composability-kill-001"
RUN_ID = "26090501-depth3-4-5"
OUTPUT = Path("artifacts/research/pcu-sparse-path-depth-001/engineering") / RUN_ID
PREVIOUS = Path("artifacts/research/pcu-cross-layer-readout-001/engineering/26090501-l7k64-plus-l23k16")
SEED_REGISTRY = Path("research/formal_seed_registry.json")
FORMAL_REGISTRY_SHA = "71a3015a7d54e795538b3aa6750860f0b9168cb3"
FORMAL_SEEDS = (26090511, 26090512, 26090513)
DEPTHS = (3, 4, 5)
VALID_STATUSES = {
    "SPARSE_PATH_DEPTH_3_RESCUES_NATIVE_GENERATION",
    "SPARSE_PATH_DEPTH_4_RESCUES_NATIVE_GENERATION",
    "SPARSE_PATH_DEPTH_5_RESCUES_NATIVE_GENERATION",
    "DEEPER_SPARSE_PATH_IMPROVES_BUT_DOES_NOT_RESCUE",
    "DEEPER_SPARSE_PATH_DID_NOT_IMPROVE",
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


def validate_previous() -> None:
    decision = load_json(PREVIOUS / "DECISION.json")
    if decision.get("status") != "CROSS_LAYER_READOUT_IMPROVES_BUT_DOES_NOT_RESCUE":
        raise RuntimeError("sparse path publisher requires previous cross-layer non-rescue")
    if abs(float(decision.get("cross_layer_direct_accuracy", -1)) - 0.15625) > 1e-12:
        raise RuntimeError("previous cross-layer direct baseline changed")
    if abs(float(decision.get("cross_layer_ranking_accuracy", -1)) - 0.828125) > 1e-12:
        raise RuntimeError("previous cross-layer ranking baseline changed")
    if decision.get("formal_execution_not_started") is not True:
        raise RuntimeError("previous cross-layer evidence crossed formal boundary")


def _validate_depth(depth: int, payload: dict, summary: dict, source_commit: str) -> None:
    if payload.get("experiment") != "PCU-SPARSE-PATH-DEPTH-001" or int(payload.get("depth", -1)) != depth:
        raise RuntimeError(f"depth{depth} identity mismatch")
    if payload.get("valid_run") is not True or payload.get("formal_execution_not_started") is not True:
        raise RuntimeError(f"depth{depth} is invalid/formal")
    if payload.get("scientific_evidence") is not False:
        raise RuntimeError(f"depth{depth} engineering result mislabeled as formal evidence")
    worker_source = payload.get("source", {})
    if worker_source.get("source_dirty") is not False or worker_source.get("source_commit") != source_commit:
        raise RuntimeError(f"depth{depth} source provenance differs from run identity")
    if payload.get("l7_reproduction", {}).get("exact") is not True:
        raise RuntimeError(f"depth{depth} did not reproduce L7 hybrid")
    if abs(float(payload["l7_reproduction"].get("ranking_eval_accuracy", -1)) - 0.8359375) > 1e-12:
        raise RuntimeError(f"depth{depth} L7 ranking reproduction changed")
    if abs(float(payload["l7_reproduction"].get("direct_accuracy", -1)) - 0.03125) > 1e-12:
        raise RuntimeError(f"depth{depth} L7 direct reproduction changed")

    topology = payload.get("topology", {})
    layers = list(topology.get("layers", []))
    if len(layers) != depth or layers[0] != 7 or layers[-1] != 23:
        raise RuntimeError(f"depth{depth} topology endpoint/depth mismatch: {layers}")
    if int(topology.get("total_added_k", -1)) != 32:
        raise RuntimeError(f"depth{depth} Cell budget changed")
    if int(topology.get("total_added_steps", -1)) != 256:
        raise RuntimeError(f"depth{depth} optimizer budget changed")
    if int(topology.get("readout_k", -1)) != 16 or int(topology.get("readout_steps", -1)) != 128:
        raise RuntimeError(f"depth{depth} readout budget changed")
    if sum(int(v) for v in topology.get("transport_k", [])) != 16:
        raise RuntimeError(f"depth{depth} transport Cell budget changed")
    if sum(int(v) for v in topology.get("transport_steps", [])) != 128:
        raise RuntimeError(f"depth{depth} transport step budget changed")

    stages = list(payload.get("stages", []))
    if len(stages) != depth - 1:
        raise RuntimeError(f"depth{depth} stage count changed")
    if sum(int(stage.get("selected_k", -1)) for stage in stages) != 32:
        raise RuntimeError(f"depth{depth} realized Cell budget changed")
    if sum(int(stage.get("optimizer_steps", -1)) for stage in stages) != 256:
        raise RuntimeError(f"depth{depth} realized optimizer budget changed")
    last = stages[-1]
    if int(last.get("layer", -1)) != 23 or int(last.get("selected_k", -1)) != 16 or int(last.get("optimizer_steps", -1)) != 128:
        raise RuntimeError(f"depth{depth} final readout stage changed")
    for stage in stages:
        mass = float(stage.get("gradient_mass_at_k", -1))
        effective = float(stage.get("effective_count", -1))
        if not (0.0 < mass <= 1.0) or effective <= 0.0:
            raise RuntimeError(f"depth{depth} invalid stage allocation geometry")
        training = stage.get("training", {})
        if int(training.get("training_steps", -1)) != int(stage.get("optimizer_steps", -2)):
            raise RuntimeError(f"depth{depth} stage training step mismatch")
        if list(training.get("selected_cells", [])) != list(stage.get("selected_cells", [])):
            raise RuntimeError(f"depth{depth} stage Cell identity mismatch")

    metrics = payload.get("metrics", {})
    for key in (
        "direct_accuracy",
        "ranking_eval_accuracy",
        "first_token_top1_accuracy",
        "later_token_top1_accuracy",
        "sequence_all_tokens_top1_accuracy",
    ):
        if abs(float(metrics.get(key, -1)) - float(summary.get(key, -2))) > 1e-12:
            raise RuntimeError(f"depth{depth} summary/worker metric mismatch: {key}")


def validate_final() -> dict:
    required = ["RUN_IDENTITY.json", "DESIGN.json", "RESULT.json", "DECISION.json"] + [f"DEPTH_{d}.json" for d in DEPTHS]
    missing = [name for name in required if not (OUTPUT / name).is_file()]
    if missing:
        raise RuntimeError(f"missing sparse path evidence: {missing}")
    identity = load_json(OUTPUT / "RUN_IDENTITY.json")
    design = load_json(OUTPUT / "DESIGN.json")
    result = load_json(OUTPUT / "RESULT.json")
    decision = load_json(OUTPUT / "DECISION.json")

    if identity.get("experiment") != "PCU-SPARSE-PATH-DEPTH-001" or int(identity.get("seed", -1)) != 26090501:
        raise RuntimeError("wrong sparse path run identity")
    if identity.get("formal_execution_not_started") is not True:
        raise RuntimeError("sparse path run crossed formal boundary")
    if identity.get("dual_gpu_execution") is not True or int(identity.get("gpu_count", -1)) != 2:
        raise RuntimeError("sparse path run did not certify dual-GPU execution")
    source = identity.get("source", {})
    if source.get("source_dirty") is not False or not source.get("source_commit") or not source.get("source_tree"):
        raise RuntimeError("sparse path run lacks clean immutable provenance")

    if design.get("causal_question") != "does_distributing_fixed_transport_readout_budget_across_more_nested_layers_improve_native_generation":
        raise RuntimeError("sparse path causal question changed")
    fixed = design.get("fixed", {})
    if int(fixed.get("transport_k_total", -1)) != 16 or int(fixed.get("readout_k", -1)) != 16:
        raise RuntimeError("sparse path fixed Cell budgets changed")
    if int(fixed.get("total_added_k_each", -1)) != 32 or int(fixed.get("total_added_steps_each", -1)) != 256:
        raise RuntimeError("sparse path equal-budget invariant changed")
    execution = design.get("execution", {})
    if execution.get("requires_two_gpus") is not True or execution.get("process_isolation") is not True:
        raise RuntimeError("sparse path dual-GPU/process isolation requirement changed")
    if execution.get("wave1") != {"depth3": "cuda:0", "depth4": "cuda:1"}:
        raise RuntimeError("sparse path dual-GPU wave1 schedule changed")

    if result.get("valid_run") is not True or result.get("formal_execution_not_started") is not True:
        raise RuntimeError("sparse path result invalid/formal")
    if result.get("status") not in VALID_STATUSES or decision.get("status") != result.get("status"):
        raise RuntimeError("sparse path decision/result status mismatch")
    if decision.get("nested_topologies") is not True or decision.get("dual_gpu_execution_required") is not True:
        raise RuntimeError("sparse path decision lost topology/GPU invariants")
    if int(decision.get("total_added_k_each", -1)) != 32 or int(decision.get("total_added_steps_each", -1)) != 256:
        raise RuntimeError("sparse path decision budget drift")

    summaries = result.get("depths", {})
    if set(summaries) != {"3", "4", "5"} or set(decision.get("depths", {})) != {"3", "4", "5"}:
        raise RuntimeError("sparse path result is missing a depth")
    worker_payloads: dict[int, dict] = {}
    for depth in DEPTHS:
        payload = load_json(OUTPUT / f"DEPTH_{depth}.json")
        _validate_depth(depth, payload, summaries[str(depth)], str(source["source_commit"]))
        if decision["depths"][str(depth)] != summaries[str(depth)]:
            raise RuntimeError(f"depth{depth} decision summary differs from result")
        worker_payloads[depth] = payload

    layers3 = set(worker_payloads[3]["topology"]["layers"])
    layers4 = set(worker_payloads[4]["topology"]["layers"])
    layers5 = set(worker_payloads[5]["topology"]["layers"])
    if not layers3.issubset(layers4) or not layers4.issubset(layers5):
        raise RuntimeError("sparse path topologies are not nested")
    if not (
        worker_payloads[3]["available_moe_layers"]
        == worker_payloads[4]["available_moe_layers"]
        == worker_payloads[5]["available_moe_layers"]
    ):
        raise RuntimeError("sparse path workers discovered different MoE layers")

    best_depth = max(DEPTHS, key=lambda d: float(summaries[str(d)]["direct_accuracy"]))
    if int(result.get("best_depth", -1)) != best_depth or int(decision.get("best_depth", -1)) != best_depth:
        raise RuntimeError("sparse path best-depth selection mismatch")
    if abs(float(result.get("best_direct_accuracy", -1)) - float(summaries[str(best_depth)]["direct_accuracy"])) > 1e-12:
        raise RuntimeError("sparse path best direct accuracy mismatch")
    return decision


def assert_previous_published(branch: str) -> None:
    run(["git", "fetch", "origin"])
    probe = subprocess.run(
        ["git", "show", f"origin/{branch}:{PREVIOUS / 'DECISION.json'}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode != 0:
        raise RuntimeError("previous cross-layer prerequisite is not published remotely")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()
    if run(["git", "branch", "--show-current"], capture=True) != args.branch:
        raise RuntimeError(f"expected branch {args.branch}")
    assert_formal_seeds_untouched()
    validate_previous()
    decision = validate_final()
    assert_previous_published(args.branch)

    remote_probe = subprocess.run(
        ["git", "show", f"origin/{args.branch}:{OUTPUT / 'DECISION.json'}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if remote_probe.returncode == 0:
        raise RuntimeError(f"{OUTPUT} already exists remotely; refusing to overwrite")

    paths = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.suffix.lower() in SAFE_SUFFIXES)
    if not paths:
        raise RuntimeError("no sparse path evidence found")
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
    run(["git", "commit", "-m", f"artifacts: publish PCU-SPARSE-PATH-DEPTH-001 {RUN_ID}"])
    run(["git", "rebase", f"origin/{args.branch}"])
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")
    askpass = Path("/tmp/minicells-pcu-path-depth-askpass.sh")
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
        "best_depth": decision["best_depth"],
        "commit": run(["git", "rev-parse", "HEAD"], capture=True),
        "formal_seeds": "RESERVED_UNTOUCHED",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
