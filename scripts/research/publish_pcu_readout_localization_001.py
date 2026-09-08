#!/usr/bin/env python3
"""Publish immutable PCU-READOUT-LOCALIZATION-001 engineering evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

BRANCH = "codex/pcu-composability-kill-001"
RUN_ID = "26090501-l7-k64-hybrid-readout"
OUTPUT = Path("artifacts/research/pcu-readout-localization-001/engineering") / RUN_ID
HYBRID = Path("artifacts/research/pcu-hybrid-objective-001/engineering/26090501-l7-k64-rank-plus-ce025")
SEED_REGISTRY = Path("research/formal_seed_registry.json")
FORMAL_REGISTRY_SHA = "71a3015a7d54e795538b3aa6750860f0b9168cb3"
FORMAL_SEEDS = (26090511, 26090512, 26090513)
HYBRID_SOURCE_COMMIT = "0241475a387a9114415cf7ed143670dd5c7e1b3b"
HYBRID_CORE_BLOB_SHA = "851c77cdd283def0698ebe721ea8bf216f5ed556"
VALID_STATUSES = {
    "FIRST_TOKEN_READOUT_BOTTLENECK_SUPPORTED",
    "EARLY_TOKEN_READOUT_BOTTLENECK_SUPPORTED",
    "AUTOREGRESSIVE_TRAJECTORY_INSTABILITY_SUPPORTED",
    "SINGLE_LAYER_GOLD_PREFIX_READOUT_INADEQUATE",
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


def validate_hybrid_baseline() -> dict:
    identity = load_json(HYBRID / "RUN_IDENTITY.json")
    result = load_json(HYBRID / "RESULT.json")
    decision = load_json(HYBRID / "DECISION.json")
    if identity.get("experiment") != "PCU-HYBRID-OBJECTIVE-001":
        raise RuntimeError("wrong hybrid prerequisite identity")
    if decision.get("status") != "HYBRID_OBJECTIVE_PRESERVES_ASSOCIATION_GENERATION_UNRESOLVED":
        raise RuntimeError("readout publisher requires the published hybrid readout failure")
    if decision.get("valid_run") is not True or decision.get("formal_execution_not_started") is not True:
        raise RuntimeError("hybrid prerequisite is not valid pre-formal evidence")
    expected = {
        "ranking_train_accuracy": 1.0,
        "ranking_eval_accuracy": 0.8359375,
        "direct_accuracy": 0.03125,
    }
    for key, value in expected.items():
        if abs(float(decision.get(key, -1.0)) - value) > 1e-12:
            raise RuntimeError(f"published hybrid {key} changed")
    if abs(float(decision.get("ce_weight", -1.0)) - 0.25) > 1e-12:
        raise RuntimeError("published hybrid CE weight changed")
    selected = list(result.get("selected_cells", []))
    if len(selected) != 64 or int(result.get("selected_k", -1)) != 64:
        raise RuntimeError("published hybrid prerequisite is not exact K64")
    return {
        "selected": selected,
        "dataset_manifest_sha256": str(result["dataset_manifest_sha256"]),
        "metrics": expected,
    }


def validate_final(hybrid: dict) -> dict:
    identity = load_json(OUTPUT / "RUN_IDENTITY.json")
    design = load_json(OUTPUT / "DESIGN.json")
    result = load_json(OUTPUT / "RESULT.json")
    decision = load_json(OUTPUT / "DECISION.json")
    if identity.get("experiment") != "PCU-READOUT-LOCALIZATION-001" or int(identity.get("seed", -1)) != 26090501:
        raise RuntimeError("wrong readout-localization experiment identity")
    if identity.get("formal_execution_not_started") is not True:
        raise RuntimeError("readout-localization identity crossed formal boundary")
    source = identity.get("source", {})
    if source.get("source_dirty") is not False or not source.get("source_commit") or not source.get("source_tree"):
        raise RuntimeError("readout-localization evidence lacks clean immutable source provenance")
    if identity.get("hybrid_scientific_source_commit") != HYBRID_SOURCE_COMMIT:
        raise RuntimeError("readout-localization hybrid source commit changed")
    if identity.get("hybrid_scientific_core_blob_sha") != HYBRID_CORE_BLOB_SHA:
        raise RuntimeError("readout-localization hybrid scientific core changed")

    if design.get("causal_variable") != "none_observational_readout_localization":
        raise RuntimeError("readout localization introduced a causal training variable")
    if design.get("training_changed") is not False:
        raise RuntimeError("readout localization changed training")
    replay = design.get("replayed_training", {})
    if int(replay.get("target_layer", -1)) != 7 or int(replay.get("selected_k", -1)) != 64:
        raise RuntimeError("readout-localization replay changed layer or K")
    if list(replay.get("selected_cells", [])) != hybrid["selected"]:
        raise RuntimeError("readout-localization replay changed selected Cells")
    if abs(float(replay.get("ce_weight", -1.0)) - 0.25) > 1e-12:
        raise RuntimeError("readout-localization replay changed hybrid CE weight")
    if replay.get("scientific_source_commit") != HYBRID_SOURCE_COMMIT:
        raise RuntimeError("readout-localization replay source commit changed")
    if replay.get("scientific_core_blob_sha") != HYBRID_CORE_BLOB_SHA:
        raise RuntimeError("readout-localization replay scientific core changed")

    if result.get("valid_run") is not True or result.get("formal_execution_not_started") is not True:
        raise RuntimeError("readout-localization result is not valid pre-formal evidence")
    if result.get("scientific_evidence") is not False or result.get("training_changed") is not False:
        raise RuntimeError("readout-localization result mislabeled or changed training")
    if result.get("status") not in VALID_STATUSES or decision.get("status") != result.get("status"):
        raise RuntimeError("readout-localization status invalid or inconsistent")
    if result.get("hybrid_reproduction_exact") is not True or decision.get("hybrid_reproduction_exact") is not True:
        raise RuntimeError("readout-localization did not exactly reproduce the published hybrid")
    reproduction = result.get("hybrid_reproduction", {})
    for key, expected in hybrid["metrics"].items():
        if abs(float(reproduction.get(key, -1.0)) - expected) > 1e-12:
            raise RuntimeError(f"readout-localization hybrid reproduction drifted: {key}")
    if list(result.get("selected_cells", [])) != hybrid["selected"]:
        raise RuntimeError("readout-localization runtime Cell identity drifted")
    if str(result.get("dataset_manifest_sha256")) != hybrid["dataset_manifest_sha256"]:
        raise RuntimeError("readout-localization runtime dataset changed")
    if int(result.get("selected_k", -1)) != 64:
        raise RuntimeError("readout-localization runtime K changed")
    if decision.get("selected_cells_exact_hybrid_match") is not True:
        raise RuntimeError("readout-localization decision did not certify Cell identity")

    gold = result.get("gold_prefix", {})
    forced = result.get("forced_prefix", {})
    required_gold = (
        "first_token_top1_accuracy",
        "later_token_top1_accuracy",
        "all_token_top1_accuracy",
        "sequence_all_tokens_top1_accuracy",
        "first_token_mean_target_rank",
        "later_token_mean_target_rank",
    )
    if any(key not in gold for key in required_gold):
        raise RuntimeError("readout-localization gold-prefix evidence incomplete")
    if not isinstance(forced.get("curve"), dict) or "0" not in forced["curve"] or "1" not in forced["curve"]:
        raise RuntimeError("readout-localization forced-prefix curve incomplete")
    return decision


def assert_hybrid_published(branch: str) -> None:
    run(["git", "fetch", "origin"])
    remote = f"origin/{branch}:{HYBRID}/DECISION.json"
    probe = subprocess.run(["git", "show", remote], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode != 0:
        raise RuntimeError("PCU-HYBRID-OBJECTIVE-001 prerequisite is not published remotely")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()
    if run(["git", "branch", "--show-current"], capture=True) != args.branch:
        raise RuntimeError(f"expected branch {args.branch}")
    assert_formal_seeds_untouched()
    hybrid = validate_hybrid_baseline()
    decision = validate_final(hybrid)
    assert_hybrid_published(args.branch)

    remote_probe = subprocess.run(
        ["git", "show", f"origin/{args.branch}:{OUTPUT}/DECISION.json"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if remote_probe.returncode == 0:
        raise RuntimeError(f"{OUTPUT} already exists remotely; refusing to overwrite")

    paths = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path.suffix.lower() in SAFE_SUFFIXES)
    if not paths:
        raise RuntimeError("no readout-localization evidence found")
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
    run(["git", "commit", "-m", f"artifacts: publish PCU-READOUT-LOCALIZATION-001 {RUN_ID}"])
    run(["git", "rebase", f"origin/{args.branch}"])
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing environment variable {args.token_env}")
    askpass = Path("/tmp/minicells-pcu-readout-askpass.sh")
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
