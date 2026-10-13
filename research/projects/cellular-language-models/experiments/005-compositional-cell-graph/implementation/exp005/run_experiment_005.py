#!/usr/bin/env python3
"""CPU preflight, dry-run, asset prep, or one registered manual training arm."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parents[1]
ROOT = next(parent for parent in HERE.parents
            if (parent / ".git").exists() or (parent / "pyproject.toml").exists())
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "shared/python"), str(ROOT), str(HERE.parent)]

from exp005.assets import load_assets, prepare_assets
from exp005.checkpoint import file_sha256
from exp005.edge_refresh import make_edge_refresh_callback
from exp005.models import ARMS
from exp005.preflight import run_cpu_preflight, validate_config, seed_role
from exp005.training import profile_completed_arm, train_arm


DEFAULT_CONFIG = EXPERIMENT_DIR / "config/registered.json"
DEFAULT_DEFINITION = EXPERIMENT_DIR / "experiment.yaml"
DEFAULT_PROTOCOL = EXPERIMENT_DIR / "protocol.json"


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          text=True, stdout=subprocess.PIPE).stdout.strip()


def verify_source_commit(source_commit: str) -> None:
    if subprocess.run(["git", "cat-file", "-e", source_commit + "^{commit}"],
                      cwd=ROOT, check=False).returncode:
        raise ValueError("requested source commit is not present in this checkout")
    if subprocess.run(["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
                      cwd=ROOT, check=False).returncode:
        raise ValueError("checkout does not descend from the requested source commit")
    protected = [
        "src", "shared/python/experiments/common", "shared/python/experiments/common", "research/governance/registry.yaml",
        "research/projects/cellular-language-models/experiments/005-compositional-cell-graph/README.md",
        "research/projects/cellular-language-models/experiments/005-compositional-cell-graph/experiment.yaml",
        "research/projects/cellular-language-models/experiments/005-compositional-cell-graph/protocol.json",
        "research/projects/cellular-language-models/experiments/005-compositional-cell-graph/config",
        "research/projects/cellular-language-models/experiments/005-compositional-cell-graph/registration",
        "research/projects/cellular-language-models/experiments/005-compositional-cell-graph/implementation",
        "research/projects/cellular-language-models/experiments/005-compositional-cell-graph/implementation/run_experiment_005.py",
    ]
    result = subprocess.run(["git", "diff", "--quiet", source_commit, "HEAD", "--", *protected],
                            cwd=ROOT, check=False)
    if result.returncode:
        raise ValueError("result branch changed the pinned experiment implementation or protocol")


def load_contract(path: str | Path) -> tuple[dict, dict, dict, Path, Path]:
    config_path = Path(path).resolve()
    definition_path = ROOT / DEFAULT_DEFINITION
    protocol_path = ROOT / DEFAULT_PROTOCOL
    config = json.loads(config_path.read_text(encoding="utf-8"))
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config["protocol_path"] = str(protocol_path)
    config["definition_path"] = str(definition_path)
    config["config_path"] = str(config_path)
    config["protocol_status"] = "registered" if protocol.get("status") == "REGISTERED" else "proposed"
    registry = json.loads((ROOT / "research/governance/registry.yaml").read_text(encoding="utf-8"))
    entry = next((item for item in registry["experiments"] if item["id"] == "005"), None)
    if entry is None or entry["directory"] != Path(definition["config"]).parents[1].as_posix():
        raise ValueError("Experiment 005 registry and definition path disagree")
    config["registry_status"] = entry.get("status")
    validate_config(config)
    return config, definition, protocol, config_path, protocol_path


def blockers(config: dict, definition: dict, protocol: dict) -> list[str]:
    found = list(config.get("open_protocol_items", []))
    for name in ("edge_refresh_candidate_count", "edge_refresh_probe_count", "edge_refresh_source_count",
                 "max_edge_replacements_per_refresh", "edge_refresh_min_improvement"):
        if config["training"].get(name) is None:
            found.append(f"training.{name} is not frozen")
    if config["data"].get("probe_manifest_sha256") is None:
        found.append("data probe-window manifest hash is not frozen")
    if config["data"].get("assets_manifest_sha256") is None:
        found.append("complete train/regression/holdout asset manifest hash is not frozen")
    if config["training"].get("probe_seed") is None:
        found.append("training.probe_seed is not frozen")
    if not config.get("seeds", {}).get("development"):
        found.append("no collision-checked development seed is registered")
    if not config.get("seeds", {}).get("confirmation"):
        found.append("no untouched confirmation seeds are registered")
    if definition.get("status") not in {"registered", "registered_unrun"}:
        found.append(f"experiment status is {definition.get('status')!r}, not registered")
    if config.get("registry_status") != "registered_unrun":
        found.append(f"registry status is {config.get('registry_status')!r}, not 'registered_unrun'")
    if protocol.get("status") != "REGISTERED":
        found.append(f"protocol status is {protocol.get('status')!r}, not 'REGISTERED'")
    strong_flat_rule = protocol.get("b2_strong_flat_matching_rule")
    if not isinstance(strong_flat_rule, dict) or not isinstance(strong_flat_rule.get("max_ratios"), dict):
        found.append("B2 comparison against strong S0 has no frozen matching rule")
    if protocol.get("b2_graph_diagnostics_rule") is None:
        found.append("B2 graph-change, no-edge and anti-collapse diagnostic gates are not frozen")
    runtime_rule = protocol.get("system_runtime_rule")
    if not isinstance(runtime_rule, dict) or runtime_rule.get("decode_p95_ratio_upper_ci_max") is None:
        found.append("system p95 decode degradation has no frozen paired-bootstrap non-inferiority rule")
    if not config.get("evaluation", {}).get("edge_perturbation_seeds"):
        found.append("G0-no-edge perturbation seeds are not frozen")
    if config.get("evaluation", {}).get("edge_perturbation_replacements") is None:
        found.append("G0-no-edge important-edge replacement count is not frozen")
    sources = int(config["training"].get("edge_refresh_source_count", 0) or 0)
    events = int(config["training"]["steps"]) // int(config["training"]["edge_refresh_every_steps"])
    if sources * events < int(config["model"]["cell_count"]) + 1:
        found.append("source-row schedule cannot cover all Cells plus START")
    try:
        for value in config["seeds"]["development"] + config["seeds"]["confirmation"]:
            seed_role(config, value)
    except ValueError as exc:
        found.append(str(exc))
    if len(config["seeds"]["confirmation"]) != int(protocol["gates"]["confirmation_seeds_required"]):
        found.append("confirmation seed reservation does not match the protocol matrix")
    if config.get("protocol_status") == "registered":
        evidence_path = EXPERIMENT_DIR / "registration/evidence-manifest.json"
        if not evidence_path.is_file():
            found.append("registration evidence manifest is missing")
        else:
            evidence = json.loads(evidence_path.read_text())
            for name, expected in evidence.get("files", {}).items():
                file = evidence_path.parent / name
                if not file.is_file() or file_sha256(file) != expected:
                    found.append("registration evidence changed: " + name)
            for name, field in (("config_path", "registered_config_sha256"), ("protocol_path", "registered_protocol_sha256")):
                if file_sha256(config[name]) != evidence.get(field):
                    found.append("registered contract differs from its evidence record: " + name)
    return list(dict.fromkeys(found))


def dry_run(config: dict, definition: dict, protocol: dict) -> dict:
    return {"status": "DRY_RUN_ONLY", "scientific_decision": False,
            "registration_stage": config.get("registration", {}).get("stage"),
            "development_seeds": config["seeds"]["development"],
            "confirmation_seeds_reserved": config["seeds"]["confirmation"],
            "confirmation_authorized": config.get("registration", {}).get("confirmation_authorized", False),
            "experiment": definition["id"], "protocol_status": protocol["status"],
            "training_arms": [
                {"arm": arm, "tokens": config["data"]["train_tokens_per_arm"],
                 "steps": config["training"]["steps"]} for arm in ARMS
            ],
            "posthoc_diagnostics": config["posthoc_diagnostics"],
            "tokens_per_seed": len(ARMS) * int(config["data"]["train_tokens_per_arm"]),
            "confirmatory_seed_count": len(config["seeds"]["confirmation"]),
            "blockers": blockers(config, definition, protocol),
            "gpu_started": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "dry-run", "prepare-assets", "train", "profile"),
                        required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source-commit")
    parser.add_argument("--assets")
    parser.add_argument("--asset-manifest-sha256")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-window-count", type=int)
    parser.add_argument("--probe-seed", type=int)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()
    config, definition, protocol, config_path, protocol_path = load_contract(args.config)

    if args.mode == "preflight":
        report = run_cpu_preflight(config)
        report["blockers"] = blockers(config, definition, protocol)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    if args.mode == "dry-run":
        print(json.dumps(dry_run(config, definition, protocol), indent=2, allow_nan=False))
        return 0
    if args.mode == "prepare-assets":
        if not args.output or args.probe_window_count is None or args.probe_seed is None:
            parser.error("prepare-assets requires --output, --probe-window-count and --probe-seed")
        manifest = prepare_assets(config, args.output,
                                  probe_window_count=args.probe_window_count,
                                  probe_seed=args.probe_seed)
        print(json.dumps({"status": "ASSETS_PREPARED", **manifest}, indent=2))
        return 0

    missing = [name for name, value in (("--arm", args.arm), ("--seed", args.seed),
                                        ("--source-commit", args.source_commit),
                                        ("--assets", args.assets),
                                        ("--asset-manifest-sha256", args.asset_manifest_sha256),
                                        ("--output", args.output)) if value is None]
    if missing:
        parser.error(args.mode + " requires " + ", ".join(missing))
    if blockers(config, definition, protocol):
        raise RuntimeError("full training is blocked until the proposal is registered; see --mode dry-run")
    role = seed_role(config, args.seed)
    if role == "confirmation" and not config.get("registration", {}).get("confirmation_authorized", False):
        raise RuntimeError("confirmation seeds are reserved; complete and review Gate 2 development first")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    verify_source_commit(args.source_commit)
    assets = load_assets(args.assets, expected_manifest_sha256=args.asset_manifest_sha256,
                         config=config)
    if args.mode == "profile":
        result = profile_completed_arm(config, args.arm, args.seed, args.source_commit, assets,
                                       args.output, device=args.device)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    callback = (make_edge_refresh_callback(config, assets, args.seed)
                if args.arm in {"S0", "R0", "G0"} else None)
    result = train_arm(config, args.arm, args.seed, args.source_commit, assets, args.output,
                       device=args.device, resume=args.resume, edge_refresh=callback)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
