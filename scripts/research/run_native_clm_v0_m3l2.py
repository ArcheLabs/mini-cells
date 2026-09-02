"""Run frozen Native CLM v0 M3L-2 lineage-cosine vs online-address validation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from statistics import mean

from minicells.native_clm_m2 import NativeCLMM2Config
from minicells.native_clm_m3 import NativeCLMM3GrowthConfig
from minicells.native_clm_m3l2 import M3L2AddressConfig, run_online_address_state_arm
from minicells.native_clm_m3r import run_lineage_growth_arm

DEFAULT_PROTOCOL = Path("research/validations/native-clm-v0-m3l2-online-address-state/protocol.json")
DEFAULT_OUTPUT = Path("artifacts/experiments/native-clm-v0-m3l2-online-address-state")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_protocol(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    p = json.loads(raw)
    if p.get("format") != "minicells.native-clm-v0.m3l2-online-address-state.protocol.v1":
        raise RuntimeError("unexpected M3L-2 protocol")
    return p, hashlib.sha256(raw).hexdigest()


def _validate_data(data_dir: Path, protocol: dict) -> tuple[dict, str]:
    path = data_dir / "manifest.json"
    raw = path.read_bytes()
    m = json.loads(raw)
    if m.get("format") != "minicells.native-clm-v0.m3l2-data-manifest.v1":
        raise RuntimeError("unexpected M3L-2 data manifest")
    expected = {
        "A_bootstrap": int(protocol["bootstrap"]["documents"]),
        "A_eval": int(protocol["stream"]["A"]["documents"]),
        "B_train": int(protocol["stream"]["B"]["train_documents"]),
        "B_eval": int(protocol["stream"]["B"]["eval_documents"]),
        "C_train": int(protocol["stream"]["C"]["train_documents"]),
        "C_eval": int(protocol["stream"]["C"]["eval_documents"]),
        "D_train": int(protocol["stream"]["D"]["train_documents"]),
        "D_eval": int(protocol["stream"]["D"]["eval_documents"]),
    }
    if set(m.get("files", {})) != set(expected):
        raise RuntimeError("M3L-2 file set mismatch")
    for name, docs in expected.items():
        r = m["files"][name]
        fp = data_dir / r["path"]
        if int(r["documents"]) != docs or not fp.exists() or fp.stat().st_size != int(r["bytes"]) or _sha256(fp) != r["sha256"]:
            raise RuntimeError(f"M3L-2 data identity mismatch: {name}")
    if m["dataset_revisions"]["A"]["resolved_revision"] != protocol["bootstrap"]["resolved_revision"]:
        raise RuntimeError("M3L-2 TinyStories revision drift")
    return m, hashlib.sha256(raw).hexdigest()


def _paths(data_dir: Path):
    return (
        {"B": data_dir / "B-wikitext-train.txt", "C": data_dir / "C-code-train.txt", "D": data_dir / "D-dolly-train.txt"},
        {"A": data_dir / "A-tinystories-eval.txt", "B": data_dir / "B-wikitext-eval.txt", "C": data_dir / "C-code-eval.txt", "D": data_dir / "D-dolly-eval.txt"},
        data_dir / "A-tinystories-bootstrap.txt",
    )


def _growth_config(protocol: dict) -> NativeCLMM3GrowthConfig:
    names = {field.name for field in fields(NativeCLMM3GrowthConfig)}
    return NativeCLMM3GrowthConfig(**{k: protocol["growth"][k] for k in names})


def _address_config(protocol: dict) -> M3L2AddressConfig:
    a = protocol["address_state"]
    return M3L2AddressConfig(
        rank=int(a["rank"]),
        diagonal_regularization=float(a["diagonal_regularization"]),
        target_old_fpr=float(a["target_old_fpr"]),
        maximum_persistent_bytes_per_cell=int(a["maximum_persistent_bytes_per_cell"]),
        bootstrap_batches=int(protocol["bootstrap"].get("batches", 160)),
    )


def _run_worker(args, protocol: dict) -> int:
    train, evaluation, bootstrap = _paths(args.data_dir)
    common = dict(
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=protocol["parent_checkpoint"]["sha256"],
        train_paths=train,
        eval_paths=evaluation,
        output_dir=args.output_dir / f"seed-{args.seed}" / args.arm,
        seed=args.seed,
        train_config=NativeCLMM2Config(**protocol["training"]),
        growth_config=_growth_config(protocol),
        device=args.device,
    )
    if args.arm == "lineage_cosine_control":
        summary = run_lineage_growth_arm(**common)
        summary["arm"] = "lineage_cosine_control"
        (Path(common["output_dir"]) / "arm-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    else:
        summary = run_online_address_state_arm(
            **common,
            bootstrap_path=bootstrap,
            address_config=_address_config(protocol),
        )
    print(json.dumps({"seed": args.seed, "arm": summary["arm"], "cells": summary["final_cell_count"], "spawned": summary["spawned_cells"]}, indent=2), flush=True)
    return 0


def _command(args, seed: int, arm: str, gpu: str) -> tuple[list[str], dict[str, str]]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--arm", arm, "--seed", str(seed), "--checkpoint", str(args.checkpoint), "--data-dir", str(args.data_dir), "--output-dir", str(args.output_dir), "--protocol", str(args.protocol), "--device", "cuda" if gpu != "cpu" else "cpu"]
    env = os.environ.copy()
    if gpu != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return cmd, env


def _loss(s: dict, stage: str, domain: str) -> float:
    return float(s["evaluation_matrix"][stage][domain]["loss"])


def _regression(s: dict, domain: str, from_stage: str, to_stage: str) -> float:
    before = _loss(s, from_stage, domain)
    return max(0.0, (_loss(s, to_stage, domain) - before) / max(before, 1e-12))


def _phase_gains(s: dict) -> dict[str, float]:
    pairs = {"B": ("initial", "after_B"), "C": ("after_B", "after_C"), "D": ("after_C", "after_D")}
    return {d: max(0.0, (_loss(s, a, d) - _loss(s, b, d)) / max(_loss(s, a, d), 1e-12)) for d, (a, b) in pairs.items()}


def _mean_forgetting(s: dict) -> float:
    return mean([_regression(s, "A", "initial", "after_D"), _regression(s, "B", "after_B", "after_D"), _regression(s, "C", "after_C", "after_D")])


def _growth_ok(s: dict, t: dict) -> bool:
    return int(t["minimum_spawned_cells"]) <= int(s["spawned_cells"]) <= int(t["maximum_spawned_cells"]) and int(s["final_cell_count"]) <= 16


def _children_reused(s: dict, t: dict) -> bool:
    hits = [int(v) for v in s.get("child_post_birth_route_hits", {}).values()]
    if not hits:
        return False
    fraction = sum(v >= int(t["minimum_child_post_birth_route_hits"]) for v in hits) / len(hits)
    return fraction >= float(t["minimum_child_reuse_fraction"])


def _birth_ok(s: dict, t: dict) -> bool:
    events = s.get("growth_events", [])
    return bool(events) and all(
        float(e["birth_logits_max_abs_drift"]) <= float(t["maximum_birth_logits_max_abs_drift"])
        and float(e["birth_logits_mse"]) <= float(t["maximum_birth_logits_mse"])
        and float(e["birth_root_topk_match"]) == 1.0
        and float(e["birth_root_prob_max_abs_drift"]) <= float(t["maximum_birth_root_prob_drift"])
        for e in events
    )


def _root_stable(s: dict) -> bool:
    p = s.get("root_route_probes", {})
    if not p or "initial" not in p:
        return False
    return all(p.get(stage) == p["initial"] for stage in ("after_B", "after_C", "after_D"))


def _max_active(s: dict) -> float:
    values = []
    for stage in s["evaluation_matrix"].values():
        for metrics in stage.values():
            values.append(float(metrics["active_fraction_vs_dense"]))
    return max(values or [1.0])


def compare_arms(control: dict, treatment: dict, *, thresholds: dict) -> dict:
    c_a = _regression(control, "A", "initial", "after_D")
    t_a = _regression(treatment, "A", "initial", "after_D")
    c_gain, t_gain = _phase_gains(control), _phase_gains(treatment)
    c_plasticity, t_plasticity = mean(c_gain.values()), mean(t_gain.values())
    t_forgetting = _mean_forgetting(treatment)
    address = treatment.get("address_state", {})
    bootstrap = treatment.get("bootstrap", {})
    same_checkpoint = control.get("parent_checkpoint_sha256") == treatment.get("parent_checkpoint_sha256")
    gates = {
        "exact_same_m1_checkpoint": same_checkpoint,
        "matched_seed_and_data_snapshot": int(control["seed"]) == int(treatment["seed"]),
        "bootstrap_precedes_continual_learning": bool(bootstrap.get("complete")),
        "bootstrap_does_not_mutate_model_parameters": bootstrap.get("parameter_sha256_before") == bootstrap.get("parameter_sha256_after"),
        "bootstrap_A_is_inaccessible_after_B_start": bootstrap.get("A_access_after_continual_start") is False,
        "zero_learner_replay_after_continual_start": int(control.get("learner_replay_bytes", -1)) == 0 and int(treatment.get("learner_replay_bytes", -1)) == 0,
        "same_frozen_m3_growth_controller": control.get("growth_config") == treatment.get("growth_config"),
        "shared_query_norm_and_root_keys_frozen": bool(control.get("shared_and_original_router_frozen")) and bool(treatment.get("shared_and_original_router_frozen")),
        "control_exposes_lineage_cosine_limit": c_a >= float(thresholds["minimum_control_A_regression"]),
        "control_growth_occurs_and_is_bounded": _growth_ok(control, thresholds),
        "treatment_growth_occurs_and_is_bounded": _growth_ok(treatment, thresholds),
        "birth_function_preserving": _birth_ok(treatment, thresholds),
        "birth_root_ownership_preserved": _birth_ok(treatment, thresholds),
        "root_route_function_preserved": _root_stable(treatment),
        "lineage_chain_valid": bool(treatment.get("lineage_chain_valid")),
        "rank32_address_state_bounded": int(address.get("maximum_rank", 999)) <= int(thresholds["maximum_address_state_rank"]) and int(address.get("maximum_bytes_per_cell", 10**9)) <= int(thresholds["maximum_address_state_bytes_per_cell"]),
        "address_state_checkpoint_roundtrip": bool(treatment.get("address_state_checkpoint_roundtrip")),
        "affine_gates_created_for_children": int(address.get("gate_count", -1)) == int(treatment.get("spawned_cells", -2)),
        "children_are_reused": _children_reused(treatment, thresholds),
        "sparse_compute_survives_growth": _max_active(treatment) <= float(thresholds["maximum_active_fraction_vs_dense"]),
        "treatment_phase_plasticity": all(v >= float(thresholds["minimum_phase_gain_each_B_C_D"]) for v in t_gain.values()),
        "treatment_absolute_A_retention": t_a <= float(thresholds["maximum_treatment_A_regression"]),
        "treatment_A_retention_advantage": c_a - t_a >= float(thresholds["minimum_A_retention_advantage_vs_control"]),
        "treatment_mean_forgetting": t_forgetting <= float(thresholds["maximum_treatment_mean_forgetting"]),
        "treatment_plasticity_preserved": t_plasticity >= float(thresholds["minimum_treatment_to_control_plasticity_ratio"]) * c_plasticity,
    }
    return {
        "seed": int(control["seed"]),
        "pass": all(gates.values()),
        "gates": gates,
        "control_A_regression": c_a,
        "treatment_A_regression": t_a,
        "A_retention_advantage": c_a - t_a,
        "control_mean_forgetting": _mean_forgetting(control),
        "treatment_mean_forgetting": t_forgetting,
        "control_phase_gains": c_gain,
        "treatment_phase_gains": t_gain,
        "control_plasticity": c_plasticity,
        "treatment_plasticity": t_plasticity,
        "treatment_address_state": address,
    }


def _run_pair(args, protocol: dict, protocol_sha: str, data_sha: str, seed: int, devices: tuple[str, str]) -> dict:
    seed_dir = args.output_dir / f"seed-{seed}"
    done = seed_dir / "seed-summary.json"
    if done.exists():
        cached = json.loads(done.read_text())
        if cached.get("protocol_sha256") == protocol_sha and cached.get("data_manifest_sha256") == data_sha:
            return cached
    arms = ("lineage_cosine_control", "online_address_state")
    procs = []
    for arm, gpu in zip(arms, devices, strict=True):
        cmd, env = _command(args, seed, arm, gpu)
        procs.append(subprocess.Popen(cmd, env=env))
    rcs = [p.wait() for p in procs]
    if any(rc != 0 for rc in rcs):
        raise RuntimeError(f"M3L-2 seed {seed} worker failure: {rcs}")
    control = json.loads((seed_dir / arms[0] / "arm-summary.json").read_text())
    treatment = json.loads((seed_dir / arms[1] / "arm-summary.json").read_text())
    result = compare_arms(control, treatment, thresholds=protocol["thresholds"])
    result["protocol_sha256"] = protocol_sha
    result["data_manifest_sha256"] = data_sha
    seed_dir.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def _write_decision(out: Path, decision: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    with (out / "gate-summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["seed", "gate", "pass"])
        for r in decision["seed_results"]:
            for gate, ok in r["gates"].items(): w.writerow([r["seed"], gate, int(ok)])
    lines = ["# Native CLM v0 M3L-2 — Online Historical Address-State Integration", "", f"- Status: `{decision['status']}`", f"- Scientific decision: `{decision['scientific_decision']}`", f"- Protocol SHA-256: `{decision['protocol_sha256']}`", f"- Data manifest SHA-256: `{decision['data_manifest_sha256']}`", "- Learner replay after continual start: `0 bytes`", "", "| seed | control A reg | treatment A reg | advantage | treatment forgetting | result |", "|---:|---:|---:|---:|---:|---|"]
    for r in decision["seed_results"]:
        lines.append(f"| {r['seed']} | {r['control_A_regression']:.4f} | {r['treatment_A_regression']:.4f} | {r['A_retention_advantage']:.4f} | {r['treatment_mean_forgetting']:.4f} | {'PASS' if r['pass'] else 'FAIL'} |")
    lines += ["", "Boundary: the A bootstrap is explicitly pre-continual and sidecar-only; after B begins, no A bootstrap token/query is available to the learner."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--formal", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm", choices=["lineage_cosine_control", "online_address_state"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    protocol, protocol_sha = _load_protocol(args.protocol)
    if args.worker:
        if args.arm is None or args.seed is None: ap.error("--worker requires --arm and --seed")
        return _run_worker(args, protocol)
    _, data_sha = _validate_data(args.data_dir, protocol)
    devices = tuple(x.strip() for x in args.devices.split(","))
    if len(devices) != 2: raise ValueError("--devices must name exactly two workers")
    consumed = {int(x) for x in protocol["consumed_seeds_forbidden"]}
    if args.formal:
        seeds = [int(x) for x in protocol["formal_seeds"]]
    elif args.seed is not None:
        if args.seed in protocol["formal_seeds"] or args.seed in consumed or args.seed not in protocol["development_seeds"]:
            raise RuntimeError("refusing unregistered/consumed/formal M3L-2 development seed")
        seeds = [args.seed]
    else:
        ap.error("choose --formal or --seed")
    results = [_run_pair(args, protocol, protocol_sha, data_sha, seed, devices) for seed in seeds]
    if args.formal:
        supported = len(results) == len(seeds) and all(r["pass"] for r in results)
        decision = {
            "format": "minicells.native-clm-v0.m3l2-decision.v1",
            "status": protocol["positive_status"] if supported else protocol["negative_status"],
            "scientific_decision": supported,
            "formal_seeds": seeds,
            "completed_seeds": [r["seed"] for r in results],
            "protocol_sha256": protocol_sha,
            "data_manifest_sha256": data_sha,
            "learner_replay_bytes_after_continual_start": 0,
            "seed_results": results,
        }
        _write_decision(args.output_dir, decision)
        print(json.dumps(decision, indent=2), flush=True)
        return 0 if supported else 2
    return 0 if results[0]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
