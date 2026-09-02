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

DEFAULT_PROTOCOL = Path(
    "research/validations/native-clm-v0-m3l2-online-address-state/protocol.json"
)
DEFAULT_OUTPUT = Path("artifacts/experiments/native-clm-v0-m3l2-online-address-state")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("format") != "minicells.native-clm-v0.m3l2-online-address-state.protocol.v1":
        raise RuntimeError("unexpected M3L-2 protocol")
    if protocol.get("status") != "FROZEN_UNRUN" or not protocol.get("frozen_before_formal_run"):
        raise RuntimeError("M3L-2 protocol is not a frozen pre-formal registration")
    formal = {int(seed) for seed in protocol["formal_seeds"]}
    consumed = {int(seed) for seed in protocol["consumed_seeds_forbidden"]}
    if formal & consumed:
        raise RuntimeError("M3L-2 formal seeds overlap consumed seeds")
    return protocol, hashlib.sha256(raw).hexdigest()


def _validate_data(data_dir: Path, protocol: dict) -> tuple[dict, str]:
    path = data_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("format") != "minicells.native-clm-v0.m3l2-data-manifest.v1":
        raise RuntimeError("unexpected M3L-2 data manifest")
    if manifest.get("stream") != ["B", "C", "D"]:
        raise RuntimeError("M3L-2 continual stream must be B->C->D")
    if int(manifest.get("learner_replay_bytes", -1)) != 0:
        raise RuntimeError("M3L-2 data manifest does not declare zero learner replay")
    bootstrap_meta = manifest.get("bootstrap", {})
    if (
        bootstrap_meta.get("dataset_split") != "train"
        or bootstrap_meta.get("access_after_continual_start") is not False
        or bootstrap_meta.get("native_clm_parameter_updates") is not False
    ):
        raise RuntimeError("M3L-2 bootstrap manifest boundary drift")

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
    records = manifest.get("files", {})
    if set(records) != set(expected):
        raise RuntimeError("M3L-2 file set mismatch")
    for name, documents in expected.items():
        record = records[name]
        file_path = data_dir / record["path"]
        if int(record["documents"]) != documents:
            raise RuntimeError(f"M3L-2 document-count drift: {name}")
        if not file_path.exists() or file_path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"M3L-2 file identity mismatch: {name}")
        if _sha256(file_path) != record["sha256"]:
            raise RuntimeError(f"M3L-2 file SHA mismatch: {name}")

    revisions = manifest.get("dataset_revisions", {})
    if set(revisions) != {"A", "B", "C_train", "C_eval", "D"}:
        raise RuntimeError("M3L-2 data manifest lacks exact dataset revisions")
    for name, record in revisions.items():
        if not record.get("repo_id") or not record.get("resolved_revision"):
            raise RuntimeError(f"M3L-2 dataset revision incomplete for {name}")
    if revisions["A"]["resolved_revision"] != protocol["bootstrap"]["resolved_revision"]:
        raise RuntimeError("M3L-2 TinyStories revision drift")
    return manifest, hashlib.sha256(raw).hexdigest()


def _paths(data_dir: Path):
    return (
        {
            "B": data_dir / "B-wikitext-train.txt",
            "C": data_dir / "C-code-train.txt",
            "D": data_dir / "D-dolly-train.txt",
        },
        {
            "A": data_dir / "A-tinystories-eval.txt",
            "B": data_dir / "B-wikitext-eval.txt",
            "C": data_dir / "C-code-eval.txt",
            "D": data_dir / "D-dolly-eval.txt",
        },
        data_dir / "A-tinystories-bootstrap.txt",
    )


def _growth_config(protocol: dict) -> NativeCLMM3GrowthConfig:
    names = {field.name for field in fields(NativeCLMM3GrowthConfig)}
    config = NativeCLMM3GrowthConfig(**{key: protocol["growth"][key] for key in names})
    config.validate()
    return config


def _address_config(protocol: dict) -> M3L2AddressConfig:
    if int(protocol["bootstrap"]["sampling_seed"]) != 74001:
        raise RuntimeError("registered M3L-2 bootstrap sampling seed drift")
    address = protocol["address_state"]
    if int(address["maximum_persistent_sketch_bytes_per_cell"]) != int(
        address["maximum_persistent_bytes_per_cell"]
    ):
        raise RuntimeError("registered M3L-2 sketch budget aliases disagree")
    if int(address["maximum_affine_gate_bytes_per_edge"]) != 1552:
        raise RuntimeError("registered M3L-2 affine-gate budget drift")
    if int(address["maximum_total_address_bytes_per_node"]) != 53912:
        raise RuntimeError("registered M3L-2 total-node address budget drift")
    config = M3L2AddressConfig(
        rank=int(address["rank"]),
        diagonal_regularization=float(address["diagonal_regularization"]),
        target_old_fpr=float(address["target_old_fpr"]),
        maximum_persistent_bytes_per_cell=int(address["maximum_persistent_bytes_per_cell"]),
        bootstrap_batches=int(protocol["bootstrap"]["batches"]),
        max_queries_per_cell_per_batch=int(address["max_queries_per_cell_per_batch"]),
    )
    config.validate()
    return config


def _run_worker(args, protocol: dict) -> int:
    train, evaluation, bootstrap = _paths(args.data_dir)
    common = {
        "checkpoint_path": args.checkpoint,
        "expected_checkpoint_sha256": protocol["parent_checkpoint"]["sha256"],
        "train_paths": train,
        "eval_paths": evaluation,
        "output_dir": args.output_dir / f"seed-{args.seed}" / args.arm,
        "seed": args.seed,
        "train_config": NativeCLMM2Config(**protocol["training"]),
        "growth_config": _growth_config(protocol),
        "device": args.device,
    }
    common["train_config"].validate()
    if args.arm == "lineage_cosine_control":
        summary = run_lineage_growth_arm(**common)
        summary["arm"] = "lineage_cosine_control"
        (Path(common["output_dir"]) / "arm-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        summary = run_online_address_state_arm(
            **common,
            bootstrap_path=bootstrap,
            address_config=_address_config(protocol),
        )
    print(
        json.dumps(
            {
                "seed": args.seed,
                "arm": summary["arm"],
                "cells": summary["final_cell_count"],
                "spawned": summary["spawned_cells"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _command(args, seed: int, arm: str, gpu: str) -> tuple[list[str], dict[str, str]]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--checkpoint",
        str(args.checkpoint),
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(args.output_dir),
        "--protocol",
        str(args.protocol),
        "--device",
        "cuda" if gpu != "cpu" else "cpu",
    ]
    env = os.environ.copy()
    if gpu != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return command, env


def _loss(summary: dict, stage: str, domain: str) -> float:
    return float(summary["evaluation_matrix"][stage][domain]["loss"])


def _regression(summary: dict, domain: str, from_stage: str, to_stage: str) -> float:
    before = _loss(summary, from_stage, domain)
    return max(0.0, (_loss(summary, to_stage, domain) - before) / max(before, 1e-12))


def _phase_gains(summary: dict) -> dict[str, float]:
    pairs = {
        "B": ("initial", "after_B"),
        "C": ("after_B", "after_C"),
        "D": ("after_C", "after_D"),
    }
    return {
        domain: max(
            0.0,
            (_loss(summary, before, domain) - _loss(summary, after, domain))
            / max(_loss(summary, before, domain), 1e-12),
        )
        for domain, (before, after) in pairs.items()
    }


def _mean_forgetting(summary: dict) -> float:
    return mean(
        [
            _regression(summary, "A", "initial", "after_D"),
            _regression(summary, "B", "after_B", "after_D"),
            _regression(summary, "C", "after_C", "after_D"),
        ]
    )


def _growth_ok(summary: dict, thresholds: dict) -> bool:
    return (
        int(thresholds["minimum_spawned_cells"])
        <= int(summary["spawned_cells"])
        <= int(thresholds["maximum_spawned_cells"])
        and int(summary["final_cell_count"]) <= 16
    )


def _children_reused(summary: dict, thresholds: dict) -> bool:
    hits = [int(value) for value in summary.get("child_post_birth_route_hits", {}).values()]
    if not hits:
        return False
    reused = sum(
        value >= int(thresholds["minimum_child_post_birth_route_hits"])
        for value in hits
    ) / len(hits)
    return reused >= float(thresholds["minimum_child_reuse_fraction"])


def _birth_ok(summary: dict, thresholds: dict) -> bool:
    events = summary.get("growth_events", [])
    return bool(events) and all(
        float(event["birth_logits_max_abs_drift"])
        <= float(thresholds["maximum_birth_logits_max_abs_drift"])
        and float(event["birth_logits_mse"]) <= float(thresholds["maximum_birth_logits_mse"])
        and float(event["birth_root_topk_match"]) == 1.0
        and float(event["birth_root_prob_max_abs_drift"])
        <= float(thresholds["maximum_birth_root_prob_drift"])
        for event in events
    )


def _root_stable(summary: dict) -> bool:
    probes = summary.get("root_route_probes", {})
    if not probes or "initial" not in probes:
        return False
    return all(
        probes.get(stage) == probes["initial"]
        for stage in ("after_B", "after_C", "after_D")
    )


def _max_active(summary: dict) -> float:
    values: list[float] = []
    for stage in summary["evaluation_matrix"].values():
        for metrics in stage.values():
            values.append(float(metrics["active_fraction_vs_dense"]))
    return max(values or [1.0])


def compare_arms(control: dict, treatment: dict, *, thresholds: dict) -> dict:
    if int(control["seed"]) != int(treatment["seed"]):
        raise ValueError("M3L-2 arm seeds do not match")
    if control.get("parent_checkpoint_sha256") != treatment.get("parent_checkpoint_sha256"):
        raise ValueError("M3L-2 arms did not start from the same M1 checkpoint")
    if control.get("growth_config") != treatment.get("growth_config"):
        raise ValueError("M3L-2 arms do not share the frozen M3 growth controller")

    control_a = _regression(control, "A", "initial", "after_D")
    treatment_a = _regression(treatment, "A", "initial", "after_D")
    control_gains = _phase_gains(control)
    treatment_gains = _phase_gains(treatment)
    control_plasticity = mean(control_gains.values())
    treatment_plasticity = mean(treatment_gains.values())
    treatment_forgetting = _mean_forgetting(treatment)
    address = dict(treatment.get("address_state", {}))
    bootstrap = treatment.get("bootstrap", {})
    logical_gate_bytes = 1552 if int(address.get("gate_count", 0)) > 0 else 0
    logical_total_node_bytes = int(address.get("maximum_bytes_per_cell", 10**9)) + logical_gate_bytes
    address["maximum_affine_gate_bytes_per_edge"] = logical_gate_bytes
    address["maximum_total_address_bytes_per_node"] = logical_total_node_bytes

    gates = {
        "exact_same_m1_checkpoint": True,
        "matched_seed_and_data_snapshot": True,
        "bootstrap_precedes_continual_learning": bool(bootstrap.get("complete")),
        "bootstrap_does_not_mutate_model_parameters": (
            bootstrap.get("parameter_sha256_before")
            == bootstrap.get("parameter_sha256_after")
            and bootstrap.get("parameter_sha256_before") is not None
        ),
        "bootstrap_A_is_inaccessible_after_B_start": (
            bootstrap.get("A_access_after_continual_start") is False
            and bool(bootstrap.get("access_released_before_continual_start"))
        ),
        "zero_learner_replay_after_continual_start": (
            int(control.get("learner_replay_bytes", -1)) == 0
            and int(treatment.get("learner_replay_bytes", -1)) == 0
        ),
        "same_frozen_m3_growth_controller": True,
        "shared_query_norm_and_root_keys_frozen": (
            bool(control.get("shared_and_original_router_frozen"))
            and bool(treatment.get("shared_and_original_router_frozen"))
        ),
        "control_exposes_lineage_cosine_limit": (
            control_a >= float(thresholds["minimum_control_A_regression"])
        ),
        "control_growth_occurs_and_is_bounded": _growth_ok(control, thresholds),
        "treatment_growth_occurs_and_is_bounded": _growth_ok(treatment, thresholds),
        "birth_function_preserving": _birth_ok(treatment, thresholds),
        "birth_root_ownership_preserved": _birth_ok(treatment, thresholds),
        "root_route_function_preserved": _root_stable(treatment),
        "lineage_chain_valid": bool(treatment.get("lineage_chain_valid")),
        "rank32_address_state_bounded": (
            int(address.get("maximum_rank", 999))
            <= int(thresholds["maximum_address_state_rank"])
            and int(address.get("maximum_bytes_per_cell", 10**9))
            <= int(thresholds["maximum_address_state_bytes_per_cell"])
            and logical_gate_bytes <= int(thresholds["maximum_affine_gate_bytes_per_edge"])
            and logical_total_node_bytes
            <= int(thresholds["maximum_total_address_bytes_per_node"])
        ),
        "address_state_checkpoint_roundtrip": bool(
            treatment.get("address_state_checkpoint_roundtrip")
        ),
        "affine_gates_created_for_children": (
            int(address.get("gate_count", -1)) == int(treatment.get("spawned_cells", -2))
        ),
        "children_are_reused": _children_reused(treatment, thresholds),
        "sparse_compute_survives_growth": (
            _max_active(treatment) <= float(thresholds["maximum_active_fraction_vs_dense"])
        ),
        "treatment_phase_plasticity": all(
            value >= float(thresholds["minimum_phase_gain_each_B_C_D"])
            for value in treatment_gains.values()
        ),
        "treatment_absolute_A_retention": (
            treatment_a <= float(thresholds["maximum_treatment_A_regression"])
        ),
        "treatment_A_retention_advantage": (
            control_a - treatment_a
            >= float(thresholds["minimum_A_retention_advantage_vs_control"])
        ),
        "treatment_mean_forgetting": (
            treatment_forgetting <= float(thresholds["maximum_treatment_mean_forgetting"])
        ),
        "treatment_plasticity_preserved": (
            treatment_plasticity
            >= float(thresholds["minimum_treatment_to_control_plasticity_ratio"])
            * control_plasticity
        ),
    }
    return {
        "seed": int(control["seed"]),
        "pass": all(gates.values()),
        "gates": gates,
        "control_A_regression": control_a,
        "treatment_A_regression": treatment_a,
        "A_retention_advantage": control_a - treatment_a,
        "control_mean_forgetting": _mean_forgetting(control),
        "treatment_mean_forgetting": treatment_forgetting,
        "control_phase_gains": control_gains,
        "treatment_phase_gains": treatment_gains,
        "control_plasticity": control_plasticity,
        "treatment_plasticity": treatment_plasticity,
        "treatment_address_state": address,
        "treatment_growth_events": treatment.get("growth_events", []),
    }


def _run_pair(
    args,
    protocol: dict,
    protocol_sha: str,
    data_sha: str,
    seed: int,
    devices: tuple[str, str],
) -> dict:
    seed_dir = args.output_dir / f"seed-{seed}"
    completed = seed_dir / "seed-summary.json"
    if completed.exists():
        cached = json.loads(completed.read_text(encoding="utf-8"))
        if (
            cached.get("protocol_sha256") == protocol_sha
            and cached.get("data_manifest_sha256") == data_sha
        ):
            print(f"[M3L-2 seed={seed}] reusing verified completed seed", flush=True)
            return cached
        raise RuntimeError(f"M3L-2 seed {seed} cache belongs to another protocol/data snapshot")

    arms = ("lineage_cosine_control", "online_address_state")
    processes = []
    for arm, gpu in zip(arms, devices, strict=True):
        command, env = _command(args, seed, arm, gpu)
        processes.append(subprocess.Popen(command, env=env))
    return_codes = [process.wait() for process in processes]
    if any(code != 0 for code in return_codes):
        raise RuntimeError(f"M3L-2 seed {seed} worker failure: {return_codes}")

    control = json.loads(
        (seed_dir / arms[0] / "arm-summary.json").read_text(encoding="utf-8")
    )
    treatment = json.loads(
        (seed_dir / arms[1] / "arm-summary.json").read_text(encoding="utf-8")
    )
    result = compare_arms(control, treatment, thresholds=protocol["thresholds"])
    if set(result["gates"]) != set(protocol["registered_gates"]):
        raise RuntimeError("M3L-2 implementation/registered gate set drift")
    result["protocol_sha256"] = protocol_sha
    result["data_manifest_sha256"] = data_sha
    seed_dir.mkdir(parents=True, exist_ok=True)
    completed.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def _write_decision(output: Path, decision: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "gate-summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "gate", "pass"])
        for result in decision["seed_results"]:
            for gate, passed in result["gates"].items():
                writer.writerow([result["seed"], gate, int(bool(passed))])

    lines = [
        "# Native CLM v0 M3L-2 — Online Historical Address-State Integration",
        "",
        f"- Status: `{decision['status']}`",
        f"- Scientific decision: `{decision['scientific_decision']}`",
        f"- Protocol SHA-256: `{decision['protocol_sha256']}`",
        f"- Data manifest SHA-256: `{decision['data_manifest_sha256']}`",
        f"- Formal seeds: `{decision['formal_seeds']}`",
        "- Learner replay after continual start: `0 bytes`",
        "",
        "| seed | control A reg | treatment A reg | advantage | treatment forgetting | result |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for result in decision["seed_results"]:
        lines.append(
            "| {seed} | {control:.4f} | {treatment:.4f} | {advantage:.4f} | "
            "{forgetting:.4f} | {status} |".format(
                seed=result["seed"],
                control=result["control_A_regression"],
                treatment=result["treatment_A_regression"],
                advantage=result["A_retention_advantage"],
                forgetting=result["treatment_mean_forgetting"],
                status="PASS" if result["pass"] else "FAIL",
            )
        )
    lines += [
        "",
        "Boundary: the A bootstrap is explicitly pre-continual and sidecar-only; its one-shot learner handle is released before B begins. No bootstrap token/query is available to the B→C→D learner.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=["lineage_cosine_control", "online_address_state"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    protocol, protocol_sha = _load_protocol(args.protocol)
    _address_config(protocol)
    _growth_config(protocol)
    if args.worker:
        if args.arm is None or args.seed is None:
            parser.error("--worker requires --arm and --seed")
        return _run_worker(args, protocol)

    _, data_sha = _validate_data(args.data_dir, protocol)
    devices = tuple(part.strip() for part in args.devices.split(","))
    if len(devices) != 2:
        raise ValueError("--devices must name exactly two workers")

    consumed = {int(seed) for seed in protocol["consumed_seeds_forbidden"]}
    if args.formal:
        if (args.output_dir / "decision.json").exists():
            raise RuntimeError("canonical M3L-2 decision already exists; formal seeds are consumed")
        seeds = [int(seed) for seed in protocol["formal_seeds"]]
    elif args.seed is not None:
        if args.seed in protocol["formal_seeds"]:
            raise RuntimeError("refusing to touch a formal M3L-2 seed outside --formal")
        if args.seed in consumed:
            raise RuntimeError("refusing to reuse a consumed formal seed")
        if args.seed not in protocol["development_seeds"]:
            raise RuntimeError("M3L-2 development run requires a registered development seed")
        seeds = [args.seed]
    else:
        parser.error("choose --formal or --seed <registered-development-seed>")

    results = [
        _run_pair(args, protocol, protocol_sha, data_sha, seed, devices)
        for seed in seeds
    ]
    if args.formal:
        supported = len(results) == len(seeds) and all(result["pass"] for result in results)
        decision = {
            "format": "minicells.native-clm-v0.m3l2-decision.v1",
            "status": protocol["positive_status"] if supported else protocol["negative_status"],
            "scientific_decision": supported,
            "formal_seeds": seeds,
            "completed_seeds": [result["seed"] for result in results],
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
