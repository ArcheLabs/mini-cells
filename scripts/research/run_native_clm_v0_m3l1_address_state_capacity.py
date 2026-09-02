"""Run checkpoint-only M3L-1 historical address-state capacity diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from minicells.native_clm_m3l1_capacity import (
    M3L1CapacityConfig,
    aggregate_m3l1_capacity,
    diagnose_m3l1_seed,
    write_m3l1_capacity_csv,
)

DEFAULT_PROTOCOL = Path(
    "research/validations/native-clm-v0-m3l1-address-state-capacity/protocol.json"
)
DEFAULT_IDENTITY = Path(
    "research/validations/native-clm-v0-m3l1-address-state-capacity/identity.json"
)
DEFAULT_DATA = Path("/kaggle/working/native-clm-m3r-address-data")
DEFAULT_CHECKPOINTS = Path("/kaggle/working/native-clm-m3r-address-checkpoints")
DEFAULT_OUTPUT = Path("artifacts/experiments/native-clm-v0-m3l1-address-state-capacity")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _config_from_protocol(protocol: dict) -> M3L1CapacityConfig:
    sampling = protocol["sampling"]
    grid = protocol["capacity_grid"]
    gate = protocol["gate_family"]
    oracle = protocol["offline_oracle"]
    thresholds = protocol["candidate_feasibility_thresholds"]
    return M3L1CapacityConfig(
        max_batches_per_domain=int(sampling["max_batches_per_domain"]),
        batch_size=int(sampling["batch_size"]),
        max_samples_per_domain_per_edge=int(sampling["max_samples_per_domain_per_edge"]),
        minimum_train_samples_per_side=int(sampling["minimum_train_samples_per_side"]),
        minimum_test_samples_per_side=int(sampling["minimum_test_samples_per_side"]),
        train_group_fraction=float(sampling["train_group_fraction"]),
        split_seed_base=int(sampling["split_seed_base"]),
        ranks=tuple(int(value) for value in grid["low_rank_gaussian_ranks"]),
        diagonal_regularization=float(gate["diagonal_regularization"]),
        target_old_fpr=float(gate["target_old_fpr"]),
        oracle_steps=int(oracle["steps"]),
        oracle_learning_rate=float(oracle["learning_rate"]),
        oracle_weight_decay=float(oracle["weight_decay"]),
        minimum_valid_edge_fraction=float(thresholds["minimum_valid_edge_fraction"]),
        oracle_separable_median_auc=float(thresholds["oracle_separable_median_auc"]),
        oracle_edge_auc_floor=float(thresholds["oracle_edge_auc_floor"]),
        minimum_fraction_oracle_edges_above_floor=float(
            thresholds["minimum_fraction_oracle_edges_above_floor"]
        ),
        candidate_median_auc=float(thresholds["candidate_median_auc"]),
        candidate_edge_auc_floor=float(thresholds["candidate_edge_auc_floor"]),
        minimum_fraction_candidate_edges_above_floor=float(
            thresholds["minimum_fraction_candidate_edges_above_floor"]
        ),
        median_normalized_oracle_excess_recovery=float(
            thresholds["median_normalized_oracle_excess_recovery"]
        ),
        median_old_fpr_max=float(thresholds["median_old_fpr_max"]),
        median_current_tpr_min=float(thresholds["median_current_tpr_min"]),
    )


def _checkpoint_record(checkpoint_manifest: dict, seed: int) -> dict:
    matches = [
        record
        for record in checkpoint_manifest["records"]
        if int(record["seed"]) == int(seed)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one lineage checkpoint for seed {seed}")
    return matches[0]


def _eval_paths(data_dir: Path) -> dict[str, Path]:
    return {
        "A": data_dir / "A-tinystories-eval.txt",
        "B": data_dir / "B-wikitext-eval.txt",
        "C": data_dir / "C-code-eval.txt",
        "D": data_dir / "D-dolly-eval.txt",
    }


def _validate_inputs(protocol: dict, data_dir: Path, checkpoint_dir: Path) -> tuple[dict, dict]:
    if protocol.get("format") != "minicells.native-clm-v0.m3l1-address-state-capacity.protocol.v1":
        raise RuntimeError("unexpected M3L-1 protocol")
    if protocol.get("status") != "REGISTERED_DIAGNOSTIC_READY":
        raise RuntimeError("M3L-1 protocol is not registered as ready")
    if protocol.get("scientific_decision") is not False:
        raise RuntimeError("M3L-1 protocol unexpectedly claims a scientific decision")
    boundary = protocol["learner_boundary"]
    for key in (
        "native_clm_training",
        "cell_updates",
        "router_updates",
        "certificate_updates",
        "growth",
        "new_formal_seeds",
    ):
        if boundary[key] is not False:
            raise RuntimeError(f"M3L-1 unexpectedly enables {key}")
    if boundary["consumed_m3r_checkpoints_only"] is not True:
        raise RuntimeError("M3L-1 must use consumed M3R checkpoints only")

    data_manifest = _load_json(data_dir / "manifest.json")
    if data_manifest.get("format") != "minicells.native-clm-v0.m3r-address-diagnostic-data.v1":
        raise RuntimeError("unexpected M3L-1 input data manifest")
    if not data_manifest.get("exact_parent_snapshot_verified"):
        raise RuntimeError("exact M3R data snapshot was not verified")
    if data_manifest.get("parent_manifest_sha256") != protocol["parent_m3r"]["data_manifest_sha256"]:
        raise RuntimeError("M3L-1 parent data-manifest SHA mismatch")

    checkpoint_manifest = _load_json(checkpoint_dir / "manifest.json")
    if checkpoint_manifest.get("format") != "minicells.native-clm-v0.m3r-address-diagnostic-checkpoints.v1":
        raise RuntimeError("unexpected M3L-1 checkpoint manifest")
    if checkpoint_manifest.get("revision") != protocol["parent_m3r"]["hf_revision"]:
        raise RuntimeError("M3L-1 checkpoint HF revision mismatch")
    seeds = sorted(int(record["seed"]) for record in checkpoint_manifest["records"])
    if seeds != [73611, 73612, 73613]:
        raise RuntimeError(f"unexpected consumed M3R seed set: {seeds}")
    for path in _eval_paths(data_dir).values():
        if not path.exists():
            raise FileNotFoundError(path)
    return data_manifest, checkpoint_manifest


def _identity_check(result: dict, identity: dict) -> dict:
    if identity.get("format") != "minicells.native-clm-v0.m3l1-address-state-capacity.identity.v1":
        raise RuntimeError("unexpected M3L-1 identity file")
    rank16 = result["capacity_curve"]["rank-16"]
    checks = {
        "median_auc": (
            float(rank16["median"]),
            float(identity["rank16_parent_median_auc"]),
        ),
        "fraction_auc_ge_floor": (
            float(rank16["fraction_auc_ge_floor"]),
            float(identity["rank16_parent_fraction_auc_ge_floor"]),
        ),
        "median_normalized_oracle_excess_recovery": (
            float(rank16["median_normalized_oracle_excess_recovery"]),
            float(identity["rank16_parent_median_normalized_oracle_excess_recovery"]),
        ),
        "median_old_fpr": (
            float(rank16["median_old_fpr"]),
            float(identity["rank16_parent_median_old_fpr"]),
        ),
        "median_current_tpr": (
            float(rank16["median_current_tpr"]),
            float(identity["rank16_parent_median_current_tpr"]),
        ),
    }
    tolerance = float(identity["absolute_metric_tolerance"])
    failures: list[str] = []
    observed: dict[str, dict[str, float]] = {}
    for name, (actual, expected) in checks.items():
        delta = abs(actual - expected)
        observed[name] = {"actual": actual, "expected": expected, "absolute_delta": delta}
        if not math.isfinite(actual) or delta > tolerance:
            failures.append(f"{name}: actual={actual} expected={expected} delta={delta}")
    actual_bytes = float(rank16["median_historical_address_state_bytes"])
    expected_bytes = float(identity["rank16_parent_median_sketch_bytes"])
    byte_delta = abs(actual_bytes - expected_bytes)
    observed["median_historical_address_state_bytes"] = {
        "actual": actual_bytes,
        "expected": expected_bytes,
        "absolute_delta": byte_delta,
    }
    if byte_delta > float(identity["storage_bytes_tolerance"]):
        failures.append(
            f"storage bytes: actual={actual_bytes} expected={expected_bytes} delta={byte_delta}"
        )
    if failures:
        raise RuntimeError("M3L-1 rank-16 identity drift: " + "; ".join(failures))
    return {
        "passed": True,
        "metric_tolerance": tolerance,
        "checks": observed,
    }


def _write_results_markdown(result: dict, path: Path) -> None:
    lines = [
        "# Native CLM v0 — M3L-1 Historical Address-State Capacity",
        "",
        f"- Classification: `{result['classification']}`",
        "- Scientific decision: `False` (checkpoint-only mechanism diagnostic)",
        f"- Valid edges: `{result['valid_edge_count']}/{result['edge_count']}`",
        f"- Offline oracle median AUC: `{result['offline_oracle']['median']:.4f}`",
        f"- Minimum passing low rank: `{result['minimum_passing_low_rank']}`",
        f"- Full covariance passes M3L gates: `{result['full_covariance_passes']}`",
        f"- Rank-16 parent identity: `{result['rank16_parent_identity']['passed']}`",
        "",
        "## Capacity curve",
        "",
        "| Candidate | Median AUC | >=0.85 | Recovery | Old FPR | Current TPR | Historical bytes | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    order = ["rank-0", "rank-8", "rank-16", "rank-32", "rank-64", "rank-128", "full-covariance"]
    for label in order:
        row = result["capacity_curve"][label]
        lines.append(
            f"| {label} | {row['median']:.4f} | {row['fraction_auc_ge_floor']:.3f} | "
            f"{row['median_normalized_oracle_excess_recovery']:.4f} | {row['median_old_fpr']:.4f} | "
            f"{row['median_current_tpr']:.4f} | {row['median_historical_address_state_bytes']:.0f} | "
            f"{'PASS' if row['passes_m3l_feasibility_gates'] else 'FAIL'} |"
        )
    lines.extend(["", "## Transition medians", ""])
    for transition, candidates in result["transition_capacity_curves"].items():
        compact = ", ".join(
            f"{label}={candidates[label]['median']:.4f}"
            for label in order
            if label in candidates
        )
        lines.append(f"- `{transition}`: {compact}")
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            result["interpretation"],
            "",
            (
                "Boundary: consumed M3R checkpoints only; no Native CLM parameter update, growth, "
                "or new formal seed. Raw historical queries are reduced into each registered candidate "
                "address state before gate construction and are otherwise used only by the offline evaluator."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _worker(args: argparse.Namespace) -> int:
    protocol = _load_json(args.protocol)
    config = _config_from_protocol(protocol)
    _, checkpoint_manifest = _validate_inputs(protocol, args.data_dir, args.checkpoint_dir)
    seed = int(args.worker_seed)
    record = _checkpoint_record(checkpoint_manifest, seed)
    output_path = args.output_dir / f"seed-{seed}" / "diagnostic.json"
    summary = diagnose_m3l1_seed(
        checkpoint_path=record["path"],
        expected_checkpoint_sha256=record["sha256"],
        eval_paths=_eval_paths(args.data_dir),
        output_path=output_path,
        seed=seed,
        config=config,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "seed": seed,
                "valid_edges": summary["valid_edge_count"],
                "edge_count": summary["edge_count"],
                "device": args.device,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _run_parallel(args: argparse.Namespace, seeds: list[int]) -> None:
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise RuntimeError("at least one M3L-1 diagnostic device is required")
    pending = list(seeds)
    active: list[tuple[int, subprocess.Popen, str]] = []
    while pending or active:
        busy = {device for _, _, device in active}
        free = [device for device in devices if device not in busy]
        while pending and free:
            seed = pending.pop(0)
            device = free.pop(0)
            output = args.output_dir / f"seed-{seed}" / "diagnostic.json"
            if output.exists():
                try:
                    cached = _load_json(output)
                    if (
                        cached.get("seed") == seed
                        and cached.get("format")
                        == "minicells.native-clm-v0.m3l1-address-state-capacity.seed.v1"
                    ):
                        print(f"Reusing completed M3L-1 diagnostic seed={seed}.", flush=True)
                        free.insert(0, device)
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            command = [
                sys.executable,
                __file__,
                "--protocol",
                str(args.protocol),
                "--identity",
                str(args.identity),
                "--data-dir",
                str(args.data_dir),
                "--checkpoint-dir",
                str(args.checkpoint_dir),
                "--output-dir",
                str(args.output_dir),
                "--worker-seed",
                str(seed),
                "--device",
                device,
            ]
            print("+", " ".join(command), flush=True)
            active.append((seed, subprocess.Popen(command), device))
        still_active: list[tuple[int, subprocess.Popen, str]] = []
        for seed, process, device in active:
            code = process.poll()
            if code is None:
                still_active.append((seed, process, device))
                continue
            if code != 0:
                raise RuntimeError(
                    f"M3L-1 diagnostic seed={seed} on {device} failed with {code}"
                )
        active = still_active
        if active:
            time.sleep(0.5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--worker-seed", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.worker_seed is not None:
        return _worker(args)

    protocol = _load_json(args.protocol)
    config = _config_from_protocol(protocol)
    data_manifest, _ = _validate_inputs(protocol, args.data_dir, args.checkpoint_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [73611, 73612, 73613]
    _run_parallel(args, seeds)
    seed_summaries = [
        _load_json(args.output_dir / f"seed-{seed}" / "diagnostic.json") for seed in seeds
    ]
    result = aggregate_m3l1_capacity(
        seed_summaries,
        config=config,
        parent_m3l_commit=protocol["parent_m3l"]["publish_commit"],
        parent_m3r_hf_revision=protocol["parent_m3r"]["hf_revision"],
    )
    result["rank16_parent_identity"] = _identity_check(result, _load_json(args.identity))
    result["protocol"] = asdict(config)
    result["diagnostic_data_manifest"] = data_manifest
    (args.output_dir / "diagnostic-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_m3l1_capacity_csv(seed_summaries, args.output_dir / "capacity-curve.csv")
    _write_results_markdown(result, args.output_dir / "RESULTS.md")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
