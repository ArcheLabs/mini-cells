"""Run checkpoint-only M3L replay-free query-sketch gate diagnostics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from minicells.native_clm_m3l_gate import (
    M3LQuerySketchConfig,
    aggregate_m3l_diagnostic,
    diagnose_m3l_seed,
    write_m3l_edge_csv,
)

DEFAULT_PROTOCOL = Path("research/validations/native-clm-v0-m3l-query-sketch-gate/protocol.json")
DEFAULT_DATA = Path("/kaggle/working/native-clm-m3r-address-data")
DEFAULT_CHECKPOINTS = Path("/kaggle/working/native-clm-m3r-address-checkpoints")
DEFAULT_OUTPUT = Path("artifacts/experiments/native-clm-v0-m3l-query-sketch-gate")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _config_from_protocol(protocol: dict) -> M3LQuerySketchConfig:
    sampling = protocol["sampling"]
    sketch = protocol["historical_query_sketch"]
    gate = protocol["sketch_gate"]
    oracle = protocol["offline_oracle"]
    thresholds = protocol["classification_thresholds"]
    return M3LQuerySketchConfig(
        max_batches_per_domain=int(sampling["max_batches_per_domain"]),
        batch_size=int(sampling["batch_size"]),
        max_samples_per_domain_per_edge=int(sampling["max_samples_per_domain_per_edge"]),
        minimum_train_samples_per_side=int(sampling["minimum_train_samples_per_side"]),
        minimum_test_samples_per_side=int(sampling["minimum_test_samples_per_side"]),
        train_group_fraction=float(sampling["train_group_fraction"]),
        split_seed_base=int(sampling["split_seed_base"]),
        sketch_rank=int(sketch["rank"]),
        diagonal_regularization=float(gate["diagonal_regularization"]),
        target_sketch_old_fpr=float(gate["target_sketch_old_fpr"]),
        oracle_steps=int(oracle["steps"]),
        oracle_learning_rate=float(oracle["learning_rate"]),
        oracle_weight_decay=float(oracle["weight_decay"]),
        minimum_valid_edge_fraction=float(thresholds["minimum_valid_edge_fraction"]),
        oracle_separable_median_auc=float(thresholds["oracle_separable_median_auc"]),
        oracle_edge_auc_floor=float(thresholds["oracle_edge_auc_floor"]),
        minimum_fraction_oracle_edges_above_floor=float(
            thresholds["minimum_fraction_oracle_edges_above_floor"]
        ),
        sketch_gate_median_auc=float(thresholds["sketch_gate_median_auc"]),
        sketch_gate_edge_auc_floor=float(thresholds["sketch_gate_edge_auc_floor"]),
        minimum_fraction_sketch_edges_above_floor=float(
            thresholds["minimum_fraction_sketch_edges_above_floor"]
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
    if protocol.get("format") != "minicells.native-clm-v0.m3l-query-sketch-gate.protocol.v1":
        raise RuntimeError("unexpected M3L protocol")
    if protocol.get("status") != "REGISTERED_DIAGNOSTIC_READY":
        raise RuntimeError("M3L diagnostic protocol is not registered as ready")
    boundary = protocol["learner_boundary"]
    if boundary["native_clm_training"] is not False:
        raise RuntimeError("M3L unexpectedly enables Native CLM training")
    if boundary["old_raw_tokens_allowed_during_gate_fit"] is not False:
        raise RuntimeError("M3L unexpectedly allows old-token replay")
    if boundary["old_raw_query_samples_allowed_during_gate_fit"] is not False:
        raise RuntimeError("M3L unexpectedly allows old-query replay in gate fitting")

    data_manifest = _load_json(data_dir / "manifest.json")
    if data_manifest.get("format") != "minicells.native-clm-v0.m3r-address-diagnostic-data.v1":
        raise RuntimeError("unexpected M3L input data manifest")
    if not data_manifest.get("exact_parent_snapshot_verified"):
        raise RuntimeError("exact M3R data snapshot was not verified")
    expected_data_sha = protocol["parent_m3r"]["data_manifest_sha256"]
    if data_manifest.get("parent_manifest_sha256") != expected_data_sha:
        raise RuntimeError("M3L parent data-manifest SHA mismatch")

    checkpoint_manifest = _load_json(checkpoint_dir / "manifest.json")
    if checkpoint_manifest.get("format") != "minicells.native-clm-v0.m3r-address-diagnostic-checkpoints.v1":
        raise RuntimeError("unexpected M3L checkpoint manifest")
    if checkpoint_manifest.get("revision") != protocol["parent_m3r"]["hf_revision"]:
        raise RuntimeError("M3L checkpoint HF revision mismatch")
    seeds = sorted(int(record["seed"]) for record in checkpoint_manifest["records"])
    if seeds != [73611, 73612, 73613]:
        raise RuntimeError(f"unexpected consumed M3R seed set: {seeds}")
    for path in _eval_paths(data_dir).values():
        if not path.exists():
            raise FileNotFoundError(path)
    return data_manifest, checkpoint_manifest


def _write_results_markdown(result: dict, path: Path) -> None:
    lines = [
        "# Native CLM v0 — M3L Query-Sketch Gate Diagnostic",
        "",
        f"- Classification: `{result['classification']}`",
        "- Scientific decision: `False` (mechanism diagnostic only)",
        f"- Valid edges: `{result['valid_edge_count']}/{result['edge_count']}`",
        f"- Current cosine median AUC: `{result['current_cosine']['median']:.4f}`",
        f"- Offline oracle median AUC: `{result['offline_oracle']['median']:.4f}`",
        f"- Query-sketch gate median AUC: `{result['sketch_gate']['median']:.4f}`",
        f"- Median normalized oracle recovery: `{result['sketch_gate']['median_normalized_oracle_excess_recovery']:.4f}`",
        f"- Median heldout old FPR: `{result['sketch_gate']['median_old_fpr']:.4f}`",
        f"- Median heldout current TPR: `{result['sketch_gate']['median_current_tpr']:.4f}`",
        f"- Median historical sketch bytes: `{result['sketch_gate']['median_sketch_bytes']:.0f}`",
        "",
        "Interpretation:",
        "",
        result["interpretation"],
        "",
        (
            "Boundary: consumed M3R checkpoints only; Native CLM parameters are frozen, old raw "
            "queries are used only to construct the historical sketch and heldout evaluator, and "
            "the sketch-derived gate itself receives no old token/query replay."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _worker(args: argparse.Namespace) -> int:
    protocol = _load_json(args.protocol)
    config = _config_from_protocol(protocol)
    _, checkpoint_manifest = _validate_inputs(protocol, args.data_dir, args.checkpoint_dir)
    seed = int(args.worker_seed)
    record = _checkpoint_record(checkpoint_manifest, seed)
    output_path = args.output_dir / f"seed-{seed}" / "diagnostic.json"
    summary = diagnose_m3l_seed(
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
        raise RuntimeError("at least one M3L diagnostic device is required")
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
                        == "minicells.native-clm-v0.m3l-query-sketch-gate.seed.v1"
                    ):
                        print(f"Reusing completed M3L diagnostic seed={seed}.", flush=True)
                        free.insert(0, device)
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            command = [
                sys.executable,
                __file__,
                "--protocol",
                str(args.protocol),
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
                raise RuntimeError(f"M3L diagnostic seed={seed} on {device} failed with {code}")
        active = still_active
        if active:
            time.sleep(0.5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
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
    result = aggregate_m3l_diagnostic(
        seed_summaries,
        config=config,
        parent_m3r_hf_revision=protocol["parent_m3r"]["hf_revision"],
        parent_address_commit=protocol["parent_address_diagnostic"]["publish_commit"],
    )
    result["protocol"] = asdict(config)
    result["diagnostic_data_manifest"] = data_manifest
    (args.output_dir / "diagnostic-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_m3l_edge_csv(seed_summaries, args.output_dir / "edge-metrics.csv")
    _write_results_markdown(result, args.output_dir / "RESULTS.md")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
