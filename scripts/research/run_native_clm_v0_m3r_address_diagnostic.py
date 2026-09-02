"""Run checkpoint-only M3R lineage address diagnostics, using two GPUs when available."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from minicells.native_clm_m3r_address_diag import (
    AddressDiagnosticConfig,
    aggregate_diagnostic,
    diagnose_seed,
    write_edge_csv,
)

DEFAULT_PROTOCOL = Path("research/validations/native-clm-v0-m3r-address-diagnostic/protocol.json")
DEFAULT_DATA = Path("/kaggle/working/native-clm-m3r-address-data")
DEFAULT_CHECKPOINTS = Path("/kaggle/working/native-clm-m3r-address-checkpoints")
DEFAULT_OUTPUT = Path("artifacts/experiments/native-clm-v0-m3r-address-diagnostic")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _config_from_protocol(protocol: dict) -> AddressDiagnosticConfig:
    sampling = protocol["sampling"]
    training = protocol["probe_training"]
    thresholds = protocol["classification_thresholds"]
    return AddressDiagnosticConfig(
        max_batches_per_domain=int(sampling["max_batches_per_domain"]),
        batch_size=int(sampling["batch_size"]),
        max_samples_per_class_per_edge=int(sampling["max_samples_per_class_per_edge"]),
        minimum_samples_per_class_per_edge=int(sampling["minimum_samples_per_class_per_edge"]),
        train_fraction=float(sampling["train_fraction"]),
        probe_steps=int(training["steps"]),
        probe_learning_rate=float(training["learning_rate"]),
        probe_weight_decay=float(training["weight_decay"]),
        probe_split_seed_base=int(sampling["probe_split_seed_base"]),
        minimum_valid_edge_fraction=float(thresholds["minimum_valid_edge_fraction"]),
        separable_median_auc=float(thresholds["separable_median_auc"]),
        separable_edge_auc_floor=float(thresholds["separable_edge_auc_floor"]),
        minimum_fraction_edges_above_floor=float(thresholds["minimum_fraction_edges_above_floor"]),
    )


def _checkpoint_record(checkpoint_manifest: dict, seed: int) -> dict:
    matches = [
        record
        for record in checkpoint_manifest["records"]
        if int(record["seed"]) == int(seed)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one diagnostic lineage checkpoint for seed {seed}")
    return matches[0]


def _eval_paths(data_dir: Path) -> dict[str, Path]:
    return {
        "A": data_dir / "A-tinystories-eval.txt",
        "B": data_dir / "B-wikitext-eval.txt",
        "C": data_dir / "C-code-eval.txt",
        "D": data_dir / "D-dolly-eval.txt",
    }


def _validate_inputs(protocol: dict, data_dir: Path, checkpoint_dir: Path) -> tuple[dict, dict]:
    if protocol.get("format") != "minicells.native-clm-v0.m3r-address-diagnostic.protocol.v1":
        raise RuntimeError("unexpected address-diagnostic protocol")
    if protocol.get("status") != "REGISTERED_DIAGNOSTIC_READY":
        raise RuntimeError("address diagnostic is not registered as ready")
    if protocol["learner_boundary"]["model_training"] is not False:
        raise RuntimeError("diagnostic protocol unexpectedly enables model training")

    data_manifest = _load_json(data_dir / "manifest.json")
    if data_manifest.get("format") != "minicells.native-clm-v0.m3r-address-diagnostic-data.v1":
        raise RuntimeError("unexpected diagnostic data manifest")
    if not data_manifest.get("exact_parent_snapshot_verified"):
        raise RuntimeError("exact M3R parent data snapshot was not verified")
    expected_parent_data_sha = protocol["parent_m3r"]["data_manifest_sha256"]
    if data_manifest.get("parent_manifest_sha256") != expected_parent_data_sha:
        raise RuntimeError(
            "parent data manifest SHA mismatch: "
            f"{data_manifest.get('parent_manifest_sha256')} != {expected_parent_data_sha}"
        )

    checkpoint_manifest = _load_json(checkpoint_dir / "manifest.json")
    if checkpoint_manifest.get("format") != "minicells.native-clm-v0.m3r-address-diagnostic-checkpoints.v1":
        raise RuntimeError("unexpected diagnostic checkpoint manifest")
    if checkpoint_manifest.get("revision") != protocol["parent_m3r"]["hf_revision"]:
        raise RuntimeError("M3R diagnostic checkpoint revision mismatch")
    seeds = sorted(int(record["seed"]) for record in checkpoint_manifest["records"])
    if seeds != [73611, 73612, 73613]:
        raise RuntimeError(f"unexpected M3R checkpoint seed set: {seeds}")
    for path in _eval_paths(data_dir).values():
        if not path.exists():
            raise FileNotFoundError(path)
    return data_manifest, checkpoint_manifest


def _write_results_markdown(result: dict, path: Path) -> None:
    features = result["features"]
    lines = [
        "# Native CLM v0 — M3R Address Diagnostic",
        "",
        f"- Classification: `{result['classification']}`",
        "- Scientific decision: `False` (diagnostic only)",
        f"- Valid edges: `{result['valid_edge_count']}/{result['edge_count']}`",
        f"- Current cosine median AUC: `{result['current_cosine']['median_auc']:.4f}`",
        "",
        "| feature | median AUC | mean AUC | fraction >= floor |",
        "|---|---:|---:|---:|",
    ]
    for name, metrics in features.items():
        lines.append(
            f"| {name} | {metrics['median_auc']:.4f} | {metrics['mean_auc']:.4f} | "
            f"{metrics['fraction_auc_ge_floor']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            result["interpretation"],
            "",
            (
                "Boundary: checkpoint-only offline diagnostic over consumed M3R formal checkpoints; "
                "no Native CLM training, routing update, certificate update, growth, or new formal "
                "seed consumption occurred."
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
    summary = diagnose_seed(
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
        raise RuntimeError("at least one diagnostic device is required")
    pending = list(seeds)
    active: list[tuple[int, subprocess.Popen, str]] = []
    while pending or active:
        while pending and len(active) < len(devices):
            seed = pending.pop(0)
            device = devices[len(active)]
            output = args.output_dir / f"seed-{seed}" / "diagnostic.json"
            if output.exists():
                try:
                    cached = _load_json(output)
                    if (
                        cached.get("seed") == seed
                        and cached.get("format")
                        == "minicells.native-clm-v0.m3r-address-diagnostic.seed.v1"
                    ):
                        print(f"Reusing completed diagnostic seed={seed}.", flush=True)
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
                raise RuntimeError(
                    f"diagnostic worker seed={seed} on {device} failed with code {code}"
                )
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
    result = aggregate_diagnostic(
        seed_summaries,
        config=config,
        parent_protocol_sha256=protocol["parent_m3r"]["protocol_sha256"],
        parent_data_manifest_sha256=protocol["parent_m3r"]["data_manifest_sha256"],
        hf_revision=protocol["parent_m3r"]["hf_revision"],
    )
    result["protocol"] = asdict(config)
    result["diagnostic_data_manifest"] = data_manifest
    (args.output_dir / "diagnostic-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_edge_csv(seed_summaries, args.output_dir / "edge-metrics.csv")
    _write_results_markdown(result, args.output_dir / "RESULTS.md")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
