"""Run frozen M3W-0 checkpoint-only write-drift restoration diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from minicells.native_clm_m2 import NativeCLMM2Config, evaluate_matrix, sha256_file
from minicells.native_clm_m3l2 import OnlineAddressNativeCLM
from minicells.native_clm_m3w0 import (
    M3W0Thresholds,
    analyze_factorial,
    classify_results,
    restore_operator_groups,
)
from minicells.native_clm_v0 import NativeCLM

DEFAULT_PROTOCOL = Path(
    "research/validations/native-clm-v0-m3w0-write-drift-restoration/protocol.json"
)
DEFAULT_OUTPUT = Path("artifacts/experiments/native-clm-v0-m3w0-write-drift-restoration")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("format") != "minicells.native-clm-v0.m3w0-write-drift-restoration.protocol.v1":
        raise RuntimeError("unexpected M3W-0 protocol format")
    if protocol.get("status") != "FROZEN_UNRUN" or protocol.get("scientific_decision") is not False:
        raise RuntimeError("M3W-0 protocol boundary drift")
    return protocol, hashlib.sha256(raw).hexdigest()


def _validate_data(data_dir: Path, protocol: dict) -> str:
    manifest_path = data_dir / "manifest.json"
    raw = manifest_path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    expected = str(protocol["parent_evidence"]["m3l2_data_manifest_sha256"])
    if actual != expected:
        raise RuntimeError(f"M3W-0 data manifest SHA mismatch: {actual} != {expected}")
    manifest = json.loads(raw)
    if manifest.get("format") != "minicells.native-clm-v0.m3l2-data-manifest.v1":
        raise RuntimeError("unexpected M3W-0 source data format")
    for key in ("A_eval", "B_eval", "C_eval", "D_eval"):
        record = manifest["files"][key]
        path = data_dir / record["path"]
        if not path.exists() or path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"M3W-0 data file size mismatch: {key}")
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"M3W-0 data file SHA mismatch: {key}")
    return actual


def _eval_paths(data_dir: Path) -> dict[str, Path]:
    return {
        "A": data_dir / "A-tinystories-eval.txt",
        "B": data_dir / "B-wikitext-eval.txt",
        "C": data_dir / "C-code-eval.txt",
        "D": data_dir / "D-dolly-eval.txt",
    }


def _eval_config(protocol: dict) -> NativeCLMM2Config:
    registered = protocol["evaluation"]
    return NativeCLMM2Config(
        batch_size=int(registered["batch_size"]),
        eval_batches=int(registered["eval_batches"]),
        precision=str(registered["precision"]),
        num_workers=int(registered["num_workers"]),
    )


def _thresholds(protocol: dict) -> M3W0Thresholds:
    return M3W0Thresholds(**protocol["thresholds"])


def _checkpoint_record(protocol: dict, seed: int) -> dict:
    matches = [record for record in protocol["treatment_checkpoints"] if int(record["seed"]) == seed]
    if len(matches) != 1:
        raise RuntimeError(f"missing registered M3W-0 checkpoint seed={seed}")
    return matches[0]


def _load_final(checkpoint: Path, device: torch.device) -> OnlineAddressNativeCLM:
    model, _ = OnlineAddressNativeCLM.load_checkpoint(checkpoint, map_location="cpu")
    model.to(device)
    model.eval()
    return model


def _evaluate_counterfactual(
    checkpoint: Path,
    m1: NativeCLM,
    *,
    device: torch.device,
    eval_paths: dict[str, Path],
    config: NativeCLMM2Config,
    restore_roots: bool,
    restore_descendants: bool,
) -> dict:
    model = _load_final(checkpoint, device)
    restoration = restore_operator_groups(
        model,
        m1,
        restore_roots=restore_roots,
        restore_descendants=restore_descendants,
    )
    matrix = evaluate_matrix(model, eval_paths, device=device, config=config)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"restoration": restoration, "matrix": matrix}


def run_seed(
    *,
    seed: int,
    checkpoint: Path,
    m1_path: Path,
    data_dir: Path,
    output_dir: Path,
    protocol: dict,
    protocol_sha256: str,
    data_manifest_sha256: str,
    device_name: str,
) -> dict:
    record = _checkpoint_record(protocol, seed)
    if sha256_file(checkpoint) != record["sha256"] or checkpoint.stat().st_size != int(record["bytes"]):
        raise RuntimeError(f"M3W-0 checkpoint identity mismatch seed={seed}")
    m1_expected = protocol["m1_checkpoint"]["sha256"]
    if sha256_file(m1_path) != m1_expected:
        raise RuntimeError("M3W-0 M1 checkpoint identity mismatch")

    device = torch.device(device_name)
    eval_paths = _eval_paths(data_dir)
    config = _eval_config(protocol)
    thresholds = _thresholds(protocol)

    m1, _ = NativeCLM.load_checkpoint(m1_path, map_location="cpu")
    m1.to(device)
    m1.eval()
    if m1.cell_count != int(protocol["m1_checkpoint"]["initial_cells"]):
        raise RuntimeError("M3W-0 M1 root count mismatch")
    m1_matrix = evaluate_matrix(m1, eval_paths, device=device, config=config)

    final_model = _load_final(checkpoint, device)
    if final_model.lineage_root_count != m1.cell_count:
        raise RuntimeError("M3W-0 final lineage root count mismatch")
    final_matrix = evaluate_matrix(final_model, eval_paths, device=device, config=config)
    del final_model

    root_restore = _evaluate_counterfactual(
        checkpoint,
        m1,
        device=device,
        eval_paths=eval_paths,
        config=config,
        restore_roots=True,
        restore_descendants=False,
    )
    descendant_restore = _evaluate_counterfactual(
        checkpoint,
        m1,
        device=device,
        eval_paths=eval_paths,
        config=config,
        restore_roots=False,
        restore_descendants=True,
    )
    all_restore = _evaluate_counterfactual(
        checkpoint,
        m1,
        device=device,
        eval_paths=eval_paths,
        config=config,
        restore_roots=True,
        restore_descendants=True,
    )

    result = analyze_factorial(
        seed=seed,
        m1_matrix=m1_matrix,
        final_matrix=final_matrix,
        root_restore_matrix=root_restore["matrix"],
        descendant_root_restore_matrix=descendant_restore["matrix"],
        all_lineage_restore_matrix=all_restore["matrix"],
        thresholds=thresholds,
    )
    result["protocol_sha256"] = protocol_sha256
    result["data_manifest_sha256"] = data_manifest_sha256
    result["source_checkpoint_sha256"] = record["sha256"]
    result["source_checkpoint_hf_path"] = record["hf_path"]
    result["restorations"] = {
        "root_restore": root_restore["restoration"],
        "descendant_root_restore": descendant_restore["restoration"],
        "all_lineage_restore": all_restore["restoration"],
    }

    seed_dir = output_dir / f"seed-{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "seed": seed,
                "identity_ok": result["identity_ok"],
                "A_root_fraction": result["A_root_fraction"],
                "A_descendant_fraction": result["A_descendant_fraction"],
                "root_restore_gain_retention": result["root_restore_new_domain_gain_retention"],
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def _worker(args, protocol: dict, protocol_sha: str, data_sha: str) -> int:
    manifest = json.loads((args.checkpoints_dir / "manifest.json").read_text(encoding="utf-8"))
    matches = [record for record in manifest["records"] if int(record["seed"]) == args.seed]
    if len(matches) != 1:
        raise RuntimeError(f"M3W-0 local checkpoint manifest lacks seed={args.seed}")
    run_seed(
        seed=args.seed,
        checkpoint=Path(matches[0]["path"]),
        m1_path=args.m1_checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        protocol=protocol,
        protocol_sha256=protocol_sha,
        data_manifest_sha256=data_sha,
        device_name=args.device,
    )
    return 0


def _spawn_worker(args, seed: int, gpu: str) -> subprocess.Popen:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--seed",
        str(seed),
        "--m1-checkpoint",
        str(args.m1_checkpoint),
        "--checkpoints-dir",
        str(args.checkpoints_dir),
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
    return subprocess.Popen(command, env=env)


def _aggregate(output_dir: Path, protocol: dict, protocol_sha: str, data_sha: str) -> dict:
    seeds = [int(record["seed"]) for record in protocol["treatment_checkpoints"]]
    results = [
        json.loads((output_dir / f"seed-{seed}" / "diagnostic.json").read_text(encoding="utf-8"))
        for seed in seeds
    ]
    classification = classify_results(results, _thresholds(protocol))
    decision = {
        "format": "minicells.native-clm-v0.m3w0-write-drift-restoration.result.v1",
        "classification": classification,
        "scientific_decision": False,
        "native_clm_training": False,
        "new_formal_seeds_consumed": False,
        "source_m3l2_formal_seeds": seeds,
        "protocol_sha256": protocol_sha,
        "data_manifest_sha256": data_sha,
        "seed_results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostic-result.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "write-attribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "seed",
            "root_fraction",
            "descendant_fraction",
            "root_shapley",
            "descendant_shapley",
            "all_lineage_A_recovery",
            "B_gain_retention_root_restore",
            "C_gain_retention_root_restore",
            "D_gain_retention_root_restore",
        ])
        for result in results:
            retained = result["root_restore_new_domain_gain_retention"]
            writer.writerow([
                result["seed"],
                result["A_root_fraction"],
                result["A_descendant_fraction"],
                result["A_root_shapley"],
                result["A_descendant_shapley"],
                result["A_all_lineage_excess_recovery_fraction"],
                retained["B"],
                retained["C"],
                retained["D"],
            ])

    lines = [
        "# Native CLM v0 M3W-0 — Root Write-Drift Counterfactual Restoration",
        "",
        f"- Classification: `{classification}`",
        "- Scientific decision: `False`",
        "- Native CLM training: `False`",
        "- New formal seeds consumed: `False`",
        f"- Source M3L-2 seeds: `{seeds}`",
        f"- Protocol SHA-256: `{protocol_sha}`",
        f"- Data manifest SHA-256: `{data_sha}`",
        "",
        "| seed | root fraction | descendant fraction | all-lineage A recovery | B gain kept | C gain kept | D gain kept |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        retained = result["root_restore_new_domain_gain_retention"]
        lines.append(
            "| {seed} | {root:.4f} | {desc:.4f} | {recovery:.4f} | {b:.4f} | {c:.4f} | {d:.4f} |".format(
                seed=result["seed"],
                root=result["A_root_fraction"],
                desc=result["A_descendant_fraction"],
                recovery=result["A_all_lineage_excess_recovery_fraction"],
                b=retained["B"],
                c=retained["C"],
                d=retained["D"],
            )
        )
    lines += [
        "",
        "Boundary: this is a checkpoint-only 2x2 operator-restoration diagnostic. It does not reconstruct unavailable child birth tensors and does not train or update any model.",
    ]
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"classification": classification, "seeds": seeds}, indent=2), flush=True)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--m1-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    protocol, protocol_sha = _load_protocol(args.protocol)
    data_sha = _validate_data(args.data_dir, protocol)
    if args.worker:
        if args.seed is None:
            raise RuntimeError("M3W-0 worker requires --seed")
        return _worker(args, protocol, protocol_sha, data_sha)

    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    if len(devices) < 2:
        raise RuntimeError("canonical M3W-0 runner requires two devices")
    seeds = [int(record["seed"]) for record in protocol["treatment_checkpoints"]]
    pending = list(seeds)
    active: dict[str, tuple[int, subprocess.Popen]] = {}
    while pending or active:
        for gpu in devices:
            if gpu in active or not pending:
                continue
            seed = pending.pop(0)
            active[gpu] = (seed, _spawn_worker(args, seed, gpu))
        finished: list[str] = []
        for gpu, (seed, process) in active.items():
            code = process.poll()
            if code is None:
                continue
            if code != 0:
                raise RuntimeError(f"M3W-0 seed {seed} worker failed with returncode={code}")
            finished.append(gpu)
        for gpu in finished:
            del active[gpu]
        if active and not finished:
            time.sleep(1.0)

    _aggregate(args.output_dir, protocol, protocol_sha, data_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
