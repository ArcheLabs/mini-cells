"""Run the frozen Native CLM v0 M2-R0 protected-update invariant audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from minicells.native_clm_m2 import (
    NativeCLMM2Config,
    _autocast,
    _cycle,
    _freeze_to_cell_only,
    _loader,
    _lr_factor,
    sha256_file,
)
from minicells.native_clm_m2r0 import (
    M2R0Thresholds,
    classify_optimizer_invariant,
    measure_realized_update_invariant,
    project_realized_updates_,
    snapshot_cell_weights,
    summarize_invariant_rows,
)
from minicells.native_clm_v0 import NativeCLM

DEFAULT_PROTOCOL = Path(
    "research/validations/native-clm-v0-m2r0-update-invariant-audit/protocol.json"
)
DEFAULT_OUTPUT = Path("artifacts/experiments/native-clm-v0-m2r0-update-invariant-audit")
ARMS = (
    "current_adamw_grad_projection",
    "adamw_no_decay_grad_projection",
    "sgd_no_decay_grad_projection",
    "adamw_final_update_projection",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("format") != "minicells.native-clm-v0.m2r0-update-invariant-audit.protocol.v1":
        raise RuntimeError("unexpected M2-R0 protocol format")
    if protocol.get("status") != "FROZEN_UNRUN":
        raise RuntimeError("M2-R0 protocol is not frozen/unrun")
    if protocol.get("scientific_decision") is not False:
        raise RuntimeError("M2-R0 must remain a non-decision diagnostic")
    if protocol.get("new_formal_seeds_consumed") is not False:
        raise RuntimeError("M2-R0 must consume no formal seed")
    return protocol, hashlib.sha256(raw).hexdigest()


def _validate_inputs(
    *,
    protocol: dict[str, Any],
    checkpoint: Path,
    data_dir: Path,
) -> tuple[Path, str]:
    expected_m1 = str(protocol["m1_checkpoint"]["sha256"])
    if not checkpoint.exists() or sha256_file(checkpoint) != expected_m1:
        raise RuntimeError("M2-R0 exact M1 checkpoint identity mismatch")

    train_record = protocol["data"]
    train_path = data_dir / str(train_record["file"])
    if not train_path.exists() or train_path.stat().st_size != int(train_record["bytes"]):
        raise RuntimeError("M2-R0 B-train file size mismatch")
    if _sha256(train_path) != str(train_record["sha256"]):
        raise RuntimeError("M2-R0 B-train SHA mismatch")

    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("M2-R0 data manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    revision = manifest.get("dataset_revisions", {}).get("B", {}).get("resolved_revision")
    if revision != train_record["resolved_revision"]:
        raise RuntimeError("M2-R0 WikiText revision mismatch")
    return train_path, _sha256(manifest_path)


def _m2_config(protocol: dict[str, Any]) -> NativeCLMM2Config:
    audit = protocol["audit"]
    return NativeCLMM2Config(
        batch_size=int(audit["batch_size"]),
        steps_per_phase=int(audit["registered_m2_steps_per_phase"]),
        warmup_steps=int(audit["warmup_steps"]),
        lr_cells=float(audit["lr"]),
        min_lr_ratio=float(audit["min_lr_ratio"]),
        weight_decay=float(audit["canonical_weight_decay"]),
        grad_clip=float(audit["grad_clip"]),
        precision=str(audit["precision"]),
        num_workers=0,
    )


def _thresholds(protocol: dict[str, Any]) -> M2R0Thresholds:
    return M2R0Thresholds(**protocol["thresholds"])


def _optimizer(
    arm: str,
    parameters: list[torch.nn.Parameter],
    protocol: dict[str, Any],
) -> torch.optim.Optimizer:
    arm_config = protocol["arms"][arm]
    audit = protocol["audit"]
    lr = float(audit["lr"])
    if arm_config["optimizer"] == "AdamW":
        betas = tuple(float(value) for value in audit["adam_betas"])
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            betas=betas,
            weight_decay=float(arm_config["weight_decay"]),
        )
    if arm_config["optimizer"] == "SGD":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=float(arm_config["momentum"]),
            weight_decay=float(arm_config["weight_decay"]),
        )
    raise ValueError(f"unsupported M2-R0 optimizer for arm {arm}")


def _state_sha(model: NativeCLM) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith("cellular.cells.") and name.endswith(".weight"):
            continue
        if name.endswith("usage_count"):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run_arm(
    *,
    arm: str,
    checkpoint: Path,
    train_path: Path,
    output_dir: Path,
    protocol: dict[str, Any],
    protocol_sha256: str,
    data_manifest_sha256: str,
    device_name: str,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("unregistered M2-R0 arm")

    audit = protocol["audit"]
    seed = int(audit["audit_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model, _ = NativeCLM.load_checkpoint(checkpoint, map_location="cpu")
    if model.cell_count != int(protocol["m1_checkpoint"]["initial_cells"]):
        raise RuntimeError("M2-R0 canonical Cell count mismatch")
    if model.config.active_cells != int(protocol["m1_checkpoint"]["active_cells"]):
        raise RuntimeError("M2-R0 canonical active-Cell count mismatch")
    if model.parameter_count()["total"] != int(protocol["m1_checkpoint"]["parameters"]):
        raise RuntimeError("M2-R0 canonical parameter count mismatch")
    model.to(device)
    cell_parameters = _freeze_to_cell_only(model)
    frozen_before = _state_sha(model)
    certificate_ranks_before = [cell.rank for cell in model.cellular.cells]
    if not any(rank > 0 for rank in certificate_ranks_before):
        raise RuntimeError("M2-R0 M1 checkpoint has no certificate coverage")

    config = _m2_config(protocol)
    loader = _loader(
        train_path,
        seq_len=model.config.max_seq_len,
        batch_size=config.batch_size,
        shuffle=True,
        seed=seed,
        num_workers=config.num_workers,
    )
    iterator = _cycle(loader)
    optimizer = _optimizer(arm, cell_parameters, protocol)
    scaler_enabled = device.type == "cuda" and config.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    rows: list[dict[str, Any]] = []
    projection_ratios: list[float] = []
    final_update_retained: list[float] = []
    losses: list[float] = []
    started = time.time()
    model.train()

    for step in range(1, int(audit["steps_per_arm"]) + 1):
        factor = _lr_factor(step - 1, config)
        for group in optimizer.param_groups:
            group["lr"] = config.lr_cells * factor
        x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, config.precision):
            out = model(x, y, return_info=True)
            loss = out["loss"]
        scaler.scale(loss).backward()
        if scaler_enabled:
            scaler.unscale_(optimizer)

        ratios = model.project_cell_gradients_()
        projection_ratios.extend(float(value) for value in ratios.values())
        torch.nn.utils.clip_grad_norm_(cell_parameters, config.grad_clip)
        before = snapshot_cell_weights(model)
        scaler.step(optimizer)
        scaler.update()

        if bool(protocol["arms"][arm]["actual_update_projection"]):
            final_update_retained.extend(project_realized_updates_(model, before))

        rows.extend(
            measure_realized_update_invariant(
                model,
                before,
                arm=arm,
                step=step,
            )
        )
        losses.append(float(loss.detach().cpu()))

    frozen_after = _state_sha(model)
    certificate_ranks_after = [cell.rank for cell in model.cellular.cells]
    if frozen_after != frozen_before:
        raise RuntimeError("M2-R0 changed frozen/router/certificate state")
    if certificate_ranks_after != certificate_ranks_before:
        raise RuntimeError("M2-R0 changed certificate ranks")

    summary = summarize_invariant_rows(rows)
    summary.update(
        {
            "format": "minicells.native-clm-v0.m2r0-arm-result.v1",
            "arm": arm,
            "protocol_sha256": protocol_sha256,
            "data_manifest_sha256": data_manifest_sha256,
            "M1_checkpoint_sha256": sha256_file(checkpoint),
            "audit_seed": seed,
            "steps": int(audit["steps_per_arm"]),
            "same_batches_each_arm": True,
            "mean_train_loss": sum(losses) / len(losses),
            "final_train_loss": losses[-1],
            "gradient_projection_ratio_mean": (
                sum(projection_ratios) / len(projection_ratios)
                if projection_ratios
                else 1.0
            ),
            "final_update_projection_retained_ratio_mean": (
                sum(final_update_retained) / len(final_update_retained)
                if final_update_retained
                else None
            ),
            "certificate_ranks": certificate_ranks_before,
            "certificate_updates": 0,
            "learner_replay_bytes": 0,
            "native_clm_formal_training": False,
            "elapsed_seconds": time.time() - started,
        }
    )
    arm_dir = output_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / "arm-result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (arm_dir / "update-invariant.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(
        json.dumps(
            {
                "arm": arm,
                "audited_cell_updates": summary["audited_cell_updates"],
                "violation_p95": summary["violation_ratio_p95"],
                "violation_max": summary["violation_ratio_max"],
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


def _spawn_worker(
    args: argparse.Namespace,
    *,
    arm: str,
    gpu: str,
) -> subprocess.Popen:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--arm",
        arm,
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
    print("+", " ".join(command), f"CUDA_VISIBLE_DEVICES={gpu}", flush=True)
    return subprocess.Popen(command, env=env)


def _aggregate(
    *,
    output_dir: Path,
    protocol: dict[str, Any],
    protocol_sha256: str,
    data_manifest_sha256: str,
) -> dict[str, Any]:
    summaries = {
        arm: json.loads((output_dir / arm / "arm-result.json").read_text(encoding="utf-8"))
        for arm in ARMS
    }
    classification = classify_optimizer_invariant(summaries, _thresholds(protocol))
    result = {
        "format": "minicells.native-clm-v0.m2r0-update-invariant-audit.result.v1",
        "classification": classification,
        "scientific_decision": False,
        "native_clm_training_milestone": False,
        "new_formal_seeds_consumed": False,
        "protocol_sha256": protocol_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "arms": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostic-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "arm-summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "arm",
                "audited_cell_updates",
                "rank_min",
                "rank_max",
                "violation_mean",
                "violation_median",
                "violation_p95",
                "violation_max",
                "update_norm_mean",
            ]
        )
        for arm in ARMS:
            summary = summaries[arm]
            writer.writerow(
                [
                    arm,
                    summary["audited_cell_updates"],
                    summary["certificate_rank_min"],
                    summary["certificate_rank_max"],
                    summary["violation_ratio_mean"],
                    summary["violation_ratio_median"],
                    summary["violation_ratio_p95"],
                    summary["violation_ratio_max"],
                    summary["update_norm_mean"],
                ]
            )

    lines = [
        "# Native CLM v0 M2-R0 — Protected Update Invariant Audit",
        "",
        f"- Classification: `{classification}`",
        "- Scientific decision: `False`",
        "- New formal seeds consumed: `False`",
        f"- Protocol SHA-256: `{protocol_sha256}`",
        f"- Data manifest SHA-256: `{data_manifest_sha256}`",
        "",
        "| arm | audited updates | rank min/max | mean rho | p95 rho | max rho |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        summary = summaries[arm]
        lines.append(
            "| {arm} | {n} | {rmin}/{rmax} | {mean:.6g} | {p95:.6g} | {maxv:.6g} |".format(
                arm=arm,
                n=summary["audited_cell_updates"],
                rmin=summary["certificate_rank_min"],
                rmax=summary["certificate_rank_max"],
                mean=summary["violation_ratio_mean"],
                p95=summary["violation_ratio_p95"],
                maxv=summary["violation_ratio_max"],
            )
        )
    lines += [
        "",
        "Boundary: M2-R0 audits optimizer mechanics only. It does not change the historical M2 decision and does not establish certificate coverage or continual-learning success.",
    ]
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"classification": classification}, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    protocol, protocol_sha = _load_protocol(args.protocol)
    train_path, data_manifest_sha = _validate_inputs(
        protocol=protocol,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
    )

    if args.worker:
        if args.arm is None:
            raise RuntimeError("M2-R0 worker requires --arm")
        run_arm(
            arm=args.arm,
            checkpoint=args.checkpoint,
            train_path=train_path,
            output_dir=args.output_dir,
            protocol=protocol,
            protocol_sha256=protocol_sha,
            data_manifest_sha256=data_manifest_sha,
            device_name=args.device,
        )
        return 0

    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    if len(devices) < 2:
        raise RuntimeError("canonical M2-R0 runner requires two devices")

    # Preserve identical batches across arms while using both GPUs: run two arms per wave.
    for start in range(0, len(ARMS), 2):
        processes: list[tuple[str, subprocess.Popen]] = []
        for offset, arm in enumerate(ARMS[start : start + 2]):
            gpu = devices[offset % len(devices)]
            processes.append((arm, _spawn_worker(args, arm=arm, gpu=gpu)))
        for arm, process in processes:
            code = process.wait()
            if code != 0:
                raise RuntimeError(f"M2-R0 arm {arm} failed with return code {code}")

    _aggregate(
        output_dir=args.output_dir,
        protocol=protocol,
        protocol_sha256=protocol_sha,
        data_manifest_sha256=data_manifest_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
