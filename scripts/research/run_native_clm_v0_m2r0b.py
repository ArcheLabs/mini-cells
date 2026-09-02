"""Run the frozen Native CLM v0 M2-R0b numerical-reference audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
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
from minicells.native_clm_m2r0 import project_realized_updates_, snapshot_cell_weights
from minicells.native_clm_m2r0b import (
    M2R0BNumericalThresholds,
    classify_numerical_reference,
    diagnose_optimizer_mechanics,
    measure_numerical_reference,
    snapshot_cell_gradients,
    summarize_numerical_rows,
)
from minicells.native_clm_v0 import NativeCLM

DEFAULT_PROTOCOL = Path(
    "research/validations/native-clm-v0-m2r0b-numerical-reference-audit/protocol.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/experiments/native-clm-v0-m2r0b-numerical-reference-audit"
)
ARMS = (
    "current_adamw_grad_projection",
    "adamw_no_decay_grad_projection",
    "sgd_no_decay_grad_projection",
    "sgd_with_decay_grad_projection",
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
    if protocol.get("format") != "minicells.native-clm-v0.m2r0b-numerical-reference-audit.protocol.v1":
        raise RuntimeError("unexpected M2-R0b protocol format")
    if protocol.get("status") != "FROZEN_UNRUN":
        raise RuntimeError("M2-R0b protocol is not frozen/unrun")
    if protocol.get("scientific_decision") is not False:
        raise RuntimeError("M2-R0b must remain a non-decision diagnostic")
    if protocol.get("new_formal_seeds_consumed") is not False:
        raise RuntimeError("M2-R0b must consume no formal seed")
    if protocol.get("parent_m2", {}).get("formal_seeds_forbidden_in_r0b") is not True:
        raise RuntimeError("M2-R0b formal-seed guard is not frozen")
    if tuple(protocol.get("arms", {})) != ARMS:
        raise RuntimeError("M2-R0b frozen optimizer arm order mismatch")
    return protocol, hashlib.sha256(raw).hexdigest()


def _validate_inputs(
    *,
    protocol: dict[str, Any],
    checkpoint: Path,
    data_dir: Path,
) -> tuple[Path, str]:
    expected_m1 = str(protocol["m1_checkpoint"]["sha256"])
    if not checkpoint.exists() or sha256_file(checkpoint) != expected_m1:
        raise RuntimeError("M2-R0b exact M1 checkpoint identity mismatch")

    train_record = protocol["data"]
    train_path = data_dir / str(train_record["file"])
    if not train_path.exists() or train_path.stat().st_size != int(train_record["bytes"]):
        raise RuntimeError("M2-R0b B-train file size mismatch")
    if _sha256(train_path) != str(train_record["sha256"]):
        raise RuntimeError("M2-R0b B-train SHA mismatch")

    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("M2-R0b data manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != train_record["source_manifest_format"]:
        raise RuntimeError("M2-R0b data manifest format mismatch")
    revision = manifest.get("dataset_revisions", {}).get("B", {}).get("resolved_revision")
    if revision != train_record["resolved_revision"]:
        raise RuntimeError("M2-R0b WikiText revision mismatch")
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


def _thresholds(protocol: dict[str, Any]) -> M2R0BNumericalThresholds:
    return M2R0BNumericalThresholds(**protocol["thresholds"])


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
    raise ValueError(f"unsupported M2-R0b optimizer for arm {arm}")


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
        raise ValueError("unregistered M2-R0b arm")

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
        raise RuntimeError("M2-R0b canonical Cell count mismatch")
    if model.config.active_cells != int(protocol["m1_checkpoint"]["active_cells"]):
        raise RuntimeError("M2-R0b canonical active-Cell count mismatch")
    if model.parameter_count()["total"] != int(protocol["m1_checkpoint"]["parameters"]):
        raise RuntimeError("M2-R0b canonical parameter count mismatch")
    model.to(device)
    cell_parameters = _freeze_to_cell_only(model)
    frozen_before = _state_sha(model)
    certificate_ranks_before = [cell.rank for cell in model.cellular.cells]
    if not any(rank > 0 for rank in certificate_ranks_before):
        raise RuntimeError("M2-R0b M1 checkpoint has no certificate coverage")

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
    minimum_update_norm = float(protocol["measurement"]["minimum_update_norm_for_ratio"])
    thresholds = _thresholds(protocol)

    for step in range(1, int(audit["steps_per_arm"]) + 1):
        factor = _lr_factor(step - 1, config)
        for group in optimizer.param_groups:
            group["lr"] = config.lr_cells * factor
        current_lr = float(optimizer.param_groups[0]["lr"])

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
        gradient_updates = snapshot_cell_gradients(model, lr=current_lr)
        before = snapshot_cell_weights(model)

        scaler.step(optimizer)
        scaler.update()
        raw_after = snapshot_cell_weights(model)

        if bool(protocol["arms"][arm]["actual_update_projection"]):
            final_update_retained.extend(project_realized_updates_(model, before))

        rows.extend(
            measure_numerical_reference(
                model,
                before,
                raw_after,
                gradient_updates,
                arm=arm,
                step=step,
                roundoff_bound_multiplier=thresholds.roundoff_bound_multiplier,
                minimum_update_norm=minimum_update_norm,
            )
        )
        losses.append(float(loss.detach().cpu()))

    frozen_after = _state_sha(model)
    certificate_ranks_after = [cell.rank for cell in model.cellular.cells]
    if frozen_after != frozen_before:
        raise RuntimeError("M2-R0b changed frozen/router/certificate state")
    if certificate_ranks_after != certificate_ranks_before:
        raise RuntimeError("M2-R0b changed certificate ranks")

    summary = summarize_numerical_rows(rows)
    summary.update(
        {
            "format": "minicells.native-clm-v0.m2r0b-arm-result.v1",
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
            "minimum_update_norm_for_ratio": minimum_update_norm,
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
    with (arm_dir / "numerical-reference.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(
        json.dumps(
            {
                "arm": arm,
                "audited_cell_updates": summary["audited_cell_updates"],
                "committed_rho_p95": summary["committed_rho_p95"],
                "committed_excess_factor_p95": summary["committed_excess_factor_p95"],
                "committed_excess_factor_max": summary["committed_excess_factor_max"],
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


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
    thresholds = _thresholds(protocol)
    classification = classify_numerical_reference(summaries, thresholds)
    mechanics = diagnose_optimizer_mechanics(summaries, thresholds)
    m2_r1_unblocked = (
        classification == "R0_REFERENCE_FAILURE_EXPLAINED_BY_PARAMETER_TRANSACTION_ROUNDOFF"
    )
    result = {
        "format": "minicells.native-clm-v0.m2r0b-numerical-reference-audit.result.v1",
        "classification": classification,
        "mechanics_diagnosis": mechanics,
        "m2_r1_unblocked": m2_r1_unblocked,
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

    with (output_dir / "arm-summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "arm",
                "audited_cell_updates",
                "gradient_analytic_rho_p95",
                "gradient_float_commit_rho_p95",
                "optimizer_raw_rho_p95",
                "matched_safe_float_commit_rho_p95",
                "committed_rho_p95",
                "committed_excess_factor_p95",
                "committed_excess_factor_max",
            ]
        )
        for arm in ARMS:
            summary = summaries[arm]
            writer.writerow(
                [
                    arm,
                    summary["audited_cell_updates"],
                    summary["gradient_analytic_rho_p95"],
                    summary["gradient_float_commit_rho_p95"],
                    summary["optimizer_raw_rho_p95"],
                    summary["matched_safe_float_commit_rho_p95"],
                    summary["committed_rho_p95"],
                    summary["committed_excess_factor_p95"],
                    summary["committed_excess_factor_max"],
                ]
            )

    lines = [
        "# Native CLM v0 M2-R0b — Numerical Reference Audit",
        "",
        f"- Classification: `{classification}`",
        f"- Mechanics diagnosis: `{mechanics}`",
        f"- M2-R1 unblocked: `{m2_r1_unblocked}`",
        "- Scientific decision: `False`",
        "- Native CLM training milestone: `False`",
        "- New formal seeds consumed: `False`",
        f"- Protocol SHA-256: `{protocol_sha256}`",
        f"- Data manifest SHA-256: `{data_manifest_sha256}`",
        "",
        "| arm | n | grad analytic p95 rho | grad float-commit p95 rho | optimizer raw p95 rho | matched-safe float p95 rho | committed p95 rho | excess p95 | excess max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        summary = summaries[arm]
        lines.append(
            "| {arm} | {n} | {ga:.6g} | {gf:.6g} | {raw:.6g} | {floor:.6g} | {committed:.6g} | {ep95:.6g} | {emax:.6g} |".format(
                arm=arm,
                n=summary["audited_cell_updates"],
                ga=summary["gradient_analytic_rho_p95"],
                gf=summary["gradient_float_commit_rho_p95"],
                raw=summary["optimizer_raw_rho_p95"],
                floor=summary["matched_safe_float_commit_rho_p95"],
                committed=summary["committed_rho_p95"],
                ep95=summary["committed_excess_factor_p95"],
                emax=summary["committed_excess_factor_max"],
            )
        )
    lines.extend(
        [
            "",
            "Boundary: M2-R0b is a numerical/optimizer-mechanics diagnostic only. It does not change the historical M2 decision and does not establish certificate coverage or continual-learning success.",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()

    protocol, protocol_sha256 = _load_protocol(args.protocol)
    train_path, data_manifest_sha256 = _validate_inputs(
        protocol=protocol,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arms = (args.arm,) if args.arm else ARMS
    for arm in arms:
        run_arm(
            arm=arm,
            checkpoint=args.checkpoint,
            train_path=train_path,
            output_dir=args.output_dir,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            data_manifest_sha256=data_manifest_sha256,
            device_name=args.device,
        )

    if args.arm is None:
        result = _aggregate(
            output_dir=args.output_dir,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            data_manifest_sha256=data_manifest_sha256,
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
