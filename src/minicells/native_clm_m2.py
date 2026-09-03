"""Native CLM v0 M2: replay-free continual-language evaluation.

M2 freezes the trained M1 shared substrate and router. Only persistent Cell operators
are writable. The protected arm projects Cell gradients through each Cell's
certificate nullspace; the unsafe control performs the same writes without projection.
Both arms start from the exact same M1 checkpoint and see the same B -> C -> D stream
with zero learner-side replay.
"""

from __future__ import annotations

from collections.abc import Iterator
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .native_clm_train import ByteSequenceDataset
from .native_clm_v0 import NativeCLM


@dataclass(frozen=True)
class NativeCLMM2Config:
    batch_size: int = 16
    steps_per_phase: int = 400
    eval_batches: int = 20
    log_interval: int = 50
    warmup_steps: int = 40
    lr_cells: float = 8e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    certificate_update_interval: int = 50
    precision: str = "fp16"
    num_workers: int = 0

    def validate(self) -> None:
        if self.batch_size < 1 or self.steps_per_phase < 1:
            raise ValueError("batch_size and steps_per_phase must be positive")
        if self.eval_batches < 1 or self.log_interval < 1:
            raise ValueError("eval_batches and log_interval must be positive")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_state_sha256(named_tensors: list[tuple[str, Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda pair: pair[0]):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def frozen_state_sha256(model: NativeCLM) -> str:
    cell_weight_names = {f"cellular.cells.{idx}.weight" for idx in range(model.cell_count)}
    tensors = [
        (name, value)
        for name, value in model.state_dict().items()
        if name not in cell_weight_names
        and not name.endswith("certificate_basis")
        and not name.endswith("certificate_rank")
        and not name.endswith("usage_count")
    ]
    return _tensor_state_sha256(tensors)


def cell_weight_sha256(model: NativeCLM) -> str:
    return _tensor_state_sha256(
        [(f"cell-{idx}", cell.weight) for idx, cell in enumerate(model.cellular.cells)]
    )


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _cycle(loader: DataLoader) -> Iterator[tuple[Tensor, Tensor]]:
    while True:
        yield from loader


def _loader(
    path: str | Path,
    *,
    seq_len: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = ByteSequenceDataset(path, seq_len=seq_len)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def evaluate_domain(
    model: NativeCLM,
    path: str | Path,
    *,
    device: torch.device,
    config: NativeCLMM2Config,
) -> dict[str, Any]:
    loader = _loader(
        path,
        seq_len=model.config.max_seq_len,
        batch_size=config.batch_size,
        shuffle=False,
        seed=0,
        num_workers=config.num_workers,
    )
    iterator = _cycle(loader)
    was_training = model.training
    model.eval()
    losses: list[float] = []
    entropies: list[float] = []
    usage = torch.zeros(model.cell_count, dtype=torch.long)
    active_fractions: list[float] = []
    for _ in range(config.eval_batches):
        x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        with _autocast(device, config.precision):
            out = model(x, y, return_info=True)
        losses.append(float(out["loss"].detach().cpu()))
        info = out["cell_info"]
        idx = info["top_idx"].detach().cpu().reshape(-1)
        usage += torch.bincount(idx, minlength=model.cell_count)
        entropies.append(float(info["route_entropy"]))
        active_fractions.append(float(info["active_fraction_vs_dense"]))
    if was_training:
        model.train()
    mean_loss = float(sum(losses) / len(losses))
    total = int(usage.sum().item())
    usage_share = [float(v / total) for v in usage.tolist()] if total else [0.0] * model.cell_count
    return {
        "loss": mean_loss,
        "perplexity": float(math.exp(min(20.0, mean_loss))),
        "route_entropy": float(sum(entropies) / len(entropies)),
        "active_fraction_vs_dense": float(sum(active_fractions) / len(active_fractions)),
        "cell_usage_share": usage_share,
    }


def evaluate_matrix(
    model: NativeCLM,
    eval_paths: dict[str, str | Path],
    *,
    device: torch.device,
    config: NativeCLMM2Config,
) -> dict[str, dict[str, Any]]:
    return {
        domain: evaluate_domain(model, path, device=device, config=config)
        for domain, path in eval_paths.items()
    }


def _lr_factor(step: int, config: NativeCLMM2Config) -> float:
    if step < config.warmup_steps:
        return max(1e-3, (step + 1) / max(1, config.warmup_steps))
    progress = (step - config.warmup_steps) / max(1, config.steps_per_phase - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine


def _freeze_to_cell_only(model: NativeCLM) -> list[torch.nn.Parameter]:
    groups = model.parameter_groups()
    for parameter in groups["shared"] + groups["router"]:
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in groups["cells"]:
        parameter.requires_grad_(True)
    return groups["cells"]


def _train_phase(
    model: NativeCLM,
    train_path: str | Path,
    *,
    device: torch.device,
    config: NativeCLMM2Config,
    seed: int,
    protected: bool,
    phase: str,
) -> dict[str, Any]:
    cell_parameters = _freeze_to_cell_only(model)
    optimizer = torch.optim.AdamW(
        cell_parameters,
        lr=config.lr_cells,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    loader = _loader(
        train_path,
        seq_len=model.config.max_seq_len,
        batch_size=config.batch_size,
        shuffle=True,
        seed=seed,
        num_workers=config.num_workers,
    )
    iterator = _cycle(loader)
    scaler_enabled = device.type == "cuda" and config.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    losses: list[float] = []
    projection_ratios: list[float] = []
    certificate_additions = 0
    start = time.time()
    model.train()

    for step in range(1, config.steps_per_phase + 1):
        factor = _lr_factor(step - 1, config)
        optimizer.param_groups[0]["lr"] = config.lr_cells * factor
        x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, config.precision):
            out = model(x, y, return_info=True)
            loss = out["loss"]
        scaler.scale(loss).backward()
        if scaler_enabled:
            scaler.unscale_(optimizer)

        if protected:
            ratios = model.project_cell_gradients_()
            projection_ratios.extend(ratios.values())
        else:
            projection_ratios.extend([1.0] * model.cell_count)

        torch.nn.utils.clip_grad_norm_(cell_parameters, config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if config.certificate_update_interval > 0 and step % config.certificate_update_interval == 0:
            certificate_additions += model.update_certificates(out["cell_info"])

        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % config.log_interval == 0 or step == config.steps_per_phase:
            print(
                f"[m2 {phase} {'protected' if protected else 'unsafe'}] "
                f"step={step}/{config.steps_per_phase} loss={losses[-1]:.6f}",
                flush=True,
            )

    return {
        "phase": phase,
        "steps": config.steps_per_phase,
        "mean_train_loss": float(sum(losses) / len(losses)),
        "final_train_loss": losses[-1],
        "projection_ratio_mean": float(sum(projection_ratios) / max(1, len(projection_ratios))),
        "projection_ratio_min": float(min(projection_ratios or [1.0])),
        "certificate_additions": int(certificate_additions),
        "elapsed_seconds": time.time() - start,
        "optimizer_reset_at_phase_start": True,
        "learner_replay_bytes": 0,
    }


def run_arm(
    *,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    train_paths: dict[str, str | Path],
    eval_paths: dict[str, str | Path],
    output_dir: str | Path,
    arm: str,
    seed: int,
    config: NativeCLMM2Config,
    device: str = "cuda",
) -> dict[str, Any]:
    if arm not in {"protected", "unsafe"}:
        raise ValueError("arm must be protected or unsafe")
    config.validate()
    checkpoint_path = Path(checkpoint_path)
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != expected_checkpoint_sha256:
        raise RuntimeError(
            f"M1 checkpoint SHA mismatch: expected {expected_checkpoint_sha256}, got {actual_sha}"
        )
    if tuple(train_paths) != ("B", "C", "D"):
        raise ValueError("registered M2 train stream must be B -> C -> D")
    if set(eval_paths) != {"A", "B", "C", "D"}:
        raise ValueError("registered M2 evaluation domains must be A/B/C/D")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    target_device = torch.device(device)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model, m1_extra = NativeCLM.load_checkpoint(checkpoint_path, map_location="cpu")
    if model.cell_count != 8 or model.config.active_cells != 2:
        raise RuntimeError("M2 requires canonical M1 topology: 8 Cells / 2 active")
    if model.parameter_count()["total"] != 12_154_368:
        raise RuntimeError("M2 requires canonical 12,154,368-parameter M1 model")
    model.to(target_device)
    _freeze_to_cell_only(model)

    frozen_before = frozen_state_sha256(model)
    cell_before = cell_weight_sha256(model)
    base_cell_weights = [cell.weight.detach().cpu().clone() for cell in model.cellular.cells]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matrices: dict[str, Any] = {}
    matrices["initial"] = evaluate_matrix(model, eval_paths, device=target_device, config=config)
    phase_summaries: list[dict[str, Any]] = []

    for index, phase in enumerate(("B", "C", "D")):
        phase_summary = _train_phase(
            model,
            train_paths[phase],
            device=target_device,
            config=config,
            seed=seed + 100 * (index + 1),
            protected=arm == "protected",
            phase=phase,
        )
        phase_summaries.append(phase_summary)
        matrices[f"after_{phase}"] = evaluate_matrix(
            model, eval_paths, device=target_device, config=config
        )

    frozen_after = frozen_state_sha256(model)
    drifts = []
    for baseline, cell in zip(base_cell_weights, model.cellular.cells):
        current = cell.weight.detach().cpu()
        denom = float(torch.linalg.vector_norm(baseline).item()) + 1e-12
        drifts.append(float(torch.linalg.vector_norm(current - baseline).item()) / denom)

    final_checkpoint = output / "final.pt"
    model.save_checkpoint(
        final_checkpoint,
        extra={
            "milestone": "M2",
            "arm": arm,
            "seed": seed,
            "parent_checkpoint_sha256": actual_sha,
            "stream": ["B", "C", "D"],
            "learner_replay_bytes": 0,
            "cell_only_writes": True,
            "optimizer_state_carried_between_phases": False,
        },
    )

    summary = {
        "format": "minicells.native-clm-v0.m2-arm-summary.v1",
        "arm": arm,
        "seed": seed,
        "protected": arm == "protected",
        "parent_checkpoint_sha256": actual_sha,
        "parent_m1_extra_keys": sorted(m1_extra.keys()),
        "parameter_count": model.parameter_count(),
        "cell_count": model.cell_count,
        "active_cells": model.config.active_cells,
        "cell_only_writes": True,
        "shared_and_router_frozen": frozen_before == frozen_after,
        "frozen_state_sha256_before": frozen_before,
        "frozen_state_sha256_after": frozen_after,
        "cell_weight_sha256_before": cell_before,
        "cell_weight_sha256_after": cell_weight_sha256(model),
        "cell_relative_weight_drift": drifts,
        "certificate": model.certificate_summary(),
        "stream": ["B", "C", "D"],
        "learner_replay_bytes": 0,
        "phase_summaries": phase_summaries,
        "evaluation_matrix": matrices,
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "final_checkpoint_bytes": final_checkpoint.stat().st_size,
        "m2_config": asdict(config),
    }
    (output / "arm-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_phase_csv(output / "phase-summary.csv", phase_summaries)
    return summary


def _write_phase_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _phase_gain(summary: dict[str, Any], phase: str) -> float:
    matrices = summary["evaluation_matrix"]
    before_key = {"B": "initial", "C": "after_B", "D": "after_C"}[phase]
    before = float(matrices[before_key][phase]["loss"])
    after = float(matrices[f"after_{phase}"][phase]["loss"])
    return (before - after) / max(before, 1e-12)


def _forgetting(summary: dict[str, Any]) -> dict[str, float]:
    matrices = summary["evaluation_matrix"]
    references = {
        "A": float(matrices["initial"]["A"]["loss"]),
        "B": float(matrices["after_B"]["B"]["loss"]),
        "C": float(matrices["after_C"]["C"]["loss"]),
    }
    final = matrices["after_D"]
    return {
        domain: max(0.0, float(final[domain]["loss"]) / max(ref, 1e-12) - 1.0)
        for domain, ref in references.items()
    }


def compare_arms(
    protected: dict[str, Any],
    unsafe: dict[str, Any],
    *,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    if protected["seed"] != unsafe["seed"]:
        raise ValueError("arm seeds do not match")
    if protected["parent_checkpoint_sha256"] != unsafe["parent_checkpoint_sha256"]:
        raise ValueError("arms did not start from the same checkpoint")

    p_gains = {phase: _phase_gain(protected, phase) for phase in ("B", "C", "D")}
    u_gains = {phase: _phase_gain(unsafe, phase) for phase in ("B", "C", "D")}
    p_forgetting = _forgetting(protected)
    u_forgetting = _forgetting(unsafe)
    p_mean_gain = float(sum(p_gains.values()) / 3.0)
    u_mean_gain = float(sum(u_gains.values()) / 3.0)
    p_mean_forgetting = float(sum(p_forgetting.values()) / 3.0)
    u_mean_forgetting = float(sum(u_forgetting.values()) / 3.0)

    p_a_base = float(protected["evaluation_matrix"]["initial"]["A"]["loss"])
    p_a_final = float(protected["evaluation_matrix"]["after_D"]["A"]["loss"])
    p_a_regression = max(0.0, p_a_final / max(p_a_base, 1e-12) - 1.0)
    max_active_fraction = max(
        float(metrics["active_fraction_vs_dense"])
        for stage in protected["evaluation_matrix"].values()
        for metrics in stage.values()
    )

    gates = {
        "exact_same_m1_checkpoint": protected["parent_checkpoint_sha256"]
        == unsafe["parent_checkpoint_sha256"],
        "cell_only_writes": bool(protected["cell_only_writes"] and unsafe["cell_only_writes"]),
        "shared_and_router_frozen": bool(
            protected["shared_and_router_frozen"] and unsafe["shared_and_router_frozen"]
        ),
        "replay_free_stream": protected["learner_replay_bytes"] == 0
        and unsafe["learner_replay_bytes"] == 0,
        "fixed_topology": protected["cell_count"] == unsafe["cell_count"] == 8,
        "sparse_cell_execution": max_active_fraction <= thresholds["max_active_fraction"],
        "protected_phase_plasticity": all(
            gain >= thresholds["min_phase_gain"] for gain in p_gains.values()
        ),
        "protected_absolute_A_retention": p_a_regression <= thresholds["max_A_regression"],
        "unsafe_interference_exposed": u_mean_forgetting
        >= thresholds["min_unsafe_mean_forgetting"],
        "protected_retention_advantage": (
            u_mean_forgetting - p_mean_forgetting
            >= thresholds["min_retention_advantage"]
        ),
        "protected_plasticity_preserved": p_mean_gain
        >= thresholds["min_plasticity_ratio_vs_unsafe"] * max(u_mean_gain, 1e-12),
    }
    return {
        "seed": protected["seed"],
        "protected_phase_gains": p_gains,
        "unsafe_phase_gains": u_gains,
        "protected_forgetting": p_forgetting,
        "unsafe_forgetting": u_forgetting,
        "protected_mean_plasticity": p_mean_gain,
        "unsafe_mean_plasticity": u_mean_gain,
        "protected_mean_forgetting": p_mean_forgetting,
        "unsafe_mean_forgetting": u_mean_forgetting,
        "protected_A_regression": p_a_regression,
        "retention_advantage": u_mean_forgetting - p_mean_forgetting,
        "gates": gates,
        "pass": all(gates.values()),
    }


def aggregate_formal(
    seed_results: list[dict[str, Any]],
    *,
    protocol_sha256: str,
    formal_seeds: list[int],
) -> dict[str, Any]:
    completed = sorted(int(result["seed"]) for result in seed_results)
    if completed != sorted(formal_seeds):
        raise RuntimeError(
            f"formal seed mismatch: expected {sorted(formal_seeds)}, got {completed}"
        )
    all_pass = all(result["pass"] for result in seed_results)
    status = (
        "NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_SUPPORTED"
        if all_pass
        else "NATIVE_CLM_V0_M2_REPLAY_FREE_CONTINUAL_LANGUAGE_NOT_SUPPORTED"
    )
    return {
        "format": "minicells.native-clm-v0.m2-decision.v1",
        "status": status,
        "scientific_decision": bool(all_pass),
        "protocol_sha256": protocol_sha256,
        "formal_seeds": formal_seeds,
        "completed_seeds": completed,
        "seed_results": seed_results,
        "all_registered_gates_pass": bool(all_pass),
        "claim_boundary": (
            "fixed 8-Cell topology; frozen M1 shared substrate/router; Cell-only protected "
            "writes; B->C->D sequential language stream; zero learner-side replay; no "
            "autonomous growth"
        ),
        "next_milestone_if_supported": "M3 autonomous Cell growth",
    }
