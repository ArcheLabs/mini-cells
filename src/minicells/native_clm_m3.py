"""Native CLM v0 M3: growth-restored replay-free continual language.

M3 keeps the M1 shared substrate and original read-address geometry frozen. Both
registered arms use certificate-projected Cell-local writes. The treatment arm may
spawn context-addressed child Cells when learner-visible protected-write pressure
persists; the fixed control never grows.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .native_clm_m2 import NativeCLMM2Config, evaluate_matrix, sha256_file
from .native_clm_train import ByteSequenceDataset
from .native_clm_v0 import NativeCLM


@dataclass(frozen=True)
class NativeCLMM3GrowthConfig:
    growth_check_interval: int = 50
    growth_cooldown_steps: int = 100
    max_new_cells: int = 8
    max_final_cells: int = 16
    min_parent_route_hits_per_window: int = 512
    min_parent_certificate_rank: int = 8
    max_projected_to_raw_gradient_ratio: float = 0.90
    min_window_train_loss: float = 1.50
    inherit_scale: float = 1.0

    def validate(self) -> None:
        if self.growth_check_interval < 1 or self.growth_cooldown_steps < 0:
            raise ValueError("invalid growth intervals")
        if self.max_new_cells < 1 or self.max_final_cells < 9:
            raise ValueError("growth budget must allow at least one child")
        if self.max_final_cells != 8 + self.max_new_cells:
            raise ValueError("M3 max_final_cells must equal 8 + max_new_cells")
        if self.min_parent_route_hits_per_window < 1:
            raise ValueError("min_parent_route_hits_per_window must be positive")
        if self.min_parent_certificate_rank < 0:
            raise ValueError("min_parent_certificate_rank must be non-negative")
        if not (0.0 <= self.max_projected_to_raw_gradient_ratio <= 1.0):
            raise ValueError("projection ratio threshold must be in [0, 1]")
        if self.inherit_scale != 1.0:
            raise ValueError("registered M3 requires exact parent-operator cloning")


@dataclass
class GrowthWindow:
    d_model: int
    cell_count: int

    def __post_init__(self) -> None:
        self.loss_sum = 0.0
        self.steps = 0
        self.route_hits = [0 for _ in range(self.cell_count)]
        self.ratio_weighted_sum = [0.0 for _ in range(self.cell_count)]
        self.query_sums = [torch.zeros(self.d_model, dtype=torch.float64) for _ in range(self.cell_count)]

    def ensure_cells(self, count: int) -> None:
        while len(self.route_hits) < count:
            self.route_hits.append(0)
            self.ratio_weighted_sum.append(0.0)
            self.query_sums.append(torch.zeros(self.d_model, dtype=torch.float64))
        self.cell_count = count

    @property
    def mean_loss(self) -> float:
        return self.loss_sum / max(1, self.steps)


def _tensor_state_sha256(named_tensors: list[tuple[str, Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda pair: pair[0]):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def original_read_state_sha256(model: NativeCLM, *, original_cells: int = 8) -> str:
    """Hash the shared model, router and original Cell route keys, but not writable Cell state."""

    tensors: list[tuple[str, Tensor]] = []
    for name, value in model.state_dict().items():
        if name.startswith("cellular.cells."):
            parts = name.split(".")
            cell_id = int(parts[2])
            if cell_id >= original_cells:
                continue
            leaf = parts[-1]
            if leaf in {"weight", "certificate_basis", "certificate_rank", "usage_count"}:
                continue
        tensors.append((name, value))
    return _tensor_state_sha256(tensors)


def root_cell_weight_sha256(model: NativeCLM, *, original_cells: int = 8) -> str:
    return _tensor_state_sha256(
        [(f"cell-{idx}", model.cellular.cells[idx].weight) for idx in range(original_cells)]
    )


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _cycle(loader: DataLoader):
    while True:
        yield from loader


def _loader(
    path: str | Path,
    *,
    seq_len: int,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = ByteSequenceDataset(path, seq_len=seq_len)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        generator=generator,
    )


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
    for cell in model.cellular.cells:
        cell.route_key.requires_grad_(False)
        cell.route_key.grad = None
        cell.weight.requires_grad_(True)
    return [cell.weight for cell in model.cellular.cells]


def _make_optimizer(model: NativeCLM, config: NativeCLMM2Config) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        _freeze_to_cell_only(model),
        lr=config.lr_cells,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )


def _observe_growth_window(
    model: NativeCLM,
    window: GrowthWindow,
    info: dict[str, Any],
    projection_ratios: dict[int, float],
    loss_value: float,
    child_post_birth_route_hits: dict[int, int],
) -> None:
    window.ensure_cells(model.cell_count)
    top_idx: Tensor = info["top_idx"]
    top1 = top_idx[..., 0]
    route_input: Tensor = info["route_input"]
    with torch.no_grad():
        query = F.normalize(model.cellular.query_proj(route_input), dim=-1)
        flat_top1 = top1.reshape(-1)
        flat_query = query.reshape(-1, query.size(-1))
        counts = torch.bincount(flat_top1, minlength=model.cell_count)
        sums = torch.zeros(
            model.cell_count,
            query.size(-1),
            device=query.device,
            dtype=query.dtype,
        )
        sums.index_add_(0, flat_top1, flat_query)
        counts_cpu = counts.detach().cpu().tolist()
        sums_cpu = sums.detach().to(dtype=torch.float64, device="cpu")

        active_counts = torch.bincount(top_idx.reshape(-1), minlength=model.cell_count)
        active_counts_cpu = active_counts.detach().cpu().tolist()

    for cell_id in range(model.cell_count):
        hits = int(counts_cpu[cell_id])
        if hits:
            window.route_hits[cell_id] += hits
            window.ratio_weighted_sum[cell_id] += float(projection_ratios[cell_id]) * hits
            window.query_sums[cell_id].add_(sums_cpu[cell_id])
        if cell_id in child_post_birth_route_hits:
            child_post_birth_route_hits[cell_id] += int(active_counts_cpu[cell_id])

    window.loss_sum += float(loss_value)
    window.steps += 1


def _select_parent(
    model: NativeCLM,
    window: GrowthWindow,
    growth: NativeCLMM3GrowthConfig,
) -> dict[str, Any] | None:
    if window.mean_loss < growth.min_window_train_loss:
        return None
    candidates: list[dict[str, Any]] = []
    for cell_id in range(model.cell_count):
        hits = window.route_hits[cell_id]
        if hits < growth.min_parent_route_hits_per_window:
            continue
        cell = model.cellular.cells[cell_id]
        if cell.rank < growth.min_parent_certificate_rank:
            continue
        ratio = window.ratio_weighted_sum[cell_id] / max(1, hits)
        if ratio > growth.max_projected_to_raw_gradient_ratio:
            continue
        score = hits * max(0.0, 1.0 - ratio)
        candidates.append(
            {
                "parent_id": cell_id,
                "route_hits": hits,
                "projection_ratio": float(ratio),
                "certificate_rank": cell.rank,
                "score": float(score),
            }
        )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["score"])


def maybe_spawn_from_pressure(
    model: NativeCLM,
    optimizer: torch.optim.Optimizer,
    window: GrowthWindow,
    growth: NativeCLMM3GrowthConfig,
    *,
    global_step: int,
    last_growth_step: int | None,
    spawned_count: int,
) -> dict[str, Any] | None:
    """Apply the registered learner-visible growth rule. No phase/eval/domain label is accepted."""

    if global_step % growth.growth_check_interval != 0:
        return None
    if spawned_count >= growth.max_new_cells or model.cell_count >= growth.max_final_cells:
        return None
    if last_growth_step is not None and global_step - last_growth_step < growth.growth_cooldown_steps:
        return None
    candidate = _select_parent(model, window, growth)
    if candidate is None:
        return None

    parent_id = int(candidate["parent_id"])
    hits = max(1, int(candidate["route_hits"]))
    route_key = window.query_sums[parent_id] / hits
    route_key = F.normalize(route_key.to(dtype=torch.float32), dim=0)
    child_id = model.spawn_cell(
        parent_id=parent_id,
        route_key=route_key,
        inherit_scale=growth.inherit_scale,
    )
    child = model.cellular.cells[child_id]
    child.route_key.requires_grad_(False)
    child.weight.requires_grad_(True)
    optimizer.add_param_group({"params": [child.weight], "lr": optimizer.param_groups[0]["lr"]})
    window.ensure_cells(model.cell_count)
    return {
        "global_step": int(global_step),
        "parent_id": parent_id,
        "child_id": int(child_id),
        "parent_certificate_rank": int(candidate["certificate_rank"]),
        "parent_route_hits": int(candidate["route_hits"]),
        "parent_projection_ratio": float(candidate["projection_ratio"]),
        "pressure_score": float(candidate["score"]),
        "window_mean_train_loss": float(window.mean_loss),
        "child_initial_certificate_rank": int(child.rank),
        "inherit_scale": float(growth.inherit_scale),
    }


def _train_phase(
    model: NativeCLM,
    train_path: str | Path,
    *,
    device: torch.device,
    train_config: NativeCLMM2Config,
    growth_config: NativeCLMM3GrowthConfig,
    seed: int,
    growth_enabled: bool,
    phase: str,
    global_step_offset: int,
    growth_events: list[dict[str, Any]],
    child_post_birth_route_hits: dict[int, int],
    last_growth_step: int | None,
) -> tuple[dict[str, Any], int | None]:
    optimizer = _make_optimizer(model, train_config)
    loader = _loader(
        train_path,
        seq_len=model.config.max_seq_len,
        batch_size=train_config.batch_size,
        seed=seed,
        num_workers=train_config.num_workers,
    )
    iterator = _cycle(loader)
    scaler_enabled = device.type == "cuda" and train_config.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    losses: list[float] = []
    projection_ratios_all: list[float] = []
    certificate_additions = 0
    start = time.time()
    window = GrowthWindow(model.config.d_model, model.cell_count)
    model.train()

    for step in range(1, train_config.steps_per_phase + 1):
        global_step = global_step_offset + step
        factor = _lr_factor(step - 1, train_config)
        for group in optimizer.param_groups:
            group["lr"] = train_config.lr_cells * factor

        x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, train_config.precision):
            out = model(x, y, return_info=True)
            loss = out["loss"]
        scaler.scale(loss).backward()
        if scaler_enabled:
            scaler.unscale_(optimizer)

        projection_ratios = model.project_cell_gradients_()
        projection_ratios_all.extend(projection_ratios.values())
        _observe_growth_window(
            model,
            window,
            out["cell_info"],
            projection_ratios,
            float(loss.detach().cpu()),
            child_post_birth_route_hits,
        )

        torch.nn.utils.clip_grad_norm_([cell.weight for cell in model.cellular.cells], train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if (
            train_config.certificate_update_interval > 0
            and step % train_config.certificate_update_interval == 0
        ):
            certificate_additions += model.update_certificates(out["cell_info"])

        if growth_enabled and global_step % growth_config.growth_check_interval == 0:
            event = maybe_spawn_from_pressure(
                model,
                optimizer,
                window,
                growth_config,
                global_step=global_step,
                last_growth_step=last_growth_step,
                spawned_count=len(growth_events),
            )
            if event is not None:
                growth_events.append(event)
                child_post_birth_route_hits[event["child_id"]] = 0
                last_growth_step = global_step
                print(
                    "[m3 growth] step={step} parent={parent} child={child} "
                    "ratio={ratio:.4f} loss={loss:.4f}".format(
                        step=global_step,
                        parent=event["parent_id"],
                        child=event["child_id"],
                        ratio=event["parent_projection_ratio"],
                        loss=event["window_mean_train_loss"],
                    ),
                    flush=True,
                )
            window = GrowthWindow(model.config.d_model, model.cell_count)

        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % train_config.log_interval == 0 or step == train_config.steps_per_phase:
            print(
                f"[m3 {phase} {'growth' if growth_enabled else 'fixed'}] "
                f"step={step}/{train_config.steps_per_phase} cells={model.cell_count} "
                f"loss={losses[-1]:.6f}",
                flush=True,
            )

    return (
        {
            "phase": phase,
            "steps": train_config.steps_per_phase,
            "mean_train_loss": float(sum(losses) / len(losses)),
            "final_train_loss": float(losses[-1]),
            "projection_ratio_mean": float(
                sum(projection_ratios_all) / max(1, len(projection_ratios_all))
            ),
            "projection_ratio_min": float(min(projection_ratios_all or [1.0])),
            "certificate_additions": int(certificate_additions),
            "elapsed_seconds": time.time() - start,
            "optimizer_reset_at_phase_start": True,
            "learner_replay_bytes": 0,
            "cell_count_end": model.cell_count,
        },
        last_growth_step,
    )


def run_arm(
    *,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    train_paths: dict[str, str | Path],
    eval_paths: dict[str, str | Path],
    output_dir: str | Path,
    arm: str,
    seed: int,
    train_config: NativeCLMM2Config,
    growth_config: NativeCLMM3GrowthConfig,
    device: str = "cuda",
) -> dict[str, Any]:
    if arm not in {"fixed_protected", "growth_protected"}:
        raise ValueError("arm must be fixed_protected or growth_protected")
    train_config.validate()
    growth_config.validate()
    checkpoint_path = Path(checkpoint_path)
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != expected_checkpoint_sha256:
        raise RuntimeError(
            f"M1 checkpoint SHA mismatch: expected {expected_checkpoint_sha256}, got {actual_sha}"
        )
    if tuple(train_paths) != ("B", "C", "D"):
        raise ValueError("registered M3 train stream must be B -> C -> D")
    if set(eval_paths) != {"A", "B", "C", "D"}:
        raise ValueError("registered M3 evaluation domains must be A/B/C/D")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    target_device = torch.device(device)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model, m1_extra = NativeCLM.load_checkpoint(checkpoint_path, map_location="cpu")
    if model.cell_count != 8 or model.config.active_cells != 2:
        raise RuntimeError("M3 requires canonical M1 topology: 8 Cells / 2 active")
    if model.parameter_count()["total"] != 12_154_368:
        raise RuntimeError("M3 requires canonical 12,154,368-parameter M1 model")
    model.to(target_device)
    _freeze_to_cell_only(model)

    original_read_before = original_read_state_sha256(model)
    root_weights_before = root_cell_weight_sha256(model)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    matrices: dict[str, Any] = {
        "initial": evaluate_matrix(model, eval_paths, device=target_device, config=train_config)
    }
    phase_summaries: list[dict[str, Any]] = []
    growth_events: list[dict[str, Any]] = []
    child_post_birth_route_hits: dict[int, int] = {}
    last_growth_step: int | None = None
    growth_enabled = arm == "growth_protected"

    for index, phase in enumerate(("B", "C", "D")):
        phase_summary, last_growth_step = _train_phase(
            model,
            train_paths[phase],
            device=target_device,
            train_config=train_config,
            growth_config=growth_config,
            seed=seed + 100 * (index + 1),
            growth_enabled=growth_enabled,
            phase=phase,
            global_step_offset=index * train_config.steps_per_phase,
            growth_events=growth_events,
            child_post_birth_route_hits=child_post_birth_route_hits,
            last_growth_step=last_growth_step,
        )
        phase_summaries.append(phase_summary)
        matrices[f"after_{phase}"] = evaluate_matrix(
            model, eval_paths, device=target_device, config=train_config
        )

    original_read_after = original_read_state_sha256(model)
    final_checkpoint = output / "final.pt"
    model.save_checkpoint(
        final_checkpoint,
        extra={
            "milestone": "M3",
            "arm": arm,
            "seed": seed,
            "parent_checkpoint_sha256": actual_sha,
            "stream": ["B", "C", "D"],
            "learner_replay_bytes": 0,
            "growth_enabled": growth_enabled,
            "growth_events": growth_events,
        },
    )

    summary = {
        "format": "minicells.native-clm-v0.m3-arm-summary.v1",
        "arm": arm,
        "seed": seed,
        "growth_enabled": growth_enabled,
        "growth_controller_uses_phase_or_eval_labels": False,
        "parent_checkpoint_sha256": actual_sha,
        "parent_m1_extra_keys": sorted(m1_extra.keys()),
        "initial_cell_count": 8,
        "final_cell_count": model.cell_count,
        "active_cells": model.config.active_cells,
        "learner_replay_bytes": 0,
        "cell_only_writes": True,
        "shared_and_original_router_frozen": original_read_before == original_read_after,
        "original_read_state_sha256_before": original_read_before,
        "original_read_state_sha256_after": original_read_after,
        "root_cell_weight_sha256_before": root_weights_before,
        "root_cell_weight_sha256_after": root_cell_weight_sha256(model),
        "growth_events": growth_events,
        "spawned_cells": len(growth_events),
        "child_post_birth_route_hits": {
            str(key): int(value) for key, value in sorted(child_post_birth_route_hits.items())
        },
        "certificate": model.certificate_summary(),
        "stream": ["B", "C", "D"],
        "phase_summaries": phase_summaries,
        "evaluation_matrix": matrices,
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "final_checkpoint_bytes": final_checkpoint.stat().st_size,
        "training_config": asdict(train_config),
        "growth_config": asdict(growth_config),
    }
    (output / "arm-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


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
    fixed: dict[str, Any],
    growth: dict[str, Any],
    *,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    if fixed["seed"] != growth["seed"]:
        raise ValueError("M3 arm seeds do not match")
    if fixed["parent_checkpoint_sha256"] != growth["parent_checkpoint_sha256"]:
        raise ValueError("M3 arms did not start from the same checkpoint")

    fixed_gains = {phase: _phase_gain(fixed, phase) for phase in ("B", "C", "D")}
    growth_gains = {phase: _phase_gain(growth, phase) for phase in ("B", "C", "D")}
    fixed_forgetting = _forgetting(fixed)
    growth_forgetting = _forgetting(growth)
    fixed_mean_gain = float(sum(fixed_gains.values()) / 3.0)
    growth_mean_gain = float(sum(growth_gains.values()) / 3.0)
    fixed_mean_forgetting = float(sum(fixed_forgetting.values()) / 3.0)
    growth_mean_forgetting = float(sum(growth_forgetting.values()) / 3.0)
    fixed_A = float(fixed_forgetting["A"])
    growth_A = float(growth_forgetting["A"])

    max_active_fraction = max(
        float(metrics["active_fraction_vs_dense"])
        for stage in growth["evaluation_matrix"].values()
        for metrics in stage.values()
    )
    child_hits = [int(value) for value in growth["child_post_birth_route_hits"].values()]
    reused = sum(
        hit >= thresholds["minimum_child_post_birth_route_hits"] for hit in child_hits
    )
    reuse_fraction = float(reused / max(1, len(child_hits))) if child_hits else 0.0

    gates = {
        "exact_same_m1_checkpoint": fixed["parent_checkpoint_sha256"]
        == growth["parent_checkpoint_sha256"],
        "matched_seed_and_data_snapshot": fixed["seed"] == growth["seed"],
        "zero_learner_replay": fixed["learner_replay_bytes"] == 0
        and growth["learner_replay_bytes"] == 0,
        "shared_and_original_router_frozen": bool(
            fixed["shared_and_original_router_frozen"]
            and growth["shared_and_original_router_frozen"]
        ),
        "fixed_control_remains_8_cells": fixed["final_cell_count"] == 8,
        "fixed_control_exposes_capacity_limit": fixed_A
        >= thresholds["minimum_fixed_A_regression_to_expose_capacity_limit"],
        "growth_is_learner_visible_only": not growth[
            "growth_controller_uses_phase_or_eval_labels"
        ],
        "growth_occurs_and_is_bounded": thresholds["minimum_spawned_cells"]
        <= growth["spawned_cells"]
        <= thresholds["maximum_spawned_cells"],
        "children_are_reused": reuse_fraction >= thresholds["minimum_child_reuse_fraction"],
        "sparse_compute_survives_growth": max_active_fraction
        <= thresholds["maximum_active_fraction_vs_dense"],
        "growth_phase_plasticity": all(
            gain >= thresholds["minimum_phase_gain_each_B_C_D"]
            for gain in growth_gains.values()
        ),
        "growth_absolute_A_retention": growth_A
        <= thresholds["maximum_growth_A_regression"],
        "growth_A_retention_advantage": fixed_A - growth_A
        >= thresholds["minimum_A_retention_advantage_vs_fixed"],
        "growth_mean_forgetting": growth_mean_forgetting
        <= thresholds["maximum_growth_mean_forgetting"],
        "growth_plasticity_preserved": growth_mean_gain
        >= thresholds["minimum_growth_to_fixed_plasticity_ratio"] * max(fixed_mean_gain, 1e-12),
    }
    return {
        "seed": fixed["seed"],
        "fixed_phase_gains": fixed_gains,
        "growth_phase_gains": growth_gains,
        "fixed_forgetting": fixed_forgetting,
        "growth_forgetting": growth_forgetting,
        "fixed_mean_plasticity": fixed_mean_gain,
        "growth_mean_plasticity": growth_mean_gain,
        "fixed_mean_forgetting": fixed_mean_forgetting,
        "growth_mean_forgetting": growth_mean_forgetting,
        "fixed_A_regression": fixed_A,
        "growth_A_regression": growth_A,
        "A_retention_advantage": fixed_A - growth_A,
        "spawned_cells": growth["spawned_cells"],
        "final_growth_cell_count": growth["final_cell_count"],
        "child_reuse_fraction": reuse_fraction,
        "gates": gates,
        "pass": all(gates.values()),
    }


def aggregate_formal(
    seed_results: list[dict[str, Any]],
    *,
    protocol_sha256: str,
    formal_seeds: list[int],
    data_manifest_sha256: str,
) -> dict[str, Any]:
    completed = sorted(int(result["seed"]) for result in seed_results)
    if completed != sorted(formal_seeds):
        raise RuntimeError("formal M3 seed set is incomplete or unexpected")
    all_pass = all(result["pass"] for result in seed_results)
    return {
        "format": "minicells.native-clm-v0.m3-decision.v1",
        "status": (
            "NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_SUPPORTED"
            if all_pass
            else "NATIVE_CLM_V0_M3_GROWTH_RESTORED_CONTINUAL_LANGUAGE_NOT_SUPPORTED"
        ),
        "scientific_decision": bool(all_pass),
        "protocol_sha256": protocol_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "formal_seeds": formal_seeds,
        "completed_seeds": completed,
        "seed_results": seed_results,
        "all_registered_gates_pass": bool(all_pass),
        "claim_boundary": (
            "12.15M Native CLM v0; frozen shared substrate/original router; protected Cell-local "
            "writes; fixed-vs-context-addressed growth; B->C->D; zero learner replay"
        ),
        "next_milestone_if_supported": "M4 Cell ontology / specialization analysis",
    }
