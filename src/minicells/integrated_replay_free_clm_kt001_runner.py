"""Continual phase engine for Integrated Replay-Free CLM Kill Test 001.

This module composes the frozen KT001 causal switches with the canonical Native
CLM mechanisms. It deliberately separates *phase execution* from replay-oracle
batch construction and formal orchestration so each integration layer can be
reviewed and committed independently.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .integrated_replay_free_clm_kt001 import KT001ArmConfig
from .integrated_replay_free_clm_kt001_mechanics import (
    capture_pre_step_cell_weights,
    commit_historical_address_state_,
    finalize_realized_adamw_transaction_,
    force_shadow_expansion_,
    observe_historical_address_queries,
    structural_state_metadata,
)
from .native_clm_m2 import NativeCLMM2Config, evaluate_matrix, sha256_file
from .native_clm_m3 import (
    NativeCLMM3GrowthConfig,
    _autocast,
    _cycle,
    _freeze_to_cell_only,
    _loader,
    _lr_factor,
)
from .native_clm_m3l2 import (
    M3L2AddressConfig,
    OnlineAddressNativeCLM,
    bootstrap_address_state,
)
from .native_clm_v0 import NativeCLM


PHASES = ("B", "C", "D")


@dataclass(frozen=True)
class KT001RunnerConfig:
    """Implementation-time schedule to be sealed before any formal seed runs."""

    calibration_batches: int = 64
    forced_shadow_expansions_per_phase: int = 1
    bootstrap_sampling_seed: int = 74001

    def validate(self) -> None:
        if self.calibration_batches < 1:
            raise ValueError("KT001 calibration_batches must be positive")
        if self.forced_shadow_expansions_per_phase != 1:
            raise ValueError("KT001 implementation currently registers one forced Shadow per phase")
        if self.bootstrap_sampling_seed != 74001:
            raise ValueError("KT001 must reuse canonical M3L-2 bootstrap sampling seed 74001")


@dataclass
class PhaseArtifacts:
    phase: str
    phase_summary: dict[str, Any]
    shadow_event: dict[str, Any] | None
    structural_before_shadow: dict[str, Any] | None
    structural_after_shadow: dict[str, Any] | None
    phase_close_address_state: dict[str, Any] | None
    invariant_rows: list[dict[str, Any]]


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def load_arm_model(
    checkpoint_path: str | Path,
    *,
    expected_checkpoint_sha256: str,
    arm: KT001ArmConfig,
    device: torch.device,
) -> tuple[NativeCLM, dict[str, Any]]:
    """Load the exact M1 bytes into the model family required by one KT001 arm."""

    checkpoint_path = Path(checkpoint_path)
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != expected_checkpoint_sha256:
        raise RuntimeError(
            f"KT001 M1 checkpoint SHA mismatch: expected {expected_checkpoint_sha256}, got {actual_sha}"
        )

    if arm.historical_address_read:
        model, extra = OnlineAddressNativeCLM.load_checkpoint(checkpoint_path, map_location="cpu")
    else:
        model, extra = NativeCLM.load_checkpoint(checkpoint_path, map_location="cpu")

    if model.cell_count != 8 or model.config.active_cells != 2:
        raise RuntimeError("KT001 requires canonical M1 topology: 8 Cells / 2 active")
    if model.parameter_count()["total"] != 12_154_368:
        raise RuntimeError("KT001 requires canonical 12,154,368-parameter M1 checkpoint")

    model.to(device)
    _freeze_to_cell_only(model)
    return model, extra


def make_phase_optimizer(model: NativeCLM, config: NativeCLMM2Config) -> torch.optim.AdamW:
    """Use the canonical Native M2/M3 AdamW Cell-only optimizer family."""

    parameters = [cell.weight for cell in model.cellular.cells if cell.weight.requires_grad]
    if not parameters:
        raise RuntimeError("KT001 phase has no writable Cell parameters")
    return torch.optim.AdamW(
        parameters,
        lr=config.lr_cells,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )


def bootstrap_historical_address_state(
    model: NativeCLM,
    bootstrap_path: str | Path,
    *,
    arm: KT001ArmConfig,
    device: torch.device,
    train_config: NativeCLMM2Config,
    address_config: M3L2AddressConfig,
    runner_config: KT001RunnerConfig,
) -> dict[str, Any] | None:
    """Consume the one allowed pre-continual A bootstrap for address-enabled arms."""

    if not arm.historical_address_read:
        return None
    if not isinstance(model, OnlineAddressNativeCLM):
        raise TypeError("address-enabled KT001 arm did not load OnlineAddressNativeCLM")
    runner_config.validate()
    address_config.validate()
    result = bootstrap_address_state(
        model,
        bootstrap_path,
        device=device,
        train_config=train_config,
        address_config=address_config,
        sampling_seed=runner_config.bootstrap_sampling_seed,
    )
    # The canonical M3L-2 patched runner releases its one-shot bootstrap lease
    # immediately after bootstrap. KT001 performs that lifecycle explicitly.
    model.bootstrap_access_released = True
    return {
        **result,
        "bootstrap_access_released_before_continual_start": True,
        "bootstrap_path_retained_by_model": False,
    }


def calibrate_current_phase_address(
    model: NativeCLM,
    train_path: str | Path,
    *,
    arm: KT001ArmConfig,
    device: torch.device,
    train_config: NativeCLMM2Config,
    runner_config: KT001RunnerConfig,
    seed: int,
) -> dict[str, Any] | None:
    """Observe current-phase learner queries without updating model parameters."""

    if not arm.historical_address_read:
        return None
    runner_config.validate()
    loader = _loader(
        train_path,
        seq_len=model.config.max_seq_len,
        batch_size=train_config.batch_size,
        seed=seed,
        num_workers=train_config.num_workers,
    )
    iterator = _cycle(loader)
    was_training = model.training
    model.eval()
    sampled_batches = 0
    sampled_tokens = 0
    probe_tokens: Tensor | None = None
    with torch.no_grad():
        for _ in range(runner_config.calibration_batches):
            x, _ = next(iterator)
            x = x.to(device)
            out = model(x, return_info=True)
            observe_historical_address_queries(model, out["cell_info"])
            sampled_batches += 1
            sampled_tokens += int(x.numel())
            probe_tokens = x
    model.train(was_training)
    if probe_tokens is None:
        raise RuntimeError("KT001 calibration produced no probe tokens")
    return {
        "batches": sampled_batches,
        "tokens": sampled_tokens,
        "seed": int(seed),
        "parameter_updates": 0,
        "probe_tokens": probe_tokens,
    }


def _current_phase_iterator(
    model: NativeCLM,
    train_path: str | Path,
    *,
    train_config: NativeCLMM2Config,
    seed: int,
) -> Iterator[tuple[Tensor, Tensor]]:
    return _cycle(
        _loader(
            train_path,
            seq_len=model.config.max_seq_len,
            batch_size=train_config.batch_size,
            seed=seed,
            num_workers=train_config.num_workers,
        )
    )


def train_phase(
    model: NativeCLM,
    *,
    phase: str,
    arm: KT001ArmConfig,
    optimizer: torch.optim.AdamW,
    batch_iterator: Iterator[tuple[Tensor, Tensor]],
    device: torch.device,
    train_config: NativeCLMM2Config,
    global_step_offset: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute one matched continual phase with the registered write transaction."""

    if phase not in PHASES:
        raise ValueError(f"unregistered KT001 phase: {phase}")
    train_config.validate()
    scaler_enabled = device.type == "cuda" and train_config.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    losses: list[float] = []
    gradient_projection_ratios: list[float] = []
    final_update_retained: list[float] = []
    invariant_rows: list[dict[str, Any]] = []
    certificate_additions = 0
    started = time.time()
    model.train()

    for step in range(1, train_config.steps_per_phase + 1):
        global_step = global_step_offset + step
        factor = _lr_factor(step - 1, train_config)
        for group in optimizer.param_groups:
            group["lr"] = train_config.lr_cells * factor

        x, y = next(batch_iterator)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, train_config.precision):
            out = model(x, y, return_info=True)
            loss = out["loss"]
        scaler.scale(loss).backward()
        if scaler_enabled:
            scaler.unscale_(optimizer)

        if arm.legacy_gradient_projection:
            ratios = model.project_cell_gradients_()
            gradient_projection_ratios.extend(float(value) for value in ratios.values())
        else:
            gradient_projection_ratios.extend([1.0] * model.cell_count)

        if arm.historical_address_read:
            observe_historical_address_queries(model, out["cell_info"])

        cell_parameters = [cell.weight for cell in model.cellular.cells if cell.weight.requires_grad]
        torch.nn.utils.clip_grad_norm_(cell_parameters, train_config.grad_clip)
        before = capture_pre_step_cell_weights(model)
        scaler.step(optimizer)
        scaler.update()
        transaction = finalize_realized_adamw_transaction_(
            model,
            before,
            arm=arm,
            step=global_step,
        )
        invariant_rows.extend(transaction["invariant_rows"])
        retained = transaction["retained_update_norm_ratios"]
        if retained is not None:
            final_update_retained.extend(float(value) for value in retained)

        if (
            train_config.certificate_update_interval > 0
            and step % train_config.certificate_update_interval == 0
        ):
            certificate_additions += model.update_certificates(out["cell_info"])

        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % train_config.log_interval == 0 or step == train_config.steps_per_phase:
            print(
                f"[KT001 {arm.name} {phase}] step={step}/{train_config.steps_per_phase} "
                f"cells={model.cell_count} loss={losses[-1]:.6f}",
                flush=True,
            )

    return (
        {
            "phase": phase,
            "steps": train_config.steps_per_phase,
            "mean_train_loss": float(sum(losses) / len(losses)),
            "final_train_loss": float(losses[-1]),
            "gradient_projection_ratio_mean": float(
                sum(gradient_projection_ratios) / max(1, len(gradient_projection_ratios))
            ),
            "final_update_projection_retained_ratio_mean": (
                float(sum(final_update_retained) / len(final_update_retained))
                if final_update_retained
                else None
            ),
            "certificate_additions": int(certificate_additions),
            "realized_update_projection": bool(arm.realized_update_write_safety),
            "learner_raw_replay": bool(arm.raw_replay),
            "cell_count_end": int(model.cell_count),
            "elapsed_seconds": time.time() - started,
        },
        invariant_rows,
    )


def run_phase(
    model: NativeCLM,
    train_path: str | Path,
    *,
    phase: str,
    arm: KT001ArmConfig,
    device: torch.device,
    train_config: NativeCLMM2Config,
    growth_config: NativeCLMM3GrowthConfig,
    runner_config: KT001RunnerConfig,
    seed: int,
    global_step_offset: int,
    batch_iterator: Iterator[tuple[Tensor, Tensor]] | None = None,
) -> PhaseArtifacts:
    """Prepare, optionally expand, train, and close one continual phase."""

    runner_config.validate()
    optimizer = make_phase_optimizer(model, train_config)
    shadow_event = None
    structural_before = None
    structural_after = None

    if arm.historical_address_read:
        calibration = calibrate_current_phase_address(
            model,
            train_path,
            arm=arm,
            device=device,
            train_config=train_config,
            runner_config=runner_config,
            seed=seed + 17,
        )
        if calibration is None:
            raise RuntimeError("KT001 address-enabled phase produced no calibration")
        probe_tokens = calibration.pop("probe_tokens")
        structural_before = {
            **structural_state_metadata(model, arm=arm),
            "calibration": calibration,
        }
        shadow_event = force_shadow_expansion_(
            model,
            optimizer,
            growth_config=growth_config,
            global_step=global_step_offset,
            probe_tokens=probe_tokens,
        )
        structural_after = structural_state_metadata(model, arm=arm)

    iterator = batch_iterator or _current_phase_iterator(
        model,
        train_path,
        train_config=train_config,
        seed=seed,
    )
    phase_summary, invariant_rows = train_phase(
        model,
        phase=phase,
        arm=arm,
        optimizer=optimizer,
        batch_iterator=iterator,
        device=device,
        train_config=train_config,
        global_step_offset=global_step_offset,
    )

    phase_close = None
    if arm.historical_address_read:
        phase_close = commit_historical_address_state_(model)

    phase_summary["shadow_event"] = shadow_event
    phase_summary["structural_before_shadow"] = structural_before
    phase_summary["structural_after_shadow"] = structural_after
    phase_summary["phase_close_address_state"] = phase_close
    return PhaseArtifacts(
        phase=phase,
        phase_summary=phase_summary,
        shadow_event=shadow_event,
        structural_before_shadow=structural_before,
        structural_after_shadow=structural_after,
        phase_close_address_state=phase_close,
        invariant_rows=invariant_rows,
    )


def run_non_oracle_stream(
    *,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    bootstrap_path: str | Path,
    train_paths: dict[str, str | Path],
    eval_paths: dict[str, str | Path],
    output_dir: str | Path,
    arm: KT001ArmConfig,
    seed: int,
    train_config: NativeCLMM2Config,
    growth_config: NativeCLMM3GrowthConfig,
    address_config: M3L2AddressConfig,
    runner_config: KT001RunnerConfig,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run one zero-replay KT001 arm from M1 through B->C->D.

    The replay oracle is intentionally excluded from this function so raw-history
    access cannot leak into a non-oracle call path.
    """

    if arm.raw_replay:
        raise ValueError("run_non_oracle_stream rejects replay-enabled arms")
    if tuple(train_paths) != PHASES or set(eval_paths) != {"A", "B", "C", "D"}:
        raise ValueError("KT001 stream must be B->C->D with A/B/C/D evaluation")
    runner_config.validate()
    train_config.validate()
    growth_config.validate()
    address_config.validate()

    target_device = torch.device(device)
    _seed_everything(seed, target_device)
    model, m1_extra = load_arm_model(
        checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        arm=arm,
        device=target_device,
    )
    bootstrap = bootstrap_historical_address_state(
        model,
        bootstrap_path,
        arm=arm,
        device=target_device,
        train_config=train_config,
        address_config=address_config,
        runner_config=runner_config,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matrices: dict[str, Any] = {
        "initial": evaluate_matrix(model, eval_paths, device=target_device, config=train_config)
    }
    phase_artifacts: list[PhaseArtifacts] = []
    all_invariant_rows: list[dict[str, Any]] = []

    for index, phase in enumerate(PHASES):
        artifact = run_phase(
            model,
            train_paths[phase],
            phase=phase,
            arm=arm,
            device=target_device,
            train_config=train_config,
            growth_config=growth_config,
            runner_config=runner_config,
            seed=seed + 100 * (index + 1),
            global_step_offset=index * train_config.steps_per_phase,
        )
        phase_artifacts.append(artifact)
        all_invariant_rows.extend(artifact.invariant_rows)
        matrices[f"after_{phase}"] = evaluate_matrix(
            model,
            eval_paths,
            device=target_device,
            config=train_config,
        )

    final_checkpoint = output / "final.pt"
    model.save_checkpoint(
        final_checkpoint,
        extra={
            "experiment": "KT001",
            "arm": arm.name,
            "seed": int(seed),
            "parent_checkpoint_sha256": expected_checkpoint_sha256,
            "stream": list(PHASES),
            "learner_replay_bytes": 0,
        },
    )
    summary = {
        "format": "minicells.kt001-arm-summary.v1",
        "arm": arm.name,
        "arm_switches": arm.metadata_switches(),
        "seed": int(seed),
        "parent_checkpoint_sha256": expected_checkpoint_sha256,
        "parent_m1_extra_keys": sorted(m1_extra.keys()),
        "stream": list(PHASES),
        "learner_replay_bytes": 0,
        "bootstrap": bootstrap,
        "runner_config": asdict(runner_config),
        "training_config": asdict(train_config),
        "growth_config": asdict(growth_config),
        "address_config": asdict(address_config),
        "phase_summaries": [artifact.phase_summary for artifact in phase_artifacts],
        "evaluation_matrix": matrices,
        "structural_final": structural_state_metadata(model, arm=arm),
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "final_checkpoint_bytes": final_checkpoint.stat().st_size,
        "realized_update_invariant_rows": len(all_invariant_rows),
    }
    return summary
