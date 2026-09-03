"""Matched raw-replay oracle for KT001.

Raw historical examples are intentionally confined to this module. Zero-replay
arms use ``run_non_oracle_stream`` from the phase engine and never receive the
historical path mapping accepted here.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

import torch
from torch import Tensor

from .integrated_replay_free_clm_kt001 import KT001ArmConfig
from .integrated_replay_free_clm_kt001_mechanics import structural_state_metadata
from .integrated_replay_free_clm_kt001_runner import (
    PHASES,
    KT001RunnerConfig,
    _seed_everything,
    _write_jsonl,
    bootstrap_historical_address_state,
    load_arm_model,
    run_phase,
)
from .native_clm_m2 import NativeCLMM2Config, evaluate_matrix, sha256_file
from .native_clm_m3 import NativeCLMM3GrowthConfig, _cycle, _loader
from .native_clm_m3l2 import M3L2AddressConfig


class MatchedReplayIterator(Iterator[tuple[Tensor, Tensor]]):
    """Yield a deterministic 50/50 current/history batch mixture."""

    def __init__(
        self,
        *,
        model,
        current_path: str | Path,
        historical_paths: dict[str, str | Path],
        train_config: NativeCLMM2Config,
        seed: int,
    ) -> None:
        if not historical_paths:
            raise ValueError("KT001 replay oracle requires at least one historical domain")
        if train_config.batch_size < 2 or train_config.batch_size % 2:
            raise ValueError("KT001 matched replay requires an even batch_size >= 2")

        self.current_batch_size = train_config.batch_size // 2
        self.replay_batch_size = train_config.batch_size - self.current_batch_size
        self._rng = random.Random(seed + 770_001)
        self._history_names = tuple(sorted(historical_paths))
        self._current = _cycle(
            _loader(
                current_path,
                seq_len=model.config.max_seq_len,
                batch_size=self.current_batch_size,
                seed=seed + 1,
                num_workers=train_config.num_workers,
            )
        )
        self._history = {
            name: _cycle(
                _loader(
                    path,
                    seq_len=model.config.max_seq_len,
                    batch_size=self.replay_batch_size,
                    seed=seed + 1000 + index,
                    num_workers=train_config.num_workers,
                )
            )
            for index, (name, path) in enumerate(sorted(historical_paths.items()))
        }
        self.steps = 0
        self.current_examples = 0
        self.replay_examples = 0
        self.history_domain_steps = {name: 0 for name in self._history_names}

    def __next__(self) -> tuple[Tensor, Tensor]:
        current_x, current_y = next(self._current)
        history_name = self._rng.choice(self._history_names)
        history_x, history_y = next(self._history[history_name])
        self.steps += 1
        self.current_examples += int(current_x.size(0))
        self.replay_examples += int(history_x.size(0))
        self.history_domain_steps[history_name] += 1
        return (
            torch.cat((current_x, history_x), dim=0),
            torch.cat((current_y, history_y), dim=0),
        )

    def metadata(self) -> dict[str, Any]:
        total = self.current_examples + self.replay_examples
        return {
            "policy": "50pct_current_50pct_uniform_historical_domain",
            "steps": int(self.steps),
            "current_batch_size": int(self.current_batch_size),
            "replay_batch_size": int(self.replay_batch_size),
            "current_examples": int(self.current_examples),
            "replay_examples": int(self.replay_examples),
            "replay_example_fraction": float(self.replay_examples / max(1, total)),
            "historical_domains": list(self._history_names),
            "historical_domain_steps": dict(self.history_domain_steps),
        }


def _history_for_phase(
    phase: str,
    *,
    bootstrap_path: str | Path,
    train_paths: dict[str, str | Path],
) -> dict[str, str | Path]:
    if phase == "B":
        return {"A": bootstrap_path}
    if phase == "C":
        return {"A": bootstrap_path, "B": train_paths["B"]}
    if phase == "D":
        return {
            "A": bootstrap_path,
            "B": train_paths["B"],
            "C": train_paths["C"],
        }
    raise ValueError(f"unregistered KT001 phase: {phase}")


def _history_file_budget(paths: dict[str, str | Path]) -> dict[str, Any]:
    sizes = {name: Path(path).stat().st_size for name, path in sorted(paths.items())}
    return {
        "files": {name: str(path) for name, path in sorted(paths.items())},
        "bytes_by_domain": sizes,
        "total_accessible_raw_history_bytes": int(sum(sizes.values())),
    }


def run_replay_oracle_stream(
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
    """Run the matched replay upper bound on the same structural/write substrate."""

    if not arm.raw_replay or arm.name != "matched_replay_oracle":
        raise ValueError("run_replay_oracle_stream requires matched_replay_oracle")
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
    phase_summaries: list[dict[str, Any]] = []
    replay_audit: dict[str, Any] = {}
    all_invariant_rows: list[dict[str, Any]] = []

    for index, phase in enumerate(PHASES):
        history_paths = _history_for_phase(
            phase,
            bootstrap_path=bootstrap_path,
            train_paths=train_paths,
        )
        replay_iterator = MatchedReplayIterator(
            model=model,
            current_path=train_paths[phase],
            historical_paths=history_paths,
            train_config=train_config,
            seed=seed + 100 * (index + 1),
        )
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
            batch_iterator=replay_iterator,
            phase_output_dir=output / f"phase-{phase}",
        )
        phase_summary = dict(artifact.phase_summary)
        phase_summary["replay"] = replay_iterator.metadata()
        phase_summaries.append(phase_summary)
        all_invariant_rows.extend(artifact.invariant_rows)
        replay_audit[phase] = {
            **_history_file_budget(history_paths),
            **replay_iterator.metadata(),
        }
        matrices[f"after_{phase}"] = evaluate_matrix(
            model,
            eval_paths,
            device=target_device,
            config=train_config,
        )

    invariant_path = output / "realized-update-invariant.jsonl"
    _write_jsonl(invariant_path, all_invariant_rows)
    final_checkpoint = output / "final.pt"
    model.save_checkpoint(
        final_checkpoint,
        extra={
            "experiment": "KT001",
            "arm": arm.name,
            "seed": int(seed),
            "parent_checkpoint_sha256": expected_checkpoint_sha256,
            "stream": list(PHASES),
            "raw_replay_oracle": True,
        },
    )
    total_accessible = max(
        int(record["total_accessible_raw_history_bytes"])
        for record in replay_audit.values()
    )
    summary = {
        "format": "minicells.kt001-arm-summary.v1",
        "arm": arm.name,
        "arm_switches": arm.metadata_switches(),
        "seed": int(seed),
        "parent_checkpoint_sha256": expected_checkpoint_sha256,
        "parent_m1_extra_keys": sorted(m1_extra.keys()),
        "stream": list(PHASES),
        "learner_replay_bytes": total_accessible,
        "replay_audit": replay_audit,
        "bootstrap": bootstrap,
        "runner_config": asdict(runner_config),
        "training_config": asdict(train_config),
        "growth_config": asdict(growth_config),
        "address_config": asdict(address_config),
        "phase_summaries": phase_summaries,
        "evaluation_matrix": matrices,
        "structural_final": structural_state_metadata(model, arm=arm),
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "final_checkpoint_bytes": final_checkpoint.stat().st_size,
        "realized_update_invariant_rows": len(all_invariant_rows),
        "realized_update_invariant_sha256": sha256_file(invariant_path),
        "realized_update_invariant_bytes": invariant_path.stat().st_size,
    }
    (output / "arm-summary-core.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
