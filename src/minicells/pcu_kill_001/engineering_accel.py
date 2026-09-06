"""Engineering-only execution acceleration for PCU-KILL-001.

This module changes scheduling, not science:

* deterministic generation still uses greedy decoding with the registered
  ``max_new_tokens``;
* A/B branch mutations remain independently trained from the same caches;
* every registered K/LR trial is still evaluated;
* a second GPU is used only as an equivalent evaluation replica;
* completed capacity trials are checkpointed and may be reused only under an
  exact source/model/dataset identity match.

Formal execution never installs this accelerator.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .cache import CachedTailRunner
from .cellular import GraniteArchitectureInspector, patch_moe_block
from .evaluation import EvaluationSummary, SampleEvaluation
from .governance import git_provenance, write_json
from .model import load_granite, target_module
from .overlay import ExpertsOverlayModel
from .registry import module_tensor_hash
from .synthetic import POSITIVE_CONTROL_VERSION
from .task_training import TaskBranchResult
from .training import ForkedCellularExperts


ACCEL_SCHEMA = "minicells.pcu-kill-001.engineering-acceleration.v1"
CAPACITY_CHECKPOINT_SCHEMA = "minicells.pcu-kill-001.capacity-checkpoint.v1"


def _summary_from_dict(value: Mapping[str, Any]) -> EvaluationSummary:
    rows = tuple(
        SampleEvaluation(
            sample_id=str(row["sample_id"]),
            expected=str(row["expected"]),
            generated=str(row["generated"]),
            exact=bool(row["exact"]),
            relay_exact=row.get("relay_exact"),
            terminal_exact=row.get("terminal_exact"),
            both_exact=row.get("both_exact"),
        )
        for row in value.get("rows", [])
    )
    return EvaluationSummary(
        split=str(value["split"]),
        exact=float(value["exact"]),
        relay_exact=(None if value.get("relay_exact") is None else float(value["relay_exact"])),
        terminal_exact=(None if value.get("terminal_exact") is None else float(value["terminal_exact"])),
        both_exact=(None if value.get("both_exact") is None else float(value["both_exact"])),
        rows=rows,
    )


def _lr_tag(value: float) -> str:
    return format(float(value), ".12g").replace("+", "p").replace("-", "m").replace(".", "d")


def _selected_map(selected_cells: Sequence[str], layer: int) -> dict[int, list[int]]:
    prefix = f"L{int(layer)}:E"
    result: dict[int, list[int]] = {}
    for cell_id in selected_cells:
        text = str(cell_id)
        if not text.startswith(prefix) or ":C" not in text:
            raise ValueError(f"invalid selected Cell ID: {cell_id}")
        expert_text, cell_text = text[len(prefix):].split(":C", 1)
        result.setdefault(int(expert_text), []).append(int(cell_text))
    return result


class EngineeringAcceleration:
    """Temporarily accelerate one engineering run and restore all globals after."""

    def __init__(
        self,
        experiment_module: Any,
        *,
        primary_original: Any,
        primary_tokenizer: Any,
        world: Any,
        model_repo: str,
        model_revision: str,
        foundation_hash: str,
        inspector: GraniteArchitectureInspector,
        output: Path,
        primary_device: str = "cuda:0",
        secondary_device: str = "cuda:1",
    ) -> None:
        self.experiment = experiment_module
        self.primary_original = primary_original
        self.primary_tokenizer = primary_tokenizer
        self.world = world
        self.model_repo = str(model_repo)
        self.model_revision = str(model_revision)
        self.foundation_hash = str(foundation_hash)
        self.inspector = inspector
        self.output = Path(output)
        self.primary_device = str(primary_device)
        self.secondary_device = str(secondary_device)

        self._orig_eval_samples = experiment_module.evaluate_samples
        self._orig_eval_matrix = experiment_module.evaluate_matrix
        self._orig_model_with = experiment_module._model_with_experts
        self._orig_train_branch = experiment_module.train_cached_branch

        self._secondary_tokenizer: Any | None = None
        self._secondary_cellular: Any | None = None
        self._secondary_base: Any | None = None
        self._overlay_queue: list[Any] = []
        self._cached_eval: dict[tuple[int, str], EvaluationSummary] = {}
        self._base_b_cache: EvaluationSummary | None = None
        self._active_trial_key: tuple[float, int] | None = None
        self._active_trial_results: dict[str, TaskBranchResult] = {}
        self._current_run_keys: set[tuple[float, int]] = set()
        self._resume_branch_count: dict[tuple[float, int], int] = {}
        self._checkpoint_cache: dict[tuple[float, int], dict[str, Any]] = {}
        self._installed = False

        source = git_provenance(Path(__file__).resolve().parents[3])
        self.identity = {
            "source_commit": source.get("source_commit"),
            "source_tree": source.get("source_tree"),
            "model_repo": self.model_repo,
            "model_revision": self.model_revision,
            "foundation_tensor_sha256": self.foundation_hash,
            "dataset_manifest_sha256": world.manifest_sha256(),
            "positive_control_version": POSITIVE_CONTROL_VERSION,
            "architecture": asdict(inspector),
        }

    @property
    def checkpoint_dir(self) -> Path:
        return self.output / "CAPACITY_PROGRESS"

    def _checkpoint_path(self, learning_rate: float, k: int) -> Path:
        return self.checkpoint_dir / f"lr-{_lr_tag(learning_rate)}-k{int(k)}.json"

    def _load_checkpoint(self, key: tuple[float, int]) -> dict[str, Any] | None:
        if key in self._checkpoint_cache:
            return self._checkpoint_cache[key]
        path = self._checkpoint_path(*key)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != CAPACITY_CHECKPOINT_SCHEMA:
            raise RuntimeError(f"capacity checkpoint schema mismatch: {path}")
        if payload.get("identity") != self.identity:
            raise RuntimeError(
                "capacity checkpoint identity mismatch; remove the stale partial run or use a new run-id"
            )
        self._checkpoint_cache[key] = payload
        return payload

    def _write_checkpoint(
        self,
        key: tuple[float, int],
        result_a: TaskBranchResult,
        result_b: TaskBranchResult,
        direct_a: EvaluationSummary,
        direct_b: EvaluationSummary,
    ) -> None:
        learning_rate, k = key
        payload = {
            "schema": CAPACITY_CHECKPOINT_SCHEMA,
            "identity": self.identity,
            "learning_rate": float(learning_rate),
            "k": int(k),
            "cells_a": list(result_a.selected_cells),
            "cells_b": list(result_b.selected_cells),
            "training_a": result_a.to_dict(),
            "training_b": result_b.to_dict(),
            "direct_a": direct_a.to_dict(),
            "direct_b": direct_b.to_dict(),
            "passes": bool(direct_a.exact >= 0.80 and direct_b.exact >= 0.80),
        }
        path = self._checkpoint_path(learning_rate, k)
        write_json(path, payload)
        self._checkpoint_cache[key] = payload

    def _ensure_secondary(self) -> None:
        if self._secondary_cellular is not None:
            return
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("dual-GPU engineering acceleration requires at least two CUDA devices")
        tokenizer, model, manifest = load_granite(
            self.model_repo,
            revision=self.model_revision,
            device=self.secondary_device,
        )
        inspector = GraniteArchitectureInspector.inspect(model, require_granite=True)
        if asdict(inspector) != asdict(self.inspector):
            raise RuntimeError("secondary GPU architecture does not match primary Granite foundation")
        if str(manifest.get("foundation_tensor_sha256")) != self.foundation_hash:
            raise RuntimeError("secondary GPU foundation hash does not match primary foundation")
        model.requires_grad_(False)
        block = target_module(model, inspector.target_path)
        exact_parent = block.experts
        patch_moe_block(block, inspector.partition)
        model.eval()
        self._secondary_tokenizer = tokenizer
        self._secondary_cellular = model
        self._secondary_base = ExpertsOverlayModel(model, inspector.target_path, exact_parent).eval()
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(self.output / "EXECUTION_ACCELERATION.json", {
            "schema": ACCEL_SCHEMA,
            "dual_gpu": True,
            "primary_device": self.primary_device,
            "secondary_device": self.secondary_device,
            "generation": {
                "method": "hf_generate_greedy",
                "batch_size": 16,
                "use_cache": True,
            },
            "capacity_trial_checkpointing": True,
            "scientific_semantics_changed": False,
            "identity": self.identity,
        })

    def _secondary_view(self, model: Any) -> Any:
        self._ensure_secondary()
        assert self._secondary_cellular is not None
        if model is self.primary_original:
            return self._secondary_base
        if not isinstance(model, ExpertsOverlayModel):
            raise TypeError("dual-GPU evaluator requires an ExpertsOverlayModel view")
        experts = deepcopy(model.experts).to(self.secondary_device)
        if hasattr(experts, "eval"):
            experts.eval()
        return ExpertsOverlayModel(
            self._secondary_cellular,
            self.inspector.target_path,
            experts,
        ).eval()

    def _paired_evaluate(
        self,
        model_a: Any,
        samples_a: Sequence[Any],
        split_a: str,
        model_b: Any,
        samples_b: Sequence[Any],
        split_b: str,
        *,
        max_new_tokens: int,
    ) -> tuple[EvaluationSummary, EvaluationSummary]:
        secondary_b = self._secondary_view(model_b)
        assert self._secondary_tokenizer is not None
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(
                self._orig_eval_samples,
                model_a,
                self.primary_tokenizer,
                samples_a,
                split=split_a,
                device=self.primary_device,
                max_new_tokens=max_new_tokens,
            )
            future_b = pool.submit(
                self._orig_eval_samples,
                secondary_b,
                self._secondary_tokenizer,
                samples_b,
                split=split_b,
                device=self.secondary_device,
                max_new_tokens=max_new_tokens,
            )
            return future_a.result(), future_b.result()

    def _dummy_result_from_checkpoint(
        self,
        parent_experts: Any,
        selected_cells: Sequence[str],
        branch: str,
        layer: int,
        payload: Mapping[str, Any],
    ) -> TaskBranchResult:
        runtime = ForkedCellularExperts(parent_experts, _selected_map(selected_cells, layer))
        runtime.to(next(parent_experts.parameters()).device)
        section = payload[f"training_{branch.lower()}"]
        return TaskBranchResult(
            branch=str(branch),
            selected_cells=tuple(str(value) for value in selected_cells),
            training_steps=int(section["training_steps"]),
            training_tokens=int(section["training_tokens"]),
            final_loss=float(section["final_loss"]),
            runtime=runtime,
        )

    def train_cached_branch(self, parent_experts: Any, runner: Any, cache: Any, selected_cells: Sequence[str], *, layer: int, branch: str, config: Any) -> TaskBranchResult:
        key = (float(config.learning_rate), len(tuple(selected_cells)))
        checkpoint = self._load_checkpoint(key)
        count = self._resume_branch_count.get(key, 0)
        if checkpoint is not None and key not in self._current_run_keys and count < 2:
            expected_cells = checkpoint[f"cells_{str(branch).lower()}"]
            if list(selected_cells) != list(expected_cells):
                raise RuntimeError("capacity checkpoint selected Cells do not match recomputed allocation")
            result = self._dummy_result_from_checkpoint(
                parent_experts,
                selected_cells,
                branch,
                layer,
                checkpoint,
            )
            self._resume_branch_count[key] = count + 1
        else:
            result = self._orig_train_branch(
                parent_experts,
                runner,
                cache,
                selected_cells,
                layer=layer,
                branch=branch,
                config=config,
            )
            self._current_run_keys.add(key)
        self._active_trial_results[str(branch)] = result
        if str(branch) == "B" and "A" in self._active_trial_results:
            self._active_trial_key = key
        return result

    def model_with_experts(self, model: Any, inspector: Any, experts: Any) -> Any:
        view = self._orig_model_with(model, inspector, experts)
        self._overlay_queue.append(view)
        if len(self._overlay_queue) > 2:
            self._overlay_queue = self._overlay_queue[-2:]
        return view

    def evaluate_samples(self, model: Any, tokenizer: Any, samples: Sequence[Any], *, split: str, device: Any, max_new_tokens: int = 16, **kwargs: Any) -> EvaluationSummary:
        cached = self._cached_eval.pop((id(model), str(split)), None)
        if cached is not None:
            return cached

        # Base A/B are the first post-oracle full-model evaluations. Pair them
        # to initialize and immediately use GPU1 only after the oracle passed.
        if model is self.primary_original and str(split) == "A_eval" and self._base_b_cache is None:
            direct_a, direct_b = self._paired_evaluate(
                model,
                samples,
                "A_eval",
                self.primary_original,
                self.world.splits["B_eval"],
                "B_eval",
                max_new_tokens=max_new_tokens,
            )
            self._base_b_cache = direct_b
            return direct_a
        if model is self.primary_original and str(split) == "B_eval" and self._base_b_cache is not None:
            value = self._base_b_cache
            self._base_b_cache = None
            return value

        # A/B capacity trial models are constructed consecutively. Evaluate A
        # on GPU0 and B on GPU1 in one wall-clock interval. Resume checkpoints
        # bypass both forwards but preserve the exact stored EvaluationSummary.
        if (
            str(split) == "A_eval"
            and len(self._overlay_queue) == 2
            and model is self._overlay_queue[0]
            and self._active_trial_key is not None
            and "A" in self._active_trial_results
            and "B" in self._active_trial_results
        ):
            key = self._active_trial_key
            checkpoint = self._load_checkpoint(key)
            resumed = checkpoint is not None and key not in self._current_run_keys
            if resumed:
                direct_a = _summary_from_dict(checkpoint["direct_a"])
                direct_b = _summary_from_dict(checkpoint["direct_b"])
            else:
                model_b = self._overlay_queue[1]
                direct_a, direct_b = self._paired_evaluate(
                    model,
                    samples,
                    "A_eval",
                    model_b,
                    self.world.splits["B_eval"],
                    "B_eval",
                    max_new_tokens=max_new_tokens,
                )
                self._write_checkpoint(
                    key,
                    self._active_trial_results["A"],
                    self._active_trial_results["B"],
                    direct_a,
                    direct_b,
                )
            self._cached_eval[(id(self._overlay_queue[1]), "B_eval")] = direct_b
            self._overlay_queue.clear()
            self._active_trial_results = {}
            self._active_trial_key = None
            return direct_a

        return self._orig_eval_samples(
            model,
            tokenizer,
            samples,
            split=split,
            device=device,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    def evaluate_matrix(self, models: Mapping[str, Any], tokenizer: Any, splits: Mapping[str, Sequence[Any]], *, device: Any, max_new_tokens: int = 16, **kwargs: Any) -> dict[str, dict[str, EvaluationSummary]]:
        """Evaluate the complete matrix with at most one task per GPU at once."""
        self._ensure_secondary()
        tasks = [
            (model_name, split_name)
            for model_name in models
            for split_name in ("A_eval", "B_eval", "AB_eval")
        ]
        matrix: dict[str, dict[str, EvaluationSummary]] = {name: {} for name in models}
        secondary_views: dict[int, Any] = {}

        def secondary_for(model: Any) -> Any:
            key = id(model)
            if key not in secondary_views:
                secondary_views[key] = self._secondary_view(model)
            return secondary_views[key]

        index = 0
        while index < len(tasks):
            primary_task = tasks[index]
            secondary_task = tasks[index + 1] if index + 1 < len(tasks) else None
            model_name_a, split_a = primary_task
            with ThreadPoolExecutor(max_workers=2) as pool:
                future_a = pool.submit(
                    self._orig_eval_samples,
                    models[model_name_a],
                    self.primary_tokenizer,
                    splits.get(split_a, ()),
                    split=split_a,
                    device=self.primary_device,
                    max_new_tokens=max_new_tokens,
                )
                future_b = None
                if secondary_task is not None:
                    model_name_b, split_b = secondary_task
                    future_b = pool.submit(
                        self._orig_eval_samples,
                        secondary_for(models[model_name_b]),
                        self._secondary_tokenizer,
                        splits.get(split_b, ()),
                        split=split_b,
                        device=self.secondary_device,
                        max_new_tokens=max_new_tokens,
                    )
                matrix[model_name_a][split_a] = future_a.result()
                if future_b is not None and secondary_task is not None:
                    model_name_b, split_b = secondary_task
                    matrix[model_name_b][split_b] = future_b.result()
            index += 2
        return matrix

    def __enter__(self) -> "EngineeringAcceleration":
        if self._installed:
            return self
        self.experiment.evaluate_samples = self.evaluate_samples
        self.experiment.evaluate_matrix = self.evaluate_matrix
        self.experiment._model_with_experts = self.model_with_experts
        self.experiment.train_cached_branch = self.train_cached_branch
        self._installed = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self._installed:
            return
        self.experiment.evaluate_samples = self._orig_eval_samples
        self.experiment.evaluate_matrix = self._orig_eval_matrix
        self.experiment._model_with_experts = self._orig_model_with
        self.experiment.train_cached_branch = self._orig_train_branch
        self._installed = False
        if self._secondary_cellular is not None:
            del self._secondary_cellular
            self._secondary_cellular = None
            self._secondary_base = None
            self._secondary_tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def maybe_engineering_acceleration(
    experiment_module: Any,
    *,
    primary_original: Any,
    primary_tokenizer: Any,
    world: Any,
    model_repo: str,
    model_revision: str,
    foundation_hash: str,
    inspector: GraniteArchitectureInspector,
    output: Path,
) -> EngineeringAcceleration | None:
    """Return a dual-GPU accelerator only when two CUDA devices are available."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return None
    return EngineeringAcceleration(
        experiment_module,
        primary_original=primary_original,
        primary_tokenizer=primary_tokenizer,
        world=world,
        model_repo=model_repo,
        model_revision=model_revision,
        foundation_hash=foundation_hash,
        inspector=inspector,
        output=output,
    )
