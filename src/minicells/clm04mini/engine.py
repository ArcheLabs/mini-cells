"""Protocol-faithful continual transaction engine for CLM-0.4-mini M1."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import time
from typing import Iterable

import torch
from torch.nn import functional as F

from .examples import ScoredTokenExample, collate_scored
from .model import MiniCLMConfig, TinyCLMDecoder
from .protocol import CandidateOptimizerConfig, canonical_json_hash
from .state import CellRegistry, DependencyIndex, model_state_hash
from .tokenizer import TokenizerBundle
from .training import mean_scored_nll, train_only_cells


VARIANTS = ("local_always", "local_tx", "local_tx_growth")


def _relative_change(before: float, after: float) -> float:
    return (after - before) / max(before, 1e-8)


def _new_gain(before: float, after: float) -> float:
    return (before - after) / max(before, 1e-8)


def _peak_memory(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def _scored_logits(
    model: TinyCLMDecoder,
    examples: Iterable[ScoredTokenExample],
    *,
    tokenizer: TokenizerBundle,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    model.eval()
    with torch.no_grad():
        for example in examples:
            x, _, mask, addresses = collate_scored(
                [example], pad_id=tokenizer.pad_id, device=device
            )
            logits = model(x, addresses)[0]
            result[example.example_id] = logits[mask[0]].detach().cpu()
    return result


def token_accuracy(
    model: TinyCLMDecoder,
    examples: Iterable[ScoredTokenExample],
    *,
    tokenizer: TokenizerBundle,
    device: torch.device,
) -> float:
    items = list(examples)
    if not items:
        return 1.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(items), 64):
            batch = items[start : start + 64]
            x, y, mask, addresses = collate_scored(
                batch, pad_id=tokenizer.pad_id, device=device
            )
            logits = model(x, addresses)
            prediction = logits.argmax(dim=-1)
            correct += int(((prediction == y) & mask).sum().item())
            total += int(mask.sum().item())
    return correct / float(max(1, total))


def logical_state_hash(
    model: TinyCLMDecoder,
    dependency_index: DependencyIndex,
    probes: dict[str, ScoredTokenExample],
) -> str:
    return canonical_json_hash(
        {
            "model_state_hash": model_state_hash(model),
            "dependency_index": dependency_index.to_dict(),
            "probes": {key: probes[key].to_dict() for key in sorted(probes)},
        }
    )


class VariantHarness:
    """Committed state for one registered primary M1 variant."""

    def __init__(
        self,
        *,
        variant: str,
        model: TinyCLMDecoder,
        tokenizer: TokenizerBundle,
        device: torch.device,
        thresholds: dict[str, float],
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown M1 variant {variant}")
        self.variant = variant
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.thresholds = dict(thresholds)
        self.dependency_index = DependencyIndex()
        self.probes: dict[str, ScoredTokenExample] = {}
        self.probe_reference_accuracy: dict[str, float] = {}
        self.registry = CellRegistry(model)
        self.records: list[dict] = []
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def protected_examples(self, ids: Iterable[str] | None = None) -> list[ScoredTokenExample]:
        if ids is None:
            keys = sorted(self.probes)
        else:
            keys = sorted(set(str(value) for value in ids) & set(self.probes))
        return [self.probes[key] for key in keys]

    def remove_knowledge_key(self, knowledge_key: str | None) -> list[str]:
        if not knowledge_key:
            return []
        removed = [
            probe_id
            for probe_id, probe in self.probes.items()
            if probe.knowledge_key == knowledge_key
        ]
        for probe_id in removed:
            self.dependency_index.remove(probe_id)
            self.probes.pop(probe_id, None)
            self.probe_reference_accuracy.pop(probe_id, None)
        return sorted(removed)

    def admit_probes(self, examples: Iterable[ScoredTokenExample]) -> None:
        for example in examples:
            self.probes[example.example_id] = example
            cells = self.model.active_cell_ids(example.address_id)
            self.dependency_index.register(example.example_id, cells)
            self.probe_reference_accuracy[example.example_id] = token_accuracy(
                self.model, [example], tokenizer=self.tokenizer, device=self.device
            )

    def _active_by_layer(self, address_id: str) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for layer_id in (3, 4):
            layer = self.model.sparse_layer(layer_id)
            cells = [layer.base_cell_id(value) for value in layer.base_route(address_id)]
            if layer.has_private(address_id):
                cells.append(layer.private_cell_id(address_id))
            result[str(layer_id)] = cells
        return result

    def _attempt(
        self,
        *,
        candidate: TinyCLMDecoder,
        candidate_kind: str,
        touched_cells: list[str],
        new_validation: list[ScoredTokenExample],
        train_stats: dict,
    ) -> dict:
        started = time.perf_counter()
        all_ids = set(self.probes)
        local_ids = self.dependency_index.scope(touched_cells)
        outside_ids = all_ids - local_ids
        local_examples = self.protected_examples(local_ids)
        global_examples = self.protected_examples(all_ids)
        outside_examples = self.protected_examples(outside_ids)

        new_before = mean_scored_nll(
            self.model, new_validation, tokenizer=self.tokenizer, device=self.device
        )
        new_after = mean_scored_nll(
            candidate, new_validation, tokenizer=self.tokenizer, device=self.device
        )
        local_before = mean_scored_nll(
            self.model, local_examples, tokenizer=self.tokenizer, device=self.device
        )
        local_after = mean_scored_nll(
            candidate, local_examples, tokenizer=self.tokenizer, device=self.device
        )
        global_before = mean_scored_nll(
            self.model, global_examples, tokenizer=self.tokenizer, device=self.device
        )
        global_after = mean_scored_nll(
            candidate, global_examples, tokenizer=self.tokenizer, device=self.device
        )
        gain = _new_gain(new_before, new_after)
        local_regression = _relative_change(local_before, local_after) if local_examples else 0.0
        global_regression = _relative_change(global_before, global_after) if global_examples else 0.0
        local_pass = (
            gain >= self.thresholds["minimum_new_gain"]
            and local_regression <= self.thresholds["maximum_local_old_regression"]
        )
        oracle_pass = (
            gain >= self.thresholds["minimum_new_gain"]
            and global_regression <= self.thresholds["maximum_global_old_regression"]
        )

        before_logits = _scored_logits(
            self.model, outside_examples, tokenizer=self.tokenizer, device=self.device
        )
        after_logits = _scored_logits(
            candidate, outside_examples, tokenizer=self.tokenizer, device=self.device
        )
        escapes = 0
        tolerance = self.thresholds["structural_logit_tolerance"]
        for probe_id, before in before_logits.items():
            if float((before - after_logits[probe_id]).abs().max()) > tolerance:
                escapes += 1

        touched_parameters = sum(
            parameter.numel()
            for module in candidate.modules_for_cell_ids(touched_cells)
            for parameter in module.parameters()
        )
        total_parameters = sum(parameter.numel() for parameter in candidate.parameters())
        validation_tokens = sum(sum(item.target_mask) for item in new_validation)
        validation_tokens += sum(sum(item.target_mask) for item in local_examples)
        validation_tokens += sum(sum(item.target_mask) for item in global_examples)
        return {
            "attempt_index": 0,
            "candidate_kind": candidate_kind,
            "touched_cells": list(touched_cells),
            "touched_parameter_count": int(touched_parameters),
            "touched_parameter_fraction": touched_parameters / float(max(1, total_parameters)),
            "local_dependency_probe_count": len(local_ids),
            "local_dependency_coverage": len(local_ids) / float(max(1, len(all_ids))),
            "new_metrics_before": {"nll": new_before},
            "new_metrics_candidate": {"nll": new_after},
            "local_old_metrics_before": {"nll": local_before, "probe_count": len(local_examples)},
            "local_old_metrics_candidate": {"nll": local_after, "probe_count": len(local_examples)},
            "global_old_metrics_before": {"nll": global_before, "probe_count": len(global_examples)},
            "global_old_metrics_candidate": {"nll": global_after, "probe_count": len(global_examples)},
            "new_gain": float(gain),
            "local_regression": float(local_regression),
            "global_regression": float(global_regression),
            "local_pass": bool(local_pass),
            "oracle_pass": bool(oracle_pass),
            "false_safe": bool(local_pass and not oracle_pass),
            "structural_escape_count": int(escapes),
            "structural_escape_rate": escapes / float(max(1, len(outside_examples))),
            "training_tokens": int(train_stats["training_tokens"]),
            "validation_tokens": int(validation_tokens),
            "optimizer_steps": int(train_stats["optimizer_steps"]),
            "candidate_wall_seconds": float(train_stats["wall_seconds"]),
            "validation_wall_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": _peak_memory(self.device),
        }

    def _candidate(
        self,
        *,
        cell_ids: list[str],
        train_examples: list[ScoredTokenExample],
        optimizer_config: CandidateOptimizerConfig,
        rng_seed: int,
    ) -> tuple[TinyCLMDecoder, dict]:
        candidate = copy.deepcopy(self.model).to(self.device)
        stats = train_only_cells(
            candidate,
            cell_ids=cell_ids,
            examples=train_examples,
            tokenizer=self.tokenizer,
            optimizer_config=optimizer_config,
            device=self.device,
            rng_seed=rng_seed,
        )
        return candidate, stats

    def execute(
        self,
        *,
        transaction_id: int,
        operation: str,
        address_id: str,
        knowledge_key: str | None,
        supersedes_key: str | None,
        train_examples: list[ScoredTokenExample],
        validation_examples: list[ScoredTokenExample],
        probe_examples: list[ScoredTokenExample],
        direct_optimizer: CandidateOptimizerConfig,
        growth_optimizer: CandidateOptimizerConfig,
        rng_seed: int,
    ) -> dict:
        started = time.perf_counter()
        state_before = logical_state_hash(self.model, self.dependency_index, self.probes)
        removed = self.remove_knowledge_key(supersedes_key if operation == "supersede" else None)
        base_routes = self.model.base_routes(address_id)
        attempts: list[dict] = []
        births: list[str] = []
        deletions: list[str] = []
        committed_attempt: dict | None = None
        private_before = self.model.has_private_bundle(address_id)

        if self.variant == "local_tx_growth" and private_before:
            touched = self.model.private_cell_ids(address_id)
            candidate, train_stats = self._candidate(
                cell_ids=touched,
                train_examples=train_examples,
                optimizer_config=growth_optimizer,
                rng_seed=rng_seed,
            )
            attempt = self._attempt(
                candidate=candidate,
                candidate_kind="private-reuse",
                touched_cells=touched,
                new_validation=validation_examples,
                train_stats=train_stats,
            )
            attempts.append(attempt)
            if attempt["local_pass"]:
                self.model = candidate
                self.registry.record_accepted(touched, transaction_id=transaction_id, reuse=True)
                final_decision = "private-reuse-commit"
                committed_attempt = attempt
            else:
                self.registry.record_rejected(touched)
                final_decision = "rollback"
        else:
            touched = self.model.base_cell_ids(address_id)
            candidate, train_stats = self._candidate(
                cell_ids=touched,
                train_examples=train_examples,
                optimizer_config=direct_optimizer,
                rng_seed=rng_seed,
            )
            direct_attempt = self._attempt(
                candidate=candidate,
                candidate_kind="direct",
                touched_cells=touched,
                new_validation=validation_examples,
                train_stats=train_stats,
            )
            attempts.append(direct_attempt)
            direct_commit = self.variant == "local_always" or direct_attempt["local_pass"]
            if direct_commit:
                self.model = candidate
                self.registry.record_accepted(touched, transaction_id=transaction_id)
                final_decision = "direct-commit"
                committed_attempt = direct_attempt
            elif self.variant != "local_tx_growth":
                self.registry.record_rejected(touched)
                final_decision = "rollback"
            else:
                self.registry.record_rejected(touched)
                rollback_hash = model_state_hash(self.model)
                growth = copy.deepcopy(self.model).to(self.device)
                growth_ids = growth.spawn_growth_bundle(address_id)
                # Route addition must be function-preserving before private training.
                comparison = [*validation_examples, *self.protected_examples()]
                before_logits = _scored_logits(
                    self.model, comparison, tokenizer=self.tokenizer, device=self.device
                )
                after_logits = _scored_logits(
                    growth, comparison, tokenizer=self.tokenizer, device=self.device
                )
                zero_delta = max(
                    [
                        float((before_logits[key] - after_logits[key]).abs().max())
                        for key in before_logits
                    ]
                    or [0.0]
                )
                if zero_delta > self.thresholds["structural_logit_tolerance"]:
                    raise RuntimeError(
                        f"zero-output growth changed scored logits before training: {zero_delta}"
                    )
                train_stats = train_only_cells(
                    growth,
                    cell_ids=growth_ids,
                    examples=train_examples,
                    tokenizer=self.tokenizer,
                    optimizer_config=growth_optimizer,
                    device=self.device,
                    rng_seed=rng_seed + 1,
                )
                growth_attempt = self._attempt(
                    candidate=growth,
                    candidate_kind="spawn",
                    touched_cells=growth_ids,
                    new_validation=validation_examples,
                    train_stats=train_stats,
                )
                growth_attempt["attempt_index"] = 1
                growth_attempt["zero_output_pretrain_max_logit_delta"] = zero_delta
                attempts.append(growth_attempt)
                if growth_attempt["local_pass"]:
                    self.model = growth
                    births = self.registry.add_growth_bundle(
                        self.model,
                        address_id=address_id,
                        transaction_id=transaction_id,
                    )
                    final_decision = "growth-commit"
                    committed_attempt = growth_attempt
                else:
                    deletions = list(growth_ids)
                    if model_state_hash(self.model) != rollback_hash:
                        raise RuntimeError("failed probationary growth mutated committed model")
                    final_decision = "rollback"

        if committed_attempt is not None:
            self.registry.record_activation(
                self.model.active_cell_ids(address_id), amount=len(train_examples)
            )
            self.admit_probes(probe_examples)
        state_after = logical_state_hash(self.model, self.dependency_index, self.probes)
        record = {
            "transaction_id": int(transaction_id),
            "operation": operation,
            "address_id": address_id,
            "knowledge_key": knowledge_key,
            "superseded_probe_ids": removed,
            "train_manifest_ids": [item.example_id for item in train_examples],
            "validation_manifest_ids": [item.example_id for item in validation_examples],
            "probe_manifest_ids": [item.example_id for item in probe_examples],
            "state_hash_before": state_before,
            "base_route_cells_by_layer": base_routes,
            "private_bundle_id_before": f"bundle:{address_id}" if private_before else None,
            "attempts": attempts,
            "final_decision": final_decision,
            "growth_attempted": any(item["candidate_kind"] == "spawn" for item in attempts),
            "cell_births": births,
            "cell_deletions": deletions,
            "state_hash_after": state_after,
            "active_cells_by_layer": self._active_by_layer(address_id),
            "transaction_wall_seconds": time.perf_counter() - started,
        }
        self.records.append(record)
        return record

    def summary(self) -> dict:
        commits = [
            record
            for record in self.records
            if record["final_decision"] != "rollback"
        ]
        committed_attempts = [
            record["attempts"][-1]
            for record in commits
        ]
        local_pass_attempts = [
            attempt
            for record in self.records
            for attempt in record["attempts"]
            if attempt["local_pass"]
        ]
        false_safe = sum(1 for attempt in local_pass_attempts if attempt["false_safe"])
        growth_attempts = [record for record in self.records if record["growth_attempted"]]
        growth_commits = [record for record in self.records if record["final_decision"] == "growth-commit"]
        reuse_attempts = [
            record for record in self.records
            if record["attempts"] and record["attempts"][0]["candidate_kind"] == "private-reuse"
        ]
        reuse_commits = [record for record in reuse_attempts if record["final_decision"] == "private-reuse-commit"]
        direct_attempts = [
            attempt
            for record in self.records
            for attempt in record["attempts"]
            if attempt["candidate_kind"] == "direct"
        ]
        base_parameters = sum(
            parameter.numel()
            for name, parameter in self.model.named_parameters()
            if "private_cells" not in name
        )
        growth_parameters = sum(
            entry["parameter_count"]
            for entry in self.registry.stats.values()
            if entry["cell_type"] == "private-growth"
        )
        admission = sum(self.probe_reference_accuracy.values()) / float(
            max(1, len(self.probe_reference_accuracy))
        )
        final_accuracy = token_accuracy(
            self.model,
            self.protected_examples(),
            tokenizer=self.tokenizer,
            device=self.device,
        )
        return {
            "variant": self.variant,
            "transactions": len(self.records),
            "effective_commits": len(commits),
            "effective_acceptance_rate": len(commits) / float(max(1, len(self.records))),
            "committed_new_gain": sum(item["new_gain"] for item in committed_attempts),
            "positive_global_regression_damage": sum(
                max(0.0, item["global_regression"]) for item in committed_attempts
            ),
            "false_safe_rate": false_safe / float(max(1, len(local_pass_attempts))),
            "maximum_structural_escape_rate": max(
                [attempt["structural_escape_rate"] for record in self.records for attempt in record["attempts"]]
                or [0.0]
            ),
            "growth_attempts": len(growth_attempts),
            "growth_commits": len(growth_commits),
            "growth_rescue_rate": len(growth_commits) / float(max(1, len(growth_attempts))),
            "private_reuse_attempts": len(reuse_attempts),
            "private_reuse_commits": len(reuse_commits),
            "private_reuse_acceptance_rate": len(reuse_commits) / float(max(1, len(reuse_attempts))),
            "spawned_bundles": len(self.model.private_addresses()),
            "spawned_bundles_per_effective_commit": len(self.model.private_addresses()) / float(max(1, len(commits))),
            "growth_parameter_overhead_ratio": growth_parameters / float(max(1, base_parameters)),
            "mean_direct_dependency_coverage": sum(item["local_dependency_coverage"] for item in direct_attempts) / float(max(1, len(direct_attempts))),
            "protected_probe_count": len(self.probes),
            "protected_admission_token_accuracy": admission,
            "final_protected_token_accuracy": final_accuracy,
            "final_protected_retention_ratio": final_accuracy / float(max(admission, 1e-8)),
            "final_state_hash": logical_state_hash(self.model, self.dependency_index, self.probes),
        }

    def checkpoint_payload(self) -> dict:
        return {
            "format": "minicells.clm-0.4-mini.m1-variant-checkpoint.v1",
            "variant": self.variant,
            "config": self.model.cfg.to_dict(),
            "private_addresses": self.model.private_addresses(),
            "model_state": self.model.state_dict(),
            "dependency_index": self.dependency_index.to_dict(),
            "probes": {key: value.to_dict() for key, value in sorted(self.probes.items())},
            "probe_reference_accuracy": self.probe_reference_accuracy,
            "registry": self.registry.to_dict(),
            "records": self.records,
            "logical_state_hash": logical_state_hash(self.model, self.dependency_index, self.probes),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        tokenizer: TokenizerBundle,
        device: torch.device,
        thresholds: dict[str, float],
    ) -> "VariantHarness":
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg = MiniCLMConfig.from_dict(payload["config"])
        model = TinyCLMDecoder(cfg).to(device)
        for address_id in payload["private_addresses"]:
            model.spawn_growth_bundle(address_id)
        model.load_state_dict(payload["model_state"])
        harness = cls(
            variant=str(payload["variant"]),
            model=model,
            tokenizer=tokenizer,
            device=device,
            thresholds=thresholds,
        )
        harness.dependency_index = DependencyIndex.from_dict(payload["dependency_index"])
        harness.probes = {
            key: ScoredTokenExample.from_dict(value)
            for key, value in payload["probes"].items()
        }
        harness.probe_reference_accuracy = {
            str(key): float(value)
            for key, value in payload["probe_reference_accuracy"].items()
        }
        harness.registry = CellRegistry.from_dict(model, payload["registry"])
        harness.records = list(payload["records"])
        actual = logical_state_hash(model, harness.dependency_index, harness.probes)
        if actual != payload["logical_state_hash"]:
            raise RuntimeError("M1 checkpoint logical-state hash mismatch")
        return harness
