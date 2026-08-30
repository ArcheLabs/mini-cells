"""CLM-0.4-mini M0 execution smoke.

M0 is deliberately branch-driven: it exercises rollback, growth commit, private
reuse, direct commit, journaling, checkpoint restore, and replay. It never emits
a scientific decision and never uses development/formal seeds.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .model import MiniCLMConfig, TinyCLMDecoder
from .state import CellRegistry, DependencyIndex, TokenExample, model_state_hash


M0_SEED = 90400
RESERVED_SEEDS = {90401, 90411, 90412, 90413}
MIN_NEW_GAIN = 0.02
MAX_OLD_REGRESSION = 0.005
STRUCTURAL_TOLERANCE = 1e-5


@dataclass(frozen=True)
class M0Config:
    model: MiniCLMConfig = MiniCLMConfig()
    pretrain_steps: int = 12
    direct_steps: int = 4
    growth_steps: int = 8
    reuse_steps: int = 4
    learning_rate: float = 0.03


def _stable_int(text: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def make_examples(
    *,
    address_id: str,
    family: str,
    target_token: int,
    count: int,
    vocab_size: int,
    start_index: int = 0,
    knowledge_key: str | None = None,
) -> list[TokenExample]:
    examples: list[TokenExample] = []
    for index in range(start_index, start_index + count):
        rng = random.Random(_stable_int(f"{family}|{index}"))
        cue0 = 4 + rng.randrange(max(2, vocab_size - 20))
        cue1 = 4 + rng.randrange(max(2, vocab_size - 20))
        target = int(target_token) % vocab_size
        tokens = (
            1,
            cue0 % vocab_size,
            target,
            cue1 % vocab_size,
            target,
            (cue0 + cue1) % vocab_size,
            target,
            2,
        )
        examples.append(
            TokenExample(
                example_id=f"{family}:{index:03d}",
                address_id=address_id,
                tokens=tokens,
                knowledge_key=knowledge_key,
            )
        )
    return examples


def _batch(examples: list[TokenExample], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    token_ids = torch.tensor(
        [list(example.tokens) for example in examples], dtype=torch.long, device=device
    )
    return token_ids[:, :-1], token_ids[:, 1:], [example.address_id for example in examples]


def mean_nll(model: TinyCLMDecoder, examples: Iterable[TokenExample], device: torch.device) -> float:
    items = list(examples)
    if not items:
        return 0.0
    model.eval()
    x, y, addresses = _batch(items, device)
    with torch.no_grad():
        logits = model(x, addresses)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    return float(loss.detach().cpu())


def scored_logits(
    model: TinyCLMDecoder, examples: Iterable[TokenExample], device: torch.device
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    model.eval()
    with torch.no_grad():
        for example in examples:
            x, _, addresses = _batch([example], device)
            result[example.example_id] = model(x, addresses).detach().cpu()
    return result


def max_logit_delta(
    before: TinyCLMDecoder,
    after: TinyCLMDecoder,
    examples: Iterable[TokenExample],
    device: torch.device,
) -> float:
    items = list(examples)
    if not items:
        return 0.0
    a = scored_logits(before, items, device)
    b = scored_logits(after, items, device)
    return max(float((a[key] - b[key]).abs().max()) for key in a)


def _relative_change(before: float, after: float) -> float:
    return (after - before) / max(before, 1e-8)


def _new_gain(before: float, after: float) -> float:
    return (before - after) / max(before, 1e-8)


def _training_token_count(examples: list[TokenExample], steps: int) -> int:
    return sum(max(0, len(example.tokens) - 1) for example in examples) * int(steps)


def _evaluation_token_count(groups: Iterable[Iterable[TokenExample]]) -> int:
    total = 0
    for group in groups:
        total += sum(max(0, len(example.tokens) - 1) for example in group)
    return total


def _peak_memory(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def _train_only_cells(
    model: TinyCLMDecoder,
    *,
    cell_ids: list[str],
    examples: list[TokenExample],
    steps: int,
    learning_rate: float,
    device: torch.device,
) -> float:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = model.modules_for_cell_ids(cell_ids)
    params: list[nn.Parameter] = []
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            params.append(parameter)
    if not params:
        raise RuntimeError("candidate has no trainable Cell parameters")
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0)
    x, y, addresses = _batch(examples, device)
    started = time.perf_counter()
    model.train()
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, addresses)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return elapsed


def _pretrain(
    model: TinyCLMDecoder,
    examples: list[TokenExample],
    *,
    steps: int,
    device: torch.device,
) -> None:
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    x, y, addresses = _batch(examples, device)
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, addresses)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


class M0Harness:
    def __init__(self, model: TinyCLMDecoder, *, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.dependency_index = DependencyIndex()
        self.probes: dict[str, TokenExample] = {}
        self.registry = CellRegistry(model)
        self.records: list[dict] = []

    def protected_examples(self, ids: Iterable[str] | None = None) -> list[TokenExample]:
        if ids is None:
            return [self.probes[key] for key in sorted(self.probes)]
        return [self.probes[key] for key in sorted(set(ids)) if key in self.probes]

    def admit_probes(self, examples: Iterable[TokenExample]) -> None:
        for example in examples:
            self.probes[example.example_id] = example
            self.dependency_index.register(
                example.example_id,
                self.model.active_cell_ids(example.address_id),
            )

    def _attempt(
        self,
        *,
        candidate: TinyCLMDecoder,
        candidate_kind: str,
        touched_cells: list[str],
        new_validation: list[TokenExample],
        optimizer_steps: int,
        training_tokens: int,
        candidate_wall_seconds: float,
        smoke_override: str,
    ) -> dict:
        validation_started = time.perf_counter()
        all_probe_ids = set(self.probes)
        local_ids = self.dependency_index.scope(touched_cells)
        outside_ids = all_probe_ids - local_ids
        local_examples = self.protected_examples(local_ids)
        global_examples = self.protected_examples(all_probe_ids)
        outside_examples = self.protected_examples(outside_ids)

        new_before = mean_nll(self.model, new_validation, self.device)
        new_after = mean_nll(candidate, new_validation, self.device)
        local_before = mean_nll(self.model, local_examples, self.device)
        local_after = mean_nll(candidate, local_examples, self.device)
        global_before = mean_nll(self.model, global_examples, self.device)
        global_after = mean_nll(candidate, global_examples, self.device)

        gain = _new_gain(new_before, new_after)
        local_regression = _relative_change(local_before, local_after) if local_examples else 0.0
        global_regression = _relative_change(global_before, global_after) if global_examples else 0.0
        local_pass = gain >= MIN_NEW_GAIN and local_regression <= MAX_OLD_REGRESSION
        oracle_pass = gain >= MIN_NEW_GAIN and global_regression <= MAX_OLD_REGRESSION

        before_logits = scored_logits(self.model, outside_examples, self.device)
        after_logits = scored_logits(candidate, outside_examples, self.device)
        escapes = 0
        for probe_id in before_logits:
            if (
                float((before_logits[probe_id] - after_logits[probe_id]).abs().max())
                > STRUCTURAL_TOLERANCE
            ):
                escapes += 1
        structural_rate = escapes / float(max(1, len(outside_examples)))
        validation_seconds = time.perf_counter() - validation_started
        touched_parameters = sum(
            parameter.numel()
            for module in candidate.modules_for_cell_ids(touched_cells)
            for parameter in module.parameters()
        )
        total_parameters = sum(parameter.numel() for parameter in candidate.parameters())

        return {
            "attempt_index": 0,
            "candidate_kind": candidate_kind,
            "touched_cells": list(touched_cells),
            "touched_parameter_count": int(touched_parameters),
            "touched_parameter_fraction": touched_parameters / float(max(1, total_parameters)),
            "local_dependency_probe_count": len(local_ids),
            "local_dependency_coverage": len(local_ids) / float(max(1, len(all_probe_ids))),
            "new_metrics_before": {"nll": new_before},
            "new_metrics_candidate": {"nll": new_after},
            "local_old_metrics_before": {"nll": local_before, "probe_count": len(local_examples)},
            "local_old_metrics_candidate": {"nll": local_after, "probe_count": len(local_examples)},
            "global_old_metrics_before": {"nll": global_before, "probe_count": len(global_examples)},
            "global_old_metrics_candidate": {"nll": global_after, "probe_count": len(global_examples)},
            "new_gain": gain,
            "local_regression": local_regression,
            "global_regression": global_regression,
            "local_pass": bool(local_pass),
            "oracle_pass": bool(oracle_pass),
            "false_safe": bool(local_pass and not oracle_pass),
            "structural_escape_count": int(escapes),
            "structural_escape_rate": float(structural_rate),
            "training_tokens": int(training_tokens),
            "validation_tokens": int(
                _evaluation_token_count([new_validation, local_examples, global_examples])
            ),
            "optimizer_steps": int(optimizer_steps),
            "candidate_wall_seconds": float(candidate_wall_seconds),
            "validation_wall_seconds": float(validation_seconds),
            "peak_gpu_memory_bytes": _peak_memory(self.device),
            "smoke_path_override": smoke_override,
        }

    def _base_active_by_layer(self, address_id: str) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for layer_id in (3, 4):
            layer = self.model.sparse_layer(layer_id)
            route = layer.base_route(address_id)
            cells = [layer.base_cell_id(index) for index in route]
            if layer.has_private(address_id):
                cells.append(layer.private_cell_id(address_id))
            result[str(layer_id)] = cells
        return result

    def _checkpoint_payload(self) -> dict:
        return {
            "format": "minicells.clm-0.4-mini.m0-checkpoint.v1",
            "config": self.model.cfg.to_dict(),
            "private_addresses": self.model.private_addresses(),
            "model_state": self.model.state_dict(),
            "dependency_index": self.dependency_index.to_dict(),
            "probes": {key: value.to_dict() for key, value in sorted(self.probes.items())},
            "registry": self.registry.to_dict(),
            "records": self.records,
            "state_hash": model_state_hash(self.model),
        }

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint_payload(), path)

    @classmethod
    def load_checkpoint(cls, path: Path, *, device: torch.device) -> "M0Harness":
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg = MiniCLMConfig.from_dict(payload["config"])
        model = TinyCLMDecoder(cfg).to(device)
        for address_id in payload["private_addresses"]:
            model.spawn_growth_bundle(address_id)
        model.load_state_dict(payload["model_state"])
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        harness = cls(model, device=device)
        harness.dependency_index = DependencyIndex.from_dict(payload["dependency_index"])
        harness.probes = {
            key: TokenExample.from_dict(value) for key, value in payload["probes"].items()
        }
        harness.registry = CellRegistry.from_dict(model, payload["registry"])
        harness.records = list(payload["records"])
        actual = model_state_hash(model)
        if actual != payload["state_hash"]:
            raise RuntimeError("checkpoint state hash mismatch")
        return harness

    def _write_checkpoint_and_finish_record(self, *, record: dict, result_dir: Path) -> None:
        checkpoint_rel = f"checkpoints/tx-{record['transaction_id']:03d}.pt"
        record["checkpoint_reference"] = checkpoint_rel
        record["journal_reference"] = "transactions.jsonl"
        self.records.append(record)
        self.save_checkpoint(result_dir / checkpoint_rel)

    def execute_growth_transaction(
        self,
        *,
        transaction_id: int,
        address_id: str,
        operation: str,
        train_examples: list[TokenExample],
        validation_examples: list[TokenExample],
        probe_examples: list[TokenExample],
        direct_steps: int,
        growth_steps: int,
        learning_rate: float,
        result_dir: Path,
    ) -> dict:
        started = time.perf_counter()
        state_before = model_state_hash(self.model)
        base_routes = self.model.base_routes(address_id)
        direct_cells = self.model.base_cell_ids(address_id)

        direct = copy.deepcopy(self.model).to(self.device)
        direct_seconds = _train_only_cells(
            direct,
            cell_ids=direct_cells,
            examples=train_examples,
            steps=direct_steps,
            learning_rate=learning_rate,
            device=self.device,
        )
        direct_attempt = self._attempt(
            candidate=direct,
            candidate_kind="direct",
            touched_cells=direct_cells,
            new_validation=validation_examples,
            optimizer_steps=direct_steps,
            training_tokens=_training_token_count(train_examples, direct_steps),
            candidate_wall_seconds=direct_seconds,
            smoke_override="force-direct-reject-to-exercise-growth",
        )
        self.registry.record_rejected(direct_cells)

        if model_state_hash(self.model) != state_before:
            raise RuntimeError("direct rollback mutated committed model state")

        growth = copy.deepcopy(self.model).to(self.device)
        growth_ids = growth.spawn_growth_bundle(address_id)
        zero_delta = max_logit_delta(
            self.model,
            growth,
            [*validation_examples, *self.protected_examples()],
            self.device,
        )
        if zero_delta > 1e-6:
            raise RuntimeError(f"zero-output growth changed logits before training: {zero_delta}")
        growth_seconds = _train_only_cells(
            growth,
            cell_ids=growth_ids,
            examples=train_examples,
            steps=growth_steps,
            learning_rate=learning_rate,
            device=self.device,
        )
        growth_attempt = self._attempt(
            candidate=growth,
            candidate_kind="spawn",
            touched_cells=growth_ids,
            new_validation=validation_examples,
            optimizer_steps=growth_steps,
            training_tokens=_training_token_count(train_examples, growth_steps),
            candidate_wall_seconds=growth_seconds,
            smoke_override="force-growth-commit-after-zero-output-invariant",
        )
        growth_attempt["attempt_index"] = 1
        growth_attempt["zero_output_pretrain_max_logit_delta"] = float(zero_delta)

        self.model = growth
        births = self.registry.add_growth_bundle(
            self.model, address_id=address_id, transaction_id=transaction_id
        )
        self.registry.record_activation(
            self.model.active_cell_ids(address_id), amount=len(train_examples)
        )
        self.admit_probes(probe_examples)
        state_after = model_state_hash(self.model)
        record = {
            "transaction_id": int(transaction_id),
            "operation": operation,
            "address_id": address_id,
            "knowledge_key": None,
            "train_manifest_ids": [x.example_id for x in train_examples],
            "validation_manifest_ids": [x.example_id for x in validation_examples],
            "probe_manifest_ids": [x.example_id for x in probe_examples],
            "state_hash_before": state_before,
            "base_route_cells_by_layer": base_routes,
            "private_bundle_id_before": None,
            "attempts": [direct_attempt, growth_attempt],
            "final_decision": "growth-commit",
            "growth_attempted": True,
            "cell_births": births,
            "cell_deletions": [],
            "state_hash_after": state_after,
            "active_cells_by_layer": self._base_active_by_layer(address_id),
            "transaction_wall_seconds": time.perf_counter() - started,
            "smoke_only": True,
        }
        self._write_checkpoint_and_finish_record(record=record, result_dir=result_dir)
        return record

    def execute_reuse_transaction(
        self,
        *,
        transaction_id: int,
        address_id: str,
        train_examples: list[TokenExample],
        validation_examples: list[TokenExample],
        probe_examples: list[TokenExample],
        steps: int,
        learning_rate: float,
        result_dir: Path,
    ) -> dict:
        if not self.model.has_private_bundle(address_id):
            raise RuntimeError("private-reuse smoke path requires an existing bundle")
        started = time.perf_counter()
        state_before = model_state_hash(self.model)
        base_routes = self.model.base_routes(address_id)
        private_ids = self.model.private_cell_ids(address_id)
        candidate = copy.deepcopy(self.model).to(self.device)
        train_seconds = _train_only_cells(
            candidate,
            cell_ids=private_ids,
            examples=train_examples,
            steps=steps,
            learning_rate=learning_rate,
            device=self.device,
        )
        attempt = self._attempt(
            candidate=candidate,
            candidate_kind="private-reuse",
            touched_cells=private_ids,
            new_validation=validation_examples,
            optimizer_steps=steps,
            training_tokens=_training_token_count(train_examples, steps),
            candidate_wall_seconds=train_seconds,
            smoke_override="force-private-reuse-commit",
        )
        self.model = candidate
        self.registry.record_accepted(private_ids, transaction_id=transaction_id, reuse=True)
        self.registry.record_activation(
            self.model.active_cell_ids(address_id), amount=len(train_examples)
        )
        self.admit_probes(probe_examples)
        state_after = model_state_hash(self.model)
        record = {
            "transaction_id": int(transaction_id),
            "operation": "capability",
            "address_id": address_id,
            "knowledge_key": None,
            "train_manifest_ids": [x.example_id for x in train_examples],
            "validation_manifest_ids": [x.example_id for x in validation_examples],
            "probe_manifest_ids": [x.example_id for x in probe_examples],
            "state_hash_before": state_before,
            "base_route_cells_by_layer": base_routes,
            "private_bundle_id_before": f"bundle:{address_id}",
            "attempts": [attempt],
            "final_decision": "private-reuse-commit",
            "growth_attempted": False,
            "cell_births": [],
            "cell_deletions": [],
            "state_hash_after": state_after,
            "active_cells_by_layer": self._base_active_by_layer(address_id),
            "transaction_wall_seconds": time.perf_counter() - started,
            "smoke_only": True,
        }
        self._write_checkpoint_and_finish_record(record=record, result_dir=result_dir)
        return record

    def execute_direct_transaction(
        self,
        *,
        transaction_id: int,
        address_id: str,
        operation: str,
        train_examples: list[TokenExample],
        validation_examples: list[TokenExample],
        probe_examples: list[TokenExample],
        steps: int,
        learning_rate: float,
        result_dir: Path,
        knowledge_key: str | None = None,
    ) -> dict:
        if self.model.has_private_bundle(address_id):
            raise RuntimeError("direct smoke path expects an address without private growth")
        started = time.perf_counter()
        state_before = model_state_hash(self.model)
        base_routes = self.model.base_routes(address_id)
        touched = self.model.base_cell_ids(address_id)
        candidate = copy.deepcopy(self.model).to(self.device)
        train_seconds = _train_only_cells(
            candidate,
            cell_ids=touched,
            examples=train_examples,
            steps=steps,
            learning_rate=learning_rate,
            device=self.device,
        )
        attempt = self._attempt(
            candidate=candidate,
            candidate_kind="direct",
            touched_cells=touched,
            new_validation=validation_examples,
            optimizer_steps=steps,
            training_tokens=_training_token_count(train_examples, steps),
            candidate_wall_seconds=train_seconds,
            smoke_override="force-direct-commit",
        )
        self.model = candidate
        self.registry.record_accepted(touched, transaction_id=transaction_id)
        self.registry.record_activation(
            self.model.active_cell_ids(address_id), amount=len(train_examples)
        )
        self.admit_probes(probe_examples)
        state_after = model_state_hash(self.model)
        record = {
            "transaction_id": int(transaction_id),
            "operation": operation,
            "address_id": address_id,
            "knowledge_key": knowledge_key,
            "train_manifest_ids": [x.example_id for x in train_examples],
            "validation_manifest_ids": [x.example_id for x in validation_examples],
            "probe_manifest_ids": [x.example_id for x in probe_examples],
            "state_hash_before": state_before,
            "base_route_cells_by_layer": base_routes,
            "private_bundle_id_before": None,
            "attempts": [attempt],
            "final_decision": "direct-commit",
            "growth_attempted": False,
            "cell_births": [],
            "cell_deletions": [],
            "state_hash_after": state_after,
            "active_cells_by_layer": self._base_active_by_layer(address_id),
            "transaction_wall_seconds": time.perf_counter() - started,
            "smoke_only": True,
        }
        self._write_checkpoint_and_finish_record(record=record, result_dir=result_dir)
        return record


def _seed_base_probes(harness: M0Harness, cfg: MiniCLMConfig) -> None:
    for index in range(4):
        harness.admit_probes(
            make_examples(
                address_id=f"base/{index}",
                family=f"base-probe-{index}",
                target_token=10 + index,
                count=2,
                vocab_size=cfg.vocab_size,
                start_index=100,
            )
        )


def _base_training_examples(cfg: MiniCLMConfig) -> list[TokenExample]:
    examples: list[TokenExample] = []
    for index in range(4):
        examples.extend(
            make_examples(
                address_id=f"base/{index}",
                family=f"base-train-{index}",
                target_token=10 + index,
                count=4,
                vocab_size=cfg.vocab_size,
            )
        )
    return examples


def replay_journal(result_dir: Path, *, device: torch.device) -> dict:
    records = [
        json.loads(line)
        for line in (result_dir / "transactions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    errors: list[str] = []
    previous_after: str | None = None
    for record in records:
        if previous_after is not None and record["state_hash_before"] != previous_after:
            errors.append(f"state-chain mismatch at transaction {record['transaction_id']}")
        restored = M0Harness.load_checkpoint(
            result_dir / record["checkpoint_reference"], device=device
        )
        if model_state_hash(restored.model) != record["state_hash_after"]:
            errors.append(f"checkpoint mismatch at transaction {record['transaction_id']}")
        previous_after = record["state_hash_after"]
    return {
        "valid": not errors,
        "transactions_replayed": len(records),
        "errors": errors,
        "final_state_hash": previous_after,
    }


def run_m0(
    out_dir: Path,
    *,
    device: str | torch.device = "cpu",
    seed: int = M0_SEED,
    config: M0Config | None = None,
) -> dict:
    if int(seed) in RESERVED_SEEDS:
        raise ValueError("M0 cannot use development or formal CLM-0.4-mini seeds")
    cfg = config or M0Config()
    device_obj = torch.device(device)
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device_obj)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = TinyCLMDecoder(cfg.model).to(device_obj)
    _pretrain(model, _base_training_examples(cfg.model), steps=cfg.pretrain_steps, device=device_obj)
    harness = M0Harness(model, device=device_obj)
    _seed_base_probes(harness, cfg.model)
    initial_hash = model_state_hash(model)

    math_train = make_examples(
        address_id="math/mul", family="m0-math-train-a", target_token=31, count=4,
        vocab_size=cfg.model.vocab_size,
    )
    math_val = make_examples(
        address_id="math/mul", family="m0-math-val-a", target_token=31, count=3,
        vocab_size=cfg.model.vocab_size, start_index=20,
    )
    math_probe = make_examples(
        address_id="math/mul", family="m0-math-probe-a", target_token=31, count=2,
        vocab_size=cfg.model.vocab_size, start_index=40,
    )
    harness.execute_growth_transaction(
        transaction_id=0,
        address_id="math/mul",
        operation="capability",
        train_examples=math_train,
        validation_examples=math_val,
        probe_examples=math_probe,
        direct_steps=cfg.direct_steps,
        growth_steps=cfg.growth_steps,
        learning_rate=cfg.learning_rate,
        result_dir=out_dir,
    )

    math_train_b = make_examples(
        address_id="math/mul", family="m0-math-train-b", target_token=31, count=4,
        vocab_size=cfg.model.vocab_size,
    )
    math_val_b = make_examples(
        address_id="math/mul", family="m0-math-val-b", target_token=31, count=3,
        vocab_size=cfg.model.vocab_size, start_index=20,
    )
    math_probe_b = make_examples(
        address_id="math/mul", family="m0-math-probe-b", target_token=31, count=2,
        vocab_size=cfg.model.vocab_size, start_index=40,
    )
    harness.execute_reuse_transaction(
        transaction_id=1,
        address_id="math/mul",
        train_examples=math_train_b,
        validation_examples=math_val_b,
        probe_examples=math_probe_b,
        steps=cfg.reuse_steps,
        learning_rate=cfg.learning_rate,
        result_dir=out_dir,
    )

    story_train = make_examples(
        address_id="story/world-1", family="m0-story-train", target_token=41, count=4,
        vocab_size=cfg.model.vocab_size, knowledge_key="world-1:location:mira",
    )
    story_val = make_examples(
        address_id="story/world-1", family="m0-story-val", target_token=41, count=3,
        vocab_size=cfg.model.vocab_size, start_index=20,
        knowledge_key="world-1:location:mira",
    )
    story_probe = make_examples(
        address_id="story/world-1", family="m0-story-probe", target_token=41, count=2,
        vocab_size=cfg.model.vocab_size, start_index=40,
        knowledge_key="world-1:location:mira",
    )
    harness.execute_direct_transaction(
        transaction_id=2,
        address_id="story/world-1",
        operation="append",
        train_examples=story_train,
        validation_examples=story_val,
        probe_examples=story_probe,
        steps=cfg.direct_steps,
        learning_rate=cfg.learning_rate,
        result_dir=out_dir,
        knowledge_key="world-1:location:mira",
    )

    (out_dir / "transactions.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in harness.records),
        encoding="utf-8",
    )
    registry_entries = harness.registry.snapshot(harness.model, harness.dependency_index)
    (out_dir / "cell-registry.json").write_text(
        json.dumps(registry_entries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    replay = replay_journal(out_dir, device=device_obj)
    (out_dir / "replay.json").write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not replay["valid"]:
        raise RuntimeError(f"M0 replay failed: {replay['errors']}")

    final_hash = model_state_hash(harness.model)
    decisions = [record["final_decision"] for record in harness.records]
    decision = {
        "experiment_id": "clm-0.4-mini-language-validation",
        "phase": "M0",
        "mode": "smoke",
        "status": "SMOKE_ONLY",
        "pass": None,
        "scientific_decision": False,
        "seed": int(seed),
        "reason": "M0 validates execution paths only; it cannot support or reject the M1 hypothesis.",
        "paths_exercised": decisions,
    }
    summary = {
        "format": "minicells.clm-0.4-mini.m0-summary.v1",
        "decision": decision,
        "initial_state_hash": initial_hash,
        "final_state_hash": final_hash,
        "transaction_count": len(harness.records),
        "private_addresses": harness.model.private_addresses(),
        "dependency_probe_count": len(harness.probes),
        "cell_registry_entries": len(registry_entries),
        "replay_valid": replay["valid"],
        "model_config": cfg.model.to_dict(),
    }
    (out_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
