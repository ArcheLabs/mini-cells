from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from .continual_learning import (
    MARGIN_Q,
    PARAMETER_COUNT,
    PARAMETER_MAX_Q,
    PARAMETER_MIN_Q,
    TaskBatch,
    candidate,
    combine_batches,
    delta_vector,
    exact_logits,
    load_q88_model,
    model_hash,
    save_q88_model,
)
from .vocab import CharVocab

Objective = Literal["loss", "accuracy-lex"]
ApplyMode = Literal["legacy-step", "evaluated-candidate", "step-recheck"]

MASK64 = (1 << 64) - 1
WORDS = ("mini", "cells", "jam", "hello", "world", "learn", "echo", "small", "local", "neural")
PRODUCTION_BATCH_DOMAIN = b"mini-cells:batch:v1"
EXTRA_BATCH_DOMAIN = b"mini-cells:batch-extra:v1"
FIXED_PROBE_DOMAIN = b"mini-cells:local-probe:v1"


@dataclass(frozen=True)
class EvalStats:
    loss: int
    correct: int
    total: int

    @property
    def token_accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def loss_per_token(self) -> float:
        return self.loss / self.total if self.total else 0.0


@dataclass(frozen=True)
class SearchConfig:
    block_size: int
    perturbation_q: int
    step_q: int
    objective: Objective = "loss"
    batch_groups: int = 1
    apply_mode: ApplyMode = "step-recheck"

    def __post_init__(self) -> None:
        if not 1 <= self.block_size <= PARAMETER_COUNT:
            raise ValueError("block_size out of range")
        if self.perturbation_q <= 0 or self.step_q <= 0:
            raise ValueError("perturbation_q and step_q must be positive")
        if self.batch_groups not in (1, 2, 4, 8):
            raise ValueError("batch_groups must be one of 1, 2, 4, 8")
        if self.objective not in ("loss", "accuracy-lex"):
            raise ValueError("unsupported objective")
        if self.apply_mode not in ("legacy-step", "evaluated-candidate", "step-recheck"):
            raise ValueError("unsupported apply mode")
        if self.apply_mode == "evaluated-candidate" and self.step_q != self.perturbation_q:
            raise ValueError("evaluated-candidate requires step_q == perturbation_q")

    @property
    def id(self) -> str:
        block = "global" if self.block_size == PARAMETER_COUNT else f"b{self.block_size}"
        objective = "loss" if self.objective == "loss" else "acclex"
        mode = {
            "legacy-step": "legacy",
            "evaluated-candidate": "evalcand",
            "step-recheck": "recheck",
        }[self.apply_mode]
        return (
            f"{block}-q{self.perturbation_q}-s{self.step_q}"
            f"-{objective}-g{self.batch_groups}-{mode}"
        )

    @property
    def production_cost_class(self) -> str:
        if self.apply_mode == "legacy-step":
            return "invalid-retained-guard"
        if self.apply_mode == "step-recheck" and self.step_q != self.perturbation_q:
            return "requires-extra-proposal-metrics"
        if self.block_size < PARAMETER_COUNT or self.batch_groups > 1 or self.objective != "loss":
            return "runtime-semantic-change-no-result-growth"
        return "drop-in-v2"


@dataclass
class GenerationRecord:
    config_id: str
    generation: int
    parent_model_hash: str
    next_model_hash: str
    base_loss: int
    base_correct: int
    plus_loss: int
    plus_correct: int
    minus_loss: int
    minus_correct: int
    proposal_loss: int
    proposal_correct: int
    retained_loss: int
    retained_correct: int
    total_tokens: int
    candidate_direction: int
    decision: str
    accepted: bool
    moved_parameters: int
    bound_parameters: int


@dataclass
class ProbeRecord:
    config_id: str
    generation: int
    model_hash: str
    total_loss: int
    correct_tokens: int
    total_tokens: int
    token_accuracy: float
    loss_per_token: float


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def range(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper must be positive")
        return self.next() % upper


def _seed32(domain: bytes, parent_hash: bytes, generation: int, group: int = 0) -> int:
    h = hashlib.blake2b(digest_size=32)
    h.update(domain)
    h.update(parent_hash)
    h.update(int(generation).to_bytes(8, "little", signed=False))
    if domain == EXTRA_BATCH_DOMAIN:
        h.update(int(group).to_bytes(4, "little", signed=False))
    return int.from_bytes(h.digest()[:8], "little", signed=False)


def _encoded_echo_batch(parent_hash: bytes, generation: int, size: int, group: int) -> list[list[int]]:
    if not 1 <= size <= 4:
        raise ValueError("each canonical microbatch must have 1..4 samples")
    vocab = CharVocab()
    domain = PRODUCTION_BATCH_DOMAIN if group == 0 else EXTRA_BATCH_DOMAIN
    rng = SplitMix64(_seed32(domain, parent_hash, generation, group))
    rows: list[list[int]] = []
    space_id = vocab.token_to_id[" "]
    for _ in range(size):
        length = 1 + rng.range(32)
        row = [0] * length
        if rng.range(10) < 7:
            for pos in range(length):
                row[pos] = 1 + rng.range(len(vocab.SYMBOLS))
        else:
            pos = 0
            while pos < length:
                if pos > 0:
                    row[pos] = space_id
                    pos += 1
                    if pos >= length:
                        break
                word = WORDS[rng.range(len(WORDS))]
                for char in word:
                    if pos >= length:
                        break
                    row[pos] = vocab.token_to_id[char]
                    pos += 1
        rows.append(row)
    return rows


def canonical_echo_batch(parent_hash: bytes, generation: int, size: int = 4, group: int = 0) -> TaskBatch:
    rows = _encoded_echo_batch(parent_hash, generation, size, group)
    ids = torch.zeros((len(rows), 64), dtype=torch.long)
    lengths = torch.tensor([len(row) for row in rows], dtype=torch.long)
    mask = torch.arange(64).unsqueeze(0) < lengths.unsqueeze(1)
    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return TaskBatch(
        input_ids=ids,
        target_ids=ids.clone(),
        mask=mask,
        lengths=lengths,
        changed_mask=torch.zeros_like(mask),
    )


def training_batch(parent_hash: bytes, generation: int, groups: int) -> TaskBatch:
    if groups not in (1, 2, 4, 8):
        raise ValueError("unsupported group count")
    return combine_batches(
        [canonical_echo_batch(parent_hash, generation, 4, group) for group in range(groups)]
    )


@torch.no_grad()
def evaluate_stats(flat: torch.Tensor, batch: TaskBatch, margin_q: int = MARGIN_Q) -> EvalStats:
    logits = exact_logits(flat, batch.input_ids)
    targets = batch.target_ids.to(dtype=torch.long, device="cpu")
    mask = batch.mask.to(dtype=torch.bool, device="cpu")
    predictions = logits.argmax(dim=-1)
    correct = int(((predictions == targets) & mask).sum().item())
    total = int(mask.sum().item())
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    competitors = logits.clone()
    competitors.scatter_(-1, targets.unsqueeze(-1), -(1 << 60))
    other = competitors.max(dim=-1).values
    loss = (margin_q - (target_logits - other)).clamp_min(0)
    return EvalStats(int(loss[mask].sum().item()), correct, total)


def objective_score(stats: EvalStats, objective: Objective) -> tuple[int, ...]:
    if objective == "loss":
        return (stats.loss,)
    if objective == "accuracy-lex":
        return (-stats.correct, stats.loss)
    raise ValueError(objective)


def choose_candidate(
    base: EvalStats,
    plus: EvalStats,
    minus: EvalStats,
    objective: Objective,
) -> int:
    base_score = objective_score(base, objective)
    plus_score = objective_score(plus, objective)
    minus_score = objective_score(minus, objective)
    if plus_score < base_score and plus_score < minus_score:
        return 1
    if minus_score < base_score and minus_score < plus_score:
        return -1
    return 0


def fixed_probe(flat: torch.Tensor, config_id: str = "probe", generation: int = 0) -> ProbeRecord:
    h = hashlib.blake2b(FIXED_PROBE_DOMAIN, digest_size=32).digest()
    total_loss = 0
    correct = 0
    total = 0
    for batch_generation in range(32):
        stats = evaluate_stats(flat, canonical_echo_batch(h, batch_generation, 4, 0))
        total_loss += stats.loss
        correct += stats.correct
        total += stats.total
    return ProbeRecord(
        config_id=config_id,
        generation=generation,
        model_hash="0x" + model_hash(flat).hex(),
        total_loss=total_loss,
        correct_tokens=correct,
        total_tokens=total,
        token_accuracy=correct / total if total else 0.0,
        loss_per_token=total_loss / total if total else 0.0,
    )


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _step(
    flat: torch.Tensor,
    generation: int,
    config: SearchConfig,
) -> tuple[torch.Tensor, GenerationRecord]:
    parent = model_hash(flat)
    batch = training_batch(parent, generation, config.batch_groups)
    delta = delta_vector(parent, generation, config.block_size)
    base = evaluate_stats(flat, batch)
    plus_model = candidate(flat, delta, 1, config.perturbation_q)
    minus_model = candidate(flat, delta, -1, config.perturbation_q)
    plus = evaluate_stats(plus_model, batch)
    minus = evaluate_stats(minus_model, batch)
    direction = choose_candidate(base, plus, minus, config.objective)

    accepted = False
    decision = "keep"
    proposal = flat
    proposal_stats = base
    retained_stats = base

    if direction:
        decision = "plus" if direction > 0 else "minus"
        if config.apply_mode == "evaluated-candidate":
            proposal = plus_model if direction > 0 else minus_model
            proposal_stats = plus if direction > 0 else minus
            accepted = True
        else:
            proposal = candidate(flat, delta, direction, config.step_q)
            proposal_stats = evaluate_stats(proposal, batch)
            if config.apply_mode == "legacy-step":
                accepted = True
            elif config.apply_mode == "step-recheck":
                accepted = objective_score(proposal_stats, config.objective) < objective_score(
                    base, config.objective
                )
            else:
                raise ValueError(config.apply_mode)
        if accepted:
            retained_stats = proposal_stats
        else:
            decision = "keep"

    next_flat = proposal if accepted else flat
    moved = int((next_flat != flat).sum().item())
    bounds = int(
        ((next_flat <= PARAMETER_MIN_Q) | (next_flat >= PARAMETER_MAX_Q)).sum().item()
    )
    record = GenerationRecord(
        config_id=config.id,
        generation=generation + 1,
        parent_model_hash="0x" + parent.hex(),
        next_model_hash="0x" + model_hash(next_flat).hex(),
        base_loss=base.loss,
        base_correct=base.correct,
        plus_loss=plus.loss,
        plus_correct=plus.correct,
        minus_loss=minus.loss,
        minus_correct=minus.correct,
        proposal_loss=proposal_stats.loss,
        proposal_correct=proposal_stats.correct,
        retained_loss=retained_stats.loss,
        retained_correct=retained_stats.correct,
        total_tokens=base.total,
        candidate_direction=direction,
        decision=decision,
        accepted=accepted,
        moved_parameters=moved,
        bound_parameters=bounds,
    )
    return next_flat, record


def summarize_run(run_dir: Path, config: SearchConfig, target_generation: int) -> dict:
    metrics = _read_jsonl(run_dir / "metrics.jsonl")
    probes = _read_jsonl(run_dir / "probes.jsonl")
    if not probes:
        raise ValueError("missing probe records")
    initial = probes[0]
    final = max(
        (row for row in probes if row["generation"] <= target_generation),
        key=lambda row: row["generation"],
    )
    eligible = [row for row in probes if row["generation"] <= target_generation]
    best = min(eligible, key=lambda row: row["total_loss"])
    accepted_rows = [
        row for row in metrics
        if row["generation"] <= target_generation and bool(row["accepted"])
    ]
    accepted = len(accepted_rows)
    unsafe_actual_proposals = 0
    for row in accepted_rows:
        base_stats = EvalStats(
            int(row["base_loss"]), int(row["base_correct"]), int(row["total_tokens"])
        )
        proposal_stats = EvalStats(
            int(row["proposal_loss"]), int(row["proposal_correct"]), int(row["total_tokens"])
        )
        if objective_score(proposal_stats, config.objective) >= objective_score(
            base_stats, config.objective
        ):
            unsafe_actual_proposals += 1
    quarter_start = max(0, target_generation - max(1, target_generation // 4))
    late = [
        row for row in metrics
        if quarter_start < row["generation"] <= target_generation
    ]
    late_accepted = sum(bool(row["accepted"]) for row in late)
    loss_improvement = (
        (initial["total_loss"] - final["total_loss"]) / initial["total_loss"]
        if initial["total_loss"]
        else 0.0
    )
    best_loss_improvement = (
        (initial["total_loss"] - best["total_loss"]) / initial["total_loss"]
        if initial["total_loss"]
        else 0.0
    )
    return {
        "config_id": config.id,
        **asdict(config),
        "production_cost_class": config.production_cost_class,
        "target_generation": target_generation,
        "initial_probe_loss": initial["total_loss"],
        "final_probe_loss": final["total_loss"],
        "best_probe_loss": best["total_loss"],
        "initial_probe_accuracy": initial["token_accuracy"],
        "final_probe_accuracy": final["token_accuracy"],
        "best_probe_accuracy": max(row["token_accuracy"] for row in eligible),
        "loss_improvement": loss_improvement,
        "best_loss_improvement": best_loss_improvement,
        "accuracy_delta": final["token_accuracy"] - initial["token_accuracy"],
        "accepted_updates": accepted,
        "accepted_actual_proposals_not_improving_base": unsafe_actual_proposals,
        "acceptance_rate": accepted / target_generation if target_generation else 0.0,
        "late_acceptance_rate": late_accepted / len(late) if late else 0.0,
        "final_to_best_loss": (
            final["total_loss"] / best["total_loss"] if best["total_loss"] else 1.0
        ),
        "final_model_hash": final["model_hash"],
        "final_bound_parameters": (
            next(
                (
                    row["bound_parameters"]
                    for row in reversed(metrics)
                    if row["generation"] <= target_generation
                ),
                0,
            )
        ),
    }


def run_config(
    initial_model_path: Path,
    run_dir: Path,
    config: SearchConfig,
    target_generation: int,
    probe_every: int = 64,
) -> dict:
    if target_generation <= 0:
        raise ValueError("target_generation must be positive")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    state_path = run_dir / "state.json"
    model_path = run_dir / "model.bin"
    metrics_path = run_dir / "metrics.jsonl"
    probes_path = run_dir / "probes.jsonl"

    config_payload = {**asdict(config), "config_id": config.id}
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config_payload:
            raise ValueError(f"config mismatch in {run_dir}")
    else:
        config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        generation = int(state["generation"])
        flat = load_q88_model(model_path)
    else:
        generation = 0
        flat = load_q88_model(initial_model_path)
        save_q88_model(model_path, flat)
        state_path.write_text(
            json.dumps({"generation": 0, "model_hash": "0x" + model_hash(flat).hex()}, indent=2),
            encoding="utf-8",
        )
        _append_jsonl(probes_path, asdict(fixed_probe(flat, config.id, 0)))

    if generation > target_generation:
        return summarize_run(run_dir, config, target_generation)

    while generation < target_generation:
        flat, record = _step(flat, generation, config)
        generation += 1
        _append_jsonl(metrics_path, asdict(record))
        if generation % probe_every == 0 or generation == target_generation:
            existing = _read_jsonl(probes_path)
            if not any(row["generation"] == generation for row in existing):
                _append_jsonl(probes_path, asdict(fixed_probe(flat, config.id, generation)))
        save_q88_model(model_path, flat)
        state_path.write_text(
            json.dumps(
                {"generation": generation, "model_hash": "0x" + model_hash(flat).hex()},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    summary = summarize_run(run_dir, config, target_generation)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def baseline_config() -> SearchConfig:
    return SearchConfig(
        block_size=PARAMETER_COUNT,
        perturbation_q=4,
        step_q=1,
        objective="loss",
        batch_groups=1,
        apply_mode="legacy-step",
    )


def stage1_configs() -> list[SearchConfig]:
    configs: list[SearchConfig] = []
    global_pairs = [
        (2, 1), (4, 1), (4, 2), (8, 1), (8, 2), (8, 4),
        (16, 2), (16, 4), (16, 8), (32, 4), (32, 8),
    ]
    for q, step in global_pairs:
        configs.append(
            SearchConfig(PARAMETER_COUNT, q, step, "loss", 1, "step-recheck")
        )
    for q in (1, 2, 4, 8):
        configs.append(
            SearchConfig(PARAMETER_COUNT, q, q, "loss", 1, "evaluated-candidate")
        )
    for block in (64, 128, 256, 512):
        for q, step in ((4, 1), (8, 2), (16, 4)):
            configs.append(SearchConfig(block, q, step, "loss", 1, "step-recheck"))
    for block in (128, 256, 512):
        configs.append(SearchConfig(block, 4, 4, "loss", 1, "evaluated-candidate"))
    seen: set[str] = set()
    unique: list[SearchConfig] = []
    for config in configs:
        if config.id not in seen:
            unique.append(config)
            seen.add(config.id)
    return unique


def expand_stage2(configs: list[SearchConfig]) -> list[SearchConfig]:
    expanded: list[SearchConfig] = []
    for base in configs:
        for objective in ("loss", "accuracy-lex"):
            for groups in (1, 2, 4):
                expanded.append(
                    SearchConfig(
                        block_size=base.block_size,
                        perturbation_q=base.perturbation_q,
                        step_q=base.step_q,
                        objective=objective,
                        batch_groups=groups,
                        apply_mode=base.apply_mode,
                    )
                )
    seen: set[str] = set()
    out: list[SearchConfig] = []
    for config in expanded:
        if config.id not in seen:
            out.append(config)
            seen.add(config.id)
    return out


def rank_summaries(rows: list[dict]) -> list[dict]:
    # Primary goal is actual Echo accuracy, then loss improvement and stability.
    # Complexity is only a final tie breaker; a better algorithm should remain visible.
    cost_rank = {
        "drop-in-v2": 0,
        "runtime-semantic-change-no-result-growth": 1,
        "requires-extra-proposal-metrics": 2,
        "invalid-retained-guard": 3,
    }
    return sorted(
        rows,
        key=lambda row: (
            -float(row["final_probe_accuracy"]),
            -float(row["loss_improvement"]),
            float(row["final_to_best_loss"]),
            cost_rank.get(str(row["production_cost_class"]), 9),
            -float(row["acceptance_rate"]),
        ),
    )


def production_candidate_gates(summary: dict, solved_regression_pass: bool) -> dict[str, bool]:
    return {
        "final_loss_improvement_ge_10_percent": float(summary["loss_improvement"]) >= 0.10,
        "accuracy_delta_ge_2pp": float(summary["accuracy_delta"]) >= 0.02,
        "final_within_15_percent_of_best": float(summary["final_to_best_loss"]) <= 1.15,
        "accepted_updates_ge_4": int(summary["accepted_updates"]) >= 4,
        "solved_regression_pass": bool(solved_regression_pass),
        "not_legacy_step": summary["apply_mode"] != "legacy-step",
    }


def run_solved_regression(
    solved_model_path: Path,
    root: Path,
    config: SearchConfig,
    generations: int = 128,
) -> dict:
    summary = run_config(
        solved_model_path,
        root / config.id,
        config,
        generations,
        probe_every=max(16, generations // 4),
    )
    start_acc = float(summary["initial_probe_accuracy"])
    end_acc = float(summary["final_probe_accuracy"])
    start_loss = int(summary["initial_probe_loss"])
    end_loss = int(summary["final_probe_loss"])
    passed = end_acc >= start_acc - 0.01 and end_loss <= int(start_loss * 1.02)
    return {
        "config_id": config.id,
        "generations": generations,
        "initial_accuracy": start_acc,
        "final_accuracy": end_acc,
        "initial_loss": start_loss,
        "final_loss": end_loss,
        "pass": passed,
    }
