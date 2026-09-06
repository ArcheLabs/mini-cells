"""Observational readout localization for the frozen L7/K64 hybrid Cell mutation.

PCU-READOUT-LOCALIZATION-001 does not introduce a new training condition. It
replays the published PCU-HYBRID-OBJECTIVE-001 training path byte-for-byte via
``_train_hybrid_branch`` and first requires exact reproduction of the published
ranking/direct accuracies. It then inspects the trained model without any more
updates.

Two diagnostics localize the remaining autoregressive failure:

1. Gold-prefix next-token readout. For every A_eval answer token, measure the
   target token's full-vocabulary rank/top-1 status when every preceding answer
   token is teacher-forced correctly using the original CE tokenization.
2. Forced-prefix recovery. Force the first k correct answer tokens, then greedy
   generate the remaining answer-token suffix. This asks how much correct prefix
   is needed before the frozen decoder follows a stable trajectory.

The experiment is engineering-only and never consumes formal PCU seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .cellular import CellPartition, extract_expert_projections, patch_moe_block
from .evaluation import evaluate_samples
from .governance import git_provenance, write_json
from .hybrid_objective import (
    CE_WEIGHT,
    TARGET_K,
    _load_baselines,
    _train_hybrid_branch,
)
from .layer_placement import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_OPTIMIZER_STEPS,
    MAX_TRAINING_TOKENS,
    _assert_only_selected_deltas_trainable,
    _validate_foundation_manifest,
)
from .locality_width import ENGINEERING_SEED, TARGET_LAYER
from .model import load_granite, target_module
from .objective_alignment import evaluate_candidate_ranking
from .synthetic import audit_dataset, generate_world
from .task import TaskSequences, build_task_sequences, validate_answer_only_labels
from .training import BranchTrainingConfig


EXPERIMENT_ID = "PCU-READOUT-LOCALIZATION-001"
HYBRID_BASELINE_ROOT = Path(
    "artifacts/research/pcu-hybrid-objective-001/engineering/26090501-l7-k64-rank-plus-ce025"
)
OBJECTIVE_BASELINE_ROOT = Path(
    "artifacts/research/pcu-objective-alignment-001/engineering/26090501-l7-k64-ranking"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research/pcu-readout-localization-001/engineering/26090501-l7-k64-hybrid-readout"
)
HYBRID_SCIENTIFIC_SOURCE_COMMIT = "0241475a387a9114415cf7ed143670dd5c7e1b3b"
HYBRID_CORE_BLOB_SHA = "851c77cdd283def0698ebe721ea8bf216f5ed556"
EXPECTED_HYBRID_RANKING_TRAIN = 1.0
EXPECTED_HYBRID_RANKING_EVAL = 0.8359375
EXPECTED_HYBRID_DIRECT = 0.03125
READOUT_FLOOR = 0.80
DIAGNOSTIC_BATCH_SIZE = 8
FORCED_PREFIX_BATCH_SIZE = 16


@dataclass(frozen=True)
class PublishedHybrid:
    selected_cells: tuple[str, ...]
    dataset_manifest_sha256: str
    ranking_train_accuracy: float
    ranking_eval_accuracy: float
    direct_accuracy: float
    source: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranking_train_accuracy": float(self.ranking_train_accuracy),
            "ranking_eval_accuracy": float(self.ranking_eval_accuracy),
            "direct_accuracy": float(self.direct_accuracy),
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=_repo_root(), text=True, stderr=subprocess.DEVNULL
    ).strip()


def _assert_hybrid_scientific_core_unchanged() -> None:
    blob = _git("rev-parse", "HEAD:src/minicells/pcu_kill_001/hybrid_objective.py")
    if blob != HYBRID_CORE_BLOB_SHA:
        raise RuntimeError(f"HYBRID_SCIENTIFIC_CORE_DRIFT: {blob}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", HYBRID_SCIENTIFIC_SOURCE_COMMIT, "HEAD"],
        cwd=_repo_root(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("hybrid scientific source is not an ancestor of HEAD")


def _load_published_hybrid(root: Path) -> PublishedHybrid:
    root = Path(root)
    for name in ("RUN_IDENTITY.json", "DESIGN.json", "RESULT.json", "DECISION.json"):
        if not (root / name).is_file():
            raise RuntimeError(f"readout localization requires published hybrid baseline: missing {name}")
    identity = json.loads((root / "RUN_IDENTITY.json").read_text(encoding="utf-8"))
    design = json.loads((root / "DESIGN.json").read_text(encoding="utf-8"))
    result = json.loads((root / "RESULT.json").read_text(encoding="utf-8"))
    decision = json.loads((root / "DECISION.json").read_text(encoding="utf-8"))
    if identity.get("experiment") != "PCU-HYBRID-OBJECTIVE-001":
        raise RuntimeError("wrong hybrid baseline identity")
    if identity.get("formal_execution_not_started") is not True:
        raise RuntimeError("published hybrid baseline crossed formal boundary")
    if decision.get("status") != "HYBRID_OBJECTIVE_PRESERVES_ASSOCIATION_GENERATION_UNRESOLVED":
        raise RuntimeError("readout localization requires the published hybrid readout failure")
    if decision.get("valid_run") is not True or decision.get("formal_execution_not_started") is not True:
        raise RuntimeError("published hybrid result is not valid pre-formal evidence")
    if abs(float(decision.get("ce_weight", -1.0)) - float(CE_WEIGHT)) > 1e-12:
        raise RuntimeError("published hybrid CE weight changed")
    if design.get("causal_variable") != "ce_readout_regularizer_weight_only":
        raise RuntimeError("published hybrid design identity changed")
    selected = tuple(str(value) for value in result.get("selected_cells", ()))
    if len(selected) != TARGET_K or int(result.get("selected_k", -1)) != TARGET_K:
        raise RuntimeError("published hybrid result is not exact K64")
    if decision.get("selected_cells_exact_baseline_match") is not True:
        raise RuntimeError("published hybrid decision did not certify allocation identity")
    values = (
        (float(decision["ranking_train_accuracy"]), EXPECTED_HYBRID_RANKING_TRAIN, "ranking_train"),
        (float(decision["ranking_eval_accuracy"]), EXPECTED_HYBRID_RANKING_EVAL, "ranking_eval"),
        (float(decision["direct_accuracy"]), EXPECTED_HYBRID_DIRECT, "direct"),
    )
    for actual, expected, label in values:
        if abs(actual - expected) > 1e-12:
            raise RuntimeError(f"published hybrid {label} changed: expected {expected}, got {actual}")
    return PublishedHybrid(
        selected_cells=selected,
        dataset_manifest_sha256=str(result["dataset_manifest_sha256"]),
        ranking_train_accuracy=float(decision["ranking_train_accuracy"]),
        ranking_eval_accuracy=float(decision["ranking_eval_accuracy"]),
        direct_accuracy=float(decision["direct_accuracy"]),
        source=dict(identity.get("source", {})),
    )


def _token_text(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:
        return f"<token:{int(token_id)}>"


def gold_prefix_token_readout(
    model: nn.Module,
    tokenizer: Any,
    sequences: TaskSequences,
    *,
    device: str,
    batch_size: int = DIAGNOSTIC_BATCH_SIZE,
) -> dict[str, Any]:
    """Measure target-token rank under the exact correct answer prefix."""
    validate_answer_only_labels(sequences)
    rows: list[dict[str, Any]] = []
    for start in range(0, int(sequences.input_ids.shape[0]), int(batch_size)):
        end = min(int(sequences.input_ids.shape[0]), start + int(batch_size))
        input_ids = sequences.input_ids[start:end].to(device)
        attention = sequences.attention_mask[start:end].to(device)
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
            logits = getattr(output, "logits", output)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if not isinstance(logits, Tensor):
                raise RuntimeError("gold-prefix readout forward produced no logits tensor")
            log_probs = torch.log_softmax(logits.float(), dim=-1)

        for local, global_row in enumerate(range(start, end)):
            prompt_length = int(sequences.prompt_lengths[global_row])
            answer_length = int(sequences.answer_lengths[global_row])
            for answer_position in range(answer_length):
                target_position = prompt_length + answer_position
                logit_position = target_position - 1
                target_id = int(sequences.labels[global_row, target_position])
                token_logits = logits[local, logit_position].float()
                target_logit = token_logits[target_id]
                top1_id = int(token_logits.argmax())
                target_rank = 1 + int((token_logits > target_logit).sum())
                target_logprob = float(log_probs[local, logit_position, target_id])
                top1_logprob = float(log_probs[local, logit_position, top1_id])
                rows.append({
                    "sample_id": str(sequences.sample_ids[global_row]),
                    "answer_position": int(answer_position),
                    "answer_length": answer_length,
                    "target_id": target_id,
                    "target_text": _token_text(tokenizer, target_id),
                    "top1_id": top1_id,
                    "top1_text": _token_text(tokenizer, top1_id),
                    "target_rank": target_rank,
                    "target_top1": top1_id == target_id,
                    "target_logprob": target_logprob,
                    "top1_logprob": top1_logprob,
                    "target_vs_top1_logprob_margin": target_logprob - top1_logprob,
                })

    if not rows:
        raise RuntimeError("gold-prefix readout produced no answer-token rows")
    first = [row for row in rows if int(row["answer_position"]) == 0]
    later = [row for row in rows if int(row["answer_position"]) > 0]
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault(str(row["sample_id"]), []).append(row)
    sequence_all_top1 = sum(all(bool(row["target_top1"]) for row in values) for values in by_sample.values())

    def accuracy(values: Sequence[Mapping[str, Any]]) -> float:
        return sum(bool(value["target_top1"]) for value in values) / max(1, len(values))

    def mean_rank(values: Sequence[Mapping[str, Any]]) -> float:
        return sum(int(value["target_rank"]) for value in values) / max(1, len(values))

    return {
        "schema": "minicells.pcu-readout-localization-001.gold-prefix.v1",
        "sample_count": len(by_sample),
        "answer_token_count": len(rows),
        "first_token_count": len(first),
        "later_token_count": len(later),
        "first_token_top1_accuracy": accuracy(first),
        "later_token_top1_accuracy": accuracy(later),
        "all_token_top1_accuracy": accuracy(rows),
        "sequence_all_tokens_top1_accuracy": sequence_all_top1 / max(1, len(by_sample)),
        "first_token_mean_target_rank": mean_rank(first),
        "later_token_mean_target_rank": mean_rank(later),
        "all_token_mean_target_rank": mean_rank(rows),
        "rows": rows,
    }


def _greedy_suffix_batch(
    model: nn.Module,
    tokenizer: Any,
    prefixes: Sequence[Sequence[int]],
    *,
    steps: int,
    device: str,
) -> list[list[int]]:
    if steps <= 0:
        return [[] for _ in prefixes]
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0
    eos_id = getattr(tokenizer, "eos_token_id", None)
    width = max(len(value) for value in prefixes)
    input_ids = torch.full((len(prefixes), width), int(pad_id), dtype=torch.long, device=device)
    attention = torch.zeros((len(prefixes), width), dtype=torch.long, device=device)
    for row, values in enumerate(prefixes):
        values = [int(value) for value in values]
        input_ids[row, width - len(values):] = torch.tensor(values, dtype=torch.long, device=device)
        attention[row, width - len(values):] = 1
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention,
        "max_new_tokens": int(steps),
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": int(pad_id),
    }
    if eos_id is not None:
        kwargs["eos_token_id"] = int(eos_id)
    with torch.inference_mode():
        generated = model.generate(**kwargs)
    if not isinstance(generated, Tensor):
        generated = getattr(generated, "sequences", None)
    if not isinstance(generated, Tensor):
        raise RuntimeError("forced-prefix generate returned no token tensor")
    suffix = generated[:, width:]
    return [[int(value) for value in row.tolist()] for row in suffix]


def forced_prefix_recovery(
    model: nn.Module,
    tokenizer: Any,
    sequences: TaskSequences,
    *,
    device: str,
    batch_size: int = FORCED_PREFIX_BATCH_SIZE,
) -> dict[str, Any]:
    """Force k gold answer tokens and greedily test the exact remaining suffix."""
    cases: list[dict[str, Any]] = []
    for row in range(int(sequences.input_ids.shape[0])):
        prompt_length = int(sequences.prompt_lengths[row])
        answer_length = int(sequences.answer_lengths[row])
        prompt_ids = [int(value) for value in sequences.input_ids[row, :prompt_length].tolist()]
        answer_ids = [
            int(value)
            for value in sequences.labels[row, prompt_length:prompt_length + answer_length].tolist()
        ]
        cases.append({
            "sample_id": str(sequences.sample_ids[row]),
            "prompt_ids": prompt_ids,
            "answer_ids": answer_ids,
        })

    max_answer_length = max(len(case["answer_ids"]) for case in cases)
    curve: dict[str, dict[str, Any]] = {}
    detail_rows: list[dict[str, Any]] = []
    for forced in range(max_answer_length):
        eligible = [case for case in cases if len(case["answer_ids"]) > forced]
        exact_count = 0
        total = 0
        by_remaining: dict[int, list[dict[str, Any]]] = {}
        for case in eligible:
            remaining = len(case["answer_ids"]) - forced
            by_remaining.setdefault(remaining, []).append(case)
        for remaining, group in sorted(by_remaining.items()):
            for start in range(0, len(group), int(batch_size)):
                chunk = group[start:start + int(batch_size)]
                prefixes = [
                    case["prompt_ids"] + case["answer_ids"][:forced]
                    for case in chunk
                ]
                generated = _greedy_suffix_batch(
                    model,
                    tokenizer,
                    prefixes,
                    steps=remaining,
                    device=device,
                )
                for case, produced in zip(chunk, generated):
                    expected = case["answer_ids"][forced:]
                    produced = produced[:remaining]
                    exact = produced == expected
                    exact_count += int(exact)
                    total += 1
                    detail_rows.append({
                        "sample_id": case["sample_id"],
                        "forced_answer_tokens": int(forced),
                        "answer_length": len(case["answer_ids"]),
                        "remaining_tokens": remaining,
                        "expected_suffix_ids": expected,
                        "generated_suffix_ids": produced,
                        "expected_suffix_text": _token_text(tokenizer, expected[0]) if len(expected) == 1 else str(tokenizer.decode(expected, skip_special_tokens=False)),
                        "generated_suffix_text": str(tokenizer.decode(produced, skip_special_tokens=False)),
                        "exact": exact,
                    })
        curve[str(forced)] = {
            "forced_answer_tokens": int(forced),
            "eligible_samples": total,
            "exact_samples": exact_count,
            "suffix_exact_accuracy": exact_count / max(1, total),
        }

    minimal_forced = None
    for key in sorted(curve, key=int):
        row = curve[key]
        if int(row["forced_answer_tokens"]) == 0:
            continue
        if int(row["eligible_samples"]) > 0 and float(row["suffix_exact_accuracy"]) >= READOUT_FLOOR:
            minimal_forced = int(row["forced_answer_tokens"])
            break
    return {
        "schema": "minicells.pcu-readout-localization-001.forced-prefix.v1",
        "readout_floor": READOUT_FLOOR,
        "curve": curve,
        "minimal_forced_tokens_reaching_floor": minimal_forced,
        "rows": detail_rows,
    }


def _curve_accuracy(forced_prefix: Mapping[str, Any], forced: int) -> float | None:
    row = forced_prefix.get("curve", {}).get(str(int(forced)))
    if not isinstance(row, Mapping) or int(row.get("eligible_samples", 0)) <= 0:
        return None
    return float(row["suffix_exact_accuracy"])


def _classify(
    *,
    first_token_top1: float,
    later_token_top1: float,
    force1_suffix: float | None,
    force2_suffix: float | None,
) -> str:
    if later_token_top1 < READOUT_FLOOR:
        return "SINGLE_LAYER_GOLD_PREFIX_READOUT_INADEQUATE"
    if first_token_top1 < READOUT_FLOOR and force1_suffix is not None and force1_suffix >= READOUT_FLOOR:
        return "FIRST_TOKEN_READOUT_BOTTLENECK_SUPPORTED"
    if force2_suffix is not None and force2_suffix >= READOUT_FLOOR:
        return "EARLY_TOKEN_READOUT_BOTTLENECK_SUPPORTED"
    return "AUTOREGRESSIVE_TRAJECTORY_INSTABILITY_SUPPORTED"


def run_readout_localization_diagnostic(
    *,
    output: Path = DEFAULT_OUTPUT,
    hybrid_root: Path = HYBRID_BASELINE_ROOT,
    objective_root: Path = OBJECTIVE_BASELINE_ROOT,
    device: str = "cuda:0",
    seed: int = ENGINEERING_SEED,
) -> dict[str, Any]:
    if int(seed) != ENGINEERING_SEED:
        raise ValueError("PCU-READOUT-LOCALIZATION-001 is engineering-seed only")
    parsed = torch.device(str(device))
    if parsed.type != "cuda" or parsed.index is None:
        raise ValueError("readout localization requires an explicit CUDA device")
    if not torch.cuda.is_available() or parsed.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device unavailable: {device}")
    torch.cuda.set_device(parsed.index)
    _assert_hybrid_scientific_core_unchanged()

    published = _load_published_hybrid(Path(hybrid_root))
    baseline = _load_baselines(Path(objective_root))
    if baseline.selected_cells != published.selected_cells:
        raise RuntimeError("readout localization baseline Cell identity drifted")
    if baseline.dataset_manifest_sha256 != published.dataset_manifest_sha256:
        raise RuntimeError("readout localization baseline dataset identity drifted")

    source = git_provenance(_repo_root())
    if source.get("source_dirty") is not False:
        raise RuntimeError("readout localization requires a clean source tree")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    design = {
        "schema": "minicells.pcu-readout-localization-001.design.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "causal_variable": "none_observational_readout_localization",
        "training_changed": False,
        "replayed_training": {
            "experiment": "PCU-HYBRID-OBJECTIVE-001",
            "scientific_source_commit": HYBRID_SCIENTIFIC_SOURCE_COMMIT,
            "scientific_core_blob_sha": HYBRID_CORE_BLOB_SHA,
            "target_layer": TARGET_LAYER,
            "selected_k": TARGET_K,
            "selected_cells": list(published.selected_cells),
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "max_training_tokens": MAX_TRAINING_TOKENS,
            "effective_batch_size": BATCH_SIZE,
            "ranking_weight": 1.0,
            "ce_weight": float(CE_WEIGHT),
        },
        "diagnostics": {
            "gold_prefix_target_token_full_vocab_rank": True,
            "gold_prefix_target_token_top1": True,
            "forced_prefix_greedy_suffix_recovery": True,
            "tokenization": "original_answer_token_CE_encoding_matching_native_greedy_prompt_boundary",
            "readout_floor": READOUT_FLOOR,
        },
        "published_hybrid_baseline": published.to_dict(),
        "formal_execution_not_started": True,
        "scientific_evidence": False,
    }
    write_json(output / "DESIGN.json", design)
    write_json(output / "RUN_IDENTITY.json", {
        "schema": "minicells.pcu-readout-localization-001.run-identity.v1",
        "experiment": EXPERIMENT_ID,
        "phase": "engineering_diagnostic",
        "seed": ENGINEERING_SEED,
        "run_id": output.name,
        "source": source,
        "hybrid_scientific_source_commit": HYBRID_SCIENTIFIC_SOURCE_COMMIT,
        "hybrid_scientific_core_blob_sha": HYBRID_CORE_BLOB_SHA,
        "formal_execution_not_started": True,
    })

    tokenizer, model, manifest = load_granite(
        str(baseline.foundation["model_repo"]),
        revision=str(baseline.foundation["model_revision"]),
        device=str(device),
    )
    try:
        _validate_foundation_manifest(manifest, baseline.foundation)
        model.requires_grad_(False)
        block = target_module(model, baseline.target_path)
        projection = extract_expert_projections(block.experts, 0)
        cellular_experts = patch_moe_block(block, CellPartition(projection.intermediate_size, 4))
        model.requires_grad_(False)
        cellular_experts.requires_grad_(False)

        world = generate_world(ENGINEERING_SEED, count=128, tokenizer=tokenizer)
        audit = audit_dataset(world)
        if not audit.passed:
            raise RuntimeError(f"readout localization dataset audit failed: {audit.errors}")
        if world.manifest_sha256() != published.dataset_manifest_sha256:
            raise RuntimeError("readout localization dataset differs from published hybrid")
        train_samples = list(world.splits["A_train"])
        eval_samples = list(world.splits["A_eval"])
        candidate_universe = tuple(item.v for item in world.triples)

        config = BranchTrainingConfig(
            optimizer="AdamW",
            learning_rate=LEARNING_RATE,
            max_optimizer_steps=MAX_OPTIMIZER_STEPS,
            max_training_tokens=MAX_TRAINING_TOKENS,
            batch_size=BATCH_SIZE,
            seed=ENGINEERING_SEED,
        )
        print("[pcu-readout] replaying frozen hybrid L7/K64 training", flush=True)
        runtime, training = _train_hybrid_branch(
            model,
            block,
            cellular_experts,
            tokenizer,
            train_samples,
            published.selected_cells,
            device=str(device),
            config=config,
        )
        _assert_only_selected_deltas_trainable(model, runtime)
        if tuple(training["selected_cells"]) != published.selected_cells:
            raise RuntimeError("READOUT_LOCALIZATION_ALLOCATION_DRIFT")

        print("[pcu-readout] verifying exact published hybrid reproduction", flush=True)
        train_ranking = evaluate_candidate_ranking(
            model, tokenizer, train_samples, candidate_universe, device=str(device)
        )
        eval_ranking = evaluate_candidate_ranking(
            model, tokenizer, eval_samples, candidate_universe, device=str(device)
        )
        direct = evaluate_samples(
            model,
            tokenizer,
            eval_samples,
            split="A_eval",
            device=str(device),
            max_new_tokens=16,
            batch_size=16,
        )
        reproduction = {
            "ranking_train_accuracy": float(train_ranking.accuracy),
            "ranking_eval_accuracy": float(eval_ranking.accuracy),
            "direct_accuracy": float(direct.exact),
        }
        expected_reproduction = published.to_dict()
        for key, expected in expected_reproduction.items():
            actual = reproduction[key]
            if abs(actual - float(expected)) > 1e-12:
                raise RuntimeError(
                    f"HYBRID_REPRODUCTION_MISMATCH {key}: expected {expected}, got {actual}"
                )

        eval_sequences = build_task_sequences(tokenizer, eval_samples, "A_eval", max_length=128)
        validate_answer_only_labels(eval_sequences)
        print("[pcu-readout] measuring gold-prefix target-token ranks", flush=True)
        gold = gold_prefix_token_readout(
            model, tokenizer, eval_sequences, device=str(device), batch_size=DIAGNOSTIC_BATCH_SIZE
        )
        print("[pcu-readout] measuring forced-prefix greedy suffix recovery", flush=True)
        forced = forced_prefix_recovery(
            model, tokenizer, eval_sequences, device=str(device), batch_size=FORCED_PREFIX_BATCH_SIZE
        )
        force1 = _curve_accuracy(forced, 1)
        force2 = _curve_accuracy(forced, 2)
        status = _classify(
            first_token_top1=float(gold["first_token_top1_accuracy"]),
            later_token_top1=float(gold["later_token_top1_accuracy"]),
            force1_suffix=force1,
            force2_suffix=force2,
        )
        result = {
            "schema": "minicells.pcu-readout-localization-001.result.v1",
            "experiment": EXPERIMENT_ID,
            "phase": "engineering_diagnostic",
            "status": status,
            "valid_run": True,
            "scientific_evidence": False,
            "formal_execution_not_started": True,
            "source": source,
            "foundation": dict(manifest),
            "dataset_manifest_sha256": world.manifest_sha256(),
            "selected_k": TARGET_K,
            "selected_cells": list(published.selected_cells),
            "training_changed": False,
            "replayed_training": training,
            "published_hybrid_baseline": published.to_dict(),
            "hybrid_reproduction": reproduction,
            "hybrid_reproduction_exact": True,
            "gold_prefix": gold,
            "forced_prefix": forced,
            "readout_floor": READOUT_FLOOR,
        }
        write_json(output / "RESULT.json", result)
        decision = {
            "schema": "minicells.pcu-readout-localization-001.decision.v1",
            "experiment": EXPERIMENT_ID,
            "phase": "engineering_diagnostic",
            "status": status,
            "valid_run": True,
            "scientific_evidence": False,
            "formal_execution_not_started": True,
            "training_changed": False,
            "hybrid_reproduction_exact": True,
            "first_token_top1_accuracy": gold["first_token_top1_accuracy"],
            "later_token_top1_accuracy": gold["later_token_top1_accuracy"],
            "all_token_top1_accuracy": gold["all_token_top1_accuracy"],
            "sequence_all_tokens_top1_accuracy": gold["sequence_all_tokens_top1_accuracy"],
            "first_token_mean_target_rank": gold["first_token_mean_target_rank"],
            "later_token_mean_target_rank": gold["later_token_mean_target_rank"],
            "force0_suffix_exact_accuracy": _curve_accuracy(forced, 0),
            "force1_suffix_exact_accuracy": force1,
            "force2_suffix_exact_accuracy": force2,
            "minimal_forced_tokens_reaching_floor": forced["minimal_forced_tokens_reaching_floor"],
            "readout_floor": READOUT_FLOOR,
            "selected_k": TARGET_K,
            "selected_cells_exact_hybrid_match": True,
            "source": source,
        }
        write_json(output / "DECISION.json", decision)
        return result
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except torch.AcceleratorError:
                pass


__all__ = [
    "EXPERIMENT_ID",
    "HYBRID_BASELINE_ROOT",
    "OBJECTIVE_BASELINE_ROOT",
    "DEFAULT_OUTPUT",
    "HYBRID_SCIENTIFIC_SOURCE_COMMIT",
    "HYBRID_CORE_BLOB_SHA",
    "READOUT_FLOOR",
    "gold_prefix_token_readout",
    "forced_prefix_recovery",
    "run_readout_localization_diagnostic",
]
