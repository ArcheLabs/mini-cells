from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

import sequence as seq
from minicells.granite_moe_layout import identify_packed_expert_tensors
from minicells.moe_multicoordinate import (
    capture_coordinate_set,
    restore_coordinate_set_,
    save_mutation_set,
)
from minicells.moe_subexpert import group_delta, validate_group_shapes

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "research" / "validations" / "jam-knowledge-mutation-001" / "protocol.json"
HC_PROTOCOL_PATH = ROOT / "research" / "validations" / "history-compression-001" / "protocol.json"
DATASET_ROOT = ROOT / "research" / "datasets" / "jam-knowledge-v0.1"
DATASET_BUILDER = ROOT / "scripts" / "research" / "jam_knowledge_v0_1" / "build_dataset.py"
ORACLE_ENGINE_PATH = ROOT / "scripts" / "research" / "functional_boundary_oracle_001" / "run_seed.py"
RESULTS_ROOT = ROOT / "results" / "jam-knowledge-mutation-001"
WORK_ROOT = ROOT / "results" / "jam-knowledge-mutation-001-work"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_oracle_engine() -> ModuleType:
    spec = importlib.util.spec_from_file_location("minicells_jam_mutation_oracle", ORACLE_ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Functional Boundary Oracle 001 engine")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _quiet_libraries(huggingface_hub: Any, transformers: Any) -> None:
    try:
        huggingface_hub.utils.disable_progress_bars()
    except Exception:
        pass
    try:
        transformers.logging.disable_progress_bar()
        transformers.logging.set_verbosity_error()
    except Exception:
        pass


def _progress(seed: int, message: str, *, capacity: int | None = None) -> None:
    prefix = f"[jam001][seed={seed}]"
    if capacity is not None:
        prefix += f"[capacity={capacity}]"
    print(f"{prefix} {message}", flush=True)


def _validate_dataset(protocol: dict[str, Any]) -> dict[str, Any]:
    manifest_path = DATASET_ROOT / "manifest.json"
    manifest = _load_json(manifest_path)
    expected = protocol["dataset"]
    if manifest.get("dataset") != expected["id"]:
        raise RuntimeError("JAM dataset id mismatch")
    if manifest.get("status") != expected["required_manifest_status"]:
        raise RuntimeError("JAM dataset status mismatch")
    if manifest.get("source_pin") != expected["source_pin"]:
        raise RuntimeError("JAM source pin mismatch")
    if int(manifest.get("concept_count", -1)) != int(expected["concept_count"]):
        raise RuntimeError("JAM concept count mismatch")
    mismatches: list[str] = []
    for relative, record in manifest["canonical_files"].items():
        path = DATASET_ROOT / relative
        if not path.is_file() or _sha256(path) != record["sha256"]:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"JAM canonical file identity mismatch: {mismatches}")
    subprocess.run([sys.executable, str(DATASET_BUILDER)], cwd=ROOT, check=True, capture_output=True, text=True)
    generated = DATASET_ROOT / "generated"
    rows = {
        "train": _load_jsonl(generated / "train.jsonl"),
        "validation": _load_jsonl(generated / "validation.jsonl"),
        "factual": _load_jsonl(generated / "evaluation" / "factual.jsonl"),
        "relational": _load_jsonl(generated / "evaluation" / "relational.jsonl"),
        "misconceptions": _load_jsonl(generated / "evaluation" / "misconceptions.jsonl"),
        "reasoning": _load_jsonl(generated / "evaluation" / "reasoning.jsonl"),
    }
    counts = {key: len(value) for key, value in rows.items()}
    if counts != {key: int(value) for key, value in expected["generated_counts"].items()}:
        raise RuntimeError(f"JAM generated split count mismatch: {counts}")
    return {"manifest": manifest, "manifest_sha256": _sha256(manifest_path), "rows": rows}


def _selection_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: hashlib.sha256(str(row["id"]).encode()).hexdigest())
    return ordered[:count]


def _last_logits_grad(model: Any, tokenizer: Any, prompts: list[str], device: str) -> torch.Tensor:
    batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
    batch = {key: value.to(device) for key, value in batch.items()}
    output = model(**batch, use_cache=False)
    positions = batch["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(len(prompts), device=device)
    return output.logits[rows, positions].float()


def _history_kl(current: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(current, dim=-1),
        F.softmax(teacher.detach(), dim=-1),
        reduction="batchmean",
    )


def _rank_coordinates(
    new_energy: torch.Tensor,
    history_energy: torch.Tensor,
    new_coverage: torch.Tensor,
    history_coverage: torch.Tensor,
    protocol: dict[str, Any],
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    minimum_coverage = float(protocol["coordinate_selection"]["minimum_jam_route_coverage"])
    candidates: list[dict[str, Any]] = []
    experts, groups = new_energy.shape
    for expert in range(experts):
        for group in range(groups):
            new_rms = math.sqrt(max(float(new_energy[expert, group].item()), 0.0))
            history_rms = math.sqrt(max(float(history_energy[expert, group].item()), 0.0))
            route_specificity = float(new_coverage[expert].item() - history_coverage[expert].item())
            score = 0.5 * math.log((new_rms + 1e-12) / (history_rms + 1e-12)) + route_specificity
            candidates.append(
                {
                    "expert_index": expert,
                    "group_index": group,
                    "jam_group_rms": new_rms,
                    "history_group_rms": history_rms,
                    "jam_route_coverage": float(new_coverage[expert].item()),
                    "history_route_coverage": float(history_coverage[expert].item()),
                    "route_specificity": route_specificity,
                    "score": score,
                    "eligible": float(new_coverage[expert].item()) >= minimum_coverage,
                }
            )
    candidates.sort(key=lambda row: (-float(row["score"]), int(row["expert_index"]), int(row["group_index"])))
    selected: list[tuple[int, int]] = []
    used_experts: set[int] = set()
    maximum = int(protocol["mutation"]["maximum_coordinate_count"])
    for row in candidates:
        expert = int(row["expert_index"])
        if not row["eligible"] or expert in used_experts:
            continue
        selected.append((expert, int(row["group_index"])))
        used_experts.add(expert)
        if len(selected) == maximum:
            break
    if len(selected) != maximum:
        raise RuntimeError(f"only {len(selected)} eligible unique-expert coordinates; need {maximum}")
    return selected, candidates


def _target_records(
    coordinates: list[tuple[int, int]],
    *,
    layer_index: int,
    group_size: int,
    intermediate: int,
    gate_up: tuple[str, torch.nn.Parameter],
    down: tuple[str, torch.nn.Parameter],
    canonical_gate_up: str,
    canonical_down: str,
) -> list[dict[str, Any]]:
    return [
        {
            "layer_index": layer_index,
            "expert_index": expert,
            "group_index": group,
            "group_size": group_size,
            "intermediate_size": intermediate,
            "gate_up_name": gate_up[0],
            "down_name": down[0],
            "gate_up_canonical_name": canonical_gate_up,
            "down_canonical_name": canonical_down,
        }
        for expert, group in coordinates
    ]


def _metrics_gain(base: dict[str, float], mutated: dict[str, float]) -> float:
    return float(base["mean_reference_nll"] - mutated["mean_reference_nll"])


def _weighted_heldout_gain(
    base: dict[str, dict[str, float]], mutated: dict[str, dict[str, float]]
) -> float:
    total_tokens = sum(float(base[name]["supervised_tokens"]) for name in base)
    base_nll = sum(
        float(base[name]["mean_reference_nll"]) * float(base[name]["supervised_tokens"])
        for name in base
    ) / total_tokens
    mutated_nll = sum(
        float(mutated[name]["mean_reference_nll"]) * float(mutated[name]["supervised_tokens"])
        for name in mutated
    ) / total_tokens
    return base_nll - mutated_nll


def _train_capacity(
    *,
    model: Any,
    tokenizer: Any,
    protocol: dict[str, Any],
    seed: int,
    capacity: int,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    base_validation: dict[str, float],
    history_prompts: list[str],
    history_teacher: torch.Tensor,
    gate_up: tuple[str, torch.nn.Parameter],
    down: tuple[str, torch.nn.Parameter],
    coordinates: list[tuple[int, int]],
    targets: list[dict[str, Any]],
    device: str,
) -> tuple[list[dict[str, torch.Tensor]] | None, list[dict[str, Any]]]:
    training = protocol["training"]
    task = protocol["sequence_task"]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gate_up[1].requires_grad_(True)
    down[1].requires_grad_(True)

    rng = random.Random(seed + capacity * 100003)
    order = list(range(len(train_rows)))
    history_order = list(range(len(history_prompts)))
    rng.shuffle(order)
    rng.shuffle(history_order)
    cursor = 0
    history_cursor = 0
    batch_size = int(training["batch_size"])
    history_batch_size = int(training["history_batch_size"])
    max_steps = int(training["max_steps_per_capacity"])
    eval_interval = int(training["candidate_eval_interval"])
    lr = float(training["learning_rate"])
    max_norm = float(training["max_selected_gradient_norm"])
    kl_weight = float(training["history_kl_weight"])
    kl_limit = float(training["maximum_history_selection_kl_for_candidate"])
    best: list[dict[str, torch.Tensor]] | None = None
    best_gain = -math.inf
    best_step = 10**9
    log: list[dict[str, Any]] = []

    def take(indices: list[int], size: int, position: int) -> tuple[list[int], int]:
        if position + size > len(indices):
            rng.shuffle(indices)
            position = 0
        batch_indices = indices[position : position + size]
        return batch_indices, position + size

    for step in range(1, max_steps + 1):
        batch_indices, cursor = take(order, batch_size, cursor)
        hist_indices, history_cursor = take(history_order, history_batch_size, history_cursor)
        batch_rows = [train_rows[index] for index in batch_indices]
        hist_prompts = [history_prompts[index] for index in hist_indices]
        hist_teacher = history_teacher[hist_indices].to(device)

        model.zero_grad(set_to_none=True)
        target_loss = seq.answer_loss(
            model,
            tokenizer,
            batch_rows,
            prompt_template=task["prompt_template"],
            max_length=int(task["max_sequence_tokens"]),
            device=device,
        )
        current_history = _last_logits_grad(model, tokenizer, hist_prompts, device)
        history_kl = _history_kl(current_history, hist_teacher)
        loss = target_loss + kl_weight * history_kl
        loss.backward()
        if gate_up[1].grad is None or down[1].grad is None:
            raise RuntimeError("selected packed tensors did not receive gradients")
        norm = seq.selected_gradient_norm(
            gate_up[1].grad,
            down[1].grad,
            coordinates,
            group_size=int(protocol["mutation"]["group_size"]),
        )
        scale = min(1.0, max_norm / max(norm, 1e-12))
        seq.apply_selected_gradients_(
            gate_up[1],
            down[1],
            coordinates,
            group_size=int(protocol["mutation"]["group_size"]),
            learning_rate=lr,
            grad_scale=scale,
        )
        model.zero_grad(set_to_none=True)
        record: dict[str, Any] = {
            "step": step,
            "target_batch_loss": float(target_loss.detach().item()),
            "history_batch_kl": float(history_kl.detach().item()),
            "selected_gradient_norm": norm,
            "grad_scale": scale,
            "train_batch_indices": batch_indices,
            "history_batch_indices": hist_indices,
        }
        if step % eval_interval == 0:
            validation = seq.evaluate_rows(
                model,
                tokenizer,
                validation_rows,
                prompt_template=task["prompt_template"],
                max_length=int(task["max_sequence_tokens"]),
                device=device,
                batch_size=batch_size,
            )
            with torch.no_grad():
                current_all_history = _last_logits_grad(model, tokenizer, history_prompts, device)
            selection_kl = float(
                F.kl_div(
                    F.log_softmax(current_all_history, dim=-1),
                    F.softmax(history_teacher.to(device), dim=-1),
                    reduction="batchmean",
                ).item()
            )
            gain = _metrics_gain(base_validation, validation)
            record["candidate_validation_reference_nll_gain"] = gain
            record["candidate_history_selection_kl"] = selection_kl
            if selection_kl <= kl_limit and gain > 0.0 and (
                gain > best_gain or (gain == best_gain and step < best_step)
            ):
                best = capture_coordinate_set(dict(model.named_parameters()), targets)
                best_gain = gain
                best_step = step
                record["selected_as_best_safe_candidate"] = True
        log.append(record)
    return best, log


def _run_capacity(
    *,
    oracle: ModuleType,
    protocol: dict[str, Any],
    protocol_sha: str,
    seed: int,
    capacity: int,
    model: Any,
    tokenizer: Any,
    bundle_manifest: dict[str, Any],
    rows: dict[str, list[dict[str, Any]]],
    base_validation: dict[str, float],
    base_heldout: dict[str, dict[str, float]],
    history_selection: list[str],
    history_teacher: torch.Tensor,
    history_eval: list[str],
    base_history_eval: torch.Tensor,
    verification_prompts: list[str],
    base_verification: torch.Tensor,
    base_repeatability: float,
    gate_up: tuple[str, torch.nn.Parameter],
    down: tuple[str, torch.nn.Parameter],
    intermediate: int,
    canonical_gate_up: str,
    canonical_down: str,
    ranked: list[tuple[int, int]],
    device: str,
    result_root: Path,
    dataset_identity: dict[str, Any],
) -> dict[str, Any]:
    task = protocol["sequence_task"]
    training = protocol["training"]
    layer_index = int(protocol["mutation"]["layer_index"])
    group_size = int(protocol["mutation"]["group_size"])
    coordinates = ranked[:capacity]
    targets = _target_records(
        coordinates,
        layer_index=layer_index,
        group_size=group_size,
        intermediate=intermediate,
        gate_up=gate_up,
        down=down,
        canonical_gate_up=canonical_gate_up,
        canonical_down=canonical_down,
    )
    parameter_map = dict(model.named_parameters())
    originals = capture_coordinate_set(parameter_map, targets)
    capacity_dir = result_root / f"capacity-{capacity}"
    shutil.rmtree(capacity_dir, ignore_errors=True)
    capacity_dir.mkdir(parents=True, exist_ok=True)
    _progress(seed, f"training coordinates={coordinates}", capacity=capacity)

    best, training_log = _train_capacity(
        model=model,
        tokenizer=tokenizer,
        protocol=protocol,
        seed=seed,
        capacity=capacity,
        train_rows=rows["train"],
        validation_rows=rows["validation"],
        base_validation=base_validation,
        history_prompts=history_selection,
        history_teacher=history_teacher,
        gate_up=gate_up,
        down=down,
        coordinates=coordinates,
        targets=targets,
        device=device,
    )
    _write_jsonl(capacity_dir / "training.jsonl", training_log)
    restore_coordinate_set_(parameter_map, targets, best if best is not None else originals)

    mutated_validation = seq.evaluate_rows(
        model,
        tokenizer,
        rows["validation"],
        prompt_template=task["prompt_template"],
        max_length=int(task["max_sequence_tokens"]),
        device=device,
        batch_size=int(training["batch_size"]),
    )
    mutated_heldout = {
        name: seq.evaluate_rows(
            model,
            tokenizer,
            rows[name],
            prompt_template=task["prompt_template"],
            max_length=int(task["max_sequence_tokens"]),
            device=device,
            batch_size=int(training["batch_size"]),
        )
        for name in ("factual", "relational", "misconceptions", "reasoning")
    }
    mutated_history_eval = oracle._next_logits(
        model,
        tokenizer,
        history_eval,
        device=device,
        batch_size=int(training["batch_size"]),
    )
    mutated_history_selection = oracle._next_logits(
        model,
        tokenizer,
        history_selection,
        device=device,
        batch_size=int(training["batch_size"]),
    )
    history_selection_kl = oracle._kl(history_teacher, mutated_history_selection)
    history_eval_kl = oracle._kl(base_history_eval, mutated_history_eval)
    history_eval_top1 = oracle._top1_identity(base_history_eval, mutated_history_eval)

    current = capture_coordinate_set(parameter_map, targets)
    coordinate_payloads: list[dict[str, Any]] = []
    delta_sq = 0.0
    for target, current_group, original_group in zip(targets, current, originals, strict=True):
        deltas = group_delta(current_group, original_group)
        delta_sq += sum(float(value.float().square().sum().item()) for value in deltas.values())
        coordinate_payloads.append({"target": target, "deltas": deltas})
    mutation_manifest = save_mutation_set(
        capacity_dir / "mutation",
        base_manifest_identity=bundle_manifest["identity_sha256"],
        source_model_id=protocol["base"]["model_id"],
        source_revision=protocol["base"]["revision"],
        coordinates=coordinate_payloads,
        metadata={
            "experiment": protocol["experiment"],
            "protocol_sha256": protocol_sha,
            "seed": seed,
            "capacity": capacity,
            "dataset_manifest_sha256": dataset_identity["manifest_sha256"],
        },
        require_unique_experts=True,
    )

    restore_coordinate_set_(parameter_map, targets, originals)
    rolled = capture_coordinate_set(parameter_map, targets)
    exact_weight_rollback = all(
        torch.equal(before[key], after[key])
        for before, after in zip(originals, rolled, strict=True)
        for key in before
    )
    rollback_logits = oracle._next_logits(
        model,
        tokenizer,
        verification_prompts,
        device=device,
        batch_size=int(training["batch_size"]),
    )
    rollback_error = float((rollback_logits - base_verification).abs().max().item())
    rollback_excess = max(0.0, rollback_error - base_repeatability)

    gains = {
        "validation": _metrics_gain(base_validation, mutated_validation),
        **{
            name: _metrics_gain(base_heldout[name], mutated_heldout[name])
            for name in mutated_heldout
        },
    }
    overall_gain = _weighted_heldout_gain(base_heldout, mutated_heldout)
    thresholds = protocol["gates"]
    preliminary_gates = {
        "conversion_identity": bundle_manifest["identity_sha256"] == protocol["base"]["conversion_manifest_identity_sha256"],
        "dataset_identity": dataset_identity["manifest"]["status"] == protocol["dataset"]["required_manifest_status"],
        "history_set_disjointness": not bool(set(history_selection) & set(history_eval)),
        "validation_reference_nll_gain": gains["validation"] >= float(thresholds["minimum_validation_reference_nll_gain"]),
        "factual_reference_nll_gain": gains["factual"] >= float(thresholds["minimum_factual_reference_nll_gain"]),
        "relational_reference_nll_gain": gains["relational"] >= float(thresholds["minimum_relational_reference_nll_gain"]),
        "misconception_reference_nll_gain": gains["misconceptions"] >= float(thresholds["minimum_misconception_reference_nll_gain"]),
        "reasoning_reference_nll_gain": gains["reasoning"] >= float(thresholds["minimum_reasoning_reference_nll_gain"]),
        "overall_heldout_reference_nll_gain": overall_gain >= float(thresholds["minimum_overall_heldout_reference_nll_gain"]),
        "history_evaluation_mean_kl": history_eval_kl <= float(thresholds["maximum_history_evaluation_mean_kl"]),
        "history_evaluation_top1_identity": history_eval_top1 >= float(thresholds["minimum_history_evaluation_top1_identity"]),
        "coordinate_count": capacity <= int(thresholds["maximum_coordinate_count"]),
        "per_expert_fraction": float(group_size / intermediate) <= float(thresholds["maximum_per_expert_fraction"]),
        "unique_experts": len({expert for expert, _group in coordinates}) == len(coordinates),
        "nonzero_delta": math.sqrt(delta_sq) > 0.0,
        "exact_weight_rollback": exact_weight_rollback,
        "forward_rollback_within_repeatability": rollback_excess <= float(thresholds["maximum_forward_rollback_excess_over_base_repeatability"]),
    }
    formal_pending = {
        "target_router_topk_identity": None,
        "artifact_reapply_logit_error": None,
        "materialized_checkpoint_logit_error": None,
    }
    preliminary_status = "PASS" if all(preliminary_gates.values()) else "FAIL"
    result = {
        "experiment": protocol["experiment"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_sha,
        "seed": seed,
        "formal_seed": seed in [int(value) for value in protocol["formal_seeds"]],
        "capacity": capacity,
        "status": "PENDING_FORMAL_VERIFICATION",
        "preliminary_status": preliminary_status,
        "base": {
            "model_id": protocol["base"]["model_id"],
            "revision": protocol["base"]["revision"],
            "conversion_manifest_identity_sha256": bundle_manifest["identity_sha256"],
        },
        "dataset": {
            "id": protocol["dataset"]["id"],
            "manifest_sha256": dataset_identity["manifest_sha256"],
            "source_pin": dataset_identity["manifest"]["source_pin"],
        },
        "selection": {
            "layer_index": layer_index,
            "group_size": group_size,
            "coordinates": [list(value) for value in coordinates],
        },
        "metrics": {
            "validation_reference_nll_gain": gains["validation"],
            "factual_reference_nll_gain": gains["factual"],
            "relational_reference_nll_gain": gains["relational"],
            "misconception_reference_nll_gain": gains["misconceptions"],
            "reasoning_reference_nll_gain": gains["reasoning"],
            "overall_heldout_reference_nll_gain": overall_gain,
            "history_selection_mean_kl": history_selection_kl,
            "history_evaluation_mean_kl": history_eval_kl,
            "history_evaluation_top1_identity": history_eval_top1,
            "delta_l2_norm": math.sqrt(delta_sq),
            "exact_weight_rollback": exact_weight_rollback,
            "base_forward_repeatability_max_abs": base_repeatability,
            "rollback_max_abs_logit_error": rollback_error,
            "rollback_excess_over_base_repeatability": rollback_excess,
        },
        "evaluation": {
            "base_validation": base_validation,
            "mutated_validation": mutated_validation,
            "base_heldout": base_heldout,
            "mutated_heldout": mutated_heldout,
        },
        "mutation": {
            "schema_version": mutation_manifest["schema_version"],
            "identity_sha256": mutation_manifest["identity_sha256"],
            "coordinate_count": mutation_manifest["coordinate_count"],
            "path": "mutation",
        },
        "training": {
            "max_steps": int(training["max_steps_per_capacity"]),
            "training_log_file": "training.jsonl",
            "learner_visible_history_prompts": len(history_selection),
        },
        "gates": {**preliminary_gates, **formal_pending},
    }
    _write_json(capacity_dir / "evaluation.json", result["evaluation"])
    _write_json(capacity_dir / "result.json", result)
    _progress(
        seed,
        f"preliminary={preliminary_status} validation_gain={gains['validation']:.4f} heldout_gain={overall_gain:.4f} history_kl={history_eval_kl:.6f}",
        capacity=capacity,
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _load_json(PROTOCOL_PATH)
    protocol_sha = _sha256(PROTOCOL_PATH)
    formal_seeds = [int(value) for value in protocol["formal_seeds"]]
    if args.seed not in formal_seeds and not args.allow_nonformal_seed:
        raise SystemExit(f"seed {args.seed} is not formal; allowed={formal_seeds}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    dataset_identity = _validate_dataset(protocol)
    rows = dataset_identity["rows"]
    oracle = _load_oracle_engine()
    (
        huggingface_hub,
        safetensors,
        transformers,
        snapshot_download,
        AutoModelForCausalLM,
        AutoTokenizer,
    ) = oracle._require_lm_dependencies()
    _quiet_libraries(huggingface_hub, transformers)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    work = (args.work_dir or WORK_ROOT / f"seed-{args.seed}").resolve()
    result_root = (args.result_dir or RESULTS_ROOT / f"seed-{args.seed}").resolve()
    if args.clean:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(result_root, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    base = protocol["base"]
    _progress(args.seed, "loading frozen Granite substrate and JAM dataset")
    source_dir = Path(snapshot_download(repo_id=base["model_id"], revision=base["revision"])).resolve()
    bundle_dir = work / "clm-bundle"
    bundle_manifest = oracle.create_clm_moe_bundle(
        source_dir,
        bundle_dir,
        source_model_id=base["model_id"],
        source_revision=base["revision"],
        copy_mode="hardlink",
    )
    if bundle_manifest["identity_sha256"] != base["conversion_manifest_identity_sha256"]:
        raise RuntimeError("Conversion 001 identity mismatch")
    tokenizer = AutoTokenizer.from_pretrained(bundle_dir / "substrate")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        bundle_dir / "substrate", dtype=torch.float32, low_cpu_mem_usage=True
    ).to(args.device)
    model.eval()

    history_protocol = _load_json(HC_PROTOCOL_PATH)
    history_selection = list(history_protocol["history"]["selection_prompts"])
    history_eval = list(history_protocol["history"]["evaluation_prompts"])
    if len(history_selection) != 32 or len(history_eval) != 32 or set(history_selection) & set(history_eval):
        raise RuntimeError("registered full_32 history prompt contract changed")

    task = protocol["sequence_task"]
    training = protocol["training"]
    batch_size = int(training["batch_size"])
    selection_rows = _selection_rows(rows["train"], int(protocol["dataset"]["selection_examples"]))
    selection_prompts = [seq.prompt_for(row, task["prompt_template"]) for row in selection_rows]
    verification_prompts = selection_prompts[:16] + history_eval

    _progress(args.seed, "caching frozen-base metrics and writable-coordinate geometry")
    base_validation = seq.evaluate_rows(
        model, tokenizer, rows["validation"], prompt_template=task["prompt_template"],
        max_length=int(task["max_sequence_tokens"]), device=args.device, batch_size=batch_size
    )
    base_heldout = {
        name: seq.evaluate_rows(
            model, tokenizer, rows[name], prompt_template=task["prompt_template"],
            max_length=int(task["max_sequence_tokens"]), device=args.device, batch_size=batch_size
        )
        for name in ("factual", "relational", "misconceptions", "reasoning")
    }
    history_teacher = oracle._next_logits(
        model, tokenizer, history_selection, device=args.device, batch_size=batch_size
    )
    base_history_eval = oracle._next_logits(
        model, tokenizer, history_eval, device=args.device, batch_size=batch_size
    )
    base_verification = oracle._next_logits(
        model, tokenizer, verification_prompts, device=args.device, batch_size=batch_size
    )
    repeat = oracle._next_logits(
        model, tokenizer, verification_prompts, device=args.device, batch_size=batch_size
    )
    base_repeatability = float((base_verification - repeat).abs().max().item())

    layer_index = int(protocol["mutation"]["layer_index"])
    group_size = int(protocol["mutation"]["group_size"])
    gate_up, down = identify_packed_expert_tensors(model, layer_index)
    intermediate = validate_group_shapes(gate_up[1], down[1])
    if intermediate != int(protocol["mutation"]["expected_intermediate_size"]):
        raise RuntimeError("unexpected Granite expert intermediate width")
    canonical_gate_up, canonical_down = oracle._canonical_mapping(
        bundle_manifest, gate_up, down, layer_index
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gate_up[1].requires_grad_(True)
    down[1].requires_grad_(True)

    new_energy = seq.coordinate_gradient_energy(
        model, tokenizer, selection_rows, prompt_template=task["prompt_template"],
        max_length=int(task["max_sequence_tokens"]), device=args.device, batch_size=batch_size,
        gate_up=gate_up, down=down, group_size=group_size
    )
    history_targets = history_teacher.argmax(dim=-1).cpu()
    history_energy = oracle._gradient_energy(
        model, tokenizer, history_selection, history_targets, device=args.device,
        batch_size=batch_size, gate_up=gate_up, down=down, group_size=group_size
    )
    top_k = int(bundle_manifest["config"]["num_experts_per_tok"])
    new_router = oracle._router_last_logits(
        model, tokenizer, selection_prompts, device=args.device, batch_size=batch_size,
        layer_index=layer_index
    )
    history_router = oracle._router_last_logits(
        model, tokenizer, history_selection, device=args.device, batch_size=batch_size,
        layer_index=layer_index
    )
    ranked, coordinate_scores = _rank_coordinates(
        new_energy, history_energy, oracle._router_coverage(new_router, top_k),
        oracle._router_coverage(history_router, top_k), protocol
    )
    _write_json(result_root / "coordinate_scores.json", coordinate_scores)
    _progress(args.seed, f"ranked writable prefix={ranked}")

    results: dict[str, Any] = {}
    for capacity in [int(value) for value in protocol["mutation"]["capacity_ladder"]]:
        results[str(capacity)] = _run_capacity(
            oracle=oracle, protocol=protocol, protocol_sha=protocol_sha, seed=args.seed,
            capacity=capacity, model=model, tokenizer=tokenizer, bundle_manifest=bundle_manifest,
            rows=rows, base_validation=base_validation, base_heldout=base_heldout,
            history_selection=history_selection, history_teacher=history_teacher,
            history_eval=history_eval, base_history_eval=base_history_eval,
            verification_prompts=verification_prompts, base_verification=base_verification,
            base_repeatability=base_repeatability, gate_up=gate_up, down=down,
            intermediate=intermediate, canonical_gate_up=canonical_gate_up,
            canonical_down=canonical_down, ranked=ranked, device=args.device,
            result_root=result_root, dataset_identity=dataset_identity
        )

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "safetensors": safetensors.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": args.device,
        "dtype": "torch.float32",
    }
    summary = {
        "experiment": protocol["experiment"],
        "protocol_sha256": protocol_sha,
        "seed": args.seed,
        "formal_seed": args.seed in formal_seeds,
        "dataset_manifest_sha256": dataset_identity["manifest_sha256"],
        "ranked_coordinates": [list(value) for value in ranked],
        "environment": environment,
        "capacities": {
            key: {
                "status": value["status"],
                "preliminary_status": value["preliminary_status"],
                "overall_heldout_reference_nll_gain": value["metrics"]["overall_heldout_reference_nll_gain"],
                "history_evaluation_mean_kl": value["metrics"]["history_evaluation_mean_kl"],
            }
            for key, value in results.items()
        },
    }
    _write_json(result_root / "seed_summary.json", summary)
    _progress(args.seed, "engine complete; fresh-base artifact/router/materialization verification remains")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--allow-nonformal-seed", action="store_true")
    parser.add_argument("--fail-on-scientific-fail", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
