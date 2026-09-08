from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import os
import random
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

from minicells.functional_cellization import (
    FunctionalCellOverlay,
    disjoint_mutations,
    freeze_foundation_,
    mask_cell_gradients_,
)

ROOT = Path(__file__).resolve().parents[3]
SEQUENCE_ROOT = ROOT / "scripts" / "research" / "jam_knowledge_mutation_001"
if str(SEQUENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SEQUENCE_ROOT))
import sequence as seq  # noqa: E402

from dataset import (  # noqa: E402
    ENTITIES,
    PROTOCOLS,
    calibration_prompts,
    contextual_conflict_rows,
    entity_facts,
    formation_evaluation,
    formation_validation,
    rewrite_rows,
    training_rows,
)

PROTOCOL_PATH = (
    ROOT / "research" / "validations" / "clm-conversion-kill-test-001" / "protocol.json"
)
DATASET_PATH = ROOT / "scripts" / "research" / "clm_conversion_kill_test_001" / "dataset.py"
RESULTS_ROOT = ROOT / "results" / "clm-conversion-kill-test-001"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _progress(seed: int, message: str) -> None:
    print(f"[conversion001][seed={seed}] {message}", flush=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prompt_batch(tokenizer: Any, prompts: Sequence[str], device: str) -> dict[str, torch.Tensor]:
    batch = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )
    return {name: value.to(device) for name, value in batch.items()}


def _last_logits(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    device: str,
    overlay: FunctionalCellOverlay | None = None,
) -> torch.Tensor:
    batch = _prompt_batch(tokenizer, prompts, device)
    cm = overlay.installed(model) if overlay is not None else contextlib.nullcontext()
    with cm:
        output = model(**batch, use_cache=False)
    positions = batch["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(len(prompts), device=device)
    return output.logits[rows, positions].float()


def _history_kl(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    teacher: torch.Tensor,
    device: str,
    overlay: FunctionalCellOverlay,
) -> float:
    current = _last_logits(model, tokenizer, prompts, device, overlay)
    return float(
        F.kl_div(
            F.log_softmax(current, dim=-1),
            F.softmax(teacher.to(device).detach(), dim=-1),
            reduction="batchmean",
        ).item()
    )


def _evaluate(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, str]],
    protocol: dict[str, Any],
    device: str,
    overlay: FunctionalCellOverlay | None,
) -> dict[str, float]:
    task = protocol["sequence_task"]
    cm = overlay.installed(model) if overlay is not None else contextlib.nullcontext()
    with cm:
        return seq.evaluate_rows(
            model,
            tokenizer,
            rows,
            prompt_template=task["prompt_template"],
            max_length=int(task["max_sequence_tokens"]),
            device=device,
            batch_size=int(protocol["evaluation"]["batch_size"]),
        )


def _answer_loss_and_route(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, str]],
    protocol: dict[str, Any],
    device: str,
    overlay: FunctionalCellOverlay,
) -> tuple[torch.Tensor, torch.Tensor]:
    task = protocol["sequence_task"]
    batch = seq.encode_rows(
        tokenizer,
        rows,
        prompt_template=task["prompt_template"],
        max_length=int(task["max_sequence_tokens"]),
        device=device,
    )
    with overlay.installed(model):
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        loss, _count, _correct = seq.answer_loss_from_logits(output.logits, batch["labels"])
        supervised = batch["labels"].ne(-100)
        first_answer = supervised.float().argmax(dim=1)
        prompt_positions = first_answer - 1
        route = overlay.prompt_routes(prompt_positions).probabilities
    return loss, route


def _route_regularizer(
    probabilities: torch.Tensor,
    concept_ids: Sequence[str],
) -> torch.Tensor:
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for index, concept in enumerate(concept_ids):
        groups[str(concept)].append(index)
    means: list[torch.Tensor] = []
    same_terms: list[torch.Tensor] = []
    for indices in groups.values():
        values = probabilities[indices]
        means.append(values.mean(dim=0))
        if len(indices) > 1:
            anchor = values[0].unsqueeze(0)
            same_terms.append(1.0 - F.cosine_similarity(anchor, values[1:], dim=-1).mean())
    same = torch.stack(same_terms).mean() if same_terms else probabilities.new_tensor(0.0)
    collision_terms: list[torch.Tensor] = []
    for left in range(len(means)):
        for right in range(left + 1, len(means)):
            collision_terms.append(F.cosine_similarity(means[left], means[right], dim=0))
    collision = (
        torch.stack(collision_terms).mean()
        if collision_terms
        else probabilities.new_tensor(0.0)
    )
    return same + collision


def _formation_batches(
    rows: Sequence[dict[str, str]],
    seed: int,
    concepts_per_batch: int,
) -> Iterable[list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        grouped[row["concept_id"]].append(row)
    concepts = sorted(grouped)
    rng = random.Random(seed)
    while True:
        selected = rng.sample(concepts, k=min(concepts_per_batch, len(concepts)))
        batch: list[dict[str, str]] = []
        for concept in selected:
            batch.extend(grouped[concept])
        yield batch


def _nll_gain(base: dict[str, float], current: dict[str, float]) -> float:
    return float(base["mean_reference_nll"] - current["mean_reference_nll"])


def _train_formation(
    *,
    model: Any,
    tokenizer: Any,
    overlay: FunctionalCellOverlay,
    protocol: dict[str, Any],
    seed: int,
    device: str,
    base_validation: dict[str, float],
    validation_rows: Sequence[dict[str, str]],
    history_prompts: Sequence[str],
    history_teacher: torch.Tensor,
) -> dict[str, Any]:
    cfg = protocol["training"]["formation"]
    optimizer = torch.optim.Adam(overlay.parameters(), lr=float(cfg["learning_rate"]))
    batches = _formation_batches(
        training_rows(),
        seed,
        int(cfg["concepts_per_batch"]),
    )
    rng = random.Random(seed + 991)
    initial = overlay.snapshot()
    best = initial
    best_gain = 0.0
    best_step = 0
    candidate_log: list[dict[str, float | int | bool]] = []
    history = list(history_prompts)

    for step in range(1, int(cfg["steps"]) + 1):
        rows = next(batches)
        optimizer.zero_grad(set_to_none=True)
        target_loss, route = _answer_loss_and_route(
            model, tokenizer, rows, protocol, device, overlay
        )
        route_loss = _route_regularizer(route, [row["concept_id"] for row in rows])
        hist_rows = rng.sample(history, k=min(int(cfg["history_batch_size"]), len(history)))
        teacher_indices = [history.index(value) for value in hist_rows]
        current_history = _last_logits(model, tokenizer, hist_rows, device, overlay)
        teacher = history_teacher[teacher_indices].to(device)
        history_loss = F.kl_div(
            F.log_softmax(current_history, dim=-1),
            F.softmax(teacher.detach(), dim=-1),
            reduction="batchmean",
        )
        total = (
            target_loss
            + float(cfg["route_regularization_weight"]) * route_loss
            + float(cfg["history_kl_weight"]) * history_loss
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(overlay.parameters(), float(cfg["max_gradient_norm"]))
        optimizer.step()

        if step % int(cfg["eval_interval"]) == 0 or step == int(cfg["steps"]):
            validation = _evaluate(
                model, tokenizer, validation_rows, protocol, device, overlay
            )
            gain = _nll_gain(base_validation, validation)
            hist_kl = _history_kl(
                model, tokenizer, history_prompts, history_teacher, device, overlay
            )
            eligible = hist_kl <= float(cfg["maximum_history_kl_for_candidate"])
            candidate_log.append(
                {
                    "step": step,
                    "validation_nll_gain": gain,
                    "history_kl": hist_kl,
                    "eligible": eligible,
                }
            )
            if eligible and gain > best_gain:
                best = overlay.snapshot()
                best_gain = gain
                best_step = step

    overlay.restore_(best)
    return {
        "best_step": best_step,
        "best_validation_nll_gain": best_gain,
        "candidates": candidate_log,
    }


def _route_primary_for_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, str]],
    protocol: dict[str, Any],
    device: str,
    overlay: FunctionalCellOverlay,
) -> list[int]:
    template = protocol["sequence_task"]["prompt_template"]
    batch_size = int(protocol["evaluation"]["batch_size"])
    results: list[int] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [template.format(question=row["question"]) for row in chunk]
        batch = _prompt_batch(tokenizer, prompts, device)
        with overlay.installed(model):
            model(**batch, use_cache=False)
            positions = batch["attention_mask"].sum(dim=1) - 1
            summary = overlay.prompt_routes(positions)
            results.extend(int(value) for value in summary.primary_cell.tolist())
    return results


def _routing_metrics(
    model: Any,
    tokenizer: Any,
    routing_rows: Sequence[dict[str, str]],
    protocol: dict[str, Any],
    device: str,
    overlay: FunctionalCellOverlay,
) -> tuple[dict[str, Any], dict[str, int], dict[str, float]]:
    primary = _route_primary_for_rows(
        model, tokenizer, routing_rows, protocol, device, overlay
    )
    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for row, cell in zip(routing_rows, primary, strict=True):
        grouped[row["concept_id"]].append(cell)
    modal: dict[str, int] = {}
    agreement: dict[str, float] = {}
    for concept, cells in grouped.items():
        counts = collections.Counter(cells)
        cell, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        modal[concept] = int(cell)
        agreement[concept] = count / len(cells)
    entity_modal = {key: value for key, value in modal.items() if key.startswith("entity.")}
    counts = collections.Counter(entity_modal.values())
    metrics = {
        "mean_route_agreement": sum(agreement.values()) / max(len(agreement), 1),
        "minimum_route_agreement": min(agreement.values()) if agreement else 0.0,
        "distinct_primary_cells": len(set(entity_modal.values())),
        "maximum_primary_cell_fraction": (
            max(counts.values()) / max(len(entity_modal), 1) if counts else 1.0
        ),
        "concept_primary_cells": modal,
        "concept_route_agreement": agreement,
    }
    return metrics, modal, agreement


def _train_cell(
    *,
    model: Any,
    tokenizer: Any,
    overlay: FunctionalCellOverlay,
    rows: Sequence[dict[str, str]],
    cell_index: int,
    protocol: dict[str, Any],
    device: str,
    steps: int,
    learning_rate: float,
    max_gradient_norm: float,
    learn_key: bool,
    route_target_weight: float = 0.0,
) -> None:
    parameters = [overlay.down, overlay.up]
    if learn_key:
        parameters = [overlay.keys, *parameters]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, route = _answer_loss_and_route(
            model, tokenizer, rows, protocol, device, overlay
        )
        if route_target_weight:
            route_target = -torch.log(route[:, cell_index].float().clamp_min(1e-8)).mean()
            loss = loss + route_target_weight * route_target
        loss.backward()
        mask_cell_gradients_(overlay, [cell_index], include_keys=learn_key)
        torch.nn.utils.clip_grad_norm_(parameters, max_gradient_norm)
        optimizer.step()


def _choose_rewrite_target(
    *,
    model: Any,
    tokenizer: Any,
    overlay: FunctionalCellOverlay,
    modal: dict[str, int],
    agreement: dict[str, float],
    protocol: dict[str, Any],
    device: str,
    forbidden_cells: set[int] | None = None,
) -> dict[str, Any]:
    current = {fact.subject: fact.value for fact in entity_facts()}
    candidates: list[dict[str, Any]] = []
    forbidden = forbidden_cells or set()
    for entity in ENTITIES:
        concept = f"entity.{entity}"
        cell = int(modal[concept])
        if cell in forbidden:
            continue
        old = current[entity]
        new = PROTOCOLS[(PROTOCOLS.index(old) + 1) % len(PROTOCOLS)]
        rows = rewrite_rows(entity, new, prefix=f"rewrite.{entity.lower()}")
        train_routes = _route_primary_for_rows(
            model, tokenizer, rows["train"], protocol, device, overlay
        )
        target_fraction = sum(value == cell for value in train_routes) / len(train_routes)
        candidates.append(
            {
                "entity": entity,
                "concept_id": concept,
                "cell": cell,
                "old_protocol": old,
                "new_protocol": new,
                "agreement": float(agreement[concept]),
                "train_target_fraction": target_fraction,
                "rows": rows,
            }
        )
    if not candidates:
        raise RuntimeError("no eligible rewrite target remains")
    candidates.sort(
        key=lambda item: (
            -float(item["train_target_fraction"]),
            -float(item["agreement"]),
            str(item["entity"]),
        )
    )
    return candidates[0]


def _state_equal(snapshot: dict[str, torch.Tensor], overlay: FunctionalCellOverlay) -> bool:
    current = overlay.state_dict()
    return all(torch.equal(current[name].detach().cpu(), value) for name, value in snapshot.items())


def _metric_dict(value: dict[str, float]) -> dict[str, float]:
    return {key: float(item) for key, item in value.items()}


def _branch_phase(
    *,
    model: Any,
    tokenizer: Any,
    overlay: FunctionalCellOverlay,
    formation_snapshot: dict[str, torch.Tensor],
    modal: dict[str, int],
    agreement: dict[str, float],
    protocol: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    local_cfg = protocol["training"]["local_mutation"]
    entity_cells = {int(modal[f"entity.{entity}"]) for entity in ENTITIES}
    if len(entity_cells) < 2:
        overlay.restore_(formation_snapshot)
        return {
            "branches": [],
            "minimum_merge_retention": 0.0,
            "exact_rollback": _state_equal(formation_snapshot, overlay),
            "skipped_reason": "fewer than two distinct entity primary Cells",
        }

    branch_a = _choose_rewrite_target(
        model=model,
        tokenizer=tokenizer,
        overlay=overlay,
        modal=modal,
        agreement=agreement,
        protocol=protocol,
        device=device,
    )
    branch_b = _choose_rewrite_target(
        model=model,
        tokenizer=tokenizer,
        overlay=overlay,
        modal=modal,
        agreement=agreement,
        protocol=protocol,
        device=device,
        forbidden_cells={int(branch_a["cell"])},
    )
    branches: list[dict[str, Any]] = []
    mutations = []
    for label, target in (("A", branch_a), ("B", branch_b)):
        overlay.restore_(formation_snapshot)
        before = _evaluate(
            model, tokenizer, target["rows"]["evaluation"], protocol, device, overlay
        )
        _train_cell(
            model=model,
            tokenizer=tokenizer,
            overlay=overlay,
            rows=target["rows"]["train"],
            cell_index=int(target["cell"]),
            protocol=protocol,
            device=device,
            steps=int(local_cfg["steps"]),
            learning_rate=float(local_cfg["learning_rate"]),
            max_gradient_norm=float(local_cfg["max_gradient_norm"]),
            learn_key=False,
        )
        after = _evaluate(
            model, tokenizer, target["rows"]["evaluation"], protocol, device, overlay
        )
        mutations.append(overlay.export_mutation(int(target["cell"])))
        branches.append(
            {
                "label": label,
                "entity": target["entity"],
                "cell": int(target["cell"]),
                "before": _metric_dict(before),
                "standalone": _metric_dict(after),
                "standalone_nll_gain": _nll_gain(before, after),
                "rows": target["rows"],
            }
        )

    disjoint_mutations(*mutations)
    overlay.restore_(formation_snapshot)
    for mutation in mutations:
        overlay.apply_mutation_(mutation)
    retentions: list[float] = []
    for branch in branches:
        merged = _evaluate(
            model, tokenizer, branch["rows"]["evaluation"], protocol, device, overlay
        )
        branch["merged"] = _metric_dict(merged)
        merged_gain = float(
            branch["before"]["mean_reference_nll"] - merged["mean_reference_nll"]
        )
        standalone_gain = float(branch["standalone_nll_gain"])
        retention = merged_gain / standalone_gain if standalone_gain > 0 else 0.0
        branch["merged_nll_gain"] = merged_gain
        branch["merge_retention"] = retention
        retentions.append(retention)
        branch.pop("rows", None)
    overlay.restore_(formation_snapshot)
    return {
        "branches": branches,
        "minimum_merge_retention": min(retentions) if retentions else 0.0,
        "exact_rollback": _state_equal(formation_snapshot, overlay),
    }


def run(seed: int, device: str) -> dict[str, Any]:
    protocol = _load_json(PROTOCOL_PATH)
    if _git_blob_sha(DATASET_PATH) != protocol["dataset"]["generator_git_blob_sha"]:
        raise RuntimeError("controlled dataset generator identity mismatch")
    if seed not in [int(value) for value in protocol["formal_seeds"]]:
        raise RuntimeError(f"seed {seed} is not a frozen formal seed")
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    _seed_everything(seed)

    import transformers

    transformers.logging.set_verbosity_error()
    model_id = protocol["base"]["model_id"]
    revision = protocol["base"]["revision"]
    _progress(seed, f"loading frozen foundation {model_id}@{revision[:12]}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad nor eos token")
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
    ).to(device)
    freeze_foundation_(model)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("foundation freeze invariant failed")

    substrate = protocol["substrate"]
    overlay = FunctionalCellOverlay(
        hidden_size=int(model.config.hidden_size),
        layer_indices=tuple(int(value) for value in substrate["layer_indices"]),
        max_cells=int(substrate["max_cells"]),
        initial_active_cells=int(substrate["initial_active_cells"]),
        rank=int(substrate["rank"]),
        temperature=float(substrate["temperature"]),
        top_k=int(substrate["top_k"]),
        seed=seed,
    ).to(device=device, dtype=torch.float32)
    if not overlay.zero_output_is_exact():
        raise RuntimeError("overlay must start with exact zero residual output")

    validation_rows = formation_validation()
    eval_rows = formation_evaluation()
    history = list(calibration_prompts())
    base_validation = _evaluate(model, tokenizer, validation_rows, protocol, device, None)
    base_metrics = {
        name: _evaluate(model, tokenizer, rows, protocol, device, None)
        for name, rows in eval_rows.items()
        if name != "routing"
    }
    history_teacher = _last_logits(model, tokenizer, history, device).detach().cpu()

    compatibility_prompts = history[:4]
    repeated_a = _last_logits(model, tokenizer, compatibility_prompts, device)
    repeated_b = _last_logits(model, tokenizer, compatibility_prompts, device)
    converted = _last_logits(model, tokenizer, compatibility_prompts, device, overlay)
    repeatability = float((repeated_a - repeated_b).abs().max().item())
    converted_delta = float((repeated_a - converted).abs().max().item())
    compatibility_excess = max(0.0, converted_delta - repeatability)
    compatibility = {
        "foundation_repeatability_max_abs_logit_delta": repeatability,
        "zero_overlay_max_abs_logit_delta": converted_delta,
        "excess_over_repeatability": compatibility_excess,
        "zero_output_exact": overlay.zero_output_is_exact(),
    }
    _progress(seed, f"compatibility excess={compatibility_excess:.3e}")

    formation_log = _train_formation(
        model=model,
        tokenizer=tokenizer,
        overlay=overlay,
        protocol=protocol,
        seed=seed,
        device=device,
        base_validation=base_validation,
        validation_rows=validation_rows,
        history_prompts=history,
        history_teacher=history_teacher,
    )
    formation_snapshot = overlay.snapshot()
    selected_validation = _evaluate(
        model, tokenizer, validation_rows, protocol, device, overlay
    )
    formation_metrics = {
        name: _evaluate(model, tokenizer, rows, protocol, device, overlay)
        for name, rows in eval_rows.items()
        if name != "routing"
    }
    formation_gains = {
        name: _nll_gain(base_metrics[name], formation_metrics[name])
        for name in ("direct", "negation", "relation")
    }
    formation_history_kl = _history_kl(
        model, tokenizer, history, history_teacher, device, overlay
    )
    route_metrics, modal, agreement = _routing_metrics(
        model,
        tokenizer,
        eval_rows["routing"],
        protocol,
        device,
        overlay,
    )
    _progress(
        seed,
        "formation "
        f"direct={formation_gains['direct']:.3f} "
        f"negation={formation_gains['negation']:.3f} "
        f"relation={formation_gains['relation']:.3f} "
        f"route={route_metrics['mean_route_agreement']:.3f}",
    )

    local_cfg = protocol["training"]["local_mutation"]
    local_target = _choose_rewrite_target(
        model=model,
        tokenizer=tokenizer,
        overlay=overlay,
        modal=modal,
        agreement=agreement,
        protocol=protocol,
        device=device,
    )
    local_base = _evaluate(
        model,
        tokenizer,
        local_target["rows"]["evaluation"],
        protocol,
        device,
        overlay,
    )
    unrelated_rows = [
        row
        for row in eval_rows["direct"]
        if row["concept_id"] != local_target["concept_id"]
    ]
    unrelated_base = _evaluate(model, tokenizer, unrelated_rows, protocol, device, overlay)
    _train_cell(
        model=model,
        tokenizer=tokenizer,
        overlay=overlay,
        rows=local_target["rows"]["train"],
        cell_index=int(local_target["cell"]),
        protocol=protocol,
        device=device,
        steps=int(local_cfg["steps"]),
        learning_rate=float(local_cfg["learning_rate"]),
        max_gradient_norm=float(local_cfg["max_gradient_norm"]),
        learn_key=False,
    )
    local_after = _evaluate(
        model,
        tokenizer,
        local_target["rows"]["evaluation"],
        protocol,
        device,
        overlay,
    )
    unrelated_after = _evaluate(model, tokenizer, unrelated_rows, protocol, device, overlay)
    local_result = {
        "entity": local_target["entity"],
        "cell": int(local_target["cell"]),
        "route_agreement": float(local_target["agreement"]),
        "rewrite_train_target_fraction": float(local_target["train_target_fraction"]),
        "semantic_write_nll_gain": _nll_gain(local_base, local_after),
        "unrelated_nll_regression": float(
            unrelated_after["mean_reference_nll"] - unrelated_base["mean_reference_nll"]
        ),
        "before": _metric_dict(local_base),
        "after": _metric_dict(local_after),
    }
    overlay.restore_(formation_snapshot)
    local_result["rollback_exact"] = _state_equal(formation_snapshot, overlay)
    _progress(
        seed,
        f"local write gain={local_result['semantic_write_nll_gain']:.3f} "
        f"unrelated_regression={local_result['unrelated_nll_regression']:.3f}",
    )

    growth_cfg = protocol["training"]["growth"]
    growth_entity = str(local_target["entity"])
    growth_parent = int(local_target["cell"])
    old_protocol = str(local_target["old_protocol"])
    new_protocol = PROTOCOLS[(PROTOCOLS.index(old_protocol) + 2) % len(PROTOCOLS)]
    growth_rows = contextual_conflict_rows(growth_entity, old_protocol, new_protocol)
    alpha_base = _evaluate(model, tokenizer, growth_rows["alpha"], protocol, device, overlay)
    beta_base = _evaluate(
        model, tokenizer, growth_rows["beta_eval"], protocol, device, overlay
    )

    _train_cell(
        model=model,
        tokenizer=tokenizer,
        overlay=overlay,
        rows=growth_rows["beta_train"],
        cell_index=growth_parent,
        protocol=protocol,
        device=device,
        steps=int(growth_cfg["steps"]),
        learning_rate=float(local_cfg["learning_rate"]),
        max_gradient_norm=float(local_cfg["max_gradient_norm"]),
        learn_key=False,
    )
    parent_alpha = _evaluate(
        model, tokenizer, growth_rows["alpha"], protocol, device, overlay
    )
    parent_beta = _evaluate(
        model, tokenizer, growth_rows["beta_eval"], protocol, device, overlay
    )
    parent_control = {
        "alpha_nll_regression": float(
            parent_alpha["mean_reference_nll"] - alpha_base["mean_reference_nll"]
        ),
        "beta_nll_gain": _nll_gain(beta_base, parent_beta),
    }

    overlay.restore_(formation_snapshot)
    child = overlay.spawn_child(growth_parent)
    alpha_spawned = _evaluate(
        model, tokenizer, growth_rows["alpha"], protocol, device, overlay
    )
    beta_spawned = _evaluate(
        model, tokenizer, growth_rows["beta_eval"], protocol, device, overlay
    )
    spawn_max_nll_delta = max(
        abs(alpha_spawned["mean_reference_nll"] - alpha_base["mean_reference_nll"]),
        abs(beta_spawned["mean_reference_nll"] - beta_base["mean_reference_nll"]),
    )
    _train_cell(
        model=model,
        tokenizer=tokenizer,
        overlay=overlay,
        rows=growth_rows["beta_train"],
        cell_index=child,
        protocol=protocol,
        device=device,
        steps=int(growth_cfg["steps"]),
        learning_rate=float(local_cfg["learning_rate"]),
        max_gradient_norm=float(local_cfg["max_gradient_norm"]),
        learn_key=True,
        route_target_weight=float(local_cfg["route_target_weight"]),
    )
    child_alpha = _evaluate(
        model, tokenizer, growth_rows["alpha"], protocol, device, overlay
    )
    child_beta = _evaluate(
        model, tokenizer, growth_rows["beta_eval"], protocol, device, overlay
    )
    beta_routes = _route_primary_for_rows(
        model, tokenizer, growth_rows["beta_eval"], protocol, device, overlay
    )
    growth_result = {
        "entity": growth_entity,
        "parent_cell": growth_parent,
        "child_cell": child,
        "spawn_max_mean_nll_delta": spawn_max_nll_delta,
        "parent_only_control": parent_control,
        "beta_nll_gain": _nll_gain(beta_base, child_beta),
        "alpha_nll_regression": float(
            child_alpha["mean_reference_nll"] - alpha_base["mean_reference_nll"]
        ),
        "child_beta_route_fraction": sum(value == child for value in beta_routes)
        / max(len(beta_routes), 1),
        "alpha_before": _metric_dict(alpha_base),
        "alpha_after": _metric_dict(child_alpha),
        "beta_before": _metric_dict(beta_base),
        "beta_after": _metric_dict(child_beta),
    }
    overlay.restore_(formation_snapshot)

    branch_result = _branch_phase(
        model=model,
        tokenizer=tokenizer,
        overlay=overlay,
        formation_snapshot=formation_snapshot,
        modal=modal,
        agreement=agreement,
        protocol=protocol,
        device=device,
    )

    gates_cfg = protocol["gates"]
    gates = {
        "compatibility": compatibility_excess
        <= float(gates_cfg["maximum_compatibility_excess_over_repeatability"]),
        "direct_acquisition": formation_gains["direct"]
        >= float(gates_cfg["minimum_direct_nll_gain"]),
        "negation_acquisition": formation_gains["negation"]
        >= float(gates_cfg["minimum_negation_nll_gain"]),
        "relation_acquisition": formation_gains["relation"]
        >= float(gates_cfg["minimum_relation_nll_gain"]),
        "history_preservation": formation_history_kl <= float(gates_cfg["maximum_history_kl"]),
        "semantic_routing": float(route_metrics["mean_route_agreement"])
        >= float(gates_cfg["minimum_mean_route_agreement"]),
        "route_diversity": int(route_metrics["distinct_primary_cells"])
        >= int(gates_cfg["minimum_distinct_primary_cells"])
        and float(route_metrics["maximum_primary_cell_fraction"])
        <= float(gates_cfg["maximum_primary_cell_fraction"]),
        "semantic_local_write": float(local_result["semantic_write_nll_gain"])
        >= float(gates_cfg["minimum_semantic_write_nll_gain"]),
        "unrelated_locality": float(local_result["unrelated_nll_regression"])
        <= float(gates_cfg["maximum_unrelated_nll_regression"]),
        "growth_beta_acquisition": float(growth_result["beta_nll_gain"])
        >= float(gates_cfg["minimum_growth_beta_nll_gain"]),
        "growth_alpha_retention": float(growth_result["alpha_nll_regression"])
        <= float(gates_cfg["maximum_growth_alpha_nll_regression"]),
        "growth_child_routing": float(growth_result["child_beta_route_fraction"])
        >= float(gates_cfg["minimum_child_beta_route_fraction"]),
        "branch_merge": float(branch_result["minimum_merge_retention"])
        >= float(gates_cfg["minimum_branch_merge_retention"]),
        "rollback": bool(local_result["rollback_exact"])
        and bool(branch_result["exact_rollback"]),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    protocol_sha256 = _sha256(PROTOCOL_PATH)
    result = {
        "experiment": protocol["experiment"],
        "protocol_sha256": protocol_sha256,
        "dataset_generator_git_blob_sha": protocol["dataset"]["generator_git_blob_sha"],
        "seed": seed,
        "status": status,
        "model_id": model_id,
        "revision": revision,
        "compatibility": compatibility,
        "base_validation": _metric_dict(base_validation),
        "selected_validation": _metric_dict(selected_validation),
        "base_metrics": {name: _metric_dict(value) for name, value in base_metrics.items()},
        "formation": {
            "training": formation_log,
            "metrics": {name: _metric_dict(value) for name, value in formation_metrics.items()},
            "nll_gains": formation_gains,
            "history_kl": formation_history_kl,
            "routing": route_metrics,
        },
        "semantic_local_write": local_result,
        "growth": growth_result,
        "branch_merge": branch_result,
        "gates": gates,
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
    }
    seed_root = RESULTS_ROOT / f"seed-{seed}"
    _write_json(seed_root / "result.json", result)
    _write_json(
        seed_root / "seed_summary.json",
        {
            "experiment": protocol["experiment"],
            "protocol_sha256": protocol_sha256,
            "dataset_generator_git_blob_sha": protocol["dataset"]["generator_git_blob_sha"],
            "seed": seed,
            "status": status,
            "failed_gates": result["failed_gates"],
            "formation_nll_gains": formation_gains,
            "mean_route_agreement": route_metrics["mean_route_agreement"],
            "distinct_primary_cells": route_metrics["distinct_primary_cells"],
            "semantic_write_nll_gain": local_result["semantic_write_nll_gain"],
            "growth_beta_nll_gain": growth_result["beta_nll_gain"],
            "minimum_branch_merge_retention": branch_result["minimum_merge_retention"],
        },
    )
    _progress(seed, f"formal seed complete: {status} failed={result['failed_gates']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CLM Conversion Kill Test 001 formal seed")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.seed, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
