from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F

from minicells.functional_cellization import freeze_foundation_
from minicells.hybrid_clm import (
    HybridCellArtifact,
    HybridCellOverlay,
    HybridManifest,
    mask_address_gradients_,
    mask_transform_gradients_,
    save_cell_artifact,
)

ROOT = Path(__file__).resolve().parents[3]
SEQUENCE_ROOT = ROOT / "scripts" / "research" / "jam_knowledge_mutation_001"
CONVERSION_ROOT = ROOT / "scripts" / "research" / "clm_conversion_kill_test_001"
for path in (SEQUENCE_ROOT, CONVERSION_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sequence as seq  # noqa: E402
from dataset import (  # noqa: E402
    CANDIDATE_CODES,
    DemoFact,
    address_negative_prompts,
    address_positive_prompts,
    demo_facts,
    evaluation_rows,
    general_history_prompts,
    training_rows,
    update_rows,
)
from semantic_choice import candidate_choice_metrics  # noqa: E402

MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-base"
MODEL_REVISION = "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"
PROMPT_TEMPLATE = "Question: {question}\nAnswer:"
RESULTS_ROOT = ROOT / "results" / "granite-hybrid-clm-v0.1"

PROTOCOL: dict[str, Any] = {
    "sequence_task": {
        "prompt_template": PROMPT_TEMPLATE,
        "max_sequence_tokens": 96,
    },
    "evaluation": {"batch_size": 8},
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _progress(message: str) -> None:
    print(f"[granite-hybrid-clm-v0.1] {message}", flush=True)


def _prompt_batch(tokenizer: Any, prompts: Sequence[str], device: str) -> dict[str, torch.Tensor]:
    batch = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )
    return {name: value.to(device) for name, value in batch.items()}


def _overlay_context(
    overlay: HybridCellOverlay | None,
    shadow_slots: Sequence[int] = (),
) -> contextlib.AbstractContextManager[Any]:
    if overlay is None:
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def combined() -> Any:
        with overlay.shadow(shadow_slots):
            yield

    return combined()


def _last_logits(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    device: str,
    overlay: HybridCellOverlay | None = None,
    shadow_slots: Sequence[int] = (),
) -> torch.Tensor:
    batch = _prompt_batch(tokenizer, prompts, device)
    with _overlay_context(overlay, shadow_slots):
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
    overlay: HybridCellOverlay,
    shadow_slots: Sequence[int] = (),
) -> float:
    current = _last_logits(
        model,
        tokenizer,
        prompts,
        device,
        overlay,
        shadow_slots,
    )
    return float(
        F.kl_div(
            F.log_softmax(current, dim=-1),
            F.softmax(teacher.to(device), dim=-1),
            reduction="batchmean",
        ).item()
    )


def _evaluate_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, str]],
    device: str,
    overlay: HybridCellOverlay | None,
    shadow_slots: Sequence[int] = (),
) -> dict[str, float]:
    with _overlay_context(overlay, shadow_slots):
        cm = overlay.installed(model) if overlay is not None else contextlib.nullcontext()
        with cm:
            return seq.evaluate_rows(
                model,
                tokenizer,
                rows,
                prompt_template=PROMPT_TEMPLATE,
                max_length=96,
                device=device,
                batch_size=8,
            )


def _candidate_choice(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, str]],
    device: str,
    overlay: HybridCellOverlay | None,
    shadow_slots: Sequence[int] = (),
) -> dict[str, Any]:
    with _overlay_context(overlay, shadow_slots):
        return candidate_choice_metrics(
            model,
            tokenizer,
            rows,
            CANDIDATE_CODES,
            protocol=PROTOCOL,
            device=device,
            overlay=overlay,
            sequence_module=seq,
        )


def _extract_read_features(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    read_layer_index: int,
    device: str,
) -> torch.Tensor:
    batch = _prompt_batch(tokenizer, prompts, device)
    captured: list[torch.Tensor] = []
    module = model.get_submodule(f"model.layers.{read_layer_index}")

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach())
        return output

    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**batch, use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("read feature hook did not execute exactly once")
    positions = batch["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(len(prompts), device=device)
    return captured[0][rows, positions].float()


def _train_address(
    overlay: HybridCellOverlay,
    slot: int,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> dict[str, float | bool]:
    features = torch.cat([positive, negative], dim=0)
    labels = torch.cat(
        [
            torch.ones(len(positive), device=features.device),
            torch.zeros(len(negative), device=features.device),
        ]
    )
    optimizer = torch.optim.Adam([overlay.gate_weight, overlay.gate_bias], lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        probabilities = overlay.address_probability_for_features(features, slot)
        loss = F.binary_cross_entropy(probabilities, labels)
        loss.backward()
        mask_address_gradients_(overlay, slot)
        optimizer.step()

    with torch.no_grad():
        positive_prob = overlay.address_probability_for_features(positive, slot)
        negative_prob = overlay.address_probability_for_features(negative, slot)
    threshold = overlay.gate_threshold
    positive_recall = float((positive_prob >= threshold).float().mean().item())
    negative_false_positive = float((negative_prob >= threshold).float().mean().item())
    return {
        "positive_recall": positive_recall,
        "negative_false_positive_rate": negative_false_positive,
        "minimum_positive_probability": float(positive_prob.min().item()),
        "maximum_negative_probability": float(negative_prob.max().item()),
        "passed": positive_recall == 1.0 and negative_false_positive <= 0.02,
    }


def _answer_loss(
    model: Any,
    tokenizer: Any,
    rows: Sequence[dict[str, str]],
    device: str,
    overlay: HybridCellOverlay,
    slot: int,
) -> torch.Tensor:
    batch = seq.encode_rows(
        tokenizer,
        rows,
        prompt_template=PROMPT_TEMPLATE,
        max_length=96,
        device=device,
    )
    with overlay.shadow([slot]):
        with overlay.installed(model):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
    loss, _count, _correct = seq.answer_loss_from_logits(output.logits, batch["labels"])
    return loss


def _load_cell_state_(
    overlay: HybridCellOverlay,
    slot: int,
    state: dict[str, torch.Tensor],
) -> None:
    device = overlay.gate_weight.device
    with torch.no_grad():
        overlay.gate_weight[slot].copy_(
            state["gate_weight"].to(device=device, dtype=overlay.gate_weight.dtype)
        )
        overlay.gate_bias[slot].copy_(
            state["gate_bias"].to(device=device, dtype=overlay.gate_bias.dtype).reshape(())
        )
        overlay.down[:, slot].copy_(state["down"].to(device=device, dtype=overlay.down.dtype))
        overlay.up[:, slot].copy_(state["up"].to(device=device, dtype=overlay.up.dtype))


def _train_transform(
    *,
    model: Any,
    tokenizer: Any,
    overlay: HybridCellOverlay,
    slot: int,
    rows: Sequence[dict[str, str]],
    evaluation: Sequence[dict[str, str]],
    history_prompts: Sequence[str],
    history_teacher: torch.Tensor,
    device: str,
    steps: int,
    learning_rate: float,
    history_kl_weight: float,
    maximum_history_kl: float,
    minimum_nll_gain: float,
) -> dict[str, Any]:
    before = _evaluate_rows(model, tokenizer, evaluation, device, overlay)
    optimizer = torch.optim.Adam([overlay.down, overlay.up], lr=learning_rate)
    best_state: dict[str, torch.Tensor] | None = None
    best_gain = float("-inf")
    candidates: list[dict[str, Any]] = []

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        target_loss = _answer_loss(model, tokenizer, rows, device, overlay, slot)
        current_history = _last_logits(
            model,
            tokenizer,
            history_prompts,
            device,
            overlay,
            shadow_slots=[slot],
        )
        history_loss = F.kl_div(
            F.log_softmax(current_history, dim=-1),
            F.softmax(history_teacher.to(device), dim=-1),
            reduction="batchmean",
        )
        total = target_loss + history_kl_weight * history_loss
        total.backward()
        mask_transform_gradients_(overlay, slot)
        torch.nn.utils.clip_grad_norm_([overlay.down, overlay.up], 1.0)
        optimizer.step()

        if step % 4 == 0 or step == steps:
            after = _evaluate_rows(
                model,
                tokenizer,
                evaluation,
                device,
                overlay,
                shadow_slots=[slot],
            )
            gain = float(before["mean_reference_nll"] - after["mean_reference_nll"])
            history_kl = _history_kl(
                model,
                tokenizer,
                history_prompts,
                history_teacher,
                device,
                overlay,
                shadow_slots=[slot],
            )
            choice = _candidate_choice(
                model,
                tokenizer,
                evaluation,
                device,
                overlay,
                shadow_slots=[slot],
            )
            eligible = (
                gain >= minimum_nll_gain
                and history_kl <= maximum_history_kl
                and float(choice["strict_choice_accuracy"]) == 1.0
            )
            candidates.append(
                {
                    "step": step,
                    "nll_gain": gain,
                    "history_kl": history_kl,
                    "choice_accuracy": float(choice["strict_choice_accuracy"]),
                    "eligible": eligible,
                }
            )
            if eligible and gain > best_gain:
                best_gain = gain
                best_state = overlay.cell_state(slot)

    if best_state is not None:
        _load_cell_state_(overlay, slot, best_state)
    return {
        "before": before,
        "best_nll_gain": best_gain if best_state is not None else 0.0,
        "candidates": candidates,
        "passed": best_state is not None,
    }


def _history_questions(committed_facts: Sequence[DemoFact]) -> list[str]:
    prompts = list(general_history_prompts())
    for fact in committed_facts[-12:]:
        prompts.extend(row["question"] for row in evaluation_rows(fact))
    return [PROMPT_TEMPLATE.format(question=value) for value in prompts]


def _manifest_payload(manifest: HybridManifest) -> dict[str, Any]:
    return manifest.as_dict()


def _train_one_fact(
    *,
    model: Any,
    tokenizer: Any,
    overlay: HybridCellOverlay,
    fact: DemoFact,
    all_facts: tuple[DemoFact, ...],
    committed_facts: list[DemoFact],
    manifest: HybridManifest,
    artifacts: dict[str, HybridCellArtifact],
    output_dir: Path,
    device: str,
    address_steps: int,
    transform_steps: int,
) -> tuple[HybridManifest, dict[str, Any]]:
    slot = overlay.allocate_cell()
    cell_id = f"fact-{fact.index:03d}"
    positives = address_positive_prompts(fact)
    negatives = address_negative_prompts(
        fact,
        all_facts,
        count=min(49, len(all_facts) - 1),
    )
    positive_features = _extract_read_features(
        model,
        tokenizer,
        [PROMPT_TEMPLATE.format(question=value) for value in positives],
        read_layer_index=overlay.read_layer_index,
        device=device,
    )
    negative_features = _extract_read_features(
        model,
        tokenizer,
        [PROMPT_TEMPLATE.format(question=value) for value in negatives],
        read_layer_index=overlay.read_layer_index,
        device=device,
    )
    address = _train_address(
        overlay,
        slot,
        positive_features,
        negative_features,
        steps=address_steps,
        learning_rate=0.08,
    )
    if not bool(address["passed"]):
        return manifest, {
            "cell_id": cell_id,
            "slot": slot,
            "status": "ADDRESS_FAIL",
            "address": address,
        }
    overlay.freeze_address_(slot)

    history_prompts = _history_questions(committed_facts)
    history_teacher = _last_logits(
        model,
        tokenizer,
        history_prompts,
        device,
        overlay,
    ).detach().cpu()
    transform = _train_transform(
        model=model,
        tokenizer=tokenizer,
        overlay=overlay,
        slot=slot,
        rows=training_rows(fact),
        evaluation=evaluation_rows(fact),
        history_prompts=history_prompts,
        history_teacher=history_teacher,
        device=device,
        steps=transform_steps,
        learning_rate=0.02,
        history_kl_weight=8.0,
        maximum_history_kl=0.02,
        minimum_nll_gain=0.5,
    )
    if not bool(transform["passed"]):
        return manifest, {
            "cell_id": cell_id,
            "slot": slot,
            "status": "WRITE_FAIL",
            "address": address,
            "transform": transform,
        }

    overlay.commit_cell_(slot)
    artifact = overlay.export_artifact(slot, cell_id=cell_id)
    artifacts[cell_id] = artifact
    save_cell_artifact(output_dir / "cells" / f"{cell_id}.pt", artifact)
    manifest = manifest.add(artifact)
    committed_facts.append(fact)

    production_choice = _candidate_choice(
        model,
        tokenizer,
        evaluation_rows(fact),
        device,
        overlay,
    )
    return manifest, {
        "cell_id": cell_id,
        "slot": slot,
        "status": "COMMITTED",
        "address": address,
        "transform": transform,
        "artifact_digest": artifact.digest(),
        "production_choice_accuracy": float(production_choice["strict_choice_accuracy"]),
    }


def _contextual_child_demo(
    *,
    model: Any,
    tokenizer: Any,
    overlay: HybridCellOverlay,
    parent_fact: DemoFact,
    parent_slot: int,
    output_dir: Path,
    manifest: HybridManifest,
    device: str,
) -> tuple[HybridManifest, dict[str, Any]]:
    new_value = CANDIDATE_CODES[(CANDIDATE_CODES.index(parent_fact.value) + 3) % len(CANDIDATE_CODES)]
    rows = update_rows(parent_fact, new_value, "v2")
    child = overlay.allocate_cell(parent_slot=parent_slot)
    positives = [item["question"] for item in rows["train"] + rows["evaluation"]]
    negatives = [item["question"] for item in evaluation_rows(parent_fact)]
    positive_features = _extract_read_features(
        model,
        tokenizer,
        [PROMPT_TEMPLATE.format(question=value) for value in positives],
        read_layer_index=overlay.read_layer_index,
        device=device,
    )
    negative_features = _extract_read_features(
        model,
        tokenizer,
        [PROMPT_TEMPLATE.format(question=value) for value in negatives],
        read_layer_index=overlay.read_layer_index,
        device=device,
    )
    address = _train_address(
        overlay,
        child,
        positive_features,
        negative_features,
        steps=240,
        learning_rate=0.08,
    )
    if not bool(address["passed"]):
        return manifest, {"status": "ADDRESS_FAIL", "address": address}
    overlay.freeze_address_(child)

    history_prompts = [
        PROMPT_TEMPLATE.format(question=item["question"])
        for item in evaluation_rows(parent_fact)
    ]
    teacher = _last_logits(model, tokenizer, history_prompts, device, overlay).detach().cpu()
    transform = _train_transform(
        model=model,
        tokenizer=tokenizer,
        overlay=overlay,
        slot=child,
        rows=rows["train"],
        evaluation=rows["evaluation"],
        history_prompts=history_prompts,
        history_teacher=teacher,
        device=device,
        steps=40,
        learning_rate=0.02,
        history_kl_weight=12.0,
        maximum_history_kl=0.02,
        minimum_nll_gain=0.5,
    )
    if not bool(transform["passed"]):
        return manifest, {"status": "WRITE_FAIL", "address": address, "transform": transform}

    overlay.commit_cell_(child)
    artifact = overlay.export_artifact(
        child,
        cell_id=f"fact-{parent_fact.index:03d}-v2",
        parent_id=f"fact-{parent_fact.index:03d}",
        version=2,
    )
    save_cell_artifact(output_dir / "cells" / f"{artifact.cell_id}.pt", artifact)
    manifest = manifest.add(artifact)
    old_choice = _candidate_choice(
        model,
        tokenizer,
        evaluation_rows(parent_fact),
        device,
        overlay,
    )
    new_choice = _candidate_choice(
        model,
        tokenizer,
        rows["evaluation"],
        device,
        overlay,
    )
    overlay.uncommit_cell_(child)
    rollback_choice = _candidate_choice(
        model,
        tokenizer,
        evaluation_rows(parent_fact),
        device,
        overlay,
    )
    overlay.commit_cell_(child)
    return manifest, {
        "status": "COMMITTED",
        "cell_id": artifact.cell_id,
        "slot": child,
        "address": address,
        "transform": transform,
        "old_choice_accuracy_with_child": float(old_choice["strict_choice_accuracy"]),
        "new_choice_accuracy": float(new_choice["strict_choice_accuracy"]),
        "rollback_old_choice_accuracy": float(rollback_choice["strict_choice_accuracy"]),
    }


def run(
    *,
    device: str,
    fact_count: int,
    seed: int,
    address_steps: int,
    transform_steps: int,
    output_dir: Path,
) -> dict[str, Any]:
    _seed_everything(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    import transformers

    transformers.logging.set_verbosity_error()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
    ).to(device)
    freeze_foundation_(model)

    overlay = HybridCellOverlay(
        hidden_size=int(model.config.hidden_size),
        read_layer_index=18,
        write_layer_indices=(20, 22),
        max_cells=max(64, fact_count + 8),
        rank=16,
        gate_threshold=0.8,
        gate_temperature=1.0,
        seed=seed,
    ).to(device=device, dtype=torch.float32)

    baseline_prompts = [PROMPT_TEMPLATE.format(question=value) for value in general_history_prompts()[:4]]
    base_logits = _last_logits(model, tokenizer, baseline_prompts, device)
    converted_logits = _last_logits(model, tokenizer, baseline_prompts, device, overlay)
    compatibility_delta = float((base_logits - converted_logits).abs().max().item())

    facts = demo_facts(fact_count)
    manifest = HybridManifest(MODEL_ID, MODEL_REVISION)
    artifacts: dict[str, HybridCellArtifact] = {}
    committed_facts: list[DemoFact] = []
    cell_results: list[dict[str, Any]] = []
    cell_slots: dict[str, int] = {}

    for fact in facts:
        _progress(f"learning {fact.concept_id} ({len(committed_facts)}/{fact_count} committed)")
        manifest, result = _train_one_fact(
            model=model,
            tokenizer=tokenizer,
            overlay=overlay,
            fact=fact,
            all_facts=facts,
            committed_facts=committed_facts,
            manifest=manifest,
            artifacts=artifacts,
            output_dir=output_dir,
            device=device,
            address_steps=address_steps,
            transform_steps=transform_steps,
        )
        cell_results.append(result)
        if result["status"] == "COMMITTED":
            cell_slots[result["cell_id"]] = int(result["slot"])
        _write_json(output_dir / "progress.json", {"cells": cell_results, "manifest": manifest.as_dict()})

    retained_rows = [row for fact in committed_facts for row in evaluation_rows(fact)]
    retention_choice = (
        _candidate_choice(model, tokenizer, retained_rows, device, overlay)
        if retained_rows
        else {"strict_choice_accuracy": 0.0}
    )

    child_result: dict[str, Any] = {"status": "SKIPPED"}
    if len(committed_facts) >= 1:
        parent = committed_facts[0]
        parent_slot = cell_slots[f"fact-{parent.index:03d}"]
        manifest, child_result = _contextual_child_demo(
            model=model,
            tokenizer=tokenizer,
            overlay=overlay,
            parent_fact=parent,
            parent_slot=parent_slot,
            output_dir=output_dir,
            manifest=manifest,
            device=device,
        )

    committed = sum(result["status"] == "COMMITTED" for result in cell_results)
    status = "GRANITE_HYBRID_CLM_V01_SUPPORTED" if (
        compatibility_delta == 0.0
        and committed == fact_count
        and float(retention_choice["strict_choice_accuracy"]) >= 0.98
        and child_result.get("status") == "COMMITTED"
        and float(child_result.get("new_choice_accuracy", 0.0)) == 1.0
        and float(child_result.get("rollback_old_choice_accuracy", 0.0)) == 1.0
    ) else "GRANITE_HYBRID_CLM_V01_NOT_YET_SUPPORTED"

    result = {
        "experiment": "GRANITE_HYBRID_CLM_V0_1",
        "status": status,
        "seed": seed,
        "foundation": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "trainable": False},
        "compatibility_max_abs_logit_delta": compatibility_delta,
        "requested_facts": fact_count,
        "committed_facts": committed,
        "retention_choice_accuracy": float(retention_choice["strict_choice_accuracy"]),
        "cells": cell_results,
        "contextual_child": child_result,
        "manifest": _manifest_payload(manifest),
    }
    _write_json(output_dir / "result.json", result)
    _write_json(output_dir / "manifest.json", manifest.as_dict())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Granite Hybrid CLM v0.1 milestone")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--facts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=26090471)
    parser.add_argument("--address-steps", type=int, default=240)
    parser.add_argument("--transform-steps", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()
    result = run(
        device=args.device,
        fact_count=args.facts,
        seed=args.seed,
        address_steps=args.address_steps,
        transform_steps=args.transform_steps,
        output_dir=args.output_dir,
    )
    print(json.dumps({key: result[key] for key in ("status", "committed_facts", "retention_choice_accuracy")}, indent=2))


if __name__ == "__main__":
    main()
