from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "scripts"))

import minicells.language_proposal_utility as proposal_utility  # noqa: E402
import run_language_growing_organism_worker as e016  # noqa: E402
import run_language_localized_learning_worker as e017  # noqa: E402
import run_proposal_utility_discovery_worker as e019  # noqa: E402
from minicells.language_growing_organism import STABILITY_WEIGHT  # noqa: E402
from minicells.language_localized_learning import (  # noqa: E402
    LocalizedLearningState,
    graft_localized_tissue,
    mask_to_newborn_gradients,
    restore_structure,
    set_newborn_tissue_active,
)
from minicells.language_proposal_checkpoints import load_checkpoint, phase1_path  # noqa: E402
from minicells.language_recruitment_numerics import stable_gated_replicator_activity  # noqa: E402
from minicells.language_tissue_specificity import (  # noqa: E402
    ONE_CELL_SIZE,
    THREE_CELL_SIZE,
    TISSUE_ARMS,
    allocate_fixed_tissue,
)
from minicells.language_utility_skill_data import (  # noqa: E402
    SKILL_FAMILIES,
    batch_from_indices,
    generate_utility_skill_corpus,
    make_index_schedule,
)


proposal_utility._gated_replicator_activity = stable_gated_replicator_activity

CHECKPOINT_FORMAT = "minicells.capability-tissue-specificity-checkpoint.v1"
TRAINING_PROTOCOL = "minicells.capability-tissue-specificity-training.v1"
ARM_SIZES = {"one-cell": ONE_CELL_SIZE, "three-cell-chain": THREE_CELL_SIZE}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 020 capability-tissue-specificity replicate.")
    parser.add_argument("--replicate", type=int, choices=range(e019.N_REPLICATES), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _checkpoint_path(output_dir: Path, replicate: int, arm: str, family: str) -> Path:
    return output_dir / "checkpoints" / f"r{replicate}-{arm}-{family}.pt"


def _atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_specificity_checkpoint(path: Path, replicate: int, arm: str, family: str) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("training_protocol") != TRAINING_PROTOCOL:
        raise RuntimeError(f"Experiment 020 checkpoint protocol mismatch: {path}")
    if int(payload.get("replicate", -1)) != replicate or payload.get("arm") != arm or payload.get("family") != family:
        raise RuntimeError(f"Experiment 020 checkpoint identity mismatch: {path}")
    return payload


def _cpu_state_dict(model) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _localized_payload(state: LocalizedLearningState) -> dict[str, torch.Tensor]:
    return {
        "base_alive": state.base_alive.detach().cpu().clone(),
        "base_adjacency": state.base_adjacency.detach().cpu().clone(),
        "base_memory": state.base_memory.detach().cpu().clone(),
    }


def _localized_from_payload(payload: dict[str, torch.Tensor]) -> LocalizedLearningState:
    return LocalizedLearningState(
        base_alive=payload["base_alive"].clone(),
        base_adjacency=payload["base_adjacency"].clone(),
        base_memory=payload["base_memory"].clone(),
    )


def _language_nll(model, validation_stream, validation_starts, *, device: torch.device) -> float:
    return float(e016.evaluate_stream(model, validation_stream, validation_starts, variant="G", device=device)["nll"])


def _load_model_from_payload(vocab_size: int, payload: dict[str, Any], device: torch.device):
    model = e017.clone_from_state(vocab_size, payload["model_state"], device)
    state = _localized_from_payload(payload["localized_state"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, state


def train_or_restore_donor(
    arm: str,
    family: str,
    *,
    replicate: int,
    vocab_size: int,
    base_state: dict[str, torch.Tensor],
    validation_stream: torch.Tensor,
    validation_starts,
    output_dir: Path,
    device: torch.device,
):
    path = _checkpoint_path(output_dir, replicate, arm, family)
    cached = _load_specificity_checkpoint(path, replicate, arm, family)
    if cached is not None:
        model, state = _load_model_from_payload(vocab_size, cached, device)
        print({"replicate": replicate, "arm": arm, "family": family, "checkpoint_reuse": True})
        return model, state, cached["summary"]

    seed = e019.DONOR_SEED_BASE + 10_000 * replicate + 101 * SKILL_FAMILIES.index(family)
    train = generate_utility_skill_corpus(e019.DONOR_TRAIN_EXAMPLES, seed=seed, families=(family,))
    validation = generate_utility_skill_corpus(e019.DONOR_VALIDATION_EXAMPLES, seed=seed + 1, families=(family,))
    schedule = make_index_schedule(
        len(train.sequences),
        steps=e019.DONOR_STEPS,
        batch_size=e019.DONOR_BATCH_SIZE,
        seed=seed + 2,
    )
    model = e017.clone_from_state(vocab_size, base_state, device)
    state = LocalizedLearningState.capture(model)
    base_skill = e019.evaluate_skill(model, validation, device=device)
    base_language = _language_nll(model, validation_stream, validation_starts, device=device)
    mask = train.loss_mask.to(device)

    first_inputs, first_targets, _ = batch_from_indices(train, schedule[0], device=device)
    initial_probe = e019._probe(model, first_inputs, first_targets, mask)
    parent, newborn = allocate_fixed_tissue(
        model,
        state,
        initial_probe,
        tissue_size=ARM_SIZES[arm],
        step=0,
    )
    if len(newborn) != ARM_SIZES[arm]:
        raise RuntimeError("fixed tissue allocation produced the wrong number of newborn cells")

    model.freeze_genome()
    optimizer = torch.optim.AdamW([model.cell_memory], lr=e019.DONOR_LR, weight_decay=0.0)
    started = time.perf_counter()
    for step in range(1, e019.DONOR_STEPS + 1):
        model.train()
        inputs, targets, _ = batch_from_indices(train, schedule[step - 1], device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            result = model.forward_variable(inputs)
            main_loss = e019._masked_loss(result.output.logits, targets, mask)
            loss = main_loss + STABILITY_WEIGHT * result.stability_loss
        loss.backward()
        mask_to_newborn_gradients(model, state)
        torch.nn.utils.clip_grad_norm_([model.cell_memory], 1.0)
        optimizer.step()
    elapsed = time.perf_counter() - started

    model.eval()
    donor_skill = e019.evaluate_skill(model, validation, device=device)
    donor_language = _language_nll(model, validation_stream, validation_starts, device=device)
    saved_alive, saved_adjacency = set_newborn_tissue_active(model, state, False)
    ablated_skill = e019.evaluate_skill(model, validation, device=device)
    ablated_language = _language_nll(model, validation_stream, validation_starts, device=device)
    restore_structure(model, saved_alive, saved_adjacency)

    recipient = e017.clone_from_state(vocab_size, base_state, device)
    graft_localized_tissue(recipient, model, newborn)
    recipient_skill = e019.evaluate_skill(recipient, validation, device=device)
    recipient_language = _language_nll(recipient, validation_stream, validation_starts, device=device)

    improvement = base_skill - donor_skill
    causal_fraction = (ablated_skill - donor_skill) / improvement if abs(improvement) > 1e-9 else 0.0
    transplant_recovery = (base_skill - recipient_skill) / improvement if abs(improvement) > 1e-9 else 0.0
    summary = {
        "replicate": replicate,
        "arm": arm,
        "family": family,
        "parent": parent,
        "newborn_cells": json.dumps(newborn),
        "newborn_count": len(newborn),
        "edges": model.edge_count,
        "base_skill_nll": base_skill,
        "donor_skill_nll": donor_skill,
        "skill_improvement": improvement,
        "ablated_skill_nll": ablated_skill,
        "tissue_causal_fraction": causal_fraction,
        "recipient_skill_nll": recipient_skill,
        "transplant_recovery": transplant_recovery,
        "base_language_nll": base_language,
        "donor_language_nll": donor_language,
        "language_retention_ratio": donor_language / max(base_language, 1e-9),
        "ablated_language_nll": ablated_language,
        "recipient_language_nll": recipient_language,
        "base_memory_drift": float(state.base_memory_drift(model).cpu()),
        "elapsed_seconds": elapsed,
        "autonomous_structure_updates": 0,
    }
    _atomic_save(path, {
        "format": CHECKPOINT_FORMAT,
        "training_protocol": TRAINING_PROTOCOL,
        "replicate": replicate,
        "arm": arm,
        "family": family,
        "vocab_size": vocab_size,
        "model_state": _cpu_state_dict(model),
        "localized_state": _localized_payload(state),
        "summary": summary,
    })
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print({"replicate": replicate, "arm": arm, "family": family, "skill_improvement": improvement, "newborn": len(newborn)})
    del recipient
    return model, state, summary


@torch.no_grad()
def measure_full_utility(
    model,
    state: LocalizedLearningState,
    corpus,
    *,
    replicate: int,
    arm: str,
    input_family: str,
    candidate_family: str,
    device: torch.device,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    mask = corpus.loss_mask.to(device)
    for start in range(0, len(corpus.sequences), e019.UTILITY_BATCH_SIZE):
        stop = min(start + e019.UTILITY_BATCH_SIZE, len(corpus.sequences))
        inputs, targets, _ = batch_from_indices(corpus, torch.arange(start, stop), device=device)
        closed = proposal_utility.forward_with_fixed_recruitment(model, inputs, state, 0.0)
        full = proposal_utility.forward_with_fixed_recruitment(model, inputs, state, 1.0)
        loss0 = proposal_utility.per_example_masked_nll(closed.output.logits.float(), targets, mask).cpu()
        loss1 = proposal_utility.per_example_masked_nll(full.output.logits.float(), targets, mask).cpu()
        for local_index, example in enumerate(range(start, stop)):
            rows.append({
                "replicate": replicate,
                "arm": arm,
                "example": example,
                "input_family": input_family,
                "candidate_family": candidate_family,
                "matching_family": int(input_family == candidate_family),
                "loss_closed": float(loss0[local_index]),
                "loss_full": float(loss1[local_index]),
                "full_value": float(loss0[local_index] - loss1[local_index]),
            })
    return rows


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 020 requires CUDA")
    device = torch.device("cuda:0")
    replicate = int(args.replicate)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _, validation_stream, manifest = e016.load_cache(args.cache_dir.resolve())
    vocab_size = int(manifest["vocab_size_actual"])
    source = load_checkpoint(
        phase1_path(args.source_checkpoint_dir.resolve(), replicate),
        kind="phase1",
        replicate=replicate,
    )
    if source is None:
        raise FileNotFoundError(f"missing stable-019 Phase-1 checkpoint for replicate {replicate}")
    base_state = source["base_state"]
    validation_starts = [tuple(int(v) for v in row) for row in source["validation_starts"]]

    summaries: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    for arm in TISSUE_ARMS:
        donors = {}
        for family in SKILL_FAMILIES:
            model, state, summary = train_or_restore_donor(
                arm,
                family,
                replicate=replicate,
                vocab_size=vocab_size,
                base_state=base_state,
                validation_stream=validation_stream,
                validation_starts=validation_starts,
                output_dir=output_dir,
                device=device,
            )
            donors[family] = (model, state)
            summaries.append(summary)

        for family_index, input_family in enumerate(SKILL_FAMILIES):
            corpus = generate_utility_skill_corpus(
                e019.UTILITY_EXAMPLES_PER_FAMILY,
                seed=e019.UTILITY_SEED_BASE + 10_000 * replicate + family_index,
                families=(input_family,),
            )
            for candidate_family in SKILL_FAMILIES:
                model, state = donors[candidate_family]
                observations.extend(measure_full_utility(
                    model,
                    state,
                    corpus,
                    replicate=replicate,
                    arm=arm,
                    input_family=input_family,
                    candidate_family=candidate_family,
                    device=device,
                ))
        for model, _ in donors.values():
            del model
        torch.cuda.empty_cache()

    pd.DataFrame(summaries).to_csv(output_dir / f"r{replicate}-donor-summary.csv", index=False)
    pd.DataFrame(observations).to_csv(output_dir / f"r{replicate}-specificity-observations.csv.gz", index=False, compression="gzip")
    worker = {
        "format": "minicells.capability-tissue-specificity-worker.v1",
        "experiment": "020",
        "replicate": replicate,
        "gpu": torch.cuda.get_device_name(0),
        "arms": list(TISSUE_ARMS),
        "families": list(SKILL_FAMILIES),
        "training_protocol": TRAINING_PROTOCOL,
        "donor_steps": e019.DONOR_STEPS,
        "autonomous_structure_updates": False,
        "observations": len(observations),
        "donor_checkpoints": len(TISSUE_ARMS) * len(SKILL_FAMILIES),
    }
    (output_dir / f"r{replicate}-worker.json").write_text(json.dumps(worker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
