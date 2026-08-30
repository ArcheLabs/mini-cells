from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_language_growing_organism_worker as e016  # noqa: E402
import run_language_localized_learning_worker as e017  # noqa: E402
from minicells.language_growing_organism import STABILITY_WEIGHT, make_structural_probe  # noqa: E402
from minicells.language_localized_learning import (  # noqa: E402
    LocalizedGrowthController,
    LocalizedLearningState,
    mask_to_newborn_gradients,
    restore_structure,
    set_newborn_tissue_active,
)
from minicells.language_proposal_utility import (  # noqa: E402
    BOUNDARY_FEATURES,
    LOCAL_FEATURES,
    UTILITY_EPSILON,
    measure_proposal_batch,
)
from minicells.language_utility_skill_data import (  # noqa: E402
    SKILL_FAMILIES,
    batch_from_indices,
    generate_utility_skill_corpus,
    make_index_schedule,
)


N_REPLICATES = 3
DONOR_STEPS = 200
DONOR_BATCH_SIZE = 64
DONOR_LR = 3e-3
DONOR_STRUCTURE_INTERVAL = 50
DONOR_TRAIN_EXAMPLES = 4096
DONOR_VALIDATION_EXAMPLES = 512
UTILITY_EXAMPLES_PER_FAMILY = 256
UTILITY_BATCH_SIZE = 64
RANDOM_CONTROL = "RANDOM"
DONOR_SEED_BASE = 119_019
UTILITY_SEED_BASE = 219_019


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 019 proposal-utility replicate.")
    parser.add_argument("--replicate", type=int, choices=range(N_REPLICATES), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _masked_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, mask, :].reshape(-1, logits.shape[-1]), targets[:, mask].reshape(-1))


@torch.no_grad()
def evaluate_skill(model, corpus, *, device: torch.device) -> float:
    model.eval()
    mask = corpus.loss_mask.to(device)
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(corpus.sequences), 64):
        stop = min(start + 64, len(corpus.sequences))
        inputs, targets, _ = batch_from_indices(corpus, torch.arange(start, stop), device=device)
        logits = model.forward_variable(inputs).output.logits.float()
        selected_logits = logits[:, mask, :]
        selected_targets = targets[:, mask]
        total_loss += float(F.cross_entropy(
            selected_logits.reshape(-1, logits.shape[-1]),
            selected_targets.reshape(-1),
            reduction="sum",
        ))
        total_tokens += selected_targets.numel()
    return total_loss / max(1, total_tokens)


def _probe(model, inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    pieces_x = torch.chunk(inputs, min(4, inputs.shape[0]), dim=0)
    pieces_y = torch.chunk(targets, min(4, targets.shape[0]), dim=0)

    def loss_fn(logits: torch.Tensor, probe_targets: torch.Tensor) -> torch.Tensor:
        return _masked_loss(logits, probe_targets, mask)

    return make_structural_probe(
        model,
        [(x, y) for x, y in zip(pieces_x, pieces_y) if len(x)],
        loss_fn=loss_fn,
    )


def train_donor(
    family: str,
    *,
    replicate: int,
    vocab_size: int,
    base_state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[object, LocalizedLearningState, dict[str, object], list[dict]]:
    seed = DONOR_SEED_BASE + 10_000 * replicate + 101 * SKILL_FAMILIES.index(family)
    train = generate_utility_skill_corpus(DONOR_TRAIN_EXAMPLES, seed=seed, families=(family,))
    validation = generate_utility_skill_corpus(DONOR_VALIDATION_EXAMPLES, seed=seed + 1, families=(family,))
    schedule = make_index_schedule(
        len(train.sequences),
        steps=DONOR_STEPS,
        batch_size=DONOR_BATCH_SIZE,
        seed=seed + 2,
    )
    model = e017.clone_from_state(vocab_size, base_state, device)
    localized_state = LocalizedLearningState.capture(model)
    base_skill = evaluate_skill(model, validation, device=device)
    mask = train.loss_mask.to(device)

    first_inputs, first_targets, _ = batch_from_indices(train, schedule[0], device=device)
    initial_probe = _probe(model, first_inputs, first_targets, mask)
    controller = LocalizedGrowthController(localized_state, max_newborns=1)
    first_event = controller.allocate_initial(model, initial_probe, step=0)
    events = [{"replicate": replicate, "family": family, **first_event.as_dict()}]
    if len(localized_state.newborn_cells(model)) != 1:
        raise RuntimeError("019 donor initialization must allocate exactly one newborn")

    model.freeze_genome()
    optimizer = torch.optim.AdamW([model.cell_memory], lr=DONOR_LR, weight_decay=0.0)
    started = time.perf_counter()
    for step in range(1, DONOR_STEPS + 1):
        model.train()
        inputs, targets, _ = batch_from_indices(train, schedule[step - 1], device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            result = model.forward_variable(inputs)
            main_loss = _masked_loss(result.output.logits, targets, mask)
            loss = main_loss + STABILITY_WEIGHT * result.stability_loss
        loss.backward()
        mask_to_newborn_gradients(model, localized_state)
        torch.nn.utils.clip_grad_norm_([model.cell_memory], 1.0)
        optimizer.step()

        if step % DONOR_STRUCTURE_INTERVAL == 0:
            model.eval()
            structural_probe = _probe(model, inputs, targets, mask)
            for event in controller.apply(model, structural_probe, step=step):
                events.append({"replicate": replicate, "family": family, **event.as_dict()})

    elapsed = time.perf_counter() - started
    donor_skill = evaluate_skill(model, validation, device=device)
    newborn = localized_state.newborn_cells(model)
    if len(newborn) != 1:
        raise RuntimeError(f"019 requires exactly one newborn donor tissue, got {newborn}")
    child = newborn[0]
    parent = int(model.parent[child].item())
    saved_alive, saved_adjacency = set_newborn_tissue_active(model, localized_state, False)
    ablated_skill = evaluate_skill(model, validation, device=device)
    restore_structure(model, saved_alive, saved_adjacency)
    improvement = base_skill - donor_skill
    causal_fraction = (ablated_skill - donor_skill) / improvement if abs(improvement) > 1e-9 else 0.0
    summary = {
        "replicate": replicate,
        "family": family,
        "candidate_kind": "trained",
        "parent": parent,
        "child": child,
        "base_skill_nll": base_skill,
        "donor_skill_nll": donor_skill,
        "skill_improvement": improvement,
        "ablated_skill_nll": ablated_skill,
        "tissue_causal_fraction": causal_fraction,
        "newborn_count": 1,
        "base_memory_drift": float(localized_state.base_memory_drift(model).cpu()),
        "edges": model.edge_count,
        "elapsed_seconds": elapsed,
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, localized_state, summary, events


def make_random_control(
    *,
    replicate: int,
    vocab_size: int,
    base_state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[object, LocalizedLearningState, dict[str, object], list[dict]]:
    seed = DONOR_SEED_BASE + 10_000 * replicate + 9_999
    corpus = generate_utility_skill_corpus(128, seed=seed, families=(SKILL_FAMILIES[0],))
    model = e017.clone_from_state(vocab_size, base_state, device)
    state = LocalizedLearningState.capture(model)
    mask = corpus.loss_mask.to(device)
    indices = torch.arange(min(DONOR_BATCH_SIZE, len(corpus.sequences)))
    inputs, targets, _ = batch_from_indices(corpus, indices, device=device)
    probe = _probe(model, inputs, targets, mask)
    controller = LocalizedGrowthController(state, max_newborns=1)
    event = controller.allocate_initial(model, probe, step=0)
    newborn = state.newborn_cells(model)
    if len(newborn) != 1:
        raise RuntimeError("random control must contain exactly one newborn")
    child = newborn[0]
    parent = int(model.parent[child].item())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    summary = {
        "replicate": replicate,
        "family": RANDOM_CONTROL,
        "candidate_kind": "untrained-control",
        "parent": parent,
        "child": child,
        "base_skill_nll": None,
        "donor_skill_nll": None,
        "skill_improvement": 0.0,
        "ablated_skill_nll": None,
        "tissue_causal_fraction": 0.0,
        "newborn_count": 1,
        "base_memory_drift": float(state.base_memory_drift(model).cpu()),
        "edges": model.edge_count,
        "elapsed_seconds": 0.0,
    }
    return model, state, summary, [{"replicate": replicate, "family": RANDOM_CONTROL, **event.as_dict()}]


def measure_candidate(
    input_family: str,
    candidate_family: str,
    model,
    localized_state: LocalizedLearningState,
    corpus,
    *,
    replicate: int,
    device: torch.device,
) -> list[dict]:
    rows: list[dict] = []
    mask = corpus.loss_mask.to(device)
    for start in range(0, len(corpus.sequences), UTILITY_BATCH_SIZE):
        stop = min(start + UTILITY_BATCH_SIZE, len(corpus.sequences))
        indices = torch.arange(start, stop)
        inputs, targets, _ = batch_from_indices(corpus, indices, device=device)
        measured = measure_proposal_batch(
            model,
            localized_state,
            inputs,
            targets,
            mask,
            epsilon=UTILITY_EPSILON,
        )
        tensors = {name: value.detach().float().cpu() for name, value in measured.items()}
        for local_index, example_index in enumerate(range(start, stop)):
            row = {
                "replicate": replicate,
                "example": example_index,
                "input_family": input_family,
                "candidate_family": candidate_family,
                "candidate_kind": "untrained-control" if candidate_family == RANDOM_CONTROL else "trained",
                "matching_family": int(candidate_family == input_family),
                "oracle_gradient": float(tensors["oracle_gradient"][local_index]),
                "oracle_fd": float(tensors["oracle_fd"][local_index]),
                "loss_closed": float(tensors["loss_closed"][local_index]),
                "loss_probe": float(tensors["loss_probe"][local_index]),
            }
            for feature in BOUNDARY_FEATURES:
                row[feature] = float(tensors[feature][local_index])
            rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 019 requires CUDA")
    device = torch.device("cuda:0")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_stream, validation_stream, manifest = e016.load_cache(args.cache_dir.resolve())
    vocab_size = int(manifest["vocab_size_actual"])
    replicate = args.replicate

    phase1, base_state, checkpoints, phase1_events, phase1_wall, validation_starts = e017.train_phase1(
        replicate=replicate,
        train_stream=train_stream,
        validation_stream=validation_stream,
        vocab_size=vocab_size,
        device=device,
    )
    del phase1
    torch.cuda.empty_cache()

    donor_models: dict[str, tuple[object, LocalizedLearningState]] = {}
    donor_summaries: list[dict] = []
    donor_events: list[dict] = []
    for family in SKILL_FAMILIES:
        model, state, summary, events = train_donor(
            family,
            replicate=replicate,
            vocab_size=vocab_size,
            base_state=base_state,
            device=device,
        )
        donor_models[family] = (model, state)
        donor_summaries.append(summary)
        donor_events.extend(events)
        print({"replicate": replicate, "donor": family, "skill_improvement": summary["skill_improvement"], "parent": summary["parent"]})

    random_model, random_state, random_summary, random_events = make_random_control(
        replicate=replicate,
        vocab_size=vocab_size,
        base_state=base_state,
        device=device,
    )
    donor_models[RANDOM_CONTROL] = (random_model, random_state)
    donor_summaries.append(random_summary)
    donor_events.extend(random_events)

    observations: list[dict] = []
    for family_index, input_family in enumerate(SKILL_FAMILIES):
        corpus = generate_utility_skill_corpus(
            UTILITY_EXAMPLES_PER_FAMILY,
            seed=UTILITY_SEED_BASE + 10_000 * replicate + family_index,
            families=(input_family,),
        )
        for candidate_family, (model, state) in donor_models.items():
            observations.extend(measure_candidate(
                input_family,
                candidate_family,
                model,
                state,
                corpus,
                replicate=replicate,
                device=device,
            ))

    pd.DataFrame(checkpoints).to_csv(output_dir / f"r{replicate}-phase1-checkpoints.csv", index=False)
    pd.DataFrame(phase1_events).to_csv(output_dir / f"r{replicate}-phase1-events.csv", index=False)
    pd.DataFrame(donor_summaries).to_csv(output_dir / f"r{replicate}-donor-summary.csv", index=False)
    pd.DataFrame(donor_events).to_csv(output_dir / f"r{replicate}-donor-events.csv", index=False)
    pd.DataFrame(observations).to_csv(output_dir / f"r{replicate}-utility-observations.csv", index=False)

    worker = {
        "format": "minicells.proposal-utility-worker.v1",
        "experiment": "019",
        "replicate": replicate,
        "gpu": torch.cuda.get_device_name(0),
        "phase1_wall_seconds": phase1_wall,
        "phase1_alive_cells": int(base_state["alive_mask"].sum().item()),
        "families": list(SKILL_FAMILIES),
        "candidate_families": [*SKILL_FAMILIES, RANDOM_CONTROL],
        "donor_steps": DONOR_STEPS,
        "utility_examples_per_family": UTILITY_EXAMPLES_PER_FAMILY,
        "utility_epsilon": UTILITY_EPSILON,
        "local_features": list(LOCAL_FEATURES),
        "boundary_features": list(BOUNDARY_FEATURES),
        "observations": len(observations),
    }
    (output_dir / f"r{replicate}-worker.json").write_text(
        json.dumps(worker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
