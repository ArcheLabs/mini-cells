from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "src").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))

from minicells.language_data import batch_from_starts, fixed_validation_starts  # noqa: E402
from minicells.language_growing_organism import (  # noqa: E402
    INITIAL_CELLS,
    MAX_CELLS,
    STABILITY_WEIGHT,
    StructuralController,
    build_cellular_model,
    build_parameter_matched_small_transformer,
    make_structural_probe,
)
from minicells.language_models import count_parameters  # noqa: E402
from minicells.language_skill_data import batch_from_indices, generate_skill_corpus, make_index_schedule  # noqa: E402


VARIANTS = ("T", "F", "G")
BATCH_SIZE = 8
SEQUENCE_LENGTH = 128
TRAIN_STEPS = 1000
TOKENS_PER_STEP = BATCH_SIZE * SEQUENCE_LENGTH
BUDGET_TOKENS = TRAIN_STEPS * TOKENS_PER_STEP
CHECKPOINT_STEPS = (100, 250, 500, 750, 1000)
BASE_LR = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 50
STRUCTURE_WARMUP = 200
STRUCTURE_INTERVAL = 100
LOCAL_SKILL = "REVERSE_INC"
LOCAL_STEPS = 200
LOCAL_BATCH_SIZE = 64
LOCAL_LR = 3e-3
LOCAL_STRUCTURE_INTERVAL = 50
N_REPLICATES = 3
MODEL_SEED_BASE = 91_016
SCHEDULE_SEED_BASE = 31_016
SKILL_SEED_BASE = 71_016


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 016 model.")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--replicate", type=int, choices=range(N_REPLICATES), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def lr_multiplier(step: int) -> float:
    if step <= WARMUP_STEPS:
        return step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, TRAIN_STEPS - WARMUP_STEPS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def load_cache(cache_dir: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    train = torch.load(cache_dir / "train-tokens.pt", map_location="cpu")
    validation = torch.load(cache_dir / "validation-tokens.pt", map_location="cpu")
    manifest = json.loads((cache_dir / "corpus-manifest.json").read_text(encoding="utf-8"))
    return train, validation, manifest


def make_schedule(stream_length: int, *, seed: int) -> tuple[tuple[int, ...], ...]:
    rng = np.random.default_rng(seed)
    high = stream_length - SEQUENCE_LENGTH - 1
    values = rng.integers(0, high, size=(TRAIN_STEPS, BATCH_SIZE), endpoint=False)
    return tuple(tuple(int(value) for value in row) for row in values)


def build_model(variant: str, *, vocab_size: int, seed: int):
    torch.manual_seed(seed)
    cellular = build_cellular_model(vocab_size, "G" if variant == "G" else "F")
    target = count_parameters(cellular)
    if variant == "T":
        torch.manual_seed(seed)
        model, metadata = build_parameter_matched_small_transformer(vocab_size, target)
        return model, {"target_cellular_parameters": target, **metadata}
    return cellular, {
        "target_cellular_parameters": target,
        "parameters": target,
        "initial_cells": INITIAL_CELLS,
        "max_cells": MAX_CELLS,
    }


@torch.no_grad()
def evaluate_stream(model, token_stream: torch.Tensor, starts, *, variant: str, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    for batch_starts in starts:
        inputs, targets = batch_from_starts(token_stream, batch_starts, SEQUENCE_LENGTH, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(inputs).logits if variant == "T" else model.forward_variable(inputs).output.logits
        total_loss += float(F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"))
        total_tokens += targets.numel()
        correct += int((logits.argmax(dim=-1) == targets).sum())
    nll = total_loss / max(1, total_tokens)
    return {"nll": nll, "ppl": math.exp(min(20.0, nll)), "token_accuracy": correct / max(1, total_tokens)}


@torch.no_grad()
def organism_snapshot(model, inputs: torch.Tensor, *, step: int, phase: str) -> tuple[list[dict], list[dict]]:
    result = model.forward_variable(inputs, collect_observability=True)
    diagnostics = result.diagnostics
    assert diagnostics is not None
    activity = diagnostics.activity.float().mean(dim=(0, 1)).cpu()
    alive = diagnostics.alive_indices.cpu().tolist()
    cells = []
    for local_index, cell in enumerate(alive):
        cells.append({
            "phase": phase,
            "step": step,
            "cell": int(cell),
            "alive": 1,
            "activity": float(activity[local_index]),
            "memory_norm": float(model.cell_memory[cell].detach().float().norm().cpu()),
            "in_degree": int(model.adjacency[cell, model.alive_mask].sum().item()),
            "out_degree": int(model.adjacency[model.alive_mask, cell].sum().item()),
            "parent": int(model.parent[cell].item()),
            "birth_step": int(model.birth_step[cell].item()),
        })
    edges = []
    for receiver in alive:
        for source in alive:
            if bool(model.adjacency[receiver, source]):
                edges.append({
                    "phase": phase,
                    "step": step,
                    "receiver": int(receiver),
                    "source": int(source),
                    "protected": int(bool(model.protected_edges[receiver, source])),
                })
    return cells, edges


def structural_probe_from_batch(model, inputs: torch.Tensor, targets: torch.Tensor):
    pieces_x = torch.chunk(inputs, min(4, inputs.shape[0]), dim=0)
    pieces_y = torch.chunk(targets, min(4, targets.shape[0]), dim=0)
    return make_structural_probe(model, [(x, y) for x, y in zip(pieces_x, pieces_y) if len(x)], loss_fn=ce_loss)


@torch.no_grad()
def evaluate_structure_interventions(model, validation_stream: torch.Tensor, validation_starts, *, replicate: int, device: torch.device) -> list[dict]:
    normal = evaluate_stream(model, validation_stream, validation_starts, variant="G", device=device)["nll"]
    saved_alive = model.alive_mask.clone()
    saved_adjacency = model.adjacency.clone()
    rows = []

    model.adjacency.copy_(saved_adjacency & model.protected_edges)
    learned_off = evaluate_stream(model, validation_stream, validation_starts, variant="G", device=device)["nll"]
    rows.append({"replicate": replicate, "intervention": "learned_edges_off", "normal_nll": normal, "altered_nll": learned_off, "delta_nll": learned_off - normal})

    model.alive_mask.zero_()
    model.alive_mask[: model.initial_cells] = True
    model.adjacency.zero_()
    for left in range(model.initial_cells - 1):
        right = left + 1
        model.adjacency[left, right] = True
        model.adjacency[right, left] = True
    initial = evaluate_stream(model, validation_stream, validation_starts, variant="G", device=device)["nll"]
    rows.append({"replicate": replicate, "intervention": "initial_organism", "normal_nll": normal, "altered_nll": initial, "delta_nll": initial - normal})

    model.alive_mask.copy_(saved_alive)
    model.adjacency.copy_(saved_adjacency)
    return rows


@torch.no_grad()
def evaluate_skill(model, corpus, *, device: torch.device) -> float:
    model.eval()
    mask = corpus.loss_mask.to(device)
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(corpus.sequences), 64):
        stop = min(start + 64, len(corpus.sequences))
        inputs, targets, _ = batch_from_indices(corpus, torch.arange(start, stop), device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model.forward_variable(inputs).output.logits
        selected_logits = logits[:, mask, :].float()
        selected_targets = targets[:, mask]
        total_loss += float(F.cross_entropy(selected_logits.reshape(-1, logits.shape[-1]), selected_targets.reshape(-1), reduction="sum"))
        total_tokens += selected_targets.numel()
    return total_loss / max(1, total_tokens)


def add_ancestor_closure(model, selected: list[int]) -> list[int]:
    closure = set(int(cell) for cell in selected)
    for cell in list(closure):
        current = cell
        for _ in range(model.max_cells):
            parent = int(model.parent[current].item())
            if parent < 0 or parent == current:
                break
            closure.add(parent)
            current = parent
            if current == 0:
                break
    closure.discard(0)
    return sorted(closure)


def local_skill_phase(model, base_state: dict[str, torch.Tensor], validation_stream: torch.Tensor, validation_starts, *, replicate: int, device: torch.device) -> tuple[list[dict], list[dict], list[dict], list[dict], float]:
    started = time.perf_counter()
    skill_seed = SKILL_SEED_BASE + 1000 * replicate
    train = generate_skill_corpus(4096, seed=skill_seed, tasks=(LOCAL_SKILL,))
    validation = generate_skill_corpus(512, seed=skill_seed + 1, tasks=(LOCAL_SKILL,))
    base_skill = evaluate_skill(model, validation, device=device)
    base_language = evaluate_stream(model, validation_stream, validation_starts, variant="G", device=device)["nll"]
    base_memory = base_state["cell_memory"].detach().clone().to(device)
    base_alive = base_state["alive_mask"].detach().clone().to(device)
    base_adjacency = base_state["adjacency"].detach().clone().to(device)

    model.freeze_genome()
    optimizer = torch.optim.AdamW([model.cell_memory], lr=LOCAL_LR, weight_decay=0.0)
    controller = StructuralController(max_cells=model.max_cells)
    schedule = make_index_schedule(len(train.sequences), steps=LOCAL_STEPS, batch_size=LOCAL_BATCH_SIZE, seed=skill_seed + 2)
    mask = train.loss_mask.to(device)
    rows = []
    event_rows = []
    for step in range(1, LOCAL_STEPS + 1):
        inputs, targets, _ = batch_from_indices(train, schedule[step - 1], device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            result = model.forward_variable(inputs)
            logits = result.output.logits
            main_loss = F.cross_entropy(logits[:, mask, :].reshape(-1, logits.shape[-1]), targets[:, mask].reshape(-1))
            loss = main_loss + STABILITY_WEIGHT * result.stability_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([model.cell_memory], 1.0)
        optimizer.step()
        if step % LOCAL_STRUCTURE_INTERVAL == 0:
            def masked_loss(probe_logits: torch.Tensor, probe_targets: torch.Tensor) -> torch.Tensor:
                return F.cross_entropy(probe_logits[:, mask, :].reshape(-1, probe_logits.shape[-1]), probe_targets[:, mask].reshape(-1))
            pieces_x = torch.chunk(inputs, 4, dim=0)
            pieces_y = torch.chunk(targets, 4, dim=0)
            probe = make_structural_probe(model, list(zip(pieces_x, pieces_y)), loss_fn=masked_loss)
            for event in controller.apply(model, probe, step=step):
                event_rows.append({"phase": "local-skill", **event.as_dict()})
        if step in (1, 25, 50, 100, 150, 200):
            rows.append({
                "replicate": replicate,
                "step": step,
                "skill_nll": evaluate_skill(model, validation, device=device),
                "language_nll": evaluate_stream(model, validation_stream, validation_starts, variant="G", device=device)["nll"],
                "alive_cells": model.alive_count,
                "edges": model.edge_count,
            })

    donor_skill = evaluate_skill(model, validation, device=device)
    donor_language = evaluate_stream(model, validation_stream, validation_starts, variant="G", device=device)["nll"]
    memory_delta = (model.cell_memory.detach() - base_memory).float().norm(dim=-1)
    candidates = [int(i) for i in torch.nonzero(model.alive_mask, as_tuple=False).flatten().cpu().tolist() if int(i) != 0]
    ranked = sorted(candidates, key=lambda cell: float(memory_delta[cell]), reverse=True)
    total = float(memory_delta[candidates].sum()) if candidates else 0.0
    selected: list[int] = []
    accumulated = 0.0
    for cell in ranked:
        selected.append(cell)
        accumulated += float(memory_delta[cell])
        if total <= 1e-12 or accumulated >= 0.80 * total:
            break
    newborn = [cell for cell in range(model.max_cells) if bool(model.alive_mask[cell]) and not bool(base_alive[cell])]
    selected = add_ancestor_closure(model, sorted(set(selected + newborn)))
    if not selected and ranked:
        selected = add_ancestor_closure(model, [ranked[0]])

    recipient = build_cellular_model(model.token_embedding.num_embeddings, "G").to(device)
    recipient.load_state_dict(copy.deepcopy(base_state))
    recipient.copy_tissue_from(model, selected)
    recipient_skill = evaluate_skill(recipient, validation, device=device)
    recipient_language = evaluate_stream(recipient, validation_stream, validation_starts, variant="G", device=device)["nll"]
    denominator = base_skill - donor_skill
    recovery = (base_skill - recipient_skill) / denominator if abs(denominator) > 1e-9 else 0.0
    transplant = [{
        "replicate": replicate,
        "skill": LOCAL_SKILL,
        "base_skill_nll": base_skill,
        "donor_skill_nll": donor_skill,
        "recipient_skill_nll": recipient_skill,
        "skill_improvement": base_skill - donor_skill,
        "transplant_recovery": recovery,
        "base_language_nll": base_language,
        "donor_language_nll": donor_language,
        "recipient_language_nll": recipient_language,
        "donor_language_ratio": donor_language / base_language,
        "recipient_language_ratio": recipient_language / base_language,
        "selected_cells": json.dumps(selected),
        "selected_cell_count": len(selected),
        "donor_alive_cells": model.alive_count,
        "recipient_alive_cells": recipient.alive_count,
    }]
    localization = [{
        "replicate": replicate,
        "cell": cell,
        "memory_delta_norm": float(memory_delta[cell].cpu()),
        "selected": int(cell in selected),
        "base_alive": int(bool(base_alive[cell])),
        "donor_alive": int(bool(model.alive_mask[cell])),
        "newborn": int(cell in newborn),
        "base_in_degree": int(base_adjacency[cell].sum().item()),
        "donor_in_degree": int(model.adjacency[cell].sum().item()),
    } for cell in range(model.max_cells)]
    synchronize(device)
    return rows, event_rows, transplant, localization, time.perf_counter() - started


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 016 worker requires CUDA")
    device = torch.device("cuda:0")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_stream, validation_stream, manifest = load_cache(args.cache_dir.resolve())
    vocab_size = int(manifest["vocab_size_actual"])
    replicate = args.replicate
    variant = args.variant
    run = f"r{replicate}-{variant}"
    model_seed = MODEL_SEED_BASE + 1000 * replicate
    schedule_seed = SCHEDULE_SEED_BASE + 1000 * replicate
    model, model_config = build_model(variant, vocab_size=vocab_size, seed=model_seed)
    parameters = count_parameters(model)
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    model = model.to(device)
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    scaler = make_scaler()
    schedule = make_schedule(len(train_stream), seed=schedule_seed)
    validation_starts = fixed_validation_starts(len(validation_stream), batches=16, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH, seed=51_016 + replicate)
    controller = StructuralController(max_cells=MAX_CELLS) if variant == "G" else None
    checkpoints: list[dict] = []
    structural_rows: list[dict] = []
    cell_rows: list[dict] = []
    edge_rows: list[dict] = []
    probe_rows: list[dict] = []
    core_elapsed = 0.0
    language_wall_started = time.perf_counter()
    print({"run": run, "gpu": torch.cuda.get_device_name(0), "parameters": parameters, **model_config})

    for step in range(1, TRAIN_STEPS + 1):
        model.train()
        lr = BASE_LR * lr_multiplier(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, targets = batch_from_starts(train_stream, schedule[step - 1], SEQUENCE_LENGTH, device)
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            if variant == "T":
                logits = model(inputs).logits
                stability = logits.new_zeros(())
            else:
                result = model.forward_variable(inputs)
                logits = result.output.logits
                stability = result.stability_loss
            main_loss = ce_loss(logits, targets)
            loss = main_loss + (STABILITY_WEIGHT * stability if variant != "T" else 0.0)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        scaler.step(optimizer)
        scaler.update()
        synchronize(device)
        core_elapsed += time.perf_counter() - started

        if variant == "G" and step >= STRUCTURE_WARMUP and step % STRUCTURE_INTERVAL == 0:
            model.eval()
            probe = structural_probe_from_batch(model, inputs, targets)
            alive = model.alive_mask.detach().cpu()
            candidate = alive[:, None] & alive[None, :]
            candidate.fill_diagonal_(False)
            utility = probe.edge_utility
            for receiver in torch.nonzero(alive, as_tuple=False).flatten().tolist():
                probe_rows.append({
                    "replicate": replicate,
                    "step": step,
                    "cell": int(receiver),
                    "pressure": float(probe.pressure[receiver]),
                    "conflict": float(probe.conflict[receiver]),
                    "best_incoming_utility": float(utility[receiver][candidate[receiver]].max()) if bool(candidate[receiver].any()) else 0.0,
                })
            assert controller is not None
            for event in controller.apply(model, probe, step=step):
                structural_rows.append({"phase": "language", "replicate": replicate, **event.as_dict()})

        if step in CHECKPOINT_STEPS:
            metrics = evaluate_stream(model, validation_stream, validation_starts, variant=variant, device=device)
            checkpoint = {
                "run": run,
                "replicate": replicate,
                "variant": variant,
                "step": step,
                "tokens": step * TOKENS_PER_STEP,
                "train_loss": float(loss.detach()),
                "main_loss": float(main_loss.detach()),
                "stability_loss": float(stability.detach()),
                "validation_nll": metrics["nll"],
                "validation_ppl": metrics["ppl"],
                "validation_token_accuracy": metrics["token_accuracy"],
                "core_elapsed_seconds": core_elapsed,
                "core_seconds_per_million_tokens": core_elapsed / ((step * TOKENS_PER_STEP) / 1_000_000),
                "grad_norm": grad_norm,
                "learning_rate": lr,
                "alive_cells": model.alive_count if variant != "T" else None,
                "edges": model.edge_count if variant != "T" else None,
            }
            checkpoints.append(checkpoint)
            if variant != "T":
                snapshot_cells, snapshot_edges = organism_snapshot(model, inputs[:2], step=step, phase="language")
                for row in snapshot_cells:
                    row.update({"replicate": replicate, "variant": variant})
                for row in snapshot_edges:
                    row.update({"replicate": replicate, "variant": variant})
                cell_rows.extend(snapshot_cells)
                edge_rows.extend(snapshot_edges)
            print(checkpoint)

    final_metrics = evaluate_stream(model, validation_stream, validation_starts, variant=variant, device=device)
    synchronize(device)
    language_wall_seconds = time.perf_counter() - language_wall_started
    phase1_alive = model.alive_count if variant != "T" else None
    phase1_edges = model.edge_count if variant != "T" else None
    interventions: list[dict] = []
    local_rows: list[dict] = []
    transplant_rows: list[dict] = []
    localization_rows: list[dict] = []
    local_wall_seconds = 0.0
    if variant == "G":
        interventions = evaluate_structure_interventions(model, validation_stream, validation_starts, replicate=replicate, device=device)
        base_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        local_rows, local_events, transplant_rows, localization_rows, local_wall_seconds = local_skill_phase(model, base_state, validation_stream, validation_starts, replicate=replicate, device=device)
        structural_rows.extend({"replicate": replicate, **row} for row in local_events)

    pd.DataFrame(checkpoints).to_csv(output_dir / f"{run}-checkpoints.csv", index=False)
    pd.DataFrame(structural_rows).to_csv(output_dir / f"{run}-structural-events.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(output_dir / f"{run}-cells.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(output_dir / f"{run}-edges.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(output_dir / f"{run}-structural-probes.csv", index=False)
    pd.DataFrame(interventions).to_csv(output_dir / f"{run}-interventions.csv", index=False)
    pd.DataFrame(local_rows).to_csv(output_dir / f"{run}-local-learning.csv", index=False)
    pd.DataFrame(transplant_rows).to_csv(output_dir / f"{run}-transplantation.csv", index=False)
    pd.DataFrame(localization_rows).to_csv(output_dir / f"{run}-skill-localization.csv", index=False)

    peak = int(torch.cuda.max_memory_allocated())
    worker = {
        "format": "minicells.growing-cellular-lm-worker.v1",
        "run": run,
        "replicate": replicate,
        "variant": variant,
        "parameters": parameters,
        "model_config": model_config,
        "model_seed": model_seed,
        "schedule_seed": schedule_seed,
        "tokens": BUDGET_TOKENS,
        "training_elapsed_seconds": core_elapsed,
        "seconds_per_million_tokens": core_elapsed / (BUDGET_TOKENS / 1_000_000),
        "language_wall_seconds": language_wall_seconds,
        "wall_seconds_per_million_tokens": language_wall_seconds / (BUDGET_TOKENS / 1_000_000),
        "local_wall_seconds": local_wall_seconds,
        "peak_vram_bytes": peak,
        "final_nll": final_metrics["nll"],
        "final_ppl": final_metrics["ppl"],
        "final_token_accuracy": final_metrics["token_accuracy"],
        "phase1_alive_cells": phase1_alive,
        "phase1_edges": phase1_edges,
        "language_structural_events": sum(1 for row in structural_rows if row.get("phase") == "language"),
        "interventions": interventions,
        "local_learning": transplant_rows[0] if transplant_rows else None,
    }
    (output_dir / f"{run}-worker.json").write_text(json.dumps(worker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
