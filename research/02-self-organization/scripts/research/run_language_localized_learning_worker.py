from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "src").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_language_growing_organism_worker as e016  # noqa: E402
from minicells.language_growing_organism import STABILITY_WEIGHT, StructuralController, build_cellular_model, make_structural_probe  # noqa: E402
from minicells.language_localized_learning import (  # noqa: E402
    LocalizedGrowthController,
    LocalizedLearningState,
    graft_localized_tissue,
    mask_to_newborn_gradients,
    restore_structure,
    set_newborn_tissue_active,
)
from minicells.language_models import count_parameters  # noqa: E402
from minicells.language_skill_data import batch_from_indices, generate_skill_corpus, make_index_schedule  # noqa: E402


N_REPLICATES = 3
POLICIES = ("B", "L")
LOCAL_SKILL = e016.LOCAL_SKILL
LOCAL_STEPS = e016.LOCAL_STEPS
LOCAL_BATCH_SIZE = e016.LOCAL_BATCH_SIZE
LOCAL_LR = e016.LOCAL_LR
LOCAL_STRUCTURE_INTERVAL = e016.LOCAL_STRUCTURE_INTERVAL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 017 replicate.")
    parser.add_argument("--replicate", type=int, choices=range(N_REPLICATES), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def masked_skill_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, mask, :].reshape(-1, logits.shape[-1]), targets[:, mask].reshape(-1))


def train_phase1(
    *,
    replicate: int,
    train_stream: torch.Tensor,
    validation_stream: torch.Tensor,
    vocab_size: int,
    device: torch.device,
) -> tuple[object, dict[str, torch.Tensor], list[dict], list[dict], float, list[tuple[int, ...]]]:
    model_seed = e016.MODEL_SEED_BASE + 1000 * replicate
    schedule_seed = e016.SCHEDULE_SEED_BASE + 1000 * replicate
    model, _ = e016.build_model("G", vocab_size=vocab_size, seed=model_seed)
    model = model.to(device)
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=e016.BASE_LR, betas=(0.9, 0.95), weight_decay=e016.WEIGHT_DECAY)
    scaler = e016.make_scaler()
    schedule = e016.make_schedule(len(train_stream), seed=schedule_seed)
    validation_starts = e016.fixed_validation_starts(
        len(validation_stream),
        batches=16,
        batch_size=e016.BATCH_SIZE,
        sequence_length=e016.SEQUENCE_LENGTH,
        seed=51_016 + replicate,
    )
    controller = StructuralController(max_cells=model.max_cells)
    events: list[dict] = []
    checkpoints: list[dict] = []
    started = time.perf_counter()
    for step in range(1, e016.TRAIN_STEPS + 1):
        model.train()
        lr = e016.BASE_LR * e016.lr_multiplier(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, targets = e016.batch_from_starts(train_stream, schedule[step - 1], e016.SEQUENCE_LENGTH, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            result = model.forward_variable(inputs)
            main_loss = e016.ce_loss(result.output.logits, targets)
            loss = main_loss + STABILITY_WEIGHT * result.stability_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step >= e016.STRUCTURE_WARMUP and step % e016.STRUCTURE_INTERVAL == 0:
            model.eval()
            probe = e016.structural_probe_from_batch(model, inputs, targets)
            for event in controller.apply(model, probe, step=step):
                events.append({"phase": "language", "policy": "phase1", "replicate": replicate, **event.as_dict()})
        if step in e016.CHECKPOINT_STEPS:
            metrics = e016.evaluate_stream(model, validation_stream, validation_starts, variant="G", device=device)
            checkpoints.append({
                "replicate": replicate,
                "step": step,
                "tokens": step * e016.TOKENS_PER_STEP,
                "validation_nll": metrics["nll"],
                "validation_ppl": metrics["ppl"],
                "alive_cells": model.alive_count,
                "edges": model.edge_count,
            })
    wall = time.perf_counter() - started
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    return model, state, checkpoints, events, wall, validation_starts


def clone_from_state(vocab_size: int, state: dict[str, torch.Tensor], device: torch.device):
    model = build_cellular_model(vocab_size, "G").to(device)
    model.load_state_dict(copy.deepcopy(state))
    return model


def run_baseline_policy(
    model,
    base_state: dict[str, torch.Tensor],
    validation_stream: torch.Tensor,
    validation_starts,
    *,
    replicate: int,
    device: torch.device,
) -> dict[str, object]:
    base_memory = base_state["cell_memory"].detach().clone().to(device)
    base_alive = base_state["alive_mask"].detach().clone().to(device)
    rows, events, transplant, localization, elapsed = e016.local_skill_phase(
        model,
        base_state,
        validation_stream,
        validation_starts,
        replicate=replicate,
        device=device,
    )
    for row in rows:
        row["policy"] = "B"
    for row in events:
        row["policy"] = "B"
    for row in transplant:
        row["policy"] = "B"
    for row in localization:
        row["policy"] = "B"
    base_delta = (model.cell_memory.detach() - base_memory).float().norm(dim=-1)
    base_drift = float(base_delta[base_alive].max().cpu())
    newborn = [int(v) for v in torch.nonzero(model.alive_mask & ~base_alive, as_tuple=False).flatten().tolist()]
    summary = dict(transplant[0])
    summary.update({
        "policy": "B",
        "base_memory_drift": base_drift,
        "newborn_cells": json.dumps(newborn),
        "newborn_count": len(newborn),
        "tissue_causal_fraction": None,
        "ablation_skill_nll": None,
        "ablation_language_nll": None,
        "elapsed_seconds": elapsed,
    })
    return {"rows": rows, "events": events, "transplant": transplant, "localization": localization, "summary": summary, "ablation": []}


def run_localized_policy(
    model,
    base_state: dict[str, torch.Tensor],
    validation_stream: torch.Tensor,
    validation_starts,
    *,
    replicate: int,
    device: torch.device,
) -> dict[str, object]:
    started = time.perf_counter()
    skill_seed = e016.SKILL_SEED_BASE + 1000 * replicate
    train = generate_skill_corpus(4096, seed=skill_seed, tasks=(LOCAL_SKILL,))
    validation = generate_skill_corpus(512, seed=skill_seed + 1, tasks=(LOCAL_SKILL,))
    schedule = make_index_schedule(len(train.sequences), steps=LOCAL_STEPS, batch_size=LOCAL_BATCH_SIZE, seed=skill_seed + 2)
    mask = train.loss_mask.to(device)
    base_skill = e016.evaluate_skill(model, validation, device=device)
    base_language = e016.evaluate_stream(model, validation_stream, validation_starts, variant="G", device=device)["nll"]
    local_state = LocalizedLearningState.capture(model)
    model.freeze_genome()

    first_inputs, first_targets, _ = batch_from_indices(train, schedule[0], device=device)
    pieces_x = torch.chunk(first_inputs, 4, dim=0)
    pieces_y = torch.chunk(first_targets, 4, dim=0)

    def probe_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return masked_skill_loss(logits, targets, mask)

    initial_probe = make_structural_probe(model, list(zip(pieces_x, pieces_y)), loss_fn=probe_loss)
    controller = LocalizedGrowthController(local_state)
    initial_event = controller.allocate_initial(model, initial_probe, step=0)
    events: list[dict] = [{"phase": "local-skill", "policy": "L", "replicate": replicate, **initial_event.as_dict()}]
    optimizer = torch.optim.AdamW([model.cell_memory], lr=LOCAL_LR, weight_decay=0.0)
    rows: list[dict] = []

    for step in range(1, LOCAL_STEPS + 1):
        model.train()
        inputs, targets, _ = batch_from_indices(train, schedule[step - 1], device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            result = model.forward_variable(inputs)
            main_loss = masked_skill_loss(result.output.logits, targets, mask)
            loss = main_loss + STABILITY_WEIGHT * result.stability_loss
        loss.backward()
        mask_to_newborn_gradients(model, local_state)
        torch.nn.utils.clip_grad_norm_([model.cell_memory], 1.0)
        optimizer.step()

        if step % LOCAL_STRUCTURE_INTERVAL == 0:
            pieces_x = torch.chunk(inputs, 4, dim=0)
            pieces_y = torch.chunk(targets, 4, dim=0)
            probe = make_structural_probe(model, list(zip(pieces_x, pieces_y)), loss_fn=probe_loss)
            for event in controller.apply(model, probe, step=step):
                events.append({"phase": "local-skill", "policy": "L", "replicate": replicate, **event.as_dict()})

        if step in (1, 25, 50, 100, 150, 200):
            rows.append({
                "replicate": replicate,
                "policy": "L",
                "step": step,
                "skill_nll": e016.evaluate_skill(model, validation, device=device),
                "language_nll": e016.evaluate_stream(model, validation_stream, validation_starts, variant="G", device=device)["nll"],
                "alive_cells": model.alive_count,
                "edges": model.edge_count,
                "newborn_cells": len(local_state.newborn_cells(model)),
                "base_memory_drift": float(local_state.base_memory_drift(model).cpu()),
            })

    donor_skill = e016.evaluate_skill(model, validation, device=device)
    donor_language = e016.evaluate_stream(model, validation_stream, validation_starts, variant="G", device=device)["nll"]
    newborn = local_state.newborn_cells(model)
    if not newborn:
        raise RuntimeError("localized policy ended without newborn tissue")

    saved_alive, saved_adjacency = set_newborn_tissue_active(model, local_state, False)
    ablated_skill = e016.evaluate_skill(model, validation, device=device)
    ablated_language = e016.evaluate_stream(model, validation_stream, validation_starts, variant="G", device=device)["nll"]
    restore_structure(model, saved_alive, saved_adjacency)

    recipient = clone_from_state(model.token_embedding.num_embeddings, base_state, device)
    graft_localized_tissue(recipient, model, newborn)
    recipient_skill = e016.evaluate_skill(recipient, validation, device=device)
    recipient_language = e016.evaluate_stream(recipient, validation_stream, validation_starts, variant="G", device=device)["nll"]
    improvement = base_skill - donor_skill
    recovery = (base_skill - recipient_skill) / improvement if abs(improvement) > 1e-9 else 0.0
    causal_fraction = (ablated_skill - donor_skill) / improvement if abs(improvement) > 1e-9 else 0.0
    base_drift = float(local_state.base_memory_drift(model).cpu())

    localization = []
    memory_delta = (model.cell_memory.detach() - local_state.base_memory.to(device)).float().norm(dim=-1)
    for cell in range(model.max_cells):
        localization.append({
            "replicate": replicate,
            "policy": "L",
            "cell": cell,
            "memory_delta_norm": float(memory_delta[cell].cpu()),
            "base_alive": int(bool(local_state.base_alive[cell])),
            "donor_alive": int(bool(model.alive_mask[cell])),
            "newborn": int(cell in newborn),
            "trainable_tissue": int(cell in newborn),
        })

    transplant = [{
        "replicate": replicate,
        "policy": "L",
        "skill": LOCAL_SKILL,
        "base_skill_nll": base_skill,
        "donor_skill_nll": donor_skill,
        "recipient_skill_nll": recipient_skill,
        "skill_improvement": improvement,
        "transplant_recovery": recovery,
        "base_language_nll": base_language,
        "donor_language_nll": donor_language,
        "recipient_language_nll": recipient_language,
        "donor_language_ratio": donor_language / base_language,
        "recipient_language_ratio": recipient_language / base_language,
        "selected_cells": json.dumps(newborn),
        "selected_cell_count": len(newborn),
        "donor_alive_cells": model.alive_count,
        "recipient_alive_cells": recipient.alive_count,
    }]
    ablation = [{
        "replicate": replicate,
        "policy": "L",
        "intervention": "newborn_tissue_off",
        "normal_skill_nll": donor_skill,
        "ablated_skill_nll": ablated_skill,
        "delta_skill_nll": ablated_skill - donor_skill,
        "skill_causal_fraction": causal_fraction,
        "normal_language_nll": donor_language,
        "ablated_language_nll": ablated_language,
        "delta_language_nll": ablated_language - donor_language,
        "newborn_count": len(newborn),
    }]
    summary = dict(transplant[0])
    summary.update({
        "base_memory_drift": base_drift,
        "newborn_cells": json.dumps(newborn),
        "newborn_count": len(newborn),
        "tissue_causal_fraction": causal_fraction,
        "ablation_skill_nll": ablated_skill,
        "ablation_language_nll": ablated_language,
        "elapsed_seconds": time.perf_counter() - started,
    })
    return {"rows": rows, "events": events, "transplant": transplant, "localization": localization, "summary": summary, "ablation": ablation}


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 017 requires CUDA")
    device = torch.device("cuda:0")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_stream, validation_stream, manifest = e016.load_cache(args.cache_dir.resolve())
    vocab_size = int(manifest["vocab_size_actual"])
    replicate = args.replicate
    torch.cuda.reset_peak_memory_stats()

    phase1_model, base_state, checkpoints, phase1_events, phase1_wall, validation_starts = train_phase1(
        replicate=replicate,
        train_stream=train_stream,
        validation_stream=validation_stream,
        vocab_size=vocab_size,
        device=device,
    )
    phase1_metrics = e016.evaluate_stream(phase1_model, validation_stream, validation_starts, variant="G", device=device)
    phase1_summary = {
        "replicate": replicate,
        "parameters": count_parameters(phase1_model),
        "phase1_nll": phase1_metrics["nll"],
        "phase1_ppl": phase1_metrics["ppl"],
        "phase1_alive_cells": phase1_model.alive_count,
        "phase1_edges": phase1_model.edge_count,
        "phase1_wall_seconds": phase1_wall,
    }

    baseline_model = clone_from_state(vocab_size, base_state, device)
    localized_model = clone_from_state(vocab_size, base_state, device)
    baseline = run_baseline_policy(baseline_model, base_state, validation_stream, validation_starts, replicate=replicate, device=device)
    localized = run_localized_policy(localized_model, base_state, validation_stream, validation_starts, replicate=replicate, device=device)

    learning = baseline["rows"] + localized["rows"]
    events = phase1_events + baseline["events"] + localized["events"]
    transplant = baseline["transplant"] + localized["transplant"]
    localization = baseline["localization"] + localized["localization"]
    ablation = baseline["ablation"] + localized["ablation"]

    pd.DataFrame(checkpoints).to_csv(output_dir / f"r{replicate}-phase1-checkpoints.csv", index=False)
    pd.DataFrame(events).to_csv(output_dir / f"r{replicate}-structural-events.csv", index=False)
    pd.DataFrame(learning).to_csv(output_dir / f"r{replicate}-local-learning.csv", index=False)
    pd.DataFrame(transplant).to_csv(output_dir / f"r{replicate}-transplantation.csv", index=False)
    pd.DataFrame(localization).to_csv(output_dir / f"r{replicate}-localization.csv", index=False)
    pd.DataFrame(ablation).to_csv(output_dir / f"r{replicate}-tissue-ablation.csv", index=False)

    worker = {
        "format": "minicells.localized-cellular-learning-worker.v1",
        "replicate": replicate,
        "phase1": phase1_summary,
        "policies": {"B": baseline["summary"], "L": localized["summary"]},
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }
    (output_dir / f"r{replicate}-worker.json").write_text(json.dumps(worker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(worker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
