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
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "src").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_language_conditional_recruitment_worker as e018  # noqa: E402
import run_language_growing_organism_worker as e016  # noqa: E402
import run_language_localized_learning_worker as e017  # noqa: E402
from minicells.language_conditional_recruitment import (  # noqa: E402
    calibrate_homeostasis,
    forward_with_recruitment,
    make_recruitment_probe,
)
from minicells.language_growing_organism import STABILITY_WEIGHT, make_structural_probe  # noqa: E402
from minicells.language_localized_learning import (  # noqa: E402
    LocalizedGrowthController,
    LocalizedLearningState,
    graft_localized_tissue,
    mask_to_newborn_gradients,
    restore_structure,
    set_newborn_tissue_active,
)
from minicells.language_models import count_parameters  # noqa: E402
from minicells.language_pressure_recruitment import (  # noqa: E402
    PressureProfile,
    calibrate_pressure_homeostasis,
    forward_with_pressure_recruitment,
    make_pressure_recruitment_probe,
)
from minicells.language_skill_data import batch_from_indices, generate_skill_corpus, make_index_schedule  # noqa: E402


N_REPLICATES = 3
POLICIES = ("N", "P")
LOCAL_SKILL = e016.LOCAL_SKILL
LOCAL_STEPS = e016.LOCAL_STEPS
LOCAL_BATCH_SIZE = e016.LOCAL_BATCH_SIZE
LOCAL_LR = e016.LOCAL_LR
LOCAL_STRUCTURE_INTERVAL = e016.LOCAL_STRUCTURE_INTERVAL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 018b replicate.")
    parser.add_argument("--replicate", type=int, choices=range(N_REPLICATES), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _masked_skill_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, mask, :].reshape(-1, logits.shape[-1]), targets[:, mask].reshape(-1))


def _forward(policy: str, model, inputs, localized_state, profile, *, force_recruitment=None, edge_probe=None):
    if policy == "N":
        return forward_with_recruitment(
            model,
            inputs,
            localized_state,
            profile,
            force_recruitment=force_recruitment,
            edge_probe=edge_probe,
        )
    if policy == "P":
        return forward_with_pressure_recruitment(
            model,
            inputs,
            localized_state,
            profile,
            force_recruitment=force_recruitment,
            edge_probe=edge_probe,
        )
    raise ValueError(policy)


def _newborn_gate_mean(result, model, localized_state: LocalizedLearningState) -> float:
    newborn = localized_state.newborn_cells(model)
    if not newborn:
        return 0.0
    alive = torch.nonzero(model.alive_mask, as_tuple=False).flatten().tolist()
    positions = [alive.index(cell) for cell in newborn if cell in alive]
    if not positions:
        return 0.0
    return float(result.recruitment_trace[..., positions].float().mean().detach().cpu())


def _pressure_mean(result, model, localized_state: LocalizedLearningState) -> float | None:
    trace = getattr(result, "shadow_pressure_trace", None)
    if trace is None:
        return None
    newborn = localized_state.newborn_cells(model)
    if not newborn:
        return 0.0
    parents = sorted({int(model.parent[cell].item()) for cell in newborn})
    return float(trace[..., parents].float().mean().detach().cpu())


@torch.no_grad()
def evaluate_stream(
    policy: str,
    model,
    localized_state,
    profile,
    token_stream,
    starts,
    *,
    device: torch.device,
    force_recruitment: float | None = None,
) -> tuple[float, float, float | None]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    gates: list[float] = []
    pressures: list[float] = []
    for batch_starts in starts:
        inputs, targets = e016.batch_from_starts(token_stream, batch_starts, e016.SEQUENCE_LENGTH, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            result = _forward(
                policy,
                model,
                inputs,
                localized_state,
                profile,
                force_recruitment=force_recruitment,
            )
        logits = result.output.logits.float()
        total_loss += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"))
        total_tokens += targets.numel()
        gates.append(_newborn_gate_mean(result, model, localized_state))
        pressure = _pressure_mean(result, model, localized_state)
        if pressure is not None:
            pressures.append(pressure)
    mean_pressure = sum(pressures) / len(pressures) if pressures else None
    return total_loss / max(1, total_tokens), sum(gates) / max(1, len(gates)), mean_pressure


@torch.no_grad()
def evaluate_skill(
    policy: str,
    model,
    localized_state,
    profile,
    corpus,
    *,
    device: torch.device,
    force_recruitment: float | None = None,
) -> tuple[float, float, float | None]:
    model.eval()
    mask = corpus.loss_mask.to(device)
    total_loss = 0.0
    total_tokens = 0
    gates: list[float] = []
    pressures: list[float] = []
    for start in range(0, len(corpus.sequences), 64):
        stop = min(start + 64, len(corpus.sequences))
        inputs, targets, _ = batch_from_indices(corpus, torch.arange(start, stop), device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            result = _forward(
                policy,
                model,
                inputs,
                localized_state,
                profile,
                force_recruitment=force_recruitment,
            )
        logits = result.output.logits.float()
        selected_logits = logits[:, mask, :]
        selected_targets = targets[:, mask]
        total_loss += float(F.cross_entropy(selected_logits.reshape(-1, logits.shape[-1]), selected_targets.reshape(-1), reduction="sum"))
        total_tokens += selected_targets.numel()
        gates.append(_newborn_gate_mean(result, model, localized_state))
        pressure = _pressure_mean(result, model, localized_state)
        if pressure is not None:
            pressures.append(pressure)
    mean_pressure = sum(pressures) / len(pressures) if pressures else None
    return total_loss / max(1, total_tokens), sum(gates) / max(1, len(gates)), mean_pressure


def _make_probe(policy, model, localized_state, profile, microbatches, *, loss_fn):
    if policy == "N":
        return make_recruitment_probe(model, localized_state, profile, microbatches, loss_fn=loss_fn)
    return make_pressure_recruitment_probe(model, localized_state, profile, microbatches, loss_fn=loss_fn)


def run_policy(
    policy: str,
    model,
    base_state: dict[str, torch.Tensor],
    train_stream: torch.Tensor,
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
    localized_state = LocalizedLearningState.capture(model)
    calibration = e018.calibration_inputs(train_stream, replicate=replicate, device=device)
    if policy == "N":
        profile = calibrate_homeostasis(model, calibration)
    else:
        profile = calibrate_pressure_homeostasis(model, localized_state, calibration)
    model.freeze_genome()

    first_inputs, first_targets, _ = batch_from_indices(train, schedule[0], device=device)
    pieces_x = torch.chunk(first_inputs, 4, dim=0)
    pieces_y = torch.chunk(first_targets, 4, dim=0)

    def probe_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return _masked_skill_loss(logits, targets, mask)

    initial_probe = make_structural_probe(model, list(zip(pieces_x, pieces_y)), loss_fn=probe_loss)
    # 017 established one newborn is sufficient; 018b fixes cell count so the
    # experiment isolates the recruitment sensor rather than birth policy.
    controller = LocalizedGrowthController(localized_state, max_newborns=1)
    initial_event = controller.allocate_initial(model, initial_probe, step=0)
    events: list[dict] = [{"phase": "local-skill", "policy": policy, "replicate": replicate, **initial_event.as_dict()}]
    optimizer = torch.optim.AdamW([model.cell_memory], lr=LOCAL_LR, weight_decay=0.0)
    rows: list[dict] = []

    for step in range(1, LOCAL_STEPS + 1):
        model.train()
        inputs, targets, _ = batch_from_indices(train, schedule[step - 1], device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            result = _forward(policy, model, inputs, localized_state, profile)
            main_loss = _masked_skill_loss(result.output.logits, targets, mask)
            loss = main_loss + STABILITY_WEIGHT * result.stability_loss
        loss.backward()
        mask_to_newborn_gradients(model, localized_state)
        torch.nn.utils.clip_grad_norm_([model.cell_memory], 1.0)
        optimizer.step()

        if step % LOCAL_STRUCTURE_INTERVAL == 0:
            pieces_x = torch.chunk(inputs, 4, dim=0)
            pieces_y = torch.chunk(targets, 4, dim=0)
            probe = _make_probe(policy, model, localized_state, profile, list(zip(pieces_x, pieces_y)), loss_fn=probe_loss)
            for event in controller.apply(model, probe, step=step):
                events.append({"phase": "local-skill", "policy": policy, "replicate": replicate, **event.as_dict()})

        if step in (1, 25, 50, 100, 150, 200):
            skill_nll, skill_gate, skill_pressure = evaluate_skill(
                policy, model, localized_state, profile, validation, device=device
            )
            language_nll, language_gate, language_pressure = evaluate_stream(
                policy, model, localized_state, profile, validation_stream, validation_starts, device=device
            )
            rows.append({
                "replicate": replicate,
                "policy": policy,
                "step": step,
                "skill_nll": skill_nll,
                "language_nll": language_nll,
                "skill_recruitment": skill_gate,
                "language_recruitment": language_gate,
                "skill_shadow_pressure": skill_pressure,
                "language_shadow_pressure": language_pressure,
                "alive_cells": model.alive_count,
                "edges": model.edge_count,
                "newborn_cells": len(localized_state.newborn_cells(model)),
                "base_memory_drift": float(localized_state.base_memory_drift(model).cpu()),
            })

    donor_skill, skill_gate, skill_pressure = evaluate_skill(
        policy, model, localized_state, profile, validation, device=device
    )
    donor_language, language_gate, language_pressure = evaluate_stream(
        policy, model, localized_state, profile, validation_stream, validation_starts, device=device
    )
    newborn = localized_state.newborn_cells(model)
    if len(newborn) != 1:
        raise RuntimeError(f"018b requires exactly one newborn tissue cell, got {newborn}")

    force_off_skill, _, _ = evaluate_skill(
        policy, model, localized_state, profile, validation, device=device, force_recruitment=0.0
    )
    force_on_language, _, _ = evaluate_stream(
        policy, model, localized_state, profile, validation_stream, validation_starts, device=device, force_recruitment=1.0
    )

    saved_alive, saved_adjacency = set_newborn_tissue_active(model, localized_state, False)
    ablated_skill = e016.evaluate_skill(model, validation, device=device)
    ablated_language = e016.evaluate_stream(model, validation_stream, validation_starts, variant="G", device=device)["nll"]
    restore_structure(model, saved_alive, saved_adjacency)

    recipient = e017.clone_from_state(model.token_embedding.num_embeddings, base_state, device)
    recipient_state = LocalizedLearningState.capture(recipient)
    graft_localized_tissue(recipient, model, newborn)
    recipient_skill, recipient_skill_gate, recipient_skill_pressure = evaluate_skill(
        policy, recipient, recipient_state, profile, validation, device=device
    )
    recipient_language, recipient_language_gate, recipient_language_pressure = evaluate_stream(
        policy, recipient, recipient_state, profile, validation_stream, validation_starts, device=device
    )

    feedback_sensor_max_delta = None
    if policy == "P":
        probe_inputs, _ = e016.batch_from_starts(
            validation_stream, validation_starts[0], e016.SEQUENCE_LENGTH, device
        )
        normal = forward_with_pressure_recruitment(model, probe_inputs, localized_state, profile)
        forced = forward_with_pressure_recruitment(
            model, probe_inputs, localized_state, profile, force_recruitment=1.0
        )
        feedback_sensor_max_delta = float(
            (normal.shadow_pressure_trace - forced.shadow_pressure_trace).abs().max().detach().cpu()
        )

    improvement = base_skill - donor_skill
    recovery = (base_skill - recipient_skill) / improvement if abs(improvement) > 1e-9 else 0.0
    recruitment_fraction = (force_off_skill - donor_skill) / improvement if abs(improvement) > 1e-9 else 0.0
    tissue_fraction = (ablated_skill - donor_skill) / improvement if abs(improvement) > 1e-9 else 0.0
    base_drift = float(localized_state.base_memory_drift(model).cpu())

    memory_delta = (model.cell_memory.detach() - localized_state.base_memory.to(device)).float().norm(dim=-1)
    localization = [{
        "replicate": replicate,
        "policy": policy,
        "cell": cell,
        "memory_delta_norm": float(memory_delta[cell].cpu()),
        "base_alive": int(bool(localized_state.base_alive[cell])),
        "donor_alive": int(bool(model.alive_mask[cell])),
        "newborn": int(cell in newborn),
        "trainable_tissue": int(cell in newborn),
    } for cell in range(model.max_cells)]

    transplant = [{
        "replicate": replicate,
        "policy": policy,
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
        "donor_skill_recruitment": skill_gate,
        "donor_language_recruitment": language_gate,
        "recipient_skill_recruitment": recipient_skill_gate,
        "recipient_language_recruitment": recipient_language_gate,
        "donor_skill_shadow_pressure": skill_pressure,
        "donor_language_shadow_pressure": language_pressure,
        "recipient_skill_shadow_pressure": recipient_skill_pressure,
        "recipient_language_shadow_pressure": recipient_language_pressure,
    }]
    interventions = [{
        "replicate": replicate,
        "policy": policy,
        "intervention": "recruitment_off_skill",
        "normal_nll": donor_skill,
        "altered_nll": force_off_skill,
        "delta_nll": force_off_skill - donor_skill,
        "causal_fraction": recruitment_fraction,
    }, {
        "replicate": replicate,
        "policy": policy,
        "intervention": "recruitment_forced_on_language",
        "normal_nll": donor_language,
        "altered_nll": force_on_language,
        "delta_nll": force_on_language - donor_language,
        "causal_fraction": None,
    }, {
        "replicate": replicate,
        "policy": policy,
        "intervention": "newborn_tissue_off",
        "normal_nll": donor_skill,
        "altered_nll": ablated_skill,
        "delta_nll": ablated_skill - donor_skill,
        "causal_fraction": tissue_fraction,
        "ablated_language_nll": ablated_language,
    }]
    summary = dict(transplant[0])
    summary.update({
        "base_memory_drift": base_drift,
        "newborn_cells": json.dumps(newborn),
        "newborn_count": len(newborn),
        "tissue_causal_fraction": tissue_fraction,
        "recruitment_causal_fraction": recruitment_fraction,
        "force_off_skill_nll": force_off_skill,
        "force_on_language_nll": force_on_language,
        "force_on_language_delta_nll": force_on_language - donor_language,
        "recruitment_selectivity": skill_gate / max(1e-8, language_gate),
        "recruitment_gap": skill_gate - language_gate,
        "feedback_sensor_max_delta": feedback_sensor_max_delta,
        "elapsed_seconds": time.perf_counter() - started,
    })
    return {
        "rows": rows,
        "events": events,
        "transplant": transplant,
        "localization": localization,
        "summary": summary,
        "interventions": interventions,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 018b requires CUDA")
    device = torch.device("cuda:0")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_stream, validation_stream, manifest = e016.load_cache(args.cache_dir.resolve())
    vocab_size = int(manifest["vocab_size_actual"])
    replicate = args.replicate
    torch.cuda.reset_peak_memory_stats()

    phase1_model, base_state, checkpoints, phase1_events, phase1_wall, validation_starts = e017.train_phase1(
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

    novelty_model = e017.clone_from_state(vocab_size, base_state, device)
    pressure_model = e017.clone_from_state(vocab_size, base_state, device)
    novelty = run_policy(
        "N", novelty_model, base_state, train_stream, validation_stream, validation_starts,
        replicate=replicate, device=device
    )
    pressure = run_policy(
        "P", pressure_model, base_state, train_stream, validation_stream, validation_starts,
        replicate=replicate, device=device
    )

    learning = novelty["rows"] + pressure["rows"]
    events = phase1_events + novelty["events"] + pressure["events"]
    transplant = novelty["transplant"] + pressure["transplant"]
    localization = novelty["localization"] + pressure["localization"]
    interventions = novelty["interventions"] + pressure["interventions"]

    pd.DataFrame(checkpoints).to_csv(output_dir / f"r{replicate}-phase1-checkpoints.csv", index=False)
    pd.DataFrame(events).to_csv(output_dir / f"r{replicate}-structural-events.csv", index=False)
    pd.DataFrame(learning).to_csv(output_dir / f"r{replicate}-local-learning.csv", index=False)
    pd.DataFrame(transplant).to_csv(output_dir / f"r{replicate}-transplantation.csv", index=False)
    pd.DataFrame(localization).to_csv(output_dir / f"r{replicate}-localization.csv", index=False)
    pd.DataFrame(interventions).to_csv(output_dir / f"r{replicate}-recruitment-interventions.csv", index=False)

    worker = {
        "format": "minicells.feedback-isolated-pressure-recruitment-worker.v1",
        "replicate": replicate,
        "phase1": phase1_summary,
        "policies": {"N": novelty["summary"], "P": pressure["summary"]},
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }
    (output_dir / f"r{replicate}-worker.json").write_text(json.dumps(worker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(worker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
