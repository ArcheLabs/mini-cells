from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))

from minicells.language_conflict_differentiation import (  # noqa: E402
    BATCH_SIZE,
    PHENOTYPE_LR,
    PRETRAIN_LR,
    PRETRAIN_STEPS,
    SEQUENCE_LENGTH,
    deterministic_starts,
    language_model_loss,
    lr_multiplier,
    prepare_arithmetic_cache,
    trait_gradient,
)
from minicells.language_data import batch_from_starts, load_tokenizer  # noqa: E402
from minicells.language_online_trait_genesis import (  # noqa: E402
    FAMILIES,
    MAX_TRAITS,
    MODE_STABILITY_MIN,
    PERSISTENCE_EVALS,
    ROUTING_PURITY_MIN,
    SENSOR_BUFFER,
    SENSOR_INTERVAL,
    STRUCTURAL_PENALTY,
    GrowthEvidence,
    OnlineTraitTextNCA,
    align_growth_centroids,
    align_same_k,
    cluster_purity,
    developmental_curriculum,
    prepare_transform_cache,
    route_to_centroid,
    select_model_order,
    stage_end_steps,
    summarize_multi_identity,
    update_growth_evidence,
)


CHECKPOINT_FORMAT = "minicells.online-nonparametric-trait-genesis-checkpoint.v1"
N_REPLICATES = 3
MODEL_SEED_BASE = 123_023
SCHEDULE_SEED_BASE = 223_023
STAGES = ("A_STORY_ONLY", "B_EMERGING_MATH", "C_DUPLICATE_CONTROL", "D_THIRD_MODE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 023 replicate")
    parser.add_argument("--replicate", type=int, choices=range(N_REPLICATES), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_save(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def make_batch(stream: torch.Tensor, starts: tuple[int, ...], device: torch.device):
    return batch_from_starts(stream, starts, SEQUENCE_LENGTH, device)


def load_streams(cache_dir: Path):
    story_train = torch.load(cache_dir / "train-tokens.pt", map_location="cpu")
    story_validation = torch.load(cache_dir / "validation-tokens.pt", map_location="cpu")
    story_manifest = json.loads((cache_dir / "corpus-manifest.json").read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(cache_dir / "tokenizer.json")
    arithmetic = prepare_arithmetic_cache(cache_dir, tokenizer)
    transform = prepare_transform_cache(cache_dir, tokenizer)
    return {
        "STORY": {"train": story_train, "validation": story_validation},
        "ARITH_A": {"train": arithmetic["train"], "validation": arithmetic["validation"]},
        "ARITH_B": {"train": arithmetic["train"], "validation": arithmetic["validation"]},
        "TRANSFORM": {"train": transform["train"], "validation": transform["validation"]},
        "story_manifest": story_manifest,
        "arithmetic_manifest": arithmetic["manifest"],
        "arithmetic_manifest_path": arithmetic["path"],
        "transform_manifest": transform["manifest"],
        "transform_manifest_path": transform["path"],
        "vocab_size": int(tokenizer.get_vocab_size()),
    }


@torch.no_grad()
def evaluate_vector(
    model: OnlineTraitTextNCA,
    stream: torch.Tensor,
    starts: tuple[tuple[int, ...], ...],
    *,
    device: torch.device,
    branch: int | None,
) -> float:
    model.eval()
    losses = []
    for row in starts:
        inputs, targets = make_batch(stream, row, device)
        logits = model.forward_parent(inputs) if branch is None else model.forward_trait(inputs, branch)
        losses.append(float(language_model_loss(logits.float(), targets)))
    return float(np.mean(losses))


def pretrain_parent(
    model: OnlineTraitTextNCA,
    story_train: torch.Tensor,
    *,
    replicate: int,
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[OnlineTraitTextNCA, list[dict[str, object]], float]:
    if checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            payload.get("format") != CHECKPOINT_FORMAT
            or payload.get("kind") != "parent"
            or int(payload.get("replicate", -1)) != replicate
        ):
            raise RuntimeError(f"invalid Experiment 023 parent checkpoint: {checkpoint_path}")
        model.load_state_dict(payload["model_state"])
        return model, list(payload.get("learning_curve", [])), float(payload.get("wall_seconds", 0.0))

    schedule = deterministic_starts(
        len(story_train),
        steps=PRETRAIN_STEPS,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        seed=SCHEDULE_SEED_BASE + 1000 * replicate,
    )
    optimizer = torch.optim.AdamW(
        model.pretrain_parameters(), lr=PRETRAIN_LR, betas=(0.9, 0.95), weight_decay=0.1
    )
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    model.train()
    for step in range(1, PRETRAIN_STEPS + 1):
        for group in optimizer.param_groups:
            group["lr"] = PRETRAIN_LR * lr_multiplier(step, PRETRAIN_STEPS)
        inputs, targets = make_batch(story_train, schedule[step - 1], device)
        optimizer.zero_grad(set_to_none=True)
        loss = language_model_loss(model.forward_parent(inputs).float(), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.pretrain_parameters(), 1.0)
        optimizer.step()
        if step in (1, 50, 100, 200, PRETRAIN_STEPS):
            rows.append({"step": step, "story_train_nll": float(loss.detach())})
    wall = time.perf_counter() - started
    atomic_save(
        checkpoint_path,
        {
            "format": CHECKPOINT_FORMAT,
            "experiment": "023",
            "kind": "parent",
            "replicate": replicate,
            "model_state": cpu_state_dict(model),
            "learning_curve": rows,
            "wall_seconds": wall,
        },
    )
    return model, rows, wall


def build_start_schedules(streams: dict[str, object], curriculum: list[dict[str, object]], *, replicate: int):
    counts: dict[str, int] = {}
    for row in curriculum:
        key = str(row["stream_key"])
        counts[key] = counts.get(key, 0) + 1
    schedules = {}
    for index, key in enumerate(sorted(counts)):
        schedules[key] = deterministic_starts(
            len(streams[key]["train"]),
            steps=counts[key],
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            seed=SCHEDULE_SEED_BASE + 20_000 * replicate + index,
        )
    return schedules


def validation_schedules(streams: dict[str, object], *, replicate: int):
    mapping = {"STORY": "STORY", "ARITHMETIC": "ARITH_A", "TRANSFORM": "TRANSFORM"}
    result = {}
    for index, (family, stream_key) in enumerate(mapping.items()):
        result[family] = deterministic_starts(
            len(streams[stream_key]["validation"]),
            steps=16,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            seed=SCHEDULE_SEED_BASE + 30_000 * replicate + index,
        )
    return result


def _family_stream_key(family: str) -> str:
    return {"STORY": "STORY", "ARITHMETIC": "ARITH_A", "TRANSFORM": "TRANSFORM"}[family]


def evaluate_stage(
    model: OnlineTraitTextNCA,
    streams: dict[str, object],
    validation: dict[str, tuple[tuple[int, ...], ...]],
    parent_losses: dict[str, float],
    *,
    replicate: int,
    stage: str,
    active_k: int,
    routing_rows: list[dict[str, object]],
):
    domains = {
        "A_STORY_ONLY": ("STORY",),
        "B_EMERGING_MATH": ("STORY", "ARITHMETIC"),
        "C_DUPLICATE_CONTROL": ("STORY", "ARITHMETIC"),
        "D_THIRD_MODE": ("STORY", "ARITHMETIC", "TRANSFORM"),
    }[stage]
    losses: dict[str, tuple[float, ...]] = {}
    evaluation_rows = []
    for family in domains:
        stream_key = _family_stream_key(family)
        values = []
        for branch in range(active_k):
            nll = evaluate_vector(
                model,
                streams[stream_key]["validation"],
                validation[family],
                device=next(model.parameters()).device,
                branch=branch,
            )
            values.append(nll)
            evaluation_rows.append(
                {
                    "replicate": replicate,
                    "stage": stage,
                    "family": family,
                    "branch": branch,
                    "nll": nll,
                    "parent_nll": parent_losses[family],
                    "utility": parent_losses[family] - nll,
                }
            )
        losses[family] = tuple(values)
    identity = None
    if len(domains) >= 2 and active_k >= len(domains):
        identity = summarize_multi_identity(losses, parent_losses, domains)

    selected_routes = [row for row in routing_rows if row["stage"] == stage and int(row["active_k_after"]) == active_k]
    route_purity = None
    if active_k >= 2 and selected_routes:
        branches = torch.tensor([int(row["branch"]) for row in selected_routes], dtype=torch.long)
        labels = [str(row["family"]) for row in selected_routes]
        route_purity = cluster_purity(branches, labels)
    summary = {
        "replicate": replicate,
        "stage": stage,
        "active_k": active_k,
        "identity_pass": None if identity is None else int(identity.passes),
        "identity_assignment": None if identity is None else json.dumps(identity.assignment),
        "normalized_identity_margin": None if identity is None else identity.normalized_identity_margin,
        "routing_purity_posthoc": route_purity,
        "routing_purity_pass": None if route_purity is None else int(route_purity >= ROUTING_PURITY_MIN),
    }
    if identity is not None:
        for family, margin in zip(domains, identity.normalized_margins):
            summary[f"{family.lower()}_normalized_margin"] = margin
    return summary, evaluation_rows


def evidence_payload(evidence: GrowthEvidence) -> dict[str, object]:
    return {
        "candidate_k": evidence.candidate_k,
        "stable_evaluations": evidence.stable_evaluations,
        "previous_centroids": evidence.previous_centroids,
        "last_stability": evidence.last_stability,
    }


def evidence_from_payload(payload: dict[str, object]) -> GrowthEvidence:
    return GrowthEvidence(
        candidate_k=int(payload.get("candidate_k", 0)),
        stable_evaluations=int(payload.get("stable_evaluations", 0)),
        previous_centroids=payload.get("previous_centroids"),
        last_stability=float(payload.get("last_stability", 0.0)),
    )


def save_stage_checkpoint(
    path: Path,
    *,
    replicate: int,
    stage: str,
    completed_step: int,
    model: OnlineTraitTextNCA,
    optimizer: torch.optim.Optimizer,
    active_k: int,
    route_centroids: torch.Tensor,
    evidence: GrowthEvidence,
    sensor_gradients: list[torch.Tensor],
    sensor_labels: list[str],
    occurrence: dict[str, int],
    structural_rows: list[dict[str, object]],
    routing_rows: list[dict[str, object]],
    genesis_rows: list[dict[str, object]],
    stage_rows: list[dict[str, object]],
    evaluation_rows: list[dict[str, object]],
) -> None:
    atomic_save(
        path,
        {
            "format": CHECKPOINT_FORMAT,
            "experiment": "023",
            "kind": "stage",
            "replicate": replicate,
            "stage": stage,
            "completed_step": completed_step,
            "model_state": cpu_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "active_k": active_k,
            "route_centroids": route_centroids.detach().cpu(),
            "evidence": evidence_payload(evidence),
            "sensor_gradients": torch.stack(sensor_gradients) if sensor_gradients else torch.empty(0, model.dim),
            "sensor_labels": list(sensor_labels),
            "occurrence": dict(occurrence),
            "structural_rows": list(structural_rows),
            "routing_rows": list(routing_rows),
            "genesis_rows": list(genesis_rows),
            "stage_rows": list(stage_rows),
            "evaluation_rows": list(evaluation_rows),
        },
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 023 requires CUDA")
    device = torch.device("cuda:0")
    replicate = int(args.replicate)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    streams = load_streams(args.cache_dir.resolve())
    vocab_size = int(streams["vocab_size"])
    torch.manual_seed(MODEL_SEED_BASE + 1000 * replicate)
    torch.cuda.manual_seed_all(MODEL_SEED_BASE + 1000 * replicate)
    model = OnlineTraitTextNCA(vocab_size).to(device)
    parent_path = checkpoint_dir / f"r{replicate}-parent.pt"
    model, pretrain_curve, pretrain_wall = pretrain_parent(
        model,
        streams["STORY"]["train"],
        replicate=replicate,
        device=device,
        checkpoint_path=parent_path,
    )
    model.initialize_online_population()
    model.freeze_for_online_development()
    validation = validation_schedules(streams, replicate=replicate)
    parent_losses = {
        family: evaluate_vector(
            model,
            streams[_family_stream_key(family)]["validation"],
            validation[family],
            device=device,
            branch=None,
        )
        for family in FAMILIES
    }
    optimizer = torch.optim.AdamW([model.online_traits], lr=PHENOTYPE_LR, weight_decay=0.0)

    curriculum = developmental_curriculum(replicate)
    starts = build_start_schedules(streams, curriculum, replicate=replicate)
    end_steps = stage_end_steps()
    active_k = 1
    route_centroids = torch.zeros(1, model.dim)
    evidence = GrowthEvidence()
    sensor_gradients: list[torch.Tensor] = []
    sensor_labels: list[str] = []
    occurrence = {key: 0 for key in starts}
    structural_rows: list[dict[str, object]] = []
    routing_rows: list[dict[str, object]] = []
    genesis_rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    evaluation_rows: list[dict[str, object]] = []
    completed_step = 0

    latest_payload = None
    for stage in STAGES:
        path = checkpoint_dir / f"r{replicate}-{stage.lower()}.pt"
        if path.is_file() and path.stat().st_size > 0:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("format") != CHECKPOINT_FORMAT or payload.get("stage") != stage:
                raise RuntimeError(f"invalid Experiment 023 stage checkpoint: {path}")
            latest_payload = payload
        else:
            break
    if latest_payload is not None:
        model.load_state_dict(latest_payload["model_state"])
        optimizer.load_state_dict(latest_payload["optimizer_state"])
        active_k = int(latest_payload["active_k"])
        route_centroids = latest_payload["route_centroids"].float()
        evidence = evidence_from_payload(latest_payload["evidence"])
        sensor_tensor = latest_payload["sensor_gradients"]
        sensor_gradients = [row.clone() for row in sensor_tensor]
        sensor_labels = list(latest_payload["sensor_labels"])
        occurrence = {str(key): int(value) for key, value in latest_payload["occurrence"].items()}
        structural_rows = list(latest_payload["structural_rows"])
        routing_rows = list(latest_payload["routing_rows"])
        genesis_rows = list(latest_payload["genesis_rows"])
        stage_rows = list(latest_payload["stage_rows"])
        evaluation_rows = list(latest_payload["evaluation_rows"])
        completed_step = int(latest_payload["completed_step"])

    started = time.perf_counter()
    for row in curriculum:
        step = int(row["step"])
        if step <= completed_step:
            continue
        stage = str(row["stage"])
        stream_key = str(row["stream_key"])
        family = str(row["family"])
        schedule_index = occurrence[stream_key]
        occurrence[stream_key] += 1
        inputs, targets = make_batch(
            streams[stream_key]["train"], starts[stream_key][schedule_index], device
        )

        shadow_gradient, shadow_loss = trait_gradient(model, inputs, targets)
        sensor_gradients.append(shadow_gradient.detach().cpu())
        sensor_labels.append(family)
        if len(sensor_gradients) > SENSOR_BUFFER:
            sensor_gradients.pop(0)
            sensor_labels.pop(0)

        if active_k == 1:
            branch, route_margin = 0, 0.0
        else:
            branch, route_margin = route_to_centroid(shadow_gradient, route_centroids[:active_k])

        optimizer.zero_grad(set_to_none=True)
        loss = language_model_loss(model.forward_trait(inputs, branch).float(), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([model.online_traits], 1.0)
        optimizer.step()
        active_before_sensor = active_k

        if len(sensor_gradients) == SENSOR_BUFFER and step % SENSOR_INTERVAL == 0:
            gradient_tensor = torch.stack(sensor_gradients)
            selection = select_model_order(gradient_tensor, max_k=MAX_TRAITS)
            current_fit = selection.fit(active_k)
            if active_k == 1 or route_centroids.shape[0] != active_k:
                route_centroids = current_fit.centroids.detach().cpu()
            else:
                route_centroids = align_same_k(route_centroids, current_fit.centroids.cpu())
            evidence, ready = update_growth_evidence(
                evidence, active_k=active_k, selection=selection
            )
            selected_fit = selection.fit(selection.selected_k)
            selected_purity = cluster_purity(selected_fit.assignment.cpu(), sensor_labels)
            active_before_growth = active_k
            if ready and active_k < MAX_TRAITS:
                candidate_fit = selection.fit(active_k + 1)
                ordered, newborn, parent_branch = align_growth_centroids(
                    route_centroids, candidate_fit.centroids.cpu()
                )
                if active_k == 1:
                    model.spawn_first_bifurcation(ordered.to(device))
                else:
                    model.spawn_additional_trait(
                        new_branch=newborn,
                        parent_branch=parent_branch,
                        parent_centroid=ordered[parent_branch].to(device),
                        new_centroid=ordered[newborn].to(device),
                    )
                active_k += 1
                route_centroids = ordered
                genesis_rows.append(
                    {
                        "replicate": replicate,
                        "step": step,
                        "stage": stage,
                        "from_k": active_before_growth,
                        "to_k": active_k,
                        "selected_k": selection.selected_k,
                        "parent_branch": parent_branch,
                        "structural_penalty": STRUCTURAL_PENALTY,
                        "stability": evidence.last_stability,
                    }
                )
                evidence.reset()
            structural = {
                "replicate": replicate,
                "step": step,
                "stage": stage,
                "subphase": row["subphase"],
                "active_k_before": active_before_growth,
                "active_k_after": active_k,
                "selected_k": selection.selected_k,
                "selected_cluster_purity_posthoc": selected_purity,
                "candidate_k": evidence.candidate_k,
                "candidate_stable_evaluations": evidence.stable_evaluations,
                "candidate_last_stability": evidence.last_stability,
                "shadow_loss": shadow_loss,
            }
            for fit in selection.fits:
                structural[f"k{fit.k}_normalized_residual"] = fit.normalized_residual
                structural[f"k{fit.k}_objective"] = fit.objective
                structural[f"k{fit.k}_min_fraction"] = fit.min_cluster_fraction
            structural_rows.append(structural)

        routing_rows.append(
            {
                "replicate": replicate,
                "step": step,
                "stage": stage,
                "subphase": row["subphase"],
                "stream_key_posthoc": stream_key,
                "family": family,
                "branch": branch,
                "route_margin": route_margin,
                "active_k_before": active_before_sensor,
                "active_k_after": active_k,
            }
        )

        if step in end_steps.values():
            current_stage = next(name for name, end in end_steps.items() if end == step)
            stage_summary, stage_eval = evaluate_stage(
                model,
                streams,
                validation,
                parent_losses,
                replicate=replicate,
                stage=current_stage,
                active_k=active_k,
                routing_rows=routing_rows,
            )
            stage_summary["genesis_events_in_stage"] = sum(
                int(event["stage"] == current_stage) for event in genesis_rows
            )
            stage_summary["max_active_k_so_far"] = max(
                [1, *[int(event["to_k"]) for event in genesis_rows]]
            )
            stage_rows.append(stage_summary)
            evaluation_rows.extend(stage_eval)
            save_stage_checkpoint(
                checkpoint_dir / f"r{replicate}-{current_stage.lower()}.pt",
                replicate=replicate,
                stage=current_stage,
                completed_step=step,
                model=model,
                optimizer=optimizer,
                active_k=active_k,
                route_centroids=route_centroids,
                evidence=evidence,
                sensor_gradients=sensor_gradients,
                sensor_labels=sensor_labels,
                occurrence=occurrence,
                structural_rows=structural_rows,
                routing_rows=routing_rows,
                genesis_rows=genesis_rows,
                stage_rows=stage_rows,
                evaluation_rows=evaluation_rows,
            )

    online_wall = time.perf_counter() - started
    pd.DataFrame(structural_rows).to_csv(output_dir / f"r{replicate}-structure.csv", index=False)
    pd.DataFrame(routing_rows).to_csv(output_dir / f"r{replicate}-routing.csv", index=False)
    pd.DataFrame(genesis_rows).to_csv(output_dir / f"r{replicate}-genesis.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(output_dir / f"r{replicate}-stage-summary.csv", index=False)
    pd.DataFrame(evaluation_rows).to_csv(output_dir / f"r{replicate}-evaluation.csv", index=False)
    pd.DataFrame(pretrain_curve).to_csv(output_dir / f"r{replicate}-pretrain.csv", index=False)

    worker = {
        "format": "minicells.online-nonparametric-trait-genesis-worker.v1",
        "experiment": "023",
        "replicate": replicate,
        "gpu": torch.cuda.get_device_name(0),
        "vocab_size": vocab_size,
        "structural_penalty": STRUCTURAL_PENALTY,
        "sensor_buffer": SENSOR_BUFFER,
        "sensor_interval": SENSOR_INTERVAL,
        "persistence_evaluations": PERSISTENCE_EVALS,
        "mode_stability_min": MODE_STABILITY_MIN,
        "max_traits": MAX_TRAITS,
        "trigger_uses_task_label": False,
        "routing_uses_task_label": False,
        "shadow_sensor": "fixed parent phenotype gradient",
        "pretrain_steps": PRETRAIN_STEPS,
        "online_steps": len(curriculum),
        "parent_losses": parent_losses,
        "final_active_k": active_k,
        "genesis_events": genesis_rows,
        "pretrain_wall_seconds": pretrain_wall,
        "online_wall_seconds": online_wall,
        "checkpoint_files": 1 + len(STAGES),
    }
    (output_dir / f"r{replicate}-worker.json").write_text(
        json.dumps(worker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "replicate": replicate,
                "final_active_k": active_k,
                "genesis": [(row["stage"], row["to_k"]) for row in genesis_rows],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
