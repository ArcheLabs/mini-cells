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
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "src").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))

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
    OnlineTraitTextNCA,
    align_growth_centroids,
    align_same_k,
    cluster_purity,
    fit_k_modes,
    prepare_transform_cache,
    route_to_centroid,
    summarize_multi_identity,
)
from minicells.language_sequential_probationary_genesis import (  # noqa: E402
    MAX_TRAITS,
    PROBATION_STEPS,
    PROBATION_WINDOWS,
    PROPOSAL_BATCHES,
    ROUTING_PURITY_MIN,
    STAGES,
    STEPS_PER_WINDOW,
    capacity_shadow_branch,
    semantic_family,
    stage_schedule,
    stage_spec,
    summarize_stage_decision,
)


CHECKPOINT_FORMAT = "minicells.sequential-probationary-genesis-checkpoint.v1"
N_REPLICATES = 3
MODEL_SEED_BASE = 124_024
SCHEDULE_SEED_BASE = 224_024
ARMS = ("incumbent", "capacity-shadow", "geometry-shadow")
FAMILIES = ("STORY", "ARITHMETIC", "TRANSFORM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 024 replicate")
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


def optimizer_state_cpu(optimizer: torch.optim.Optimizer) -> dict[str, object]:
    state = copy.deepcopy(optimizer.state_dict())
    for values in state.get("state", {}).values():
        for key, value in list(values.items()):
            if torch.is_tensor(value):
                values[key] = value.detach().cpu().clone()
    return state


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


def configure_online(model: OnlineTraitTextNCA) -> None:
    model.freeze_for_online_development()


def new_optimizer(model: OnlineTraitTextNCA) -> torch.optim.AdamW:
    return torch.optim.AdamW([model.online_traits], lr=PHENOTYPE_LR, weight_decay=0.0)


def clone_model_optimizer(
    model_state: dict[str, torch.Tensor],
    optimizer_state: dict[str, object],
    *,
    vocab_size: int,
    device: torch.device,
) -> tuple[OnlineTraitTextNCA, torch.optim.AdamW]:
    model = OnlineTraitTextNCA(vocab_size, max_traits=MAX_TRAITS).to(device)
    model.load_state_dict(copy.deepcopy(model_state))
    configure_online(model)
    optimizer = new_optimizer(model)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    return model, optimizer


def inherit_optimizer_row(
    optimizer: torch.optim.AdamW,
    *,
    parent_branch: int,
    newborn_branch: int,
) -> None:
    parameter = optimizer.param_groups[0]["params"][0]
    state = optimizer.state.get(parameter)
    if not state:
        return
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(key)
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[0] > newborn_branch:
            value[newborn_branch].copy_(value[parent_branch])


def initialize_candidate(
    model: OnlineTraitTextNCA,
    optimizer: torch.optim.AdamW,
    *,
    active_k: int,
    active_centroids: torch.Tensor,
    candidate_centroids: torch.Tensor,
    parent_branch: int,
) -> None:
    if active_k == 1:
        model.spawn_first_bifurcation(candidate_centroids)
        inherit_optimizer_row(optimizer, parent_branch=0, newborn_branch=1)
        return
    newborn = active_k
    model.spawn_additional_trait(
        new_branch=newborn,
        parent_branch=parent_branch,
        parent_centroid=active_centroids[parent_branch],
        new_centroid=candidate_centroids[newborn],
    )
    inherit_optimizer_row(optimizer, parent_branch=parent_branch, newborn_branch=newborn)


def pretrain_parent(
    model: OnlineTraitTextNCA,
    story_train: torch.Tensor,
    *,
    replicate: int,
    device: torch.device,
    checkpoint_path: Path,
):
    if checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            payload.get("format") != CHECKPOINT_FORMAT
            or payload.get("kind") != "parent"
            or int(payload.get("replicate", -1)) != replicate
        ):
            raise RuntimeError(f"invalid Experiment 024 parent checkpoint: {checkpoint_path}")
        model.load_state_dict(payload["model_state"])
        configure_online(model)
        optimizer = new_optimizer(model)
        optimizer.load_state_dict(payload["optimizer_state"])
        return model, optimizer, list(payload.get("learning_curve", [])), float(payload.get("wall_seconds", 0.0))

    schedule = deterministic_starts(
        len(story_train),
        steps=PRETRAIN_STEPS,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        seed=SCHEDULE_SEED_BASE + 1000 * replicate,
    )
    optimizer_pre = torch.optim.AdamW(
        model.pretrain_parameters(), lr=PRETRAIN_LR, betas=(0.9, 0.95), weight_decay=0.1
    )
    rows = []
    started = time.perf_counter()
    model.train()
    for step in range(1, PRETRAIN_STEPS + 1):
        for group in optimizer_pre.param_groups:
            group["lr"] = PRETRAIN_LR * lr_multiplier(step, PRETRAIN_STEPS)
        inputs, targets = make_batch(story_train, schedule[step - 1], device)
        optimizer_pre.zero_grad(set_to_none=True)
        loss = language_model_loss(model.forward_parent(inputs).float(), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.pretrain_parameters(), 1.0)
        optimizer_pre.step()
        if step in (1, 50, 100, 200, PRETRAIN_STEPS):
            rows.append({"step": step, "story_train_nll": float(loss.detach())})
    wall = time.perf_counter() - started
    model.initialize_online_population()
    configure_online(model)
    optimizer = new_optimizer(model)
    atomic_save(
        checkpoint_path,
        {
            "format": CHECKPOINT_FORMAT,
            "experiment": "024",
            "kind": "parent",
            "replicate": replicate,
            "model_state": cpu_state_dict(model),
            "optimizer_state": optimizer_state_cpu(optimizer),
            "learning_curve": rows,
            "wall_seconds": wall,
        },
    )
    return model, optimizer, rows, wall


def starts_for_schedule(
    streams: dict[str, object],
    schedule: tuple[str, ...],
    *,
    replicate: int,
    seed_offset: int,
):
    result = {}
    for index, key in enumerate(sorted(set(schedule))):
        result[key] = deterministic_starts(
            len(streams[key]["train"]),
            steps=len(schedule),
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            seed=SCHEDULE_SEED_BASE + seed_offset + 10_000 * replicate + index,
        )
    return result


def validation_schedules(streams: dict[str, object], *, replicate: int):
    keys = {"STORY": "STORY", "ARITHMETIC": "ARITH_A", "TRANSFORM": "TRANSFORM"}
    result = {}
    for index, (family, key) in enumerate(keys.items()):
        result[family] = deterministic_starts(
            len(streams[key]["validation"]),
            steps=16,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            seed=SCHEDULE_SEED_BASE + 30_000 * replicate + index,
        )
    return result


@torch.no_grad()
def evaluate_online_trait(
    model: OnlineTraitTextNCA,
    stream: torch.Tensor,
    starts: tuple[tuple[int, ...], ...],
    *,
    branch: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for row in starts:
        inputs, targets = make_batch(stream, row, device)
        losses.append(float(language_model_loss(model.forward_trait(inputs, branch).float(), targets)))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_parent_trait(
    model: OnlineTraitTextNCA,
    stream: torch.Tensor,
    starts: tuple[tuple[int, ...], ...],
    *,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for row in starts:
        inputs, targets = make_batch(stream, row, device)
        losses.append(float(language_model_loss(model.forward_parent(inputs).float(), targets)))
    return float(np.mean(losses))


def proposal_geometry(
    sensor: OnlineTraitTextNCA,
    streams: dict[str, object],
    *,
    stage: str,
    active_k: int,
    previous_centroids: torch.Tensor | None,
    replicate: int,
    device: torch.device,
):
    schedule = stage_schedule(stage, replicate=replicate, steps=PROPOSAL_BATCHES)
    starts = starts_for_schedule(
        streams,
        schedule,
        replicate=replicate,
        seed_offset=100_000 + 1000 * STAGES.index(stage),
    )
    gradients = []
    labels = []
    losses = []
    for step, key in enumerate(schedule):
        inputs, targets = make_batch(streams[key]["train"], starts[key][step], device)
        gradient, loss = trait_gradient(sensor, inputs, targets)
        gradients.append(gradient.cpu())
        labels.append(semantic_family(key))
        losses.append(loss)
    tensor = torch.stack(gradients)
    active_raw, active_assignment, active_residual, active_min_fraction = fit_k_modes(tensor, active_k)
    if active_k > 1 and previous_centroids is not None:
        active_centroids = align_same_k(previous_centroids.cpu(), active_raw.cpu())
    else:
        active_centroids = active_raw.cpu()
    candidate_raw, candidate_assignment, candidate_residual, candidate_min_fraction = fit_k_modes(
        tensor, active_k + 1
    )
    candidate_centroids, newborn, parent_branch = align_growth_centroids(
        active_centroids.cpu(), candidate_raw.cpu()
    )
    gain = (active_residual - candidate_residual) / max(active_residual, 1e-12)
    purity = cluster_purity(candidate_assignment.cpu(), labels)
    proposal = {
        "replicate": replicate,
        "stage": stage,
        "active_k": active_k,
        "candidate_k": active_k + 1,
        "proposal_batches": len(schedule),
        "active_residual": active_residual,
        "candidate_residual": candidate_residual,
        "residual_gain": gain,
        "active_min_cluster_fraction": active_min_fraction,
        "candidate_min_cluster_fraction": candidate_min_fraction,
        "candidate_cluster_purity_posthoc": purity,
        "parent_branch": parent_branch,
        "newborn_branch": newborn,
        "mean_probe_nll": float(np.mean(losses)),
        "proposal_uses_task_label": 0,
    }
    return active_centroids, candidate_centroids, int(parent_branch), proposal


def evaluate_population(
    model: OnlineTraitTextNCA,
    streams: dict[str, object],
    validation: dict[str, tuple[tuple[int, ...], ...]],
    parent_losses: dict[str, float],
    *,
    active_k: int,
    replicate: int,
    stage: str,
    device: torch.device,
):
    family_to_key = {"STORY": "STORY", "ARITHMETIC": "ARITH_A", "TRANSFORM": "TRANSFORM"}
    rows = []
    losses = {}
    for family in FAMILIES:
        values = []
        for branch in range(active_k):
            nll = evaluate_online_trait(
                model,
                streams[family_to_key[family]]["validation"],
                validation[family],
                branch=branch,
                device=device,
            )
            values.append(nll)
            rows.append(
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
    identity_domains = ("STORY",) if active_k == 1 else (("STORY", "ARITHMETIC") if active_k == 2 else FAMILIES)
    identity = summarize_multi_identity(losses, parent_losses, tuple(identity_domains))
    return rows, identity


def run_stage(
    *,
    stage: str,
    organism_state: dict[str, torch.Tensor],
    organism_optimizer_state: dict[str, object],
    sensor: OnlineTraitTextNCA,
    active_k: int,
    previous_centroids: torch.Tensor | None,
    vocab_size: int,
    streams: dict[str, object],
    validation: dict[str, tuple[tuple[int, ...], ...]],
    parent_losses: dict[str, float],
    replicate: int,
    device: torch.device,
    checkpoint_path: Path,
):
    if checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            payload.get("format") != CHECKPOINT_FORMAT
            or payload.get("kind") != "stage"
            or payload.get("stage") != stage
            or int(payload.get("replicate", -1)) != replicate
        ):
            raise RuntimeError(f"invalid Experiment 024 stage checkpoint: {checkpoint_path}")
        return payload

    active_centroids, candidate_centroids, parent_branch, proposal = proposal_geometry(
        sensor,
        streams,
        stage=stage,
        active_k=active_k,
        previous_centroids=previous_centroids,
        replicate=replicate,
        device=device,
    )
    incumbent, incumbent_optimizer = clone_model_optimizer(
        organism_state, organism_optimizer_state, vocab_size=vocab_size, device=device
    )
    capacity, capacity_optimizer = clone_model_optimizer(
        organism_state, organism_optimizer_state, vocab_size=vocab_size, device=device
    )
    geometry, geometry_optimizer = clone_model_optimizer(
        organism_state, organism_optimizer_state, vocab_size=vocab_size, device=device
    )
    initialize_candidate(
        capacity,
        capacity_optimizer,
        active_k=active_k,
        active_centroids=active_centroids,
        candidate_centroids=candidate_centroids,
        parent_branch=parent_branch,
    )
    initialize_candidate(
        geometry,
        geometry_optimizer,
        active_k=active_k,
        active_centroids=active_centroids,
        candidate_centroids=candidate_centroids,
        parent_branch=parent_branch,
    )

    schedule = stage_schedule(stage, replicate=replicate)
    starts = starts_for_schedule(
        streams,
        schedule,
        replicate=replicate,
        seed_offset=200_000 + 1000 * STAGES.index(stage),
    )
    models = {"incumbent": incumbent, "capacity-shadow": capacity, "geometry-shadow": geometry}
    optimizers = {
        "incumbent": incumbent_optimizer,
        "capacity-shadow": capacity_optimizer,
        "geometry-shadow": geometry_optimizer,
    }
    current_window = {arm: [] for arm in ARMS}
    window_losses = {arm: [] for arm in ARMS}
    learning_rows = []
    routing_rows = []
    split_occurrence = {key: 0 for key in sorted(set(schedule))}
    started = time.perf_counter()

    for step, key in enumerate(schedule, start=1):
        inputs, targets = make_batch(streams[key]["train"], starts[key][step - 1], device)
        gradient, _ = trait_gradient(sensor, inputs, targets)
        gradient_cpu = gradient.cpu()
        incumbent_branch, incumbent_margin = route_to_centroid(gradient_cpu, active_centroids)
        geometry_branch, geometry_margin = route_to_centroid(gradient_cpu, candidate_centroids)
        occurrence = split_occurrence[key]
        capacity_branch = capacity_shadow_branch(
            incumbent_branch=incumbent_branch,
            parent_branch=parent_branch,
            newborn_branch=active_k,
            occurrence=occurrence,
            replicate=replicate,
        )
        if incumbent_branch == parent_branch:
            split_occurrence[key] += 1
        routes = {
            "incumbent": incumbent_branch,
            "capacity-shadow": capacity_branch,
            "geometry-shadow": geometry_branch,
        }
        margins = {
            "incumbent": incumbent_margin,
            "capacity-shadow": None,
            "geometry-shadow": geometry_margin,
        }
        for arm in ARMS:
            model = models[arm]
            optimizer = optimizers[arm]
            for group in optimizer.param_groups:
                group["lr"] = PHENOTYPE_LR * lr_multiplier(step, PROBATION_STEPS)
            optimizer.zero_grad(set_to_none=True)
            branch = routes[arm]
            pre_loss = language_model_loss(model.forward_trait(inputs, branch).float(), targets)
            current_window[arm].append(float(pre_loss.detach()))
            pre_loss.backward()
            torch.nn.utils.clip_grad_norm_([model.online_traits], 1.0)
            optimizer.step()
            routing_rows.append(
                {
                    "replicate": replicate,
                    "stage": stage,
                    "arm": arm,
                    "step": step,
                    "stream_key_posthoc": key,
                    "family_posthoc": semantic_family(key),
                    "branch": branch,
                    "route_margin": margins[arm],
                    "candidate_parent_branch": parent_branch,
                }
            )
        if step % STEPS_PER_WINDOW == 0:
            window = step // STEPS_PER_WINDOW - 1
            for arm in ARMS:
                value = float(np.mean(current_window[arm]))
                window_losses[arm].append(value)
                learning_rows.append(
                    {
                        "replicate": replicate,
                        "stage": stage,
                        "arm": arm,
                        "window": window,
                        "step_end": step,
                        "prequential_nll": value,
                    }
                )
                current_window[arm] = []

    if any(len(values) != PROBATION_WINDOWS for values in window_losses.values()):
        raise RuntimeError("Experiment 024 probation window accounting mismatch")
    decision = summarize_stage_decision(
        stage=stage,
        start_k=active_k,
        parent_window_losses=window_losses["incumbent"],
        capacity_window_losses=window_losses["capacity-shadow"],
        geometry_window_losses=window_losses["geometry-shadow"],
    )
    accepted = bool(decision.accepted)
    committed_model = geometry if accepted else incumbent
    committed_optimizer = geometry_optimizer if accepted else incumbent_optimizer
    end_k = decision.end_k
    committed_centroids = candidate_centroids if accepted else active_centroids

    evaluation_rows, identity = evaluate_population(
        committed_model,
        streams,
        validation,
        parent_losses,
        active_k=end_k,
        replicate=replicate,
        stage=stage,
        device=device,
    )
    geometry_routes = [row for row in routing_rows if row["arm"] == "geometry-shadow"]
    geometry_purity = cluster_purity(
        torch.tensor([int(row["branch"]) for row in geometry_routes], dtype=torch.long),
        [str(row["family_posthoc"]) for row in geometry_routes],
    )
    retention_domains = ("STORY", "ARITHMETIC") if end_k >= 2 else ("STORY",)
    retention_identity = summarize_multi_identity(
        {
            family: tuple(
                row["nll"]
                for row in evaluation_rows
                if row["family"] == family
            )
            for family in retention_domains
        },
        parent_losses,
        tuple(retention_domains),
    )
    summary = {
        "replicate": replicate,
        "stage": stage,
        "expected_outcome": stage_spec(stage).expected_outcome,
        "start_k": active_k,
        "end_k": end_k,
        "accepted": int(accepted),
        "parent_branch": parent_branch,
        "newborn_branch": active_k,
        "geometry_sustained_positive": int(decision.sustained_positive),
        "geometry_cumulative_positive": int(decision.cumulative_positive),
        "geometry_beats_capacity": int(decision.beats_capacity),
        "geometry_mean_net_utility_last3": decision.geometry_mean_net_utility_last3,
        "capacity_mean_net_utility_last3": decision.capacity_mean_net_utility_last3,
        "geometry_advantage_last3": decision.geometry_advantage_last3,
        "routing_purity_posthoc": geometry_purity,
        "routing_purity_pass": int(geometry_purity >= ROUTING_PURITY_MIN),
        "identity_pass": int(identity.passes),
        "identity_margin": identity.normalized_identity_margin,
        "identity_assignment": list(identity.assignment),
        "retention_identity_pass": int(retention_identity.passes),
        "retention_identity_margin": retention_identity.normalized_identity_margin,
        "wall_seconds": time.perf_counter() - started,
    }
    window_rows = []
    probation = summarize_stage_decision(
        stage=stage,
        start_k=active_k,
        parent_window_losses=window_losses["incumbent"],
        capacity_window_losses=window_losses["capacity-shadow"],
        geometry_window_losses=window_losses["geometry-shadow"],
    )
    from minicells.language_probationary_trait_genesis import summarize_probation
    utility = summarize_probation(
        window_losses["incumbent"],
        window_losses["capacity-shadow"],
        window_losses["geometry-shadow"],
    )
    for window in range(PROBATION_WINDOWS):
        window_rows.append(
            {
                "replicate": replicate,
                "stage": stage,
                "window": window,
                "incumbent_prequential_nll": window_losses["incumbent"][window],
                "capacity_prequential_nll": window_losses["capacity-shadow"][window],
                "geometry_prequential_nll": window_losses["geometry-shadow"][window],
                "capacity_net_utility": utility.capacity_window_net_utility[window],
                "geometry_net_utility": utility.geometry_window_net_utility[window],
                "geometry_advantage": utility.geometry_advantage[window],
            }
        )

    payload = {
        "format": CHECKPOINT_FORMAT,
        "experiment": "024",
        "kind": "stage",
        "replicate": replicate,
        "stage": stage,
        "proposal": proposal,
        "summary": summary,
        "window_rows": window_rows,
        "learning_rows": learning_rows,
        "routing_rows": routing_rows,
        "evaluation_rows": evaluation_rows,
        "committed_model_state": cpu_state_dict(committed_model),
        "committed_optimizer_state": optimizer_state_cpu(committed_optimizer),
        "active_k": end_k,
        "active_centroids": committed_centroids.detach().cpu(),
    }
    atomic_save(checkpoint_path, payload)
    return payload


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 024 requires CUDA")
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
    organism = OnlineTraitTextNCA(vocab_size, max_traits=MAX_TRAITS).to(device)
    parent_path = checkpoint_dir / f"r{replicate}-parent.pt"
    organism, organism_optimizer, pretrain_rows, pretrain_wall = pretrain_parent(
        organism,
        streams["STORY"]["train"],
        replicate=replicate,
        device=device,
        checkpoint_path=parent_path,
    )
    organism_state = cpu_state_dict(organism)
    organism_optimizer_state = optimizer_state_cpu(organism_optimizer)
    sensor = OnlineTraitTextNCA(vocab_size, max_traits=MAX_TRAITS).to(device)
    sensor.load_state_dict(copy.deepcopy(organism_state))
    for parameter in sensor.parameters():
        parameter.requires_grad_(False)
    sensor.parent_trait.requires_grad_(True)

    validation = validation_schedules(streams, replicate=replicate)
    parent_losses = {
        "STORY": evaluate_parent_trait(sensor, streams["STORY"]["validation"], validation["STORY"], device=device),
        "ARITHMETIC": evaluate_parent_trait(sensor, streams["ARITH_A"]["validation"], validation["ARITHMETIC"], device=device),
        "TRANSFORM": evaluate_parent_trait(sensor, streams["TRANSFORM"]["validation"], validation["TRANSFORM"], device=device),
    }

    active_k = 1
    active_centroids = None
    proposals = []
    summaries = []
    windows = []
    learning = []
    routing = []
    evaluation = []
    trajectory = [{"replicate": replicate, "point": "START", "active_k": 1}]

    for stage in STAGES:
        payload = run_stage(
            stage=stage,
            organism_state=organism_state,
            organism_optimizer_state=organism_optimizer_state,
            sensor=sensor,
            active_k=active_k,
            previous_centroids=active_centroids,
            vocab_size=vocab_size,
            streams=streams,
            validation=validation,
            parent_losses=parent_losses,
            replicate=replicate,
            device=device,
            checkpoint_path=checkpoint_dir / f"r{replicate}-{stage.lower()}.pt",
        )
        proposals.append(payload["proposal"])
        summaries.append(payload["summary"])
        windows.extend(payload["window_rows"])
        learning.extend(payload["learning_rows"])
        routing.extend(payload["routing_rows"])
        evaluation.extend(payload["evaluation_rows"])
        organism_state = payload["committed_model_state"]
        organism_optimizer_state = payload["committed_optimizer_state"]
        active_k = int(payload["active_k"])
        active_centroids = payload["active_centroids"].cpu()
        trajectory.append({"replicate": replicate, "point": stage, "active_k": active_k})

    pd.DataFrame(proposals).to_csv(output_dir / f"r{replicate}-proposal.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / f"r{replicate}-stage-summary.csv", index=False)
    pd.DataFrame(windows).to_csv(output_dir / f"r{replicate}-probation-windows.csv", index=False)
    pd.DataFrame(learning).to_csv(output_dir / f"r{replicate}-learning-curve.csv", index=False)
    pd.DataFrame(routing).to_csv(output_dir / f"r{replicate}-routing.csv", index=False)
    pd.DataFrame(evaluation).to_csv(output_dir / f"r{replicate}-evaluation.csv", index=False)
    pd.DataFrame(trajectory).to_csv(output_dir / f"r{replicate}-trajectory.csv", index=False)
    pd.DataFrame(pretrain_rows).to_csv(output_dir / f"r{replicate}-pretrain.csv", index=False)

    worker = {
        "format": "minicells.sequential-probationary-genesis-worker.v1",
        "experiment": "024",
        "replicate": replicate,
        "gpu": torch.cuda.get_device_name(0),
        "vocab_size": vocab_size,
        "stages": list(STAGES),
        "final_active_k": active_k,
        "proposal_uses_task_label": False,
        "geometry_routing_uses_task_label": False,
        "commit_uses_task_label": False,
        "capacity_control_uses_stream_identity_only_for_matched_split": True,
        "sensor": "fixed pretrained parent phenotype gradient",
        "parent_losses": parent_losses,
        "pretrain_steps": PRETRAIN_STEPS,
        "pretrain_wall_seconds": pretrain_wall,
        "checkpoint_files": 1 + len(STAGES),
    }
    (output_dir / f"r{replicate}-worker.json").write_text(
        json.dumps(worker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"replicate": replicate, "final_k": active_k, "summaries": summaries}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
