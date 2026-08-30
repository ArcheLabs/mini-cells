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
from torch.nn import functional as F

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "src").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_sequential_probationary_genesis_worker as base024  # noqa: E402
from minicells.language_conflict_differentiation import (  # noqa: E402
    BATCH_SIZE,
    FORK_EPSILON,
    PHENOTYPE_LR,
    SEQUENCE_LENGTH,
    deterministic_starts,
    language_model_loss,
    lr_multiplier,
    trait_gradient,
)
from minicells.language_data import batch_from_starts, load_tokenizer  # noqa: E402
from minicells.language_independent_third_trait import (  # noqa: E402
    CANDIDATES,
    MAX_TRAITS,
    PROBATION_WINDOWS,
    PROPOSAL_BATCHES,
    ROUTING_PURITY_MIN,
    SCREEN_STEPS,
    SCREEN_VALIDATION_BATCHES,
    STEPS_PER_WINDOW,
    prepare_candidate_caches,
    screening_score,
    selected_stage_schedule,
)
from minicells.language_online_trait_genesis import (  # noqa: E402
    OnlineTraitTextNCA,
    align_growth_centroids,
    align_same_k,
    cluster_purity,
    fit_k_modes,
    route_to_centroid,
    summarize_multi_identity,
)
from minicells.language_probationary_trait_genesis import summarize_probation  # noqa: E402


CHECKPOINT_FORMAT = "minicells.independent-third-trait-checkpoint.v1"
N_REPLICATES = 3
MODEL_SEED_BASE = 124_024
SCHEDULE_SEED_BASE = 224_024
ARMS = ("incumbent", "capacity-shadow", "geometry-shadow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 024b replicate phase")
    parser.add_argument("--replicate", type=int, choices=range(N_REPLICATES), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "challenge"), required=True)
    parser.add_argument("--candidate", choices=CANDIDATES)
    return parser.parse_args()


def atomic_save(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def make_batch(stream: torch.Tensor, starts: tuple[int, ...], device: torch.device):
    return batch_from_starts(stream, starts, SEQUENCE_LENGTH, device)


def load_all_streams(cache_dir: Path):
    base = base024.load_streams(cache_dir)
    tokenizer = load_tokenizer(cache_dir / "tokenizer.json")
    candidates = prepare_candidate_caches(cache_dir, tokenizer)
    for name, payload in candidates.items():
        base[name] = {"train": payload["train"], "validation": payload["validation"]}
    base["candidate_manifests"] = {name: payload["manifest"] for name, payload in candidates.items()}
    return base


def validation_starts(length: int, *, replicate: int, offset: int):
    return deterministic_starts(
        length,
        steps=SCREEN_VALIDATION_BATCHES,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        seed=SCHEDULE_SEED_BASE + 700_000 + 10_000 * replicate + offset,
    )


def train_one_branch(
    model: OnlineTraitTextNCA,
    optimizer: torch.optim.AdamW,
    *,
    branch: int,
    stream: torch.Tensor,
    schedule: tuple[tuple[int, ...], ...],
    device: torch.device,
) -> None:
    model.train()
    for step, starts in enumerate(schedule, start=1):
        for group in optimizer.param_groups:
            group["lr"] = PHENOTYPE_LR * lr_multiplier(step, len(schedule))
        inputs, targets = make_batch(stream, starts, device)
        optimizer.zero_grad(set_to_none=True)
        loss = language_model_loss(model.forward_trait(inputs, branch).float(), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([model.online_traits], 1.0)
        optimizer.step()


def mean_candidate_gradient(
    sensor: OnlineTraitTextNCA,
    stream: torch.Tensor,
    *,
    replicate: int,
    candidate_index: int,
    device: torch.device,
) -> torch.Tensor:
    starts = deterministic_starts(
        len(stream),
        steps=16,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        seed=SCHEDULE_SEED_BASE + 500_000 + 10_000 * replicate + candidate_index,
    )
    gradients = []
    for row in starts:
        inputs, targets = make_batch(stream, row, device)
        gradient, _ = trait_gradient(sensor, inputs, targets)
        gradients.append(gradient.detach().cpu())
    mean = torch.stack(gradients).mean(dim=0)
    return F.normalize(mean.float(), dim=0, eps=1e-8)


def screen_candidates(
    *,
    foundation: dict[str, object],
    parent_state: dict[str, torch.Tensor],
    vocab_size: int,
    streams: dict[str, object],
    replicate: int,
    device: torch.device,
):
    active_k = int(foundation["active_k"])
    if active_k != 2:
        return []
    state = foundation["committed_model_state"]
    optimizer_state = foundation["committed_optimizer_state"]
    model, _ = base024.clone_model_optimizer(
        state, optimizer_state, vocab_size=vocab_size, device=device
    )
    sensor = OnlineTraitTextNCA(vocab_size, max_traits=MAX_TRAITS).to(device)
    sensor.load_state_dict(copy.deepcopy(parent_state))
    for parameter in sensor.parameters():
        parameter.requires_grad_(False)
    sensor.parent_trait.requires_grad_(True)

    arithmetic_validation = validation_starts(
        len(streams["ARITH_A"]["validation"]), replicate=replicate, offset=1
    )
    arithmetic_losses = [
        base024.evaluate_online_trait(
            model,
            streams["ARITH_A"]["validation"],
            arithmetic_validation,
            branch=branch,
            device=device,
        )
        for branch in range(2)
    ]
    computational_branch = int(np.argmin(arithmetic_losses))
    baseline_arithmetic = float(arithmetic_losses[computational_branch])
    rows = []

    for index, candidate in enumerate(CANDIDATES):
        train_stream = streams[candidate]["train"]
        validation_stream = streams[candidate]["validation"]
        train_schedule = deterministic_starts(
            len(train_stream),
            steps=SCREEN_STEPS,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            seed=SCHEDULE_SEED_BASE + 600_000 + 10_000 * replicate + index,
        )
        candidate_validation = validation_starts(
            len(validation_stream), replicate=replicate, offset=100 + index
        )
        baseline_candidate = base024.evaluate_online_trait(
            model,
            validation_stream,
            candidate_validation,
            branch=computational_branch,
            device=device,
        )

        existing, existing_optimizer = base024.clone_model_optimizer(
            state, optimizer_state, vocab_size=vocab_size, device=device
        )
        train_one_branch(
            existing,
            existing_optimizer,
            branch=computational_branch,
            stream=train_stream,
            schedule=train_schedule,
            device=device,
        )
        existing_candidate = base024.evaluate_online_trait(
            existing,
            validation_stream,
            candidate_validation,
            branch=computational_branch,
            device=device,
        )
        existing_arithmetic = base024.evaluate_online_trait(
            existing,
            streams["ARITH_A"]["validation"],
            arithmetic_validation,
            branch=computational_branch,
            device=device,
        )

        newborn, newborn_optimizer = base024.clone_model_optimizer(
            state, optimizer_state, vocab_size=vocab_size, device=device
        )
        direction = mean_candidate_gradient(
            sensor,
            train_stream,
            replicate=replicate,
            candidate_index=index,
            device=device,
        ).to(device=newborn.online_traits.device, dtype=newborn.online_traits.dtype)
        with torch.no_grad():
            newborn.online_traits[2].copy_(
                newborn.online_traits[computational_branch] + FORK_EPSILON * direction
            )
        base024.inherit_optimizer_row(
            newborn_optimizer, parent_branch=computational_branch, newborn_branch=2
        )
        train_one_branch(
            newborn,
            newborn_optimizer,
            branch=2,
            stream=train_stream,
            schedule=train_schedule,
            device=device,
        )
        newborn_candidate = base024.evaluate_online_trait(
            newborn,
            validation_stream,
            candidate_validation,
            branch=2,
            device=device,
        )
        score = screening_score(
            baseline_candidate_nll=baseline_candidate,
            existing_candidate_nll=existing_candidate,
            newborn_candidate_nll=newborn_candidate,
            baseline_arithmetic_nll=baseline_arithmetic,
            existing_arithmetic_nll=existing_arithmetic,
        )
        rows.append(
            {
                "replicate": replicate,
                "candidate": candidate,
                "computational_branch": computational_branch,
                **score.__dict__,
            }
        )
    return rows


def run_screen_phase(args: argparse.Namespace, streams: dict[str, object], device: torch.device) -> int:
    replicate = int(args.replicate)
    output = args.output_dir.resolve()
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vocab_size = int(streams["vocab_size"])
    torch.manual_seed(MODEL_SEED_BASE + 1000 * replicate)
    torch.cuda.manual_seed_all(MODEL_SEED_BASE + 1000 * replicate)

    organism = OnlineTraitTextNCA(vocab_size, max_traits=MAX_TRAITS).to(device)
    organism, optimizer, pretrain_rows, pretrain_wall = base024.pretrain_parent(
        organism,
        streams["STORY"]["train"],
        replicate=replicate,
        device=device,
        checkpoint_path=checkpoint_dir / f"r{replicate}-parent.pt",
    )
    parent_state = base024.cpu_state_dict(organism)
    optimizer_state = base024.optimizer_state_cpu(optimizer)
    sensor = OnlineTraitTextNCA(vocab_size, max_traits=MAX_TRAITS).to(device)
    sensor.load_state_dict(copy.deepcopy(parent_state))
    for parameter in sensor.parameters():
        parameter.requires_grad_(False)
    sensor.parent_trait.requires_grad_(True)
    validation = base024.validation_schedules(streams, replicate=replicate)
    parent_losses = {
        "STORY": base024.evaluate_parent_trait(
            sensor, streams["STORY"]["validation"], validation["STORY"], device=device
        ),
        "ARITHMETIC": base024.evaluate_parent_trait(
            sensor, streams["ARITH_A"]["validation"], validation["ARITHMETIC"], device=device
        ),
        "TRANSFORM": base024.evaluate_parent_trait(
            sensor, streams["TRANSFORM"]["validation"], validation["TRANSFORM"], device=device
        ),
    }
    foundation = base024.run_stage(
        stage="B_ARITHMETIC_BIRTH",
        organism_state=parent_state,
        organism_optimizer_state=optimizer_state,
        sensor=sensor,
        active_k=1,
        previous_centroids=None,
        vocab_size=vocab_size,
        streams=streams,
        validation=validation,
        parent_losses=parent_losses,
        replicate=replicate,
        device=device,
        checkpoint_path=checkpoint_dir / f"r{replicate}-arithmetic-birth.pt",
    )
    screening_rows = screen_candidates(
        foundation=foundation,
        parent_state=parent_state,
        vocab_size=vocab_size,
        streams=streams,
        replicate=replicate,
        device=device,
    )
    columns = [
        "replicate", "candidate", "computational_branch", "baseline_candidate_nll",
        "existing_candidate_nll", "newborn_candidate_nll", "baseline_arithmetic_nll",
        "existing_arithmetic_nll", "existing_candidate_gain", "arithmetic_damage",
        "existing_value", "newborn_candidate_gain", "newborn_value",
        "independence_advantage", "absorption_ratio", "qualifies",
    ]
    pd.DataFrame(screening_rows, columns=columns).to_csv(
        output / f"r{replicate}-screening.csv", index=False
    )
    screen_checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "experiment": "024b",
        "kind": "screen",
        "replicate": replicate,
        "parent_state": parent_state,
        "foundation_model_state": foundation["committed_model_state"],
        "foundation_optimizer_state": foundation["committed_optimizer_state"],
        "active_k": int(foundation["active_k"]),
        "active_centroids": foundation["active_centroids"].cpu(),
        "foundation_summary": foundation["summary"],
        "screening_rows": screening_rows,
    }
    atomic_save(checkpoint_dir / f"r{replicate}-screen.pt", screen_checkpoint)
    worker = {
        "format": "minicells.independent-third-trait-screen-worker.v1",
        "experiment": "024b",
        "replicate": replicate,
        "gpu": torch.cuda.get_device_name(0),
        "foundation_active_k": int(foundation["active_k"]),
        "foundation_summary": foundation["summary"],
        "screening_candidates": list(CANDIDATES),
        "screening_rows": len(screening_rows),
        "pretrain_wall_seconds": pretrain_wall,
    }
    (output / f"r{replicate}-screen-worker.json").write_text(
        json.dumps(worker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(pretrain_rows).to_csv(output / f"r{replicate}-pretrain.csv", index=False)
    print(json.dumps(worker, sort_keys=True))
    return 0


def generic_starts(
    streams: dict[str, object], schedule: tuple[str, ...], *, replicate: int, offset: int
):
    result = {}
    for index, key in enumerate(sorted(set(schedule))):
        result[key] = deterministic_starts(
            len(streams[key]["train"]),
            steps=len(schedule),
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            seed=SCHEDULE_SEED_BASE + offset + 10_000 * replicate + index,
        )
    return result


def family_for_key(key: str, candidate: str) -> str:
    if key == "STORY":
        return "STORY"
    if key.startswith("ARITH"):
        return "ARITHMETIC"
    if key == candidate:
        return "SELECTED"
    return key


def challenge_validation(streams: dict[str, object], candidate: str, replicate: int):
    keys = {"STORY": "STORY", "ARITHMETIC": "ARITH_A", "SELECTED": candidate}
    return {
        family: validation_starts(
            len(streams[key]["validation"]), replicate=replicate, offset=300 + index
        )
        for index, (family, key) in enumerate(keys.items())
    }


def evaluate_three_family_population(
    model: OnlineTraitTextNCA,
    streams: dict[str, object],
    validation: dict[str, tuple[tuple[int, ...], ...]],
    parent_losses: dict[str, float],
    *,
    candidate: str,
    active_k: int,
    replicate: int,
    stage: str,
    device: torch.device,
):
    keys = {"STORY": "STORY", "ARITHMETIC": "ARITH_A", "SELECTED": candidate}
    rows = []
    losses: dict[str, tuple[float, ...]] = {}
    for family, key in keys.items():
        values = []
        for branch in range(active_k):
            nll = base024.evaluate_online_trait(
                model,
                streams[key]["validation"],
                validation[family],
                branch=branch,
                device=device,
            )
            values.append(nll)
            rows.append(
                {
                    "replicate": replicate,
                    "stage": stage,
                    "candidate": candidate,
                    "family": family,
                    "branch": branch,
                    "nll": nll,
                    "parent_nll": parent_losses[family],
                    "utility": parent_losses[family] - nll,
                }
            )
        losses[family] = tuple(values)
    domains = ("STORY", "ARITHMETIC") if active_k == 2 else ("STORY", "ARITHMETIC", "SELECTED")
    identity = summarize_multi_identity(losses, parent_losses, domains)
    retention = summarize_multi_identity(
        {key: losses[key] for key in ("STORY", "ARITHMETIC")},
        parent_losses,
        ("STORY", "ARITHMETIC"),
    )
    return rows, identity, retention


def run_selected_stage(
    *,
    stage: str,
    weak: bool,
    candidate: str,
    organism_state: dict[str, torch.Tensor],
    organism_optimizer_state: dict[str, object],
    sensor: OnlineTraitTextNCA,
    active_centroids: torch.Tensor,
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
            payload.get("format") == CHECKPOINT_FORMAT
            and payload.get("kind") == "challenge-stage"
            and payload.get("stage") == stage
            and payload.get("candidate") == candidate
            and int(payload.get("replicate", -1)) == replicate
        ):
            return payload
        raise RuntimeError(f"invalid Experiment 024b challenge checkpoint: {checkpoint_path}")

    full_schedule = selected_stage_schedule(candidate, weak=weak, replicate=replicate)
    probe_schedule = full_schedule[:PROPOSAL_BATCHES]
    probe_starts = generic_starts(
        streams, probe_schedule, replicate=replicate, offset=800_000 + (0 if weak else 10_000)
    )
    gradients = []
    labels = []
    for step, key in enumerate(probe_schedule):
        inputs, targets = make_batch(streams[key]["train"], probe_starts[key][step], device)
        gradient, _ = trait_gradient(sensor, inputs, targets)
        gradients.append(gradient.cpu())
        labels.append(family_for_key(key, candidate))
    tensor = torch.stack(gradients)
    active_raw, _, active_residual, active_min = fit_k_modes(tensor, 2)
    active_aligned = align_same_k(active_centroids.cpu(), active_raw.cpu())
    candidate_raw, candidate_assignment, candidate_residual, candidate_min = fit_k_modes(tensor, 3)
    candidate_centroids, newborn, parent_branch = align_growth_centroids(
        active_aligned.cpu(), candidate_raw.cpu()
    )
    proposal = {
        "replicate": replicate,
        "stage": stage,
        "candidate": candidate,
        "active_k": 2,
        "candidate_k": 3,
        "active_residual": active_residual,
        "candidate_residual": candidate_residual,
        "residual_gain": (active_residual - candidate_residual) / max(active_residual, 1e-12),
        "active_min_cluster_fraction": active_min,
        "candidate_min_cluster_fraction": candidate_min,
        "candidate_cluster_purity_posthoc": cluster_purity(candidate_assignment.cpu(), labels),
        "parent_branch": int(parent_branch),
        "newborn_branch": int(newborn),
        "proposal_uses_task_label": 0,
    }

    incumbent, incumbent_optimizer = base024.clone_model_optimizer(
        organism_state, organism_optimizer_state, vocab_size=vocab_size, device=device
    )
    capacity, capacity_optimizer = base024.clone_model_optimizer(
        organism_state, organism_optimizer_state, vocab_size=vocab_size, device=device
    )
    geometry, geometry_optimizer = base024.clone_model_optimizer(
        organism_state, organism_optimizer_state, vocab_size=vocab_size, device=device
    )
    for model, optimizer in ((capacity, capacity_optimizer), (geometry, geometry_optimizer)):
        base024.initialize_candidate(
            model,
            optimizer,
            active_k=2,
            active_centroids=active_aligned,
            candidate_centroids=candidate_centroids,
            parent_branch=int(parent_branch),
        )

    starts = generic_starts(
        streams, full_schedule, replicate=replicate, offset=900_000 + (0 if weak else 10_000)
    )
    models = {"incumbent": incumbent, "capacity-shadow": capacity, "geometry-shadow": geometry}
    optimizers = {
        "incumbent": incumbent_optimizer,
        "capacity-shadow": capacity_optimizer,
        "geometry-shadow": geometry_optimizer,
    }
    window = {arm: [] for arm in ARMS}
    window_losses = {arm: [] for arm in ARMS}
    routing_rows = []
    learning_rows = []
    occurrence = {key: 0 for key in sorted(set(full_schedule))}
    started = time.perf_counter()
    for step, key in enumerate(full_schedule, start=1):
        inputs, targets = make_batch(streams[key]["train"], starts[key][step - 1], device)
        gradient, _ = trait_gradient(sensor, inputs, targets)
        g = gradient.cpu()
        incumbent_branch, incumbent_margin = route_to_centroid(g, active_aligned)
        geometry_branch, geometry_margin = route_to_centroid(g, candidate_centroids)
        capacity_branch = base024.capacity_shadow_branch(
            incumbent_branch=incumbent_branch,
            parent_branch=int(parent_branch),
            newborn_branch=2,
            occurrence=occurrence[key],
            replicate=replicate,
        )
        if incumbent_branch == int(parent_branch):
            occurrence[key] += 1
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
                group["lr"] = PHENOTYPE_LR * lr_multiplier(step, len(full_schedule))
            optimizer.zero_grad(set_to_none=True)
            branch = routes[arm]
            loss = language_model_loss(model.forward_trait(inputs, branch).float(), targets)
            window[arm].append(float(loss.detach()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([model.online_traits], 1.0)
            optimizer.step()
            routing_rows.append(
                {
                    "replicate": replicate,
                    "stage": stage,
                    "candidate": candidate,
                    "arm": arm,
                    "step": step,
                    "stream_key_posthoc": key,
                    "family_posthoc": family_for_key(key, candidate),
                    "branch": branch,
                    "route_margin": margins[arm],
                    "candidate_parent_branch": int(parent_branch),
                }
            )
        if step % STEPS_PER_WINDOW == 0:
            window_index = step // STEPS_PER_WINDOW - 1
            for arm in ARMS:
                value = float(np.mean(window[arm]))
                window_losses[arm].append(value)
                learning_rows.append(
                    {
                        "replicate": replicate,
                        "stage": stage,
                        "candidate": candidate,
                        "arm": arm,
                        "window": window_index,
                        "prequential_nll": value,
                    }
                )
                window[arm] = []
    utility = summarize_probation(
        window_losses["incumbent"],
        window_losses["capacity-shadow"],
        window_losses["geometry-shadow"],
    )
    accepted = bool(utility.sustained_positive and utility.cumulative_positive and utility.beats_capacity)
    committed_model = geometry if accepted else incumbent
    committed_optimizer = geometry_optimizer if accepted else incumbent_optimizer
    end_k = 3 if accepted else 2
    committed_centroids = candidate_centroids if accepted else active_aligned
    evaluation_rows, identity, retention = evaluate_three_family_population(
        committed_model,
        streams,
        validation,
        parent_losses,
        candidate=candidate,
        active_k=end_k,
        replicate=replicate,
        stage=stage,
        device=device,
    )
    geometry_routes = [row for row in routing_rows if row["arm"] == "geometry-shadow"]
    purity = cluster_purity(
        torch.tensor([int(row["branch"]) for row in geometry_routes]),
        [str(row["family_posthoc"]) for row in geometry_routes],
    )
    summary = {
        "replicate": replicate,
        "stage": stage,
        "candidate": candidate,
        "expected_outcome": "REJECT" if weak else "ACCEPT",
        "start_k": 2,
        "end_k": end_k,
        "accepted": int(accepted),
        "parent_branch": int(parent_branch),
        "newborn_branch": 2,
        "geometry_sustained_positive": int(utility.sustained_positive),
        "geometry_cumulative_positive": int(utility.cumulative_positive),
        "geometry_beats_capacity": int(utility.beats_capacity),
        "geometry_mean_net_utility_last3": utility.geometry_mean_net_utility_last3,
        "capacity_mean_net_utility_last3": utility.capacity_mean_net_utility_last3,
        "geometry_advantage_last3": utility.geometry_advantage_last3,
        "routing_purity_posthoc": purity,
        "routing_purity_pass": int(purity >= ROUTING_PURITY_MIN),
        "identity_pass": int(identity.passes),
        "identity_margin": identity.normalized_identity_margin,
        "identity_assignment": list(identity.assignment),
        "retention_identity_pass": int(retention.passes),
        "retention_identity_margin": retention.normalized_identity_margin,
        "wall_seconds": time.perf_counter() - started,
    }
    window_rows = [
        {
            "replicate": replicate,
            "stage": stage,
            "candidate": candidate,
            "window": index,
            "incumbent_prequential_nll": window_losses["incumbent"][index],
            "capacity_prequential_nll": window_losses["capacity-shadow"][index],
            "geometry_prequential_nll": window_losses["geometry-shadow"][index],
            "capacity_net_utility": utility.capacity_window_net_utility[index],
            "geometry_net_utility": utility.geometry_window_net_utility[index],
            "geometry_advantage": utility.geometry_advantage[index],
        }
        for index in range(PROBATION_WINDOWS)
    ]
    payload = {
        "format": CHECKPOINT_FORMAT,
        "experiment": "024b",
        "kind": "challenge-stage",
        "replicate": replicate,
        "stage": stage,
        "candidate": candidate,
        "proposal": proposal,
        "summary": summary,
        "window_rows": window_rows,
        "learning_rows": learning_rows,
        "routing_rows": routing_rows,
        "evaluation_rows": evaluation_rows,
        "committed_model_state": base024.cpu_state_dict(committed_model),
        "committed_optimizer_state": base024.optimizer_state_cpu(committed_optimizer),
        "active_k": end_k,
        "active_centroids": committed_centroids.detach().cpu(),
    }
    atomic_save(checkpoint_path, payload)
    return payload


def run_challenge_phase(args: argparse.Namespace, streams: dict[str, object], device: torch.device) -> int:
    if args.candidate is None:
        raise ValueError("--candidate is required for challenge phase")
    candidate = str(args.candidate)
    replicate = int(args.replicate)
    output = args.output_dir.resolve()
    checkpoint_dir = output / "checkpoints"
    screen_path = checkpoint_dir / f"r{replicate}-screen.pt"
    if not screen_path.is_file():
        raise FileNotFoundError(screen_path)
    screen = torch.load(screen_path, map_location="cpu", weights_only=False)
    if screen.get("format") != CHECKPOINT_FORMAT or screen.get("kind") != "screen":
        raise RuntimeError(f"invalid Experiment 024b screen checkpoint: {screen_path}")
    foundation_summary = dict(screen["foundation_summary"])
    if int(screen["active_k"]) != 2:
        worker = {
            "format": "minicells.independent-third-trait-challenge-worker.v1",
            "experiment": "024b",
            "replicate": replicate,
            "candidate": candidate,
            "foundation_active_k": int(screen["active_k"]),
            "skipped": True,
            "reason": "first birth did not commit",
        }
        (output / f"r{replicate}-challenge-worker.json").write_text(
            json.dumps(worker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0

    vocab_size = int(streams["vocab_size"])
    parent_state = screen["parent_state"]
    sensor = OnlineTraitTextNCA(vocab_size, max_traits=MAX_TRAITS).to(device)
    sensor.load_state_dict(copy.deepcopy(parent_state))
    for parameter in sensor.parameters():
        parameter.requires_grad_(False)
    sensor.parent_trait.requires_grad_(True)
    validation = challenge_validation(streams, candidate, replicate)
    parent_losses = {
        "STORY": base024.evaluate_parent_trait(
            sensor, streams["STORY"]["validation"], validation["STORY"], device=device
        ),
        "ARITHMETIC": base024.evaluate_parent_trait(
            sensor, streams["ARITH_A"]["validation"], validation["ARITHMETIC"], device=device
        ),
        "SELECTED": base024.evaluate_parent_trait(
            sensor, streams[candidate]["validation"], validation["SELECTED"], device=device
        ),
    }
    organism_state = screen["foundation_model_state"]
    organism_optimizer_state = screen["foundation_optimizer_state"]
    centroids = screen["active_centroids"].cpu()
    payloads = []
    for stage, weak in (("C_WEAK_SELECTED", True), ("D_STRONG_SELECTED", False)):
        payload = run_selected_stage(
            stage=stage,
            weak=weak,
            candidate=candidate,
            organism_state=organism_state,
            organism_optimizer_state=organism_optimizer_state,
            sensor=sensor,
            active_centroids=centroids,
            vocab_size=vocab_size,
            streams=streams,
            validation=validation,
            parent_losses=parent_losses,
            replicate=replicate,
            device=device,
            checkpoint_path=checkpoint_dir / f"r{replicate}-{stage.lower()}.pt",
        )
        payloads.append(payload)
        organism_state = payload["committed_model_state"]
        organism_optimizer_state = payload["committed_optimizer_state"]
        centroids = payload["active_centroids"].cpu()
        if stage == "C_WEAK_SELECTED" and int(payload["active_k"]) != 2:
            break

    summaries = [payload["summary"] for payload in payloads]
    proposals = [payload["proposal"] for payload in payloads]
    windows = [row for payload in payloads for row in payload["window_rows"]]
    learning = [row for payload in payloads for row in payload["learning_rows"]]
    routing = [row for payload in payloads for row in payload["routing_rows"]]
    evaluation = [row for payload in payloads for row in payload["evaluation_rows"]]
    pd.DataFrame(summaries).to_csv(output / f"r{replicate}-challenge-stage-summary.csv", index=False)
    pd.DataFrame(proposals).to_csv(output / f"r{replicate}-challenge-proposal.csv", index=False)
    pd.DataFrame(windows).to_csv(output / f"r{replicate}-challenge-windows.csv", index=False)
    pd.DataFrame(learning).to_csv(output / f"r{replicate}-challenge-learning.csv", index=False)
    pd.DataFrame(routing).to_csv(output / f"r{replicate}-challenge-routing.csv", index=False)
    pd.DataFrame(evaluation).to_csv(output / f"r{replicate}-challenge-evaluation.csv", index=False)
    final_k = int(payloads[-1]["active_k"]) if payloads else 2
    worker = {
        "format": "minicells.independent-third-trait-challenge-worker.v1",
        "experiment": "024b",
        "replicate": replicate,
        "candidate": candidate,
        "foundation_summary": foundation_summary,
        "stages_completed": [payload["stage"] for payload in payloads],
        "final_active_k": final_k,
        "proposal_uses_task_label": False,
        "geometry_routing_uses_task_label": False,
        "commit_uses_task_label": False,
        "gpu": torch.cuda.get_device_name(0),
    }
    (output / f"r{replicate}-challenge-worker.json").write_text(
        json.dumps(worker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(worker, sort_keys=True))
    return 0


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 024b requires CUDA")
    device = torch.device("cuda:0")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    streams = load_all_streams(args.cache_dir.resolve())
    if args.phase == "screen":
        return run_screen_phase(args, streams, device)
    return run_challenge_phase(args, streams, device)


if __name__ == "__main__":
    raise SystemExit(main())
