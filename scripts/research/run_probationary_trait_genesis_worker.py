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
    DOMAINS,
    PHENOTYPE_LR,
    PRETRAIN_LR,
    PRETRAIN_STEPS,
    SEQUENCE_LENGTH,
    ForkableTextNCA,
    deterministic_starts,
    language_model_loss,
    lr_multiplier,
    prepare_arithmetic_cache,
    trait_gradient,
)
from minicells.language_data import batch_from_starts, load_tokenizer  # noqa: E402
from minicells.language_probationary_trait_genesis import (  # noqa: E402
    ARMS,
    CONDITIONS,
    PROBATION_STEPS,
    PROBATION_WINDOWS,
    PROPOSAL_BATCHES,
    ROUTING_PURITY_MIN,
    STEPS_PER_WINDOW,
    capacity_branch,
    condition_schedule,
    expected_condition_outcome,
    semantic_family,
    summarize_probation,
)
from minicells.language_trait_bifurcation import (  # noqa: E402
    fit_two_mode_gradient_field,
    route_gradient_to_mode,
    routing_purity_from_branches,
    summarize_identity,
)


CHECKPOINT_FORMAT = "minicells.probationary-trait-genesis-checkpoint.v1"
N_REPLICATES = 3
MODEL_SEED_BASE = 123_231
SCHEDULE_SEED_BASE = 223_231


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 023b replicate")
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
    return {
        "STORY_A": {"train": story_train, "validation": story_validation},
        "STORY_B": {"train": story_train, "validation": story_validation},
        "ARITHMETIC": {"train": arithmetic["train"], "validation": arithmetic["validation"]},
        "story_manifest": story_manifest,
        "arithmetic_manifest": arithmetic["manifest"],
        "arithmetic_manifest_path": arithmetic["path"],
        "vocab_size": int(tokenizer.get_vocab_size()),
    }


def clone_parent(parent_state: dict[str, torch.Tensor], vocab_size: int, device: torch.device) -> ForkableTextNCA:
    model = ForkableTextNCA(vocab_size).to(device)
    model.load_state_dict(copy.deepcopy(parent_state))
    return model


def configure_parent_adaptation(model: ForkableTextNCA) -> None:
    for parameter in model.base.parameters():
        parameter.requires_grad_(False)
    model.child_traits.requires_grad_(False)
    model.parent_trait.requires_grad_(True)


@torch.no_grad()
def evaluate_parent(
    model: ForkableTextNCA,
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


@torch.no_grad()
def evaluate_child(
    model: ForkableTextNCA,
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
        losses.append(float(language_model_loss(model.forward_child(inputs, branch).float(), targets)))
    return float(np.mean(losses))


def pretrain_parent(
    model: ForkableTextNCA,
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
            raise RuntimeError(f"invalid Experiment 023b parent checkpoint: {checkpoint_path}")
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
    rows = []
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
            "experiment": "023b",
            "kind": "parent",
            "replicate": replicate,
            "model_state": cpu_state_dict(model),
            "learning_curve": rows,
            "wall_seconds": wall,
        },
    )
    return model, rows, wall


def validation_schedules(streams: dict[str, object], *, replicate: int):
    result = {}
    for index, stream_key in enumerate(("STORY_A", "ARITHMETIC")):
        result[semantic_family(stream_key)] = deterministic_starts(
            len(streams[stream_key]["validation"]),
            steps=16,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            seed=SCHEDULE_SEED_BASE + 30_000 * replicate + index,
        )
    return result


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


def learn_proposal_geometry(
    sensor: ForkableTextNCA,
    streams: dict[str, object],
    *,
    condition: str,
    replicate: int,
    device: torch.device,
):
    schedule = condition_schedule(condition, replicate=replicate, steps=PROPOSAL_BATCHES)
    starts = starts_for_schedule(
        streams, schedule, replicate=replicate, seed_offset=100_000 + 1000 * CONDITIONS.index(condition)
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
    geometry = fit_two_mode_gradient_field(tensor)
    branches = [route_gradient_to_mode(row, geometry)[0] for row in tensor]
    purity = None
    if len(set(labels)) == 2:
        purity = routing_purity_from_branches(branches, labels)
    return geometry, {
        "replicate": replicate,
        "condition": condition,
        "proposal_batches": len(schedule),
        "bifurcation_gain": geometry.bifurcation_gain,
        "split_balance": geometry.split_balance,
        "centroid_separation": geometry.centroid_separation,
        "routing_purity_posthoc": purity,
        "mean_probe_nll": float(np.mean(losses)),
        "proposal_uses_task_label": 0,
    }


def train_arm(
    *,
    arm: str,
    parent_state: dict[str, torch.Tensor],
    vocab_size: int,
    geometry,
    streams: dict[str, object],
    condition: str,
    replicate: int,
    device: torch.device,
):
    model = clone_parent(parent_state, vocab_size, device)
    schedule = condition_schedule(condition, replicate=replicate)
    starts = starts_for_schedule(
        streams,
        schedule,
        replicate=replicate,
        seed_offset=200_000 + 1000 * CONDITIONS.index(condition),
    )
    routing_rows = []
    learning_rows = []
    window_losses: list[float] = []
    current_window: list[float] = []
    occurrence = {key: 0 for key in sorted(set(schedule))}

    if arm == "parent":
        configure_parent_adaptation(model)
        optimizer = torch.optim.AdamW([model.parent_trait], lr=PHENOTYPE_LR, weight_decay=0.0)
    else:
        model.initialize_children(geometry.axis, symmetry_break=True)
        model.freeze_genome_for_fork()
        optimizer = torch.optim.AdamW([model.child_traits], lr=PHENOTYPE_LR, weight_decay=0.0)

    started = time.perf_counter()
    model.train()
    for step, key in enumerate(schedule, start=1):
        for group in optimizer.param_groups:
            group["lr"] = PHENOTYPE_LR * lr_multiplier(step, PROBATION_STEPS)
        inputs, targets = make_batch(streams[key]["train"], starts[key][step - 1], device)
        family = semantic_family(key)
        score = None
        if arm == "parent":
            branch = None
            logits = model.forward_parent(inputs)
        elif arm == "capacity-shadow":
            branch = capacity_branch(occurrence[key], replicate)
            logits = model.forward_child(inputs, branch)
        elif arm == "geometry-shadow":
            gradient, _ = trait_gradient(model, inputs, targets)
            branch, score = route_gradient_to_mode(gradient.cpu(), geometry)
            logits = model.forward_child(inputs, branch)
        else:
            raise ValueError(arm)

        pre_loss = language_model_loss(logits.float(), targets)
        current_window.append(float(pre_loss.detach()))
        optimizer.zero_grad(set_to_none=True)
        pre_loss.backward()
        if arm == "parent":
            torch.nn.utils.clip_grad_norm_([model.parent_trait], 1.0)
        else:
            torch.nn.utils.clip_grad_norm_([model.child_traits], 1.0)
        optimizer.step()

        if arm != "parent":
            routing_rows.append(
                {
                    "replicate": replicate,
                    "condition": condition,
                    "arm": arm,
                    "step": step,
                    "stream_key_posthoc": key,
                    "family_posthoc": family,
                    "branch": branch,
                    "geometry_score": score,
                }
            )
        occurrence[key] += 1

        if step % STEPS_PER_WINDOW == 0:
            window = step // STEPS_PER_WINDOW - 1
            value = float(np.mean(current_window))
            window_losses.append(value)
            learning_rows.append(
                {
                    "replicate": replicate,
                    "condition": condition,
                    "arm": arm,
                    "window": window,
                    "step_end": step,
                    "prequential_nll": value,
                }
            )
            current_window = []

    if len(window_losses) != PROBATION_WINDOWS:
        raise RuntimeError("probation window accounting mismatch")
    return model, window_losses, learning_rows, routing_rows, time.perf_counter() - started


def evaluate_arm(
    model: ForkableTextNCA,
    streams: dict[str, object],
    validation: dict[str, tuple[tuple[int, ...], ...]],
    parent_losses: dict[str, float],
    *,
    arm: str,
    condition: str,
    replicate: int,
    device: torch.device,
):
    rows = []
    child_losses: dict[str, tuple[float, float]] = {}
    for family, key in (("STORY", "STORY_A"), ("ARITHMETIC", "ARITHMETIC")):
        if arm == "parent":
            nll = evaluate_parent(model, streams[key]["validation"], validation[family], device=device)
            rows.append(
                {
                    "replicate": replicate,
                    "condition": condition,
                    "arm": arm,
                    "family": family,
                    "branch": 0,
                    "nll": nll,
                    "parent_nll": parent_losses[family],
                    "utility": parent_losses[family] - nll,
                }
            )
        else:
            values = []
            for branch in (0, 1):
                nll = evaluate_child(
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
                        "condition": condition,
                        "arm": arm,
                        "family": family,
                        "branch": branch,
                        "nll": nll,
                        "parent_nll": parent_losses[family],
                        "utility": parent_losses[family] - nll,
                    }
                )
            child_losses[family] = (values[0], values[1])
    identity = None if arm == "parent" else summarize_identity(child_losses, parent_losses)
    return rows, identity


def run_condition(
    *,
    condition: str,
    parent_state: dict[str, torch.Tensor],
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
            or payload.get("kind") != "condition"
            or payload.get("condition") != condition
            or int(payload.get("replicate", -1)) != replicate
        ):
            raise RuntimeError(f"invalid Experiment 023b condition checkpoint: {checkpoint_path}")
        return payload

    sensor = clone_parent(parent_state, vocab_size, device)
    geometry, proposal = learn_proposal_geometry(
        sensor, streams, condition=condition, replicate=replicate, device=device
    )
    arm_models = {}
    arm_windows = {}
    learning_rows = []
    routing_rows = []
    wall = {}
    for arm in ARMS:
        model, windows, learning, routing, seconds = train_arm(
            arm=arm,
            parent_state=parent_state,
            vocab_size=vocab_size,
            geometry=geometry,
            streams=streams,
            condition=condition,
            replicate=replicate,
            device=device,
        )
        arm_models[arm] = model
        arm_windows[arm] = windows
        learning_rows.extend(learning)
        routing_rows.extend(routing)
        wall[arm] = seconds

    probation = summarize_probation(
        arm_windows["parent"],
        arm_windows["capacity-shadow"],
        arm_windows["geometry-shadow"],
    )
    evaluation_rows = []
    identities = {}
    for arm in ARMS:
        rows, identity = evaluate_arm(
            arm_models[arm],
            streams,
            validation,
            parent_losses,
            arm=arm,
            condition=condition,
            replicate=replicate,
            device=device,
        )
        evaluation_rows.extend(rows)
        identities[arm] = identity

    geometry_routes = [row for row in routing_rows if row["arm"] == "geometry-shadow"]
    mixed = condition in ("STORY_ARITHMETIC", "WEAK_ARITHMETIC")
    routing_purity = None
    if mixed and geometry_routes:
        routing_purity = routing_purity_from_branches(
            [int(row["branch"]) for row in geometry_routes],
            [str(row["family_posthoc"]) for row in geometry_routes],
        )
    geometry_identity = identities["geometry-shadow"]
    capacity_identity = identities["capacity-shadow"]
    committed = int(probation.accepted)
    strong_positive = int(
        condition == "STORY_ARITHMETIC"
        and probation.accepted
        and geometry_identity is not None
        and geometry_identity.passes
        and routing_purity is not None
        and routing_purity >= ROUTING_PURITY_MIN
    )
    summary = {
        "replicate": replicate,
        "condition": condition,
        "expected_outcome": expected_condition_outcome(condition),
        "accepted": committed,
        "strong_positive": strong_positive,
        "geometry_sustained_positive": int(probation.sustained_positive),
        "geometry_cumulative_positive": int(probation.cumulative_positive),
        "geometry_beats_capacity": int(probation.beats_capacity),
        "geometry_mean_net_utility_last3": float(np.mean(probation.geometry_window_net_utility[-3:])),
        "capacity_mean_net_utility_last3": float(np.mean(probation.capacity_window_net_utility[-3:])),
        "geometry_advantage_last3": float(np.mean(probation.geometry_advantage[-3:])),
        "geometry_identity_pass": None if geometry_identity is None else int(geometry_identity.passes),
        "geometry_identity_margin": None if geometry_identity is None else geometry_identity.normalized_identity_margin,
        "capacity_identity_pass": None if capacity_identity is None else int(capacity_identity.passes),
        "capacity_identity_margin": None if capacity_identity is None else capacity_identity.normalized_identity_margin,
        "routing_purity_posthoc": routing_purity,
        "routing_purity_pass": None if routing_purity is None else int(routing_purity >= ROUTING_PURITY_MIN),
    }
    window_rows = []
    for window in range(PROBATION_WINDOWS):
        window_rows.append(
            {
                "replicate": replicate,
                "condition": condition,
                "window": window,
                "parent_prequential_nll": arm_windows["parent"][window],
                "capacity_prequential_nll": arm_windows["capacity-shadow"][window],
                "geometry_prequential_nll": arm_windows["geometry-shadow"][window],
                "capacity_net_utility": probation.capacity_window_net_utility[window],
                "geometry_net_utility": probation.geometry_window_net_utility[window],
                "geometry_advantage": probation.geometry_advantage[window],
            }
        )

    payload = {
        "format": CHECKPOINT_FORMAT,
        "experiment": "023b",
        "kind": "condition",
        "replicate": replicate,
        "condition": condition,
        "proposal": proposal,
        "summary": summary,
        "window_rows": window_rows,
        "learning_rows": learning_rows,
        "routing_rows": routing_rows,
        "evaluation_rows": evaluation_rows,
        "arm_states": {arm: cpu_state_dict(model) for arm, model in arm_models.items()},
        "wall_seconds": wall,
    }
    atomic_save(checkpoint_path, payload)
    return payload


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 023b requires CUDA")
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
    parent = ForkableTextNCA(vocab_size).to(device)
    parent_path = checkpoint_dir / f"r{replicate}-parent.pt"
    parent, pretrain_rows, pretrain_wall = pretrain_parent(
        parent,
        streams["STORY_A"]["train"],
        replicate=replicate,
        device=device,
        checkpoint_path=parent_path,
    )
    parent_state = cpu_state_dict(parent)
    validation = validation_schedules(streams, replicate=replicate)
    parent_losses = {
        "STORY": evaluate_parent(
            parent, streams["STORY_A"]["validation"], validation["STORY"], device=device
        ),
        "ARITHMETIC": evaluate_parent(
            parent,
            streams["ARITHMETIC"]["validation"],
            validation["ARITHMETIC"],
            device=device,
        ),
    }

    proposals = []
    summaries = []
    windows = []
    learning = []
    routing = []
    evaluation = []
    for condition in CONDITIONS:
        payload = run_condition(
            condition=condition,
            parent_state=parent_state,
            vocab_size=vocab_size,
            streams=streams,
            validation=validation,
            parent_losses=parent_losses,
            replicate=replicate,
            device=device,
            checkpoint_path=checkpoint_dir / f"r{replicate}-{condition.lower()}.pt",
        )
        proposals.append(payload["proposal"])
        summaries.append(payload["summary"])
        windows.extend(payload["window_rows"])
        learning.extend(payload["learning_rows"])
        routing.extend(payload["routing_rows"])
        evaluation.extend(payload["evaluation_rows"])

    pd.DataFrame(proposals).to_csv(output_dir / f"r{replicate}-proposal.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / f"r{replicate}-condition-summary.csv", index=False)
    pd.DataFrame(windows).to_csv(output_dir / f"r{replicate}-probation-windows.csv", index=False)
    pd.DataFrame(learning).to_csv(output_dir / f"r{replicate}-learning-curve.csv", index=False)
    pd.DataFrame(routing).to_csv(output_dir / f"r{replicate}-routing.csv", index=False)
    pd.DataFrame(evaluation).to_csv(output_dir / f"r{replicate}-evaluation.csv", index=False)
    pd.DataFrame(pretrain_rows).to_csv(output_dir / f"r{replicate}-pretrain.csv", index=False)

    worker = {
        "format": "minicells.probationary-trait-genesis-worker.v1",
        "experiment": "023b",
        "replicate": replicate,
        "gpu": torch.cuda.get_device_name(0),
        "vocab_size": vocab_size,
        "conditions": list(CONDITIONS),
        "arms": list(ARMS),
        "proposal_uses_task_label": False,
        "geometry_routing_uses_task_label": False,
        "commit_uses_task_label": False,
        "capacity_control_uses_stream_identity_only_for_exact_balance": True,
        "parent_losses": parent_losses,
        "pretrain_steps": PRETRAIN_STEPS,
        "pretrain_wall_seconds": pretrain_wall,
        "checkpoint_files": 1 + len(CONDITIONS),
    }
    (output_dir / f"r{replicate}-worker.json").write_text(
        json.dumps(worker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"replicate": replicate, "summaries": summaries}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
