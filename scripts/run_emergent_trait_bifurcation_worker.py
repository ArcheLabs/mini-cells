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
    CALIBRATION_BATCHES_PER_DOMAIN,
    CALIBRATION_WINDOWS,
    DOMAINS,
    FORK_EPSILON,
    PHENOTYPE_LR,
    POSTFORK_STEPS,
    PRETRAIN_LR,
    PRETRAIN_STEPS,
    SEQUENCE_LENGTH,
    ForkableTextNCA,
    deterministic_starts,
    language_model_loss,
    lr_multiplier,
    mixed_domain_schedule,
    prepare_arithmetic_cache,
    trait_gradient,
)
from minicells.language_data import batch_from_starts, load_tokenizer  # noqa: E402
from minicells.language_trait_bifurcation import (  # noqa: E402
    ARMS,
    ROUTING_PURITY_MIN,
    bifurcation_window_pass,
    fit_two_mode_gradient_field,
    route_gradient_to_mode,
    routing_purity_from_branches,
    summarize_identity,
    summarize_persistent_bifurcation,
)


CHECKPOINT_FORMAT = "minicells.emergent-trait-bifurcation-checkpoint.v1"
N_REPLICATES = 3
MODEL_SEED_BASE = 122_022
SCHEDULE_SEED_BASE = 222_022
CHECKPOINT_STEPS = (0, 50, 100, 200, 400)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Experiment 022 replicate")
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


def load_streams(cache_dir: Path):
    train = torch.load(cache_dir / "train-tokens.pt", map_location="cpu")
    validation = torch.load(cache_dir / "validation-tokens.pt", map_location="cpu")
    manifest = json.loads((cache_dir / "corpus-manifest.json").read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(cache_dir / "tokenizer.json")
    arithmetic = prepare_arithmetic_cache(cache_dir, tokenizer)
    return {
        DOMAINS[0]: {"train": train, "validation": validation},
        DOMAINS[1]: {"train": arithmetic["train"], "validation": arithmetic["validation"]},
        "story_manifest": manifest,
        "arithmetic_manifest": arithmetic["manifest"],
        "arithmetic_manifest_path": arithmetic["path"],
        "vocab_size": int(tokenizer.get_vocab_size()),
    }


def make_batch(stream: torch.Tensor, starts: tuple[int, ...], device: torch.device):
    return batch_from_starts(stream, starts, SEQUENCE_LENGTH, device)


@torch.no_grad()
def evaluate_trait(
    model: ForkableTextNCA,
    stream: torch.Tensor,
    starts: tuple[tuple[int, ...], ...],
    *,
    device: torch.device,
    parent: bool = False,
    branch: int = 0,
) -> float:
    model.eval()
    losses = []
    for row in starts:
        inputs, targets = make_batch(stream, row, device)
        logits = model.forward_parent(inputs) if parent else model.forward_child(inputs, branch)
        losses.append(float(language_model_loss(logits.float(), targets)))
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
            raise RuntimeError(f"invalid Experiment 022 parent checkpoint: {checkpoint_path}")
        model.load_state_dict(payload["model_state"])
        return model, list(payload.get("learning_curve", [])), float(payload.get("wall_seconds", 0.0))

    schedule = deterministic_starts(
        len(story_train),
        steps=PRETRAIN_STEPS,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        seed=SCHEDULE_SEED_BASE + 1000 * replicate,
    )
    optimizer = torch.optim.AdamW(model.pretrain_parameters(), lr=PRETRAIN_LR, betas=(0.9, 0.95), weight_decay=0.1)
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
    atomic_save(checkpoint_path, {
        "format": CHECKPOINT_FORMAT,
        "experiment": "022",
        "kind": "parent",
        "replicate": replicate,
        "model_state": cpu_state_dict(model),
        "learning_curve": rows,
        "wall_seconds": wall,
    })
    return model, rows, wall


def calibration_batches(streams: dict[str, object], *, replicate: int, window: int, device: torch.device):
    rows: list[tuple[str, tuple[torch.Tensor, torch.Tensor]]] = []
    for domain_index, domain in enumerate(DOMAINS):
        stream = streams[domain]["train"]
        schedule = deterministic_starts(
            len(stream),
            steps=CALIBRATION_BATCHES_PER_DOMAIN,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            seed=SCHEDULE_SEED_BASE + 10_000 * replicate + 100 * window + domain_index,
        )
        for starts in schedule:
            rows.append((domain, make_batch(stream, starts, device)))
    rng = np.random.default_rng(SCHEDULE_SEED_BASE + 99_000 * replicate + window)
    order = rng.permutation(len(rows)).tolist()
    return [rows[index] for index in order]


def calibrate_bifurcation(model: ForkableTextNCA, streams: dict[str, object], *, replicate: int, device: torch.device):
    model.eval()
    all_gradients: list[torch.Tensor] = []
    all_labels: list[str] = []
    window_geometries = []
    window_rows = []

    for window in range(CALIBRATION_WINDOWS):
        batches = calibration_batches(streams, replicate=replicate, window=window, device=device)
        gradients = []
        labels = []
        losses = []
        for label, batch in batches:
            gradient, loss = trait_gradient(model, batch[0], batch[1])
            gradients.append(gradient.cpu())
            labels.append(label)
            losses.append(loss)
        gradient_tensor = torch.stack(gradients)
        geometry = fit_two_mode_gradient_field(gradient_tensor)
        branches = [route_gradient_to_mode(gradient, geometry)[0] for gradient in gradient_tensor]
        window_geometries.append(geometry)
        window_rows.append({
            "replicate": replicate,
            "window": window,
            "residual_k1": geometry.residual_k1,
            "residual_k2": geometry.residual_k2,
            "bifurcation_gain": geometry.bifurcation_gain,
            "split_balance": geometry.split_balance,
            "centroid_separation": geometry.centroid_separation,
            "routing_purity_posthoc": routing_purity_from_branches(branches, labels),
            "window_bifurcation_pass": int(bifurcation_window_pass(geometry)),
            "mean_probe_loss": float(np.mean(losses)),
        })
        all_gradients.extend(gradients)
        all_labels.extend(labels)

    combined_tensor = torch.stack(all_gradients)
    combined = fit_two_mode_gradient_field(combined_tensor)
    combined_branches = [route_gradient_to_mode(gradient, combined)[0] for gradient in combined_tensor]
    persistent = summarize_persistent_bifurcation(window_geometries, combined)
    summary = {
        "replicate": replicate,
        "windows_passed": persistent.windows_passed,
        "axis_stability": persistent.axis_stability,
        "combined_bifurcation_gain": persistent.combined_gain,
        "combined_split_balance": persistent.combined_split_balance,
        "combined_centroid_separation": combined.centroid_separation,
        "combined_routing_purity_posthoc": routing_purity_from_branches(combined_branches, all_labels),
        "persistent_multimodality": int(persistent.persistent),
    }
    return combined, window_rows, summary


def clone_parent(parent_state: dict[str, torch.Tensor], vocab_size: int, device: torch.device) -> ForkableTextNCA:
    model = ForkableTextNCA(vocab_size).to(device)
    model.load_state_dict(copy.deepcopy(parent_state))
    return model


def training_schedules(streams: dict[str, object], *, replicate: int):
    domain_schedule = mixed_domain_schedule(steps=POSTFORK_STEPS, seed=SCHEDULE_SEED_BASE + 5000 * replicate)
    starts = {}
    for domain_index, domain in enumerate(DOMAINS):
        starts[domain] = deterministic_starts(
            len(streams[domain]["train"]),
            steps=POSTFORK_STEPS,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            seed=SCHEDULE_SEED_BASE + 6000 * replicate + domain_index,
        )
    return domain_schedule, starts


def validation_schedules(streams: dict[str, object], *, replicate: int):
    schedules = {}
    for domain_index, domain in enumerate(DOMAINS):
        schedules[domain] = deterministic_starts(
            len(streams[domain]["validation"]),
            steps=16,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            seed=SCHEDULE_SEED_BASE + 7000 * replicate + domain_index,
        )
    return schedules


def stratified_capacity_branch(domain: str, occurrence: int, replicate: int) -> int:
    """Benchmark-label-aware control used only to force exactly matched exposure.

    The domain label is allowed only in this negative control so each child receives
    the same number of STORY and ARITHMETIC updates. It is not a candidate routing
    mechanism and is never used by the multimodality trigger or geometry fork.
    """
    if domain not in DOMAINS:
        raise ValueError(domain)
    return int((occurrence + replicate) % 2)


def evaluate_arm(
    model: ForkableTextNCA,
    streams: dict[str, object],
    validation: dict[str, tuple[tuple[int, ...], ...]],
    parent_losses: dict[str, float],
    *,
    arm: str,
    replicate: int,
    device: torch.device,
):
    rows = []
    losses: dict[str, tuple[float, float]] = {}
    branches = (0,) if arm == "unified" else (0, 1)
    for domain in DOMAINS:
        values = []
        for branch in branches:
            nll = evaluate_trait(
                model,
                streams[domain]["validation"],
                validation[domain],
                device=device,
                branch=branch,
            )
            values.append(nll)
            rows.append({
                "replicate": replicate,
                "arm": arm,
                "domain": domain,
                "branch": branch,
                "nll": nll,
                "parent_nll": parent_losses[domain],
                "utility": parent_losses[domain] - nll,
            })
        if arm != "unified":
            losses[domain] = (values[0], values[1])
    identity = None if arm == "unified" else summarize_identity(losses, parent_losses)
    combined_nll = float(np.mean([min(losses[d]) for d in DOMAINS])) if losses else float(np.mean([r["nll"] for r in rows]))
    return rows, identity, combined_nll


def train_arm(
    arm: str,
    parent_state: dict[str, torch.Tensor],
    vocab_size: int,
    geometry,
    streams: dict[str, object],
    validation: dict[str, tuple[tuple[int, ...], ...]],
    parent_losses: dict[str, float],
    *,
    replicate: int,
    device: torch.device,
    checkpoint_path: Path,
):
    if checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            payload.get("format") != CHECKPOINT_FORMAT
            or payload.get("arm") != arm
            or int(payload.get("replicate", -1)) != replicate
        ):
            raise RuntimeError(f"invalid Experiment 022 arm checkpoint: {checkpoint_path}")
        model = clone_parent(parent_state, vocab_size, device)
        model.load_state_dict(payload["model_state"])
        return model, payload["summary"], list(payload["evaluation"]), list(payload.get("learning_curve", [])), list(payload.get("routing", []))

    model = clone_parent(parent_state, vocab_size, device)
    model.initialize_children(geometry.axis, symmetry_break=(arm != "unified"))
    initial_distance = float((model.child_traits[0] - model.child_traits[1]).detach().norm().cpu())
    model.freeze_genome_for_fork()
    optimizer = torch.optim.AdamW([model.child_traits], lr=PHENOTYPE_LR, weight_decay=0.0)
    domain_schedule, starts = training_schedules(streams, replicate=replicate)
    learning_curve = []
    routing_rows = []
    domain_seen = {domain: 0 for domain in DOMAINS}
    exposure = {(domain, branch): 0 for domain in DOMAINS for branch in (0, 1)}
    started = time.perf_counter()

    def record(step: int) -> None:
        model.eval()
        for domain in DOMAINS:
            branches = (0,) if arm == "unified" else (0, 1)
            for branch in branches:
                nll = evaluate_trait(model, streams[domain]["validation"], validation[domain], device=device, branch=branch)
                learning_curve.append({
                    "replicate": replicate,
                    "arm": arm,
                    "step": step,
                    "domain": domain,
                    "branch": branch,
                    "nll": nll,
                    "parent_nll": parent_losses[domain],
                })

    record(0)
    model.train()
    for step in range(1, POSTFORK_STEPS + 1):
        for group in optimizer.param_groups:
            group["lr"] = PHENOTYPE_LR * lr_multiplier(step, POSTFORK_STEPS)
        domain = domain_schedule[step - 1]
        inputs, targets = make_batch(streams[domain]["train"], starts[domain][step - 1], device)
        optimizer.zero_grad(set_to_none=True)

        if arm == "unified":
            branch = 0
            routing_source = "unified"
            score = None
        elif arm == "stratified-capacity-fork":
            branch = stratified_capacity_branch(domain, domain_seen[domain], replicate)
            routing_source = "label-stratified-control"
            score = None
        elif arm == "geometry-bifurcation-fork":
            gradient, _ = trait_gradient(model, inputs, targets)
            branch, score = route_gradient_to_mode(gradient, geometry)
            routing_source = "unlabeled-gradient-k2"
        else:
            raise ValueError(arm)

        domain_seen[domain] += 1
        exposure[(domain, branch)] += 1
        routing_rows.append({
            "replicate": replicate,
            "arm": arm,
            "step": step,
            "domain_posthoc": domain,
            "branch": branch,
            "routing_source": routing_source,
            "cluster_distance_score": score,
        })
        loss = language_model_loss(model.forward_child(inputs, branch).float(), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([model.child_traits], 1.0)
        optimizer.step()
        if step in CHECKPOINT_STEPS[1:]:
            record(step)
        model.train()

    wall = time.perf_counter() - started
    evaluation, identity, combined_nll = evaluate_arm(
        model,
        streams,
        validation,
        parent_losses,
        arm=arm,
        replicate=replicate,
        device=device,
    )
    final_distance = float((model.child_traits[0] - model.child_traits[1]).detach().norm().cpu())

    route_purity = None
    if arm == "geometry-bifurcation-fork":
        route_purity = routing_purity_from_branches(
            [int(row["branch"]) for row in routing_rows],
            [str(row["domain_posthoc"]) for row in routing_rows],
        )

    summary = {
        "replicate": replicate,
        "arm": arm,
        "initial_child_distance": initial_distance,
        "final_child_distance": final_distance,
        "distance_growth_ratio": final_distance / max(initial_distance, 1e-8) if arm != "unified" else 0.0,
        "routing_purity_posthoc": route_purity,
        "wall_seconds": wall,
        "identity_pass": None if identity is None else int(identity.passes),
        "identity_assignment": None if identity is None else json.dumps(identity.assignment),
        "normalized_story_margin": None if identity is None else identity.normalized_story_margin,
        "normalized_arithmetic_margin": None if identity is None else identity.normalized_arithmetic_margin,
        "normalized_identity_margin": None if identity is None else identity.normalized_identity_margin,
        "opposite_preference": None if identity is None else int(identity.opposite_preference),
        "routing_purity_pass": None if route_purity is None else int(route_purity >= ROUTING_PURITY_MIN),
        "oracle_combined_nll": combined_nll,
        "story_branch0_updates": exposure[(DOMAINS[0], 0)],
        "story_branch1_updates": exposure[(DOMAINS[0], 1)],
        "arithmetic_branch0_updates": exposure[(DOMAINS[1], 0)],
        "arithmetic_branch1_updates": exposure[(DOMAINS[1], 1)],
    }
    atomic_save(checkpoint_path, {
        "format": CHECKPOINT_FORMAT,
        "experiment": "022",
        "kind": "arm",
        "replicate": replicate,
        "arm": arm,
        "model_state": cpu_state_dict(model),
        "summary": summary,
        "evaluation": evaluation,
        "learning_curve": learning_curve,
        "routing": routing_rows,
    })
    return model, summary, evaluation, learning_curve, routing_rows


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 022 requires CUDA")
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
    parent, pretrain_curve, pretrain_wall = pretrain_parent(
        parent,
        streams[DOMAINS[0]]["train"],
        replicate=replicate,
        device=device,
        checkpoint_path=checkpoint_dir / f"r{replicate}-parent.pt",
    )
    parent.eval()
    parent_state = cpu_state_dict(parent)

    validation = validation_schedules(streams, replicate=replicate)
    parent_losses = {
        domain: evaluate_trait(parent, streams[domain]["validation"], validation[domain], device=device, parent=True)
        for domain in DOMAINS
    }
    geometry, bifurcation_windows, bifurcation_summary = calibrate_bifurcation(
        parent, streams, replicate=replicate, device=device
    )

    summaries = []
    evaluations = []
    learning = []
    routing = []
    for arm in ARMS:
        _, summary, arm_eval, arm_learning, arm_routing = train_arm(
            arm,
            parent_state,
            vocab_size,
            geometry,
            streams,
            validation,
            parent_losses,
            replicate=replicate,
            device=device,
            checkpoint_path=checkpoint_dir / f"r{replicate}-{arm}.pt",
        )
        summaries.append({**summary, "persistent_multimodality": bifurcation_summary["persistent_multimodality"]})
        evaluations.extend(arm_eval)
        learning.extend(arm_learning)
        routing.extend(arm_routing)

    pd.DataFrame(bifurcation_windows).to_csv(output_dir / f"r{replicate}-bifurcation-windows.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / f"r{replicate}-arm-summary.csv", index=False)
    pd.DataFrame(evaluations).to_csv(output_dir / f"r{replicate}-evaluation.csv", index=False)
    pd.DataFrame(learning).to_csv(output_dir / f"r{replicate}-learning-curve.csv", index=False)
    pd.DataFrame(routing).to_csv(output_dir / f"r{replicate}-routing.csv", index=False)
    pd.DataFrame(pretrain_curve).to_csv(output_dir / f"r{replicate}-pretrain.csv", index=False)

    worker = {
        "format": "minicells.emergent-trait-bifurcation-worker.v1",
        "experiment": "022",
        "replicate": replicate,
        "gpu": torch.cuda.get_device_name(0),
        "vocab_size": vocab_size,
        "domains": list(DOMAINS),
        "arms": list(ARMS),
        "pretrain_steps": PRETRAIN_STEPS,
        "postfork_steps": POSTFORK_STEPS,
        "fork_epsilon": FORK_EPSILON,
        "trigger_uses_task_label": False,
        "geometry_routing_uses_task_label": False,
        "capacity_control_uses_labels_only_for_exact_stratification": True,
        "parent_losses": parent_losses,
        "bifurcation": bifurcation_summary,
        "pretrain_wall_seconds": pretrain_wall,
        "checkpoint_files": 1 + len(ARMS),
    }
    (output_dir / f"r{replicate}-worker.json").write_text(
        json.dumps(worker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "replicate": replicate,
        "persistent_multimodality": bifurcation_summary["persistent_multimodality"],
        "geometry_identity": next(row["identity_pass"] for row in summaries if row["arm"] == "geometry-bifurcation-fork"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
