"""Development-only trainer for the EPCLM-002-A0 learned baselines."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .data import event_tensors, latest_fact_slot
from .models import ModelConfig, build_model, model_config_dict
from .task import generate_episode


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parents[2] / "research/validations/epclm-002-a0-persistent-state-recall/protocol.json"


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _queries_and_targets(episode: Any) -> tuple[list[int], list[int], list[int]]:
    queries = sorted(episode.final_facts)
    slots = [latest_fact_slot(list(episode.events), query) for query in queries]
    targets = [episode.final_facts[query] for query in queries]
    return queries, targets, slots


def _training_episode(rng: random.Random) -> Any:
    fact_count = rng.choice((1, 2, 4, 8, 16, 32))
    return generate_episode(
        rng.randrange(1 << 30),
        fact_count=fact_count,
        distractor_count=rng.choice((0, 16, 64)),
        overwrite_count=rng.randrange(0, min(2, fact_count) + 1),
    )


def _episode_loss(model: torch.nn.Module, model_name: str, episode: Any, device: torch.device) -> torch.Tensor:
    events = list(episode.events)
    keys, values, is_fact, valid = event_tensors(events, device=device)
    if model_name == "learned-cells":
        state = model.encode_events(keys, values, is_fact, valid)  # type: ignore[attr-defined]
    else:
        state = model.write_events(keys, values, is_fact, valid)  # type: ignore[attr-defined]
    queries, targets, slots = _queries_and_targets(episode)
    query_batch = torch.tensor(
        [[(key >> bit) & 1 for bit in range(15, -1, -1)] for key in queries],
        dtype=torch.float32,
        device=device,
    )
    repeated_state = state.expand(len(queries), *state.shape[1:])
    repeated_valid = valid.expand(len(queries), -1)
    value_logits, route_scores = model(repeated_state, repeated_valid, query_batch)
    target_bits = torch.tensor(
        [[(value >> bit) & 1 for bit in range(15, -1, -1)] for value in targets],
        dtype=torch.float32,
        device=device,
    )
    loss = nn.functional.binary_cross_entropy_with_logits(value_logits, target_bits)
    if model_name == "learned-cells":
        target_slots = torch.tensor(slots, dtype=torch.long, device=device)
        loss = loss + 0.25 * nn.functional.cross_entropy(route_scores, target_slots)
    return loss


def train(
    *,
    model_name: str,
    seed: int,
    steps: int,
    batch_size: int,
    output: Path,
    device: str = "cpu",
    formal_run: bool = False,
    max_runtime_seconds: int = 7200,
    resume: bool = False,
) -> dict[str, Any]:
    if model_name not in {"learned-cells", "flat-recurrent"}:
        raise ValueError("model_name must be learned-cells or flat-recurrent")
    if not 1 <= max_runtime_seconds <= 7200:
        raise ValueError("max_runtime_seconds must be between 1 and 7200")
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max_runtime_seconds
    torch.manual_seed(seed)
    random.seed(seed)
    selected_device = torch.device(device)
    protocol_data = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    formal_seeds = set(int(value) for value in protocol_data["training"]["formal_seeds"])
    if seed in formal_seeds and not formal_run:
        raise ValueError("a reserved formal training seed requires --formal")
    if formal_run and seed not in formal_seeds:
        raise ValueError("--formal is only valid for a registered formal model seed")
    if formal_run and (
        steps != int(protocol_data["training"]["formal_max_steps"])
        or batch_size != int(protocol_data["training"]["formal_batch_size"])
    ):
        raise ValueError("formal training steps and batch size must match the frozen protocol")
    config = ModelConfig(max_events=128, cell_dim=16)
    model = build_model(model_name, config).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    protocol_sha256 = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    data_seed = int(protocol_data["task"]["splits"]["development_seed"])
    rng = random.Random(data_seed ^ seed ^ 0xA002)
    validation_episodes = [
        generate_episode(
            validation_seed * 1000 + episode_index,
            fact_count=(1, 2, 4, 8, 16, 32)[episode_index % 6],
            distractor_count=(0, 16, 64)[(episode_index // 2) % 3],
            overwrite_count=1 if episode_index % 4 == 0 else 0,
        )
        for validation_seed in protocol_data["task"]["splits"]["validation_seeds"]
        for episode_index in range(24)
    ]
    validation_interval = int(protocol_data["training"]["validation_interval_steps"])
    best_validation_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_step = 0
    progress_path = output.with_suffix(output.suffix + ".progress.pt")
    start_step = 0
    if resume and progress_path.is_file():
        saved = torch.load(progress_path, map_location=selected_device, weights_only=False)
        expected_identity = (model_name, seed, protocol_sha256, model_config_dict(config))
        saved_identity = (
            saved.get("model_name"), saved.get("training_seed"), saved.get("protocol_sha256"),
            saved.get("model_config"),
        )
        if saved_identity != expected_identity:
            raise ValueError("training progress identity differs; refusing to resume another run")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        random.setstate(saved["python_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        rng.setstate(saved["data_rng_state"])
        best_state = saved["best_state"]
        best_validation_loss = float(saved["best_validation_loss"])
        best_step = int(saved["best_step"])
        start_step = int(saved["next_step"])
        if not 0 <= start_step < steps:
            raise ValueError("saved training progress has an invalid next_step")
    model.train()

    for step in range(start_step, steps):
        if time.monotonic() >= deadline:
            model.eval()
            _atomic_torch_save(
                {
                    "model_name": model_name,
                    "model_config": model_config_dict(config),
                    "training_seed": seed,
                    "protocol_sha256": protocol_sha256,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "python_rng_state": random.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "data_rng_state": rng.getstate(),
                    "best_state": best_state,
                    "best_validation_loss": best_validation_loss,
                    "best_step": best_step,
                    "next_step": step,
                },
                progress_path,
            )
            raise TimeoutError("training reached its registered wall-time budget; progress checkpoint saved")
        episodes = [_training_episode(rng) for _ in range(batch_size)]
        optimizer.zero_grad(set_to_none=True)
        losses = [_episode_loss(model, model_name, episode, selected_device) for episode in episodes]
        batch_loss = torch.stack(losses).mean()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % max(1, steps // 10) == 0:
            print(f"step={step + 1}/{steps} loss={batch_loss.item():.5f}", flush=True)
        if (step + 1) % validation_interval == 0 or step + 1 == steps:
            model.eval()
            with torch.no_grad():
                validation_loss = torch.stack(
                    [_episode_loss(model, model_name, episode, selected_device) for episode in validation_episodes]
                ).mean().item()
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                best_step = step + 1
            _atomic_torch_save(
                {
                    "model_name": model_name,
                    "model_config": model_config_dict(config),
                    "training_seed": seed,
                    "protocol_sha256": protocol_sha256,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "python_rng_state": random.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "data_rng_state": rng.getstate(),
                    "best_state": best_state,
                    "best_validation_loss": best_validation_loss,
                    "best_step": best_step,
                    "next_step": step + 1,
                },
                progress_path,
            )
            model.train()

    model.eval()
    model.load_state_dict(best_state)
    checkpoint = {
        "format": "minicells.epclm-002-a0.model.v1",
        "model_name": model_name,
        "model_config": model_config_dict(config),
        "model_state": model.state_dict(),
        "training_seed": seed,
        "training_data_seed": data_seed,
        "training_steps": steps,
        "training_episodes": steps * batch_size,
        "training_seed_role": "formal" if formal_run else "development-only",
        "training_device": str(selected_device),
        "torch_version": str(torch.__version__),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "validation_loss": best_validation_loss,
        "selected_step": best_step,
        "protocol_sha256": protocol_sha256,
    }
    _atomic_torch_save(checkpoint, output)
    progress_path.unlink(missing_ok=True)
    metadata = {key: value for key, value in checkpoint.items() if key != "model_state"}
    metadata["checkpoint_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("learned-cells", "flat-recurrent"), default="learned-cells")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-runtime-seconds", type=int, default=7200)
    parser.add_argument("--resume", action="store_true", help="resume the exact saved development checkpoint")
    parser.add_argument("--formal", action="store_true", help="authorize a registered formal seed and freeze the checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = train(
        model_name=args.model,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        output=args.output,
        device=args.device,
        formal_run=args.formal,
        max_runtime_seconds=args.max_runtime_seconds,
        resume=args.resume,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
