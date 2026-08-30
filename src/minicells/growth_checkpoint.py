"""Versioned checkpoints and dynamic optimizer support for CLM-0.3."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch


GROWTH_CHECKPOINT_FORMAT = "minicells.clm-0.3-growth-checkpoint.v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_base_release_hash(
    model_path: str | Path,
    expected: str = "87d36c408ae3873ffd567ebf17050661b42ddae2c8d5d1bab84b2c27c3c7e7a0",
) -> str:
    observed = sha256_file(model_path)
    if observed != expected:
        raise RuntimeError(
            f"CLM-0.1 checkpoint hash mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _clone_state(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return value


def add_newborn_parameters(
    optimizer: torch.optim.Optimizer,
    parent_parameters: Iterable[torch.nn.Parameter],
    child_parameters: Iterable[torch.nn.Parameter],
) -> None:
    """Add a newborn group and copy the parent's Adam moments/step.

    Existing parameter groups are untouched.  Split-router parameters should be
    passed separately when they are created; they intentionally start fresh.
    """

    parents = list(parent_parameters)
    children = list(child_parameters)
    if len(parents) != len(children):
        raise ValueError("parent and child parameter structures differ")
    if not children:
        raise ValueError("child has no parameters")
    existing = {parameter for group in optimizer.param_groups for parameter in group["params"]}
    fresh = [parameter for parameter in children if parameter not in existing]
    if fresh:
        template = optimizer.param_groups[0] if optimizer.param_groups else {}
        group = {key: value for key, value in template.items() if key != "params"}
        group["params"] = fresh
        optimizer.add_param_group(group)
    for parent, child in zip(parents, children):
        if parent in optimizer.state:
            optimizer.state[child] = _clone_state(optimizer.state[parent])


def add_fresh_parameter_group(
    optimizer: torch.optim.Optimizer, parameters: Iterable[torch.nn.Parameter]
) -> None:
    parameters = [parameter for parameter in parameters if parameter not in {
        item for group in optimizer.param_groups for item in group["params"]
    }]
    if not parameters:
        return
    template = optimizer.param_groups[0] if optimizer.param_groups else {"lr": 0.0}
    group = {key: value for key, value in template.items() if key != "params"}
    group["params"] = parameters
    optimizer.add_param_group(group)


def inherit_optimizer_state(
    optimizer: torch.optim.Optimizer,
    parent_module: torch.nn.Module,
    child_module: torch.nn.Module,
) -> None:
    add_newborn_parameters(optimizer, parent_module.parameters(), child_module.parameters())


class GlobalLRScheduler:
    """Scheduler whose LR is a pure function of global continuation step."""

    def __init__(self, optimizer: torch.optim.Optimizer, lr_function: Callable[[int], float], step: int = 0) -> None:
        self.optimizer = optimizer
        self.lr_function = lr_function
        self.global_step = int(step)
        self.step(self.global_step)

    def step(self, global_step: int | None = None) -> float:
        if global_step is not None:
            self.global_step = int(global_step)
        lr = float(self.lr_function(self.global_step))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def add_param_group(self, group: dict[str, Any]) -> None:
        group["lr"] = float(self.lr_function(self.global_step))

    def state_dict(self) -> dict[str, Any]:
        return {"global_step": self.global_step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.global_step = int(state["global_step"])
        self.step()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _cpu_byte_rng_state(value: torch.Tensor) -> torch.Tensor:
    """Normalize serialized RNG tensors for PyTorch RNG restoration APIs.

    ``torch.load(..., map_location='cuda')`` also moves checkpoint metadata
    tensors, including RNG states, to CUDA.  PyTorch's RNG setters require CPU
    ByteTensors, so checkpoint loading must normalize them explicitly.
    """

    if not torch.is_tensor(value):
        raise TypeError("serialized torch RNG state must be a tensor")
    return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(_cpu_byte_rng_state(state["torch_cpu"]))

    saved_cuda = state.get("torch_cuda")
    if not torch.cuda.is_available() or saved_cuda is None:
        return

    cuda_states = [_cpu_byte_rng_state(item) for item in saved_cuda]
    visible_devices = torch.cuda.device_count()
    if not cuda_states:
        return
    if len(cuda_states) == visible_devices:
        torch.cuda.set_rng_state_all(cuda_states)
        return
    if len(cuda_states) == 1:
        # Workers run under CUDA_VISIBLE_DEVICES and therefore normally expose
        # exactly one logical GPU.  This also permits moving a saved worker to a
        # different physical GPU without changing its logical RNG stream.
        torch.cuda.set_rng_state(cuda_states[0], device=torch.cuda.current_device())
        return
    raise RuntimeError(
        "checkpoint CUDA RNG state count does not match visible CUDA devices: "
        f"saved={len(cuda_states)}, visible={visible_devices}"
    )


def save_growth_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    consumed_tokens: int = 0,
    training_step: int = 0,
    growth_event_index: int | None = None,
    data_schedule_state: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    structure = model.growth_structure() if hasattr(model, "growth_structure") else {}
    lineages = model.lineage_metadata() if hasattr(model, "lineage_metadata") else []
    growth_history = list(getattr(model, "growth_history", []))
    completed_events = len(growth_history)
    if growth_event_index is None:
        growth_event_index = completed_events
    if int(growth_event_index) != completed_events:
        raise ValueError("growth_event_index must equal the number of committed births")
    payload = {
        "format": GROWTH_CHECKPOINT_FORMAT,
        "base_release": "clm-0.1",
        "base_model_sha256": getattr(model, "base_model_sha256", None),
        "model_structure": structure,
        "model_state": model.state_dict(),
        "lineages": lineages,
        "growth_history": growth_history,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "consumed_tokens": int(consumed_tokens),
        "training_step": int(training_step),
        "growth_event_index": int(growth_event_index),
        "rng_state": capture_rng_state(),
        "data_schedule_state": data_schedule_state or {},
        "metrics": metrics or {},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return destination


def validate_growth_checkpoint_payload(payload: dict[str, Any]) -> None:
    history = list(payload.get("growth_history", []))
    event_index = int(payload.get("growth_event_index", -1))
    if event_index != len(history):
        raise RuntimeError(
            "growth checkpoint is inconsistent: growth_event_index does not match growth_history"
        )
    expected_indices = list(range(1, event_index + 1))
    observed_indices = [int(event.get("birth_index", -1)) for event in history]
    if observed_indices != expected_indices:
        raise RuntimeError("growth checkpoint birth indices are not contiguous")
    consumed_tokens = int(payload.get("consumed_tokens", 0))
    if any(int(event.get("token", consumed_tokens + 1)) > consumed_tokens for event in history):
        raise RuntimeError("growth checkpoint contains a birth after consumed_tokens")
    structure = payload.get("model_structure", {})
    stages = structure.get("stages", []) if isinstance(structure, dict) else []
    if stages:
        def count_leaves(node: dict[str, Any]) -> int:
            if "leaf" in node:
                return 1
            return count_leaves(node["left"]) + count_leaves(node["right"])

        expert_count = sum(
            count_leaves(root)
            for stage in stages
            for root in stage.get("roots", [])
        )
        if expert_count != 12 + event_index:
            raise RuntimeError("growth checkpoint expert tree does not match committed births")
    schedule = payload.get("data_schedule_state", {})
    if schedule:
        step = int(schedule.get("current_step", payload.get("training_step", 0)))
        consumed = int(schedule.get("consumed_tokens", payload.get("consumed_tokens", 0)))
        if step != int(payload.get("training_step", step)):
            raise RuntimeError("data schedule step does not match checkpoint training_step")
        if consumed != int(payload.get("consumed_tokens", consumed)):
            raise RuntimeError("data schedule tokens do not match checkpoint consumed_tokens")


def load_growth_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    model_factory: Callable[[dict[str, Any]], torch.nn.Module] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format") != GROWTH_CHECKPOINT_FORMAT:
        raise RuntimeError(f"unsupported growth checkpoint format: {payload.get('format')!r}")
    validate_growth_checkpoint_payload(payload)
    if model is None:
        if model_factory is None:
            raise ValueError("model or model_factory is required to reconstruct a checkpoint")
        model = model_factory(payload["model_structure"])
    if hasattr(model, "growth_history"):
        model.growth_history = list(payload.get("growth_history", []))
    if hasattr(model, "restore_growth_structure"):
        model.restore_growth_structure(payload["model_structure"], payload.get("lineages", []))
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None and payload.get("optimizer_state") is not None:
        if hasattr(model, "growth_history"):
            for event in model.growth_history:
                bank = model.stages[int(event["stage"])].program_bank
                add_fresh_parameter_group(optimizer, bank.experts[str(event["child"])].parameters())
                add_fresh_parameter_group(
                    optimizer, bank.router.split_routers[str(event["split_id"])].parameters()
                )
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if payload.get("rng_state") is not None:
        restore_rng_state(payload["rng_state"])
    return model, payload


save_checkpoint = save_growth_checkpoint
load_checkpoint = load_growth_checkpoint
