"""Tensor adapters shared by the learned model trainer and subprocess workers."""

from __future__ import annotations

from typing import Any

import torch

from .task import key_bits, value_bits


def _payload_bits(payload: int, width: int) -> list[int]:
    return [(payload >> bit) & 1 for bit in range(width - 1, -1, -1)]


def event_tensors(
    events: list[dict[str, Any]],
    *,
    key_width: int = 16,
    value_width: int = 16,
    max_events: int = 128,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not events:
        raise ValueError("an episode must contain at least one event")
    if len(events) > max_events:
        raise ValueError(f"event stream exceeds the registered {max_events}-cell limit")
    keys: list[list[int]] = []
    values: list[list[int]] = []
    facts: list[float] = []
    for event in events:
        if event["kind"] == "fact":
            keys.append(key_bits(int(event["key"]), key_width))
            values.append(value_bits(int(event["value"]), value_width))
            facts.append(1.0)
        elif event["kind"] == "distractor":
            payload = int(event["payload"])
            keys.append(_payload_bits(payload, key_width))
            values.append(_payload_bits(payload >> key_width, value_width))
            facts.append(0.0)
        else:
            raise ValueError(f"unknown event type: {event['kind']}")
    valid_count = len(events)
    padding = max_events - valid_count
    keys.extend([[0] * key_width for _ in range(padding)])
    values.extend([[0] * value_width for _ in range(padding)])
    facts.extend([0.0] * padding)
    key_tensor = torch.tensor(keys, dtype=torch.float32, device=device).unsqueeze(0)
    value_tensor = torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)
    is_fact = torch.tensor(facts, dtype=torch.float32, device=device).unsqueeze(0)
    valid = torch.zeros((1, max_events), dtype=torch.bool, device=device)
    valid[:, :valid_count] = True
    return key_tensor, value_tensor, is_fact, valid


def query_tensor(query: int, *, width: int = 16, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor([key_bits(int(query), width)], dtype=torch.float32, device=device)


def latest_fact_slot(events: list[dict[str, Any]], query: int) -> int:
    found = [i for i, event in enumerate(events) if event["kind"] == "fact" and int(event["key"]) == int(query)]
    if not found:
        raise KeyError(f"query key {query} was not written")
    return found[-1]
