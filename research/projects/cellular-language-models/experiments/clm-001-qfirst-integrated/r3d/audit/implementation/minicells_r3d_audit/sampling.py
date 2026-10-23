"""Freeze input-only unique-Story windows and per-Stage Token positions."""

import numpy as np
import torch
from .common import canonical


def plan(c, original, data, seed, role):
    if role not in c["sampling"]["roles"]:
        raise ValueError("Calibration/Development only")
    meta = data["windows"][role]["evaluation"]
    unique = {}
    for index, (start, story) in enumerate(zip(meta["starts"], meta["groups"])):
        unique.setdefault(str(story), []).append((index, int(start)))
    number = c["sampling"]["roles"][role]
    if len(unique) < number:
        raise RuntimeError("INSUFFICIENT_UNIQUE_STORY_WINDOWS: " + role)
    rng = np.random.default_rng(
        int(canonical({"seed": seed, "role": role, "sampling": c["sampling"]["seed"]})[:16], 16)
    )
    keys = sorted(unique)
    chosen = rng.choice(len(keys), size=number, replace=False)
    length = original["engine"]["training"]["sequence_length"]
    rows = []
    for k in chosen:
        story = keys[int(k)]
        index, start = unique[story][int(rng.integers(len(unique[story])))]
        rows.append(
            {
                "story": story,
                "evaluation_index": index,
                "start": start,
                "token_positions": rng.integers(length, size=c["sampling"]["stages"]).tolist(),
            }
        )
    return rows


def batches(data, original, role, device, batch_size, selected=None):
    if role not in ("calibration", "development"):
        raise ValueError("Confirmation sealed")
    meta = data["windows"][role]["evaluation"]
    rows = (
        selected
        if selected is not None
        else [
            {"story": str(story), "start": int(start)}
            for start, story in zip(meta["starts"], meta["groups"])
        ]
    )
    n = original["engine"]["training"]["sequence_length"]
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        x = torch.tensor(
            np.stack(
                [np.array(data[role][r["start"] : r["start"] + n], dtype="int64") for r in batch]
            ),
            device=device,
        )
        y = torch.tensor(
            np.stack(
                [
                    np.array(data[role][r["start"] + 1 : r["start"] + n + 1], dtype="int64")
                    for r in batch
                ]
            ),
            device=device,
        )
        yield batch, x, y
