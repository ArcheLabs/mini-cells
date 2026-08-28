"""Shared execution helpers for resumable CLM growth experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .clm_release import build_release_model
from .growth_checkpoint import GlobalLRScheduler, save_growth_checkpoint
from .language_data import batch_from_starts


def schedule_digest(starts: tuple[tuple[int, ...], ...]) -> str:
    return hashlib.sha256(json.dumps(starts, separators=(",", ":")).encode()).hexdigest()


def value_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_provenance(root: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    commit = run("rev-parse", "HEAD")
    tree = run("rev-parse", "HEAD^{tree}")
    dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    return {"code_commit": commit, "code_tree_sha": tree, "tracked_tree_dirty": dirty}


def load_ppl_history(path: Path, *, through_tokens: int) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            tokens = int(raw["tokens"])
            if tokens > through_tokens:
                continue
            row: dict[str, object] = dict(raw)
            row["replicate"] = int(raw["replicate"])
            row["tokens"] = tokens
            for key in (
                "ppl",
                "nll",
                "fixed4_ppl",
                "clm01_start_ppl",
                "textnca_frozen_ppl",
                "ppl_vs_fixed4",
                "ppl_vs_clm01",
                "ppl_vs_textnca",
            ):
                value = raw.get(key, "")
                row[key] = None if value in (None, "", "None", "nan", "NaN") else float(value)
            rows.append(row)
    return rows


def load_diagnostics(
    path: Path,
    *,
    through_tokens: int,
    growth_history: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    birth_tokens = {int(event["birth_index"]): int(event["token"]) for event in growth_history}
    rows: list[dict[str, object]] = []
    for item in raw:
        birth_index = int(item["birth_index"])
        birth_token = birth_tokens.get(birth_index)
        if birth_token is None:
            continue
        if birth_token + int(item.get("offset_tokens", 0)) <= through_tokens:
            rows.append(item)
    return rows


def persist_diagnostics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class Telemetry:
    def __init__(self, output: Path, arm: str, replicate: int) -> None:
        self.output = output
        self.arm = arm
        self.replicate = replicate
        output.mkdir(parents=True, exist_ok=True)
        self.handle = (output / "events.jsonl").open("a", encoding="utf-8")

    def write(self, event: dict[str, object]) -> None:
        payload = {"arm": self.arm, "replicate": self.replicate, "time": time.time(), **event}
        self.handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self.handle.flush()
        if payload["type"] == "training_progress":
            self.output.joinpath("progress.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"[r{self.replicate} {self.arm}] "
                f"{int(payload['consumed_tokens']):,} {payload['phase']}",
                flush=True,
            )

    def close(self) -> None:
        self.handle.close()


def release_teacher(release_dir: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(release_dir / "model.pt", map_location="cpu", weights_only=False)
    model = build_release_model(
        num_experts=int(checkpoint["num_experts"]),
        router_scale=float(checkpoint["router_scale"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def validation_starts(*, eval_batches: int, batch_size: int, sequence_length: int) -> tuple[int, ...]:
    width = sequence_length + 1
    return tuple(range(0, eval_batches * batch_size * width, width))


def validation_batches(
    stream: torch.Tensor,
    *,
    eval_batches: int,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    starts = validation_starts(
        eval_batches=eval_batches,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    return [
        batch_from_starts(stream, starts[index:index + batch_size], sequence_length, device)
        for index in range(0, len(starts), batch_size)
    ]


def checkpoint(
    path: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: GlobalLRScheduler,
    consumed: int,
    step: int,
    schedule_state: dict[str, object],
    *,
    telemetry: Telemetry | None = None,
    reason: str = "periodic",
) -> None:
    state = {**schedule_state, "current_step": int(step), "consumed_tokens": int(consumed)}
    save_growth_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        consumed_tokens=consumed,
        training_step=step,
        data_schedule_state=state,
    )
    if telemetry is not None:
        telemetry.write({
            "type": "checkpoint",
            "path": str(path),
            "reason": reason,
            "consumed_tokens": int(consumed),
            "training_step": int(step),
            "growth_event_index": len(model.growth_history),
        })
