"""Shared one-click development bookkeeping; never grants confirmation access."""

import hashlib
import json
from pathlib import Path
from .train import write_json


def seed_for(experiment, source, occupied):
    seed = (
        100_000_000
        + int.from_bytes(hashlib.sha256((experiment + source).encode()).digest()[:8], "little") % 900_000_000
    )
    while seed in occupied:
        seed += 1
    return seed


def occupied_seeds(root):
    """Read metadata only; never open checkpoints, datasets or holdout metrics."""
    occupied = set()
    files = [Path(root) / "research/governance/formal_seed_registry.json"]
    files += list((Path(root) / "research/projects").glob("**/config/*.json"))
    files += list((Path(root) / "research/projects").glob("**/configs/*.json"))

    def visit(value):
        if isinstance(value, int) and not isinstance(value, bool):
            occupied.add(value)
        elif isinstance(value, str) and value.isdecimal():
            occupied.add(int(value))
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    for path in files:
        if path.is_file():
            visit(json.loads(path.read_text()))
    return occupied


def record(folder, name, value):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    if target.exists():
        if json.loads(target.read_text()) != value:
            raise ValueError("existing development identity changed; preserve this run")
        return target
    write_json(target, value)
    return target
