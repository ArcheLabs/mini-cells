"""Private immutable fixed-state bank, versioned snapshot diagnostics and restart validation."""

from pathlib import Path
import os
import torch
from .common import write_json, file_sha256
from .audit import capture
from .specialization import snapshot, longitudinal
from clm001_r2.report import seal, verified


def bank(model, data, c, folder, identity):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "frozen-probe-bank.pt"
    meta = folder / "frozen-probe-bank.json"
    if meta.exists():
        import json

        record = json.loads(meta.read_text())
        if record["identity"] != identity or file_sha256(path) != record["sha256"]:
            raise RuntimeError("probe bank identity/checksum differs")
        return torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    result, _ = capture(model, data, c)
    temporary = path.with_suffix(".tmp")
    torch.save(result, temporary)
    os.replace(temporary, path)
    write_json(meta, {"identity": identity, "sha256": file_sha256(path)})
    return result


def record(model, data, c, initial, step, folder):
    folder = Path(folder)
    path = folder / "snapshots" / ("step-%06d.json" % step)
    if path.exists():
        return verified(path)
    previous_paths = sorted(
        p for p in (folder / "snapshots").glob("step-*.json") if not p.name.endswith(".sha256.json")
    )
    previous = verified(previous_paths[-1]) if previous_paths else None
    print("[R3] functional snapshot step " + str(step), flush=True)
    value = snapshot(model, data, c, initial, step, previous)
    seal(path, value)
    return value


def history(folder):
    paths = sorted(
        p
        for p in (Path(folder) / "snapshots").glob("step-*.json")
        if not p.name.endswith(".sha256.json")
    )
    return longitudinal([verified(path) for path in paths])
