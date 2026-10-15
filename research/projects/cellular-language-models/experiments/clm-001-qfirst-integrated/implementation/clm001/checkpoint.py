"""Reuse atomic PD checkpoint; persist accepted, private candidates, all RNGs and identities."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import os
import shutil
import torch
from pd001.checkpoint import save as pd_save, restore as pd_restore
from .common import BASE, canonical_sha256, file_sha256, write_json
from .actions import Registry
from .model import Model
from .training import Trainer
from .shadows import Candidate, Nursery


def identity(c, arm, source, assets):
    return {
        "experiment_id": c["experiment_id"],
        "arm": arm,
        "seed": c["seed"],
        "source_commit": source,
        "config_sha256": canonical_sha256(c),
        "assets_manifest_sha256": assets["asset_sha256"],
        "tokenizer_sha256": assets["tokenizer_sha256"],
        "role": "development",
        "protocol_sha256": file_sha256(BASE / "SPEC.md"),
    }


def save(folder, trainer, nursery, data_rng, run_identity):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = "checkpoint-step-%06d.pt" % trainer.step
    path = folder / name
    if path.exists():
        latest = folder / "latest.json"
        committed = json.loads(latest.read_text())["file"] if latest.exists() else None
        if committed == name:
            raise FileExistsError("immutable checkpoint step already exists")
        # A file without an atomically committed latest manifest is an interrupted orphan, never a completed checkpoint.
        path.rename(path.with_name(path.name + ".orphan"))
    candidates = []
    for candidate in nursery.candidates:
        t = candidate.trainer
        candidates.append(
            {
                "kind": candidate.kind,
                "parent_sha": candidate.parent_sha,
                "parent_registry": candidate.parent_registry,
                "state": candidate.state,
                "journal": deepcopy(candidate.journal),
                "metrics": deepcopy(candidate.metrics),
                "model": candidate.model.state_dict(),
                "registry": asdict(candidate.model.registry),
                "version": candidate.model.version,
                "optimizer": t.opt.state_dict(),
                "rng": {k: v.get_state() for k, v in t.rng.items()},
                "step": t.step,
                "data_position": t.data_position,
                "training_journal": deepcopy(t.journal),
            }
        )
    extra = {
        "registry": asdict(trainer.model.registry),
        "version": trainer.model.version,
        "data_position": trainer.data_position,
        "journal": deepcopy(trainer.journal),
        "cache": {},
        "births": nursery.births,
        "groups": nursery.groups,
        "candidates": candidates,
    }
    pd_save(
        path,
        trainer.model,
        trainer.opt,
        trainer.step,
        run_identity,
        data_rng,
        list(trainer.rng.values()),
        extra,
    )
    digest = file_sha256(path)
    write_json(path.with_suffix(".sha256.json"), {"sha256": digest, "identity": run_identity})
    # Manifest replacement is the commit point; an interrupted save leaves the preceding checkpoint readable.
    write_json(folder / "latest.json", {"file": name, "sha256": digest, "identity": run_identity})
    return digest


def restore(folder, trainer, data_rng, run_identity):
    folder = Path(folder)
    latest = json.loads((folder / "latest.json").read_text())
    name = latest["file"]
    if Path(name).name != name or not name.endswith(".pt") or latest["identity"] != run_identity:
        raise ValueError("checkpoint identity/path differs")
    path = folder / name
    if path.is_symlink() or file_sha256(path) != latest["sha256"]:
        raise ValueError("checkpoint checksum differs")
    step, extra = pd_restore(
        path,
        trainer.model,
        trainer.opt,
        run_identity,
        data_rng,
        list(trainer.rng.values()),
        device=next(trainer.model.parameters()).device,
    )
    trainer.step = step
    trainer.model.registry = Registry(**extra["registry"])
    trainer.model.version = extra["version"]
    trainer.data_position = extra["data_position"]
    trainer.journal = extra["journal"]
    trainer.cache = {}
    nursery = Nursery()
    nursery.births = extra["births"]
    nursery.groups = extra["groups"]
    global_cpu = torch.get_rng_state()
    global_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    for record in extra["candidates"]:
        model = Model(trainer.model.experiment_config, trainer.model.arm).to(
            next(trainer.model.parameters()).device
        )
        model.load_state_dict(record["model"])
        model.registry = Registry(**record["registry"])
        model.version = record["version"]
        for name, param in model.named_parameters():
            param.requires_grad_(
                name.startswith("controller.")
                or (
                    record["kind"] == "cell" and name.startswith("cells.%d." % model.registry.cells)
                )
            )
        branch = Trainer(model)
        branch.opt.load_state_dict(record["optimizer"])
        branch.step = record["step"]
        branch.data_position = record["data_position"]
        branch.journal = record["training_journal"]
        for k, value in record["rng"].items():
            branch.rng[k].set_state(value.cpu())
        nursery.candidates.append(
            Candidate(
                record["kind"],
                model,
                record["parent_sha"],
                record["parent_registry"],
                branch,
                record["state"],
                record["journal"],
                record["metrics"],
            )
        )
    torch.set_rng_state(global_cpu)
    if global_cuda is not None:
        torch.cuda.set_rng_state_all(global_cuda)
    return nursery


def export_portable(run, assets, destination, run_identity, c):
    run = Path(run)
    assets = Path(assets)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("portable output immutable")
    temp = destination.with_name(destination.name + ".tmp")
    temp.mkdir(parents=True)
    latest = json.loads((run / "latest.json").read_text())
    if latest["identity"] != run_identity:
        raise ValueError("portable checkpoint identity differs")
    chosen = [
        run / "latest.json",
        run / latest["file"],
        run / Path(latest["file"]).with_suffix(".sha256.json"),
    ]
    for path in chosen:
        shutil.copy2(path, temp / path.name)
    shutil.copytree(assets, temp / "assets")
    write_json(temp / "config.json", c)
    files = {str(p.relative_to(temp)): file_sha256(p) for p in temp.rglob("*") if p.is_file()}
    write_json(
        temp / "portable.json",
        {
            "experiment_id": c["experiment_id"],
            "identity": run_identity,
            "files": files,
            "actual_kaggle_persistence_verified": False,
        },
    )
    os.replace(temp, destination)
    return destination


def verify_portable(folder, run_identity):
    folder = Path(folder)
    meta = json.loads((folder / "portable.json").read_text())
    if meta.get("experiment_id") != "CLM-001-QFIRST" or meta["identity"] != run_identity:
        raise ValueError("portable run identity differs")
    for name, digest in meta["files"].items():
        p = folder / name
        if (
            p.is_symlink()
            or ".." in Path(name).parts
            or Path(name).is_absolute()
            or file_sha256(p) != digest
        ):
            raise ValueError("portable checksum/path differs")
    return meta
