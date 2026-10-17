"""R3 Trainer.update and Event transactions, new identity and immutable handoffs.

Workers never publish or mutate the coordinator's committed bundle. Every local
checkpoint is acknowledged by one host publisher before another update/event.
"""

from copy import deepcopy
from pathlib import Path
import os
import random
import shutil
import time
import numpy as np
import torch
from minicells_runtime import r3 as reused
from minicells_runtime.store import (
    ContinuationError,
    MountedInputStore,
    atomic,
    canonical,
    digest,
    read,
    safe,
)
from clm001_r3.model import Model
from clm001_r3.training import Trainer
from clm001_r3.shadows import Nursery
from clm001_r3 import checkpoint as cp
from clm001_r3.data import load_assets, batches
from clm001_r3.events import pending, process
from .common import ARMS, ENGINE, BASE, EXPERIMENT, identity, require_training_release
from .instrumentation import banks, snapshot


def new_run(candidate, c, assets, root, *, release=None):
    reused.require_source(c["execution"]["source_sha"])
    load_assets(c, assets)
    root = Path(root)
    if root.exists():
        raise ContinuationError("OUTPUT_ALREADY_EXISTS")
    root.mkdir(parents=True)
    shutil.copytree(assets, root / "assets")
    atomic(root / "candidate-config.json", candidate)
    atomic(root / "runtime-config.json", c)
    shutil.copyfile(BASE / "SPEC.md", root / "protocol.md")
    atomic(root / "preflight.json", release or {"passed": True, "scope": "SYNTHETIC_CPU_ONLY"})
    lock = {
        "experiment_id": c["r3b"]["experiment_id"],
        "run_id": c["r3b"]["run_id"],
        "source_sha": c["execution"]["source_sha"],
    }
    for name, key in (
        ("candidate-config.json", "candidate_config_sha256"),
        ("runtime-config.json", "runtime_config_sha256"),
        ("protocol.md", "protocol_sha256"),
        ("assets/manifest.json", "assets_manifest_sha256"),
        ("assets/tokenizer.json", "tokenizer_sha256"),
        ("preflight.json", "preflight_sha256"),
    ):
        lock[key] = digest(root / name)
    atomic(root / "experiment.lock.json", lock)
    return lock


def load_state(folder, c, arm, data, source, device="cpu", selector=None):
    trainer = Trainer(Model(c, ENGINE[arm]).to(device))
    rng = np.random.default_rng(c["seed"] + 10)
    expected = identity(c, arm, source, data)
    if read(Path(folder) / "identity.json") != expected:
        raise ContinuationError("IDENTITY_MISMATCH", arm)
    if selector is None:
        nursery = cp.restore(folder, trainer, rng, expected)
    else:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = safe(Path(folder).parents[1], selector["checkpoint"])
            shutil.copyfile(path, Path(tmp) / path.name)
            atomic(
                Path(tmp) / "latest.json",
                {"file": path.name, "sha256": selector["checkpoint_sha256"], "identity": expected},
            )
            nursery = cp.restore(tmp, trainer, rng, expected)
    check_events(c, trainer.step, nursery, expected)
    return trainer, nursery, rng, expected


def check_events(c, step, nursery, expected):
    # A pre-event/event-revision checkpoint is intentionally recoverable. Validate
    # every committed cursor, but don't require all pending events to be finished.
    for key, event in nursery.events.items():
        if (
            key
            != canonical(
                {
                    "identity": expected,
                    "step": event["step"],
                    "event": event["event"],
                    "base_registry": event["base_registry"],
                }
            )
            or event["step"] > step
        ):
            raise ContinuationError("EVENT_STATE_INCONSISTENT")


def resources(root, c, progress):
    return {
        "arms": {a: {"last_committed_step": p["last_committed_step"]} for a, p in progress.items()},
        "diagnostic_plan": c["r3b"]["diagnostics"],
        "runtime_termination": False,
        "scientific_decision": False,
    }


def inspect_state(folder, c, arm, data, source, device="cpu", selector=None):
    """Read-only publisher validation; never restore device-specific RNG on host."""
    from types import SimpleNamespace

    folder = Path(folder)
    expected = identity(c, arm, source, data)
    latest = read(folder / "latest.json") if selector is None else {"identity": expected}
    if read(folder / "identity.json") != expected or latest["identity"] != expected:
        raise ContinuationError("IDENTITY_MISMATCH", arm)
    path = (
        safe(folder.parents[1], selector["checkpoint"])
        if selector is not None
        else safe(folder, latest["file"])
    )
    wanted = selector["checkpoint_sha256"] if selector is not None else latest["sha256"]
    if digest(path) != wanted:
        raise ContinuationError("CHECKPOINT_HASH_MISMATCH", arm)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("identity") != expected:
        raise ContinuationError("IDENTITY_MISMATCH", arm)
    model = Model(c, ENGINE[arm])
    model.load_state_dict(payload["model"], strict=True)
    extra = payload["extra"]
    # RNG bytes remain opaque here. Worker restore performs the full device-specific
    # restoration; packaging still hashes every byte, including all RNG states.
    trainer = SimpleNamespace(step=payload["step"])
    nursery = SimpleNamespace(events=extra.get("events", {}))
    check_events(c, trainer.step, nursery, expected)
    return trainer, nursery, None, expected


class Adapter:
    arms = staticmethod(lambda c: ARMS)
    load_state = staticmethod(inspect_state)
    check_events = staticmethod(check_events)
    resources = staticmethod(resources)


def package(root):
    return reused.package(root, adapter=Adapter)


def verify(root, expected_source=None):
    meta = MountedInputStore(root).verify(expected_source)
    root = Path(root)
    c = read(root / "runtime-config.json")
    lock = read(root / "experiment.lock.json")
    if (
        c["r3b"]["experiment_id"] != meta["experiment_id"]
        or c["r3b"]["run_id"] != meta["run_id"]
        or digest(root / "protocol.md") != c["r3b"]["protocol_sha256"]
        or digest(root / "preflight.json") != lock["preflight_sha256"]
    ):
        raise ContinuationError("IDENTITY_MISMATCH")
    data = load_assets(c, root / "assets")
    for arm, progress in meta["arms"].items():
        t, n, _, _ = inspect_state(
            root / "arms" / arm, c, arm, data, meta["source_sha"], selector=progress
        )
        if (
            t.step != progress["step"]
            or canonical(n.events) != progress["event_state_sha256"]
            or reused.state_fingerprint(root / progress["checkpoint"])
            != progress["training_state_sha256"]
        ):
            raise ContinuationError("EVENT_STATE_INCONSISTENT", arm)
        development = read(root / "arms" / arm / "development-probe-bank.json")
        if (
            development["identity"] != read(root / "arms" / arm / "identity.json")
            or digest(root / "arms" / arm / "development-probe-bank.pt") != development["sha256"]
        ):
            raise ContinuationError("FROZEN_PROBE_MISMATCH")
    return meta


def handoff(folder, destination, *, phase):
    """Freeze only completed files while this worker is blocked at a boundary."""
    folder = Path(folder)
    latest = read(folder / "latest.json")
    target = Path(destination) / (latest["sha256"] + "-" + phase)
    if target.exists():
        document = read(target / "handoff.json")
        for name, row in document["files"].items():
            path = safe(target, name)
            if digest(path) != row["sha256"]:
                raise ContinuationError("HANDOFF_HASH_MISMATCH")
        return target
    temp = target.with_name(target.name + ".preparing")
    temp.mkdir(parents=True)
    files = {}
    import re

    def checkpoint_rank(name):
        return tuple(
            int(v or 0)
            for v in re.fullmatch(r"checkpoint-step-(\d{6})(?:-event-(\d{6}))?\.pt", name).groups()
        )

    latest_rank = checkpoint_rank(latest["file"])
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.name.endswith((".tmp", ".orphan")):
            continue
        if path.is_symlink():
            raise ContinuationError("UNSAFE_PATH")
        name = path.relative_to(folder).as_posix()
        if (
            path.name.startswith("checkpoint-step-")
            and path.suffix == ".pt"
            and not path.with_suffix(".sha256.json").exists()
        ):
            continue
        if (
            path.name.startswith("checkpoint-step-")
            and path.suffix == ".pt"
            and checkpoint_rank(path.name) > latest_rank
        ):
            continue
        if (
            path.name.endswith(".sha256.json")
            and path.name.startswith("checkpoint-step-")
            and checkpoint_rank(path.name.replace(".sha256.json", ".pt")) > latest_rank
        ):
            continue
        copy = temp / name
        copy.parent.mkdir(parents=True, exist_ok=True)
        # Checkpoint arrays and probe banks are immutable. Avoid re-copying GB of
        # historical weights at every boundary on the same Kaggle filesystem.
        if path.suffix in (".pt", ".npy"):
            os.link(path, copy)
        else:
            shutil.copyfile(path, copy)
        files[name] = {"sha256": digest(copy), "bytes": copy.stat().st_size}
    atomic(
        temp / "handoff.json",
        {"files": files, "checkpoint_sha256": latest["sha256"], "phase": phase},
    )
    os.replace(temp, target)
    return target


def segment(
    root,
    arm,
    end,
    *,
    device="cpu",
    committed=None,
    stop_requested=None,
    crash=None,
    observation_reference=False,
):
    root = Path(root)
    c = read(root / "runtime-config.json")
    source = read(root / "experiment.lock.json")["source_sha"]
    if observation_reference and (not c.get("synthetic") or device != "cpu"):
        raise ContinuationError("REFERENCE_OBSERVER_CPU_TEST_ONLY")
    reused.require_source(source)
    if not c.get("synthetic"):
        if c["r3b"]["experiment_id"] != EXPERIMENT:
            raise ContinuationError("FORK_REQUIRES_NEW_RESEARCH_RUNNER")
        release = read(root / "preflight.json")
        require_training_release(
            read(root / "candidate-config.json"),
            source,
            release["acceptance"],
            release["authorization"],
            release["pilot"],
        )
        __import__("clm001_r3.common", fromlist=["authorize"]).authorize(c, "train-dev", device)
    elif device != "cpu":
        raise ContinuationError("SYNTHETIC_CPU_ONLY")
    data = load_assets(c, root / "assets")
    folder = root / "arms" / arm
    restored = (folder / "latest.json").exists()
    if restored:
        trainer, nursery, rng, ident = load_state(folder, c, arm, data, source, device)
    else:
        if (folder / "identity.json").exists():
            raise ContinuationError("PARTIAL_ARM_CANNOT_RANDOM_INIT")
        random.seed(c["seed"])
        np.random.seed(c["seed"])
        torch.manual_seed(c["seed"])
        trainer = Trainer(Model(c, ENGINE[arm]).to(device))
        nursery = Nursery()
        rng = np.random.default_rng(c["seed"] + 10)
        ident = identity(c, arm, source, data)
        atomic(folder / "identity.json", ident)
    bank_started = time.perf_counter()
    frozen = banks(trainer.model, data, c, folder, ident, restored=restored)
    bank_seconds = time.perf_counter() - bank_started
    print(
        {"arm": arm, "RESTORED_FROM_STEP": trainer.step, "STARTED_FROM_RANDOM_INIT": not restored},
        flush=True,
    )
    checkpoint_costs = {}

    def save_local():
        started = time.perf_counter()
        cp.save(folder, trainer, nursery, rng, ident)
        checkpoint_costs[read(folder / "latest.json")["sha256"]] = time.perf_counter() - started

    if not restored:
        save_local()

    gpu_markers = []
    prior_progress = folder / "worker-progress" / (read(folder / "latest.json")["file"] + ".json")
    gpu_seconds = (
        [read(prior_progress).get("gpu_training_seconds", 0) or 0]
        if prior_progress.exists()
        else [0.0]
    )

    def deliver(phase):
        from .diagnostic_schedule import deferred, telemetry

        if deferred(c) and not observation_reference:
            telemetry(folder, trainer, c, phase)
        if gpu_markers:
            torch.cuda.synchronize(device)
            gpu_seconds[0] += sum(start.elapsed_time(end) / 1000 for start, end in gpu_markers)
            gpu_markers.clear()
        latest = read(folder / "latest.json")
        ledger_path = folder / "worker-progress" / (latest["file"] + ".json")
        if not ledger_path.exists():
            telemetry_times = [
                read(p)["telemetry_seconds"] for p in (folder / "telemetry").glob("*.json")
            ]
            atomic(
                ledger_path,
                {
                    "checkpoint_sha256": latest["sha256"],
                    "checkpoint_serialize_seconds": checkpoint_costs.get(latest["sha256"]),
                    "telemetry_host_percentiles_seconds": {
                        name: float(np.percentile(telemetry_times, q)) if telemetry_times else None
                        for name, q in (("p50", 50), ("p90", 90), ("p99", 99))
                    },
                    "telemetry_scope": "journal aggregation excluding atomic metadata write",
                    "update_host_wall_percentiles_seconds": {
                        name: float(
                            np.percentile([r.get("total_seconds", 0) for r in trainer.journal], q)
                        )
                        if trainer.journal
                        else None
                        for name, q in (("p50", 50), ("p90", 90), ("p99", 99))
                    },
                    "step": trainer.step,
                    "live_training_tokens": trainer.data_position,
                    "training_worker_host_seconds": sum(
                        r.get("total_seconds", 0) for r in trainer.journal
                    ),
                    "oracle_seconds": sum(r.get("oracle_seconds", 0) for r in trainer.journal),
                    "competition_candidate_cell_calls": sum(
                        r.get("lcg_candidate_cell_calls", 0) for r in trainer.journal
                    ),
                    "counterfactual_operator_calls": sum(
                        r.get("counterfactual_operator_calls", 0) for r in trainer.journal
                    ),
                    "shadow_private_training_tokens": sum(
                        v.trainer.data_position for v in nursery.candidates
                    ),
                    "checkpoint_bytes": (folder / latest["file"]).stat().st_size,
                    "probe_bank_capture_seconds": bank_seconds if not restored else 0,
                    "probe_bank_load_seconds": bank_seconds if restored else 0,
                    "gpu_training_seconds": gpu_seconds[0] if device.startswith("cuda") else None,
                    "gpu_timer_status": "CUDA_STREAM_ELAPSED_AT_SAFE_BOUNDARY"
                    if device.startswith("cuda")
                    else "CPU_ONLY",
                },
            )
        if committed:
            committed(folder, phase)
        if crash:
            crash(phase, trainer.step)

    def observe(phase):
        from .diagnostic_schedule import deferred, telemetry

        if deferred(c) and not observation_reference:
            telemetry(folder, trainer, c, phase)
        else:
            snapshot(trainer, data, c, frozen, folder, phase)

    event_starts = {}

    def event_callback(kind, when):
        if when == "before":
            event_starts[kind] = time.perf_counter()
        if when == "after":
            latest = read(folder / "latest.json")
            cost = folder / "semantic-costs" / (latest["sha256"] + ".json")
            if not cost.exists():
                atomic(
                    cost,
                    {
                        "category": "TRAINING_SEMANTIC",
                        "event": kind,
                        "step": trainer.step,
                        "checkpoint_sha256": latest["sha256"],
                        "mandatory_event_wall_seconds": time.perf_counter() - event_starts[kind],
                        "scope": "original_event_including_checkpoint_excluding_extra_observation_and_HF_ACK",
                    },
                )
            if kind == pending(c, trainer.step)[-1]:
                observe("post_event")
            deliver("event-" + kind)

    if pending(c, trainer.step):
        process(c, trainer, nursery, data, rng, folder, ident, crash=event_callback)
    observe("post_event")
    deliver("post_event")
    limit = min(end, c["training"]["steps"])
    while trainer.step < limit:
        if stop_requested and stop_requested():
            latest = read(folder / "latest.json")
            saved_step = int(latest["file"].split("-")[2].split(".")[0])
            if trainer.step > saved_step:
                # A different worker's storage failure can arrive between this
                # arm's scheduled boundaries. Preserve its finished local work,
                # without sending a new unconfirmed remote/durable claim.
                save_local()
            break
        rng_before = deepcopy(rng.bit_generator.state)
        try:
            if device.startswith("cuda"):
                marker_start, marker_end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                marker_start.record()
            row, _, _ = trainer.update(batches(data, c, rng, device))
        except Exception:
            rng.bit_generator.state = rng_before
            raise
        if device.startswith("cuda"):
            marker_end.record()
            gpu_markers.append((marker_start, marker_end))
        step = trainer.step
        safe_boundary = (
            step % c["training"]["checkpoint_interval"] == 0
            or pending(c, step)
            or step in c["r3b"]["diagnostics"]["steps"]
            or step == limit
            or (stop_requested and stop_requested())
        )
        if safe_boundary:
            save_local()
            if pending(c, step):
                observe("pre_event")
            else:
                observe("post_event")
            deliver("pre_event" if pending(c, step) else "post_event")
            if pending(c, step):
                process(c, trainer, nursery, data, rng, folder, ident, crash=event_callback)
        print(
            {"arm": arm, "step": step, "train_tokens": trainer.data_position, "ce": row["live_ce"]},
            flush=True,
        )
    return {
        "arm": arm,
        "step": trainer.step,
        "status": "COMPLETE" if trainer.step == c["training"]["steps"] else "PAUSED",
    }
