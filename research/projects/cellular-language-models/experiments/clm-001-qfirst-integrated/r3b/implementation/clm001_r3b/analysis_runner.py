"""Read-only, task-granular analysis of exact immutable training checkpoints."""

from copy import deepcopy
from pathlib import Path
import hashlib
import random
import time
from datetime import datetime, timezone
import numpy as np
import torch
from minicells_runtime.store import (
    MountedInputStore,
    atomic,
    canonical,
    digest,
    read,
    ContinuationError,
)
from clm001_r3.model import Model
from clm001_r3.training import Trainer
from clm001_r3.actions import Registry
from clm001_r3.data import load_assets
from clm001_r3.specialization import rebind, crossover
from .common import ENGINE
from . import runner, instrumentation as obs
from .diagnostic_schedule import validate, stages


def fingerprint(value):
    h = hashlib.sha256()

    def visit(v):
        h.update(type(v).__name__.encode())
        if isinstance(v, torch.Tensor):
            h.update(str((v.dtype, tuple(v.shape))).encode())
            h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(str((v.dtype, v.shape)).encode())
            h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                visit(k)
                visit(v[k])
        elif isinstance(v, (tuple, list)):
            for row in v:
                visit(row)
        else:
            h.update(repr(v).encode())

    visit(value)
    return h.hexdigest()


def quality(model, data, c, role, policies):
    values = {p: obs.evaluate(model, data, c, role, p) for p in policies}
    key = canonical(
        {
            "windows": data["windows"][role]["evaluation"],
            "seed": c["evaluation"]["seed"],
            "assets": data["asset_sha256"],
            "role": role,
        }
    )
    for row in values.values():
        row["original_policy_identity"] = row["paired_identity"]
        row["paired_identity"] = key
    return values


def list_checkpoints(root):
    meta = runner.verify(root)
    return [
        {"arm": arm, **record} for arm, row in meta["arms"].items() for record in row["checkpoints"]
    ]


def identity(root, ref, plan, analysis_source, arm, step, revision, phase, stage, role, policies):
    validate(plan)
    if stage not in ("quality", "functional", "action") or role not in (
        "calibration",
        "development",
    ):
        raise ContinuationError("ANALYSIS_ROLE_OR_STAGE_INVALID")
    if (
        stage == "quality"
        and (
            not policies
            or any(p not in ("greedy", "uniform", "sampled", "constant") for p in policies)
        )
    ) or (stage != "quality" and policies):
        raise ContinuationError("ANALYSIS_POLICY_INVALID")
    if stage == "action" and role != "calibration":
        raise ContinuationError("ANALYSIS_ROLE_OR_STAGE_INVALID")
    if policies != list(dict.fromkeys(policies)):
        raise ContinuationError("ANALYSIS_POLICY_INVALID")
    if phase not in ("pre_event", "post_event"):
        raise ContinuationError("ANALYSIS_PHASE_INVALID")
    meta = read(Path(root) / "continuation.json")
    if ref.get("hf_repo_type") != "model" or len(ref.get("hf_revision", "")) != 40:
        raise ContinuationError("PINNED_HF_REFERENCE_REQUIRED")
    if len(analysis_source) != 40 or any(v not in "0123456789abcdef" for v in analysis_source):
        raise ContinuationError("ANALYSIS_SOURCE_INVALID")
    selected = MountedInputStore(root).select(arm, step, revision)
    records = [r for r in meta["arms"][arm]["checkpoints"] if r["step"] == step]
    c = read(Path(root) / "runtime-config.json")
    from clm001_r3.events import pending

    event_types = {v["event"] for v in selected["event_cursor"].values()}
    if step in (400, c["shadow"]["admission_step"]) and (
        (phase == "pre_event" and selected["revision"] != 0)
        or (
            phase == "post_event"
            and (
                selected["revision"] == 0
                or selected["revision"] != max(r["revision"] for r in records)
                or not set(pending(c, step)).issubset(event_types)
            )
        )
    ):
        raise ContinuationError("EXACT_EVENT_PHASE_REQUIRED")
    c = read(Path(root) / "runtime-config.json")
    folder = Path(root) / "arms" / arm
    name = "frozen-probe-bank" if role == "calibration" else "development-probe-bank"
    witness = read(folder / (name + ".json"))
    if digest(folder / (name + ".pt")) != witness["sha256"]:
        raise ContinuationError("FROZEN_PROBE_MISMATCH")
    if role == "development" and witness.get("role", "development") != "development":
        raise ContinuationError("PILOT_BANK_IS_NOT_DEVELOPMENT")
    windows = (
        read(Path(root) / "assets/probe-windows.json")
        if (Path(root) / "assets/probe-windows.json").exists()
        else None
    )
    # load_assets validates the actual immutable role/window identities.
    data = load_assets(c, Path(root) / "assets")
    return {
        "experiment_id": meta["experiment_id"],
        "training_run_id": meta["run_id"],
        "training_source_sha": meta["source_sha"],
        "runtime_config_sha256": meta["runtime_config_sha256"],
        "assets_manifest_sha256": meta["assets_manifest_sha256"],
        "tokenizer_sha256": meta["tokenizer_sha256"],
        "training_protocol_sha256": meta["protocol_sha256"],
        "arm": arm,
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "step": step,
        "event_revision": selected["revision"],
        "phase": phase,
        "event_semantics": sorted(v["event"] for v in selected["event_cursor"].values()),
        "input_checkpoint_ref": ref,
        "analysis_source_sha": analysis_source,
        "analysis_protocol_sha256": canonical(plan),
        "frozen_sampling_protocol_sha256": canonical(c["r3b"]["diagnostics"]),
        "stage": stage,
        "role": role,
        "policies": list(policies),
        "evaluation_seed": c["evaluation"]["seed"],
        "frozen_probe_sha256": witness["sha256"],
        "dataset_window_sha256": canonical(data["windows"][role]),
        "state_scope": "FROZEN_INITIAL_STATE",
        "window_file_sha256": canonical(windows) if windows else None,
    }


def expand(root, ref, plan, analysis_source, stage=None):
    meta = runner.verify(root)
    tasks, missing = [], []
    for arm in ENGINE:
        for step in sorted(
            set(
                plan["posthoc"]["quality_primary_steps"]
                + plan["posthoc"]["functional_steps"]
                + plan["posthoc"]["quality_secondary_steps"]
            )
        ):
            for phase in ("pre_event", "post_event") if step == 400 else ("post_event",):
                records = [
                    r
                    for r in meta.get("arms", {}).get(arm, {}).get("checkpoints", [])
                    if r["step"] == step
                ]
                if not records or (
                    step == 400
                    and phase == "post_event"
                    and max(r["revision"] for r in records) == 0
                ):
                    missing.append(
                        {"arm": arm, "step": step, "phase": phase, "status": "NOT_EVALUATED"}
                    )
                    continue
                revision = 0 if phase == "pre_event" else max(r["revision"] for r in records)
                for kind, policies, role in stages(plan, step):
                    if stage and stage != kind:
                        continue
                    try:
                        tasks.append(
                            identity(
                                root,
                                ref,
                                plan,
                                analysis_source,
                                arm,
                                step,
                                revision,
                                phase,
                                kind,
                                role,
                                policies,
                            )
                        )
                    except ContinuationError as error:
                        if error.code not in (
                            "PILOT_BANK_IS_NOT_DEVELOPMENT",
                            "EXACT_EVENT_PHASE_REQUIRED",
                        ):
                            raise
                        missing.append(
                            {
                                "arm": arm,
                                "step": step,
                                "phase": phase,
                                "stage": kind,
                                "role": role,
                                "status": error.code,
                            }
                        )
    return tasks, missing


def measured_state(trainer):
    return fingerprint(
        {
            "model": trainer.model.state_dict(),
            "optimizer": trainer.opt.state_dict(),
            "registry": vars(trainer.model.registry),
            "version": trainer.model.version,
            "step": trainer.step,
            "data_position": trainer.data_position,
            "journal": trainer.journal,
            "streams": {k: v.get_state() for k, v in trainer.rng.items()},
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    )


@obs.isolated(torch.nn.Identity())
def execute(root, task, plan, *, device="cpu"):
    root = Path(root)
    # Re-derive all identity fields from verified input, never trust an arbitrary task JSON.
    runner.verify(root)
    expected = identity(
        root,
        task["input_checkpoint_ref"],
        plan,
        task["analysis_source_sha"],
        task["arm"],
        task["step"],
        task["event_revision"],
        task["phase"],
        task["stage"],
        task["role"],
        task["policies"],
    )
    if task != expected:
        raise ContinuationError("ANALYSIS_TASK_IDENTITY_MISMATCH")
    before_files = {
        p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file()
    }
    c = deepcopy(read(root / "runtime-config.json"))
    data = load_assets(c, root / "assets")
    selected = MountedInputStore(root).select(task["arm"], task["step"], task["event_revision"])
    begin = time.perf_counter()
    timings = {}
    with obs.isolated(torch.nn.Identity()):
        payload = torch.load(root / selected["checkpoint"], map_location="cpu", weights_only=False)
        model = Model(c, ENGINE[task["arm"]]).to(device)
        model.load_state_dict(payload["model"], strict=True)
        model.registry = Registry(**payload["extra"]["registry"])
        model.version = payload["extra"]["version"]
        model.experiment_config["policy"]["epsilon"] = payload["extra"]["execution_epsilon"]
        trainer = Trainer(model)
        trainer.opt.load_state_dict(payload["optimizer"])
        trainer.step, trainer.data_position = payload["step"], payload["extra"]["data_position"]
        trainer.journal = payload["extra"]["journal"]
        folder = root / "arms" / task["arm"]
        bankname = (
            "frozen-probe-bank.pt" if task["role"] == "calibration" else "development-probe-bank.pt"
        )
        frozen = torch.load(folder / bankname, map_location=device, weights_only=False)
        guard = measured_state(trainer)

        def timed(name, call):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            value = call()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            timings[name] = time.perf_counter() - start
            return value

        with obs.isolated(model, trainer):
            if task["stage"] == "quality":
                value = {"quality": {}}
                for policy in task["policies"]:
                    value["quality"].update(
                        timed(
                            "quality_" + policy,
                            lambda: quality(model, data, c, task["role"], [policy]),
                        )
                    )
            elif task["stage"] == "functional":
                value = timed(
                    "functional",
                    lambda: obs.functional_measure(
                        model, rebind(frozen, model), c, resampled=task["role"] == "calibration"
                    ),
                )
                value = {"functional": {**value, "crossover": crossover(value["records"], c)}}
            else:
                value = {
                    "router": timed("router", lambda: obs.router_rows(model, frozen, c)),
                    "action_type_calibration": timed(
                        "action", lambda: obs.action_calibration(model, frozen, c)
                    ),
                }
        if measured_state(trainer) != guard:
            raise ContinuationError("ANALYSIS_MUTATED_TRAINING_STATE")
    after_files = {
        p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file()
    }
    if before_files != after_files:
        raise ContinuationError("READ_ONLY_ANALYSIS_MUTATED_INPUT")
    samples = (
        sum(len(v["nll"]) for v in value["quality"].values())
        if "quality" in value
        else len(value["functional"]["state_records"])
        if "functional" in value
        else len(value["router"])
    )
    return {
        "schema": "minicells.r3b.analysis-result.v1",
        "identity": task,
        "analysis_mode": "POSTHOC_FROM_IMMUTABLE_CHECKPOINT",
        "task_status": "COMPLETE",
        "measurement_time": datetime.now(timezone.utc).isoformat(),
        "result": value,
        "coverage_status": "OBSERVED" if samples else "INSUFFICIENT_COVERAGE",
        "sample_count": samples,
        "optimizer_updates": 0,
        "training_state_unchanged": True,
        "elapsed_seconds": time.perf_counter() - begin,
        "timings": timings,
        "gradient_audit": obs.gradient_row(trainer, c),
        "training_exposure_by_group": {
            str(g): {
                str(a): sum(
                    row.get("cell_exposure_by_group", {})
                    .get(g, row.get("cell_exposure_by_group", {}).get(str(g), {}))
                    .get(str(a), 0)
                    for row in trainer.journal
                )
                for a in range(model.registry.slots)
            }
            for g in (0, 1)
        },
        "scientific_decision": False,
        "confirmation": False,
    }


def batch(root, output, tasks, plan, storage, *, max_tasks=3, device="cpu"):
    if max_tasks < 0:
        raise ContinuationError("TASK_BUDGET_INVALID")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    completed, ledger, consumed = [], [], 0
    from .common import safe_error

    for task in tasks:
        key = canonical(task)
        cached = storage.completed(task)
        if cached is not None:
            completed.append(cached)
            ledger.append({"task_key": key, "status": "COMPLETE", "reused": True})
            continue
        if consumed >= max_tasks:
            ledger.append({"task_key": key, "status": "SKIPPED_BUDGET"})
            continue
        consumed += 1
        atomic(output / "tasks" / (key + ".json"), {"status": "RUNNING", "identity": task})
        try:
            result = execute(root, task, plan, device=device)
            entry = storage.commit(result)
            completed.append(result)
            ledger.append({"task_key": key, "status": "COMPLETE", "reused": False, "entry": entry})
        except Exception as error:
            ledger.append({"task_key": key, "status": "BLOCKED", **safe_error(error)})
            atomic(output / "analysis-index.json", {"tasks": ledger, "status": "ANALYSIS_PARTIAL"})
            raise
        atomic(output / "tasks" / (key + ".json"), ledger[-1])
        atomic(
            output / "analysis-index.json",
            {"tasks": ledger, "status": "ANALYSIS_PARTIAL", "scientific_decision": False},
        )
    return completed, ledger
