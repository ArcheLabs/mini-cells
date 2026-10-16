"""Exact historical event selection, read-only evaluation and explicit child forks."""

from copy import deepcopy
from pathlib import Path
import shutil
from minicells_runtime.store import (
    MountedInputStore,
    atomic,
    read,
    canonical,
    digest,
    ContinuationError,
)
from clm001_r3.data import load_assets
from clm001_r3.evaluation import evaluate
from clm001_r3.training import Trainer
from clm001_r3 import checkpoint as cp
from . import runner
from .common import identity
from .instrumentation import isolated


def evaluate_checkpoint(root, output, arm, step, revision=None):
    root, output = Path(root), Path(output)
    meta = runner.verify(root)
    selected = MountedInputStore(root).select(arm, step, revision)
    if (
        output.exists()
        or root.resolve() == output.resolve()
        or root.resolve() in output.resolve().parents
    ):
        raise ContinuationError("EVALUATION_OUTPUT_MUST_BE_SEPARATE")
    before = digest(root / "continuation.json")
    c = read(root / "runtime-config.json")
    data = load_assets(c, root / "assets")
    trainer, _, _, _ = runner.load_state(
        root / "arms" / arm, c, arm, data, meta["source_sha"], selector=selected
    )
    with isolated(trainer.model, trainer):
        quality = {
            p: evaluate(trainer.model, data, c, "development", p)
            for p in ("greedy", "sampled", "uniform", "constant")
        }
    result = {
        "experiment_id": meta["experiment_id"],
        "run_id": meta["run_id"],
        "arm": arm,
        "checkpoint": selected,
        "source_sha": meta["source_sha"],
        "evaluation_source_sha": runner.reused.source_sha(),
        "evaluation_protocol": "r3b-historical-development-v1",
        "quality": quality,
        "retrospective": True,
        "exploratory": True,
        "optimizer_updates": 0,
        "scientific_decision": False,
    }
    if digest(root / "continuation.json") != before:
        raise ContinuationError("READ_ONLY_EVALUATION_MUTATED_SOURCE")
    output.mkdir(parents=True)
    atomic(output / "evaluation.json", result)
    return result


def fork_checkpoint(
    root, output, arm, step, *, revision, new_run_id, new_experiment_id, optimizer_policy
):
    root, output = Path(root), Path(output)
    meta = runner.verify(root)
    if (
        not new_run_id
        or new_run_id == meta["run_id"]
        or not new_experiment_id
        or new_experiment_id == meta["experiment_id"]
    ):
        raise ContinuationError("FORK_REQUIRES_NEW_EXPERIMENT_AND_RUN")
    if optimizer_policy not in ("inherit", "reset"):
        raise ContinuationError("FORK_OPTIMIZER_POLICY_REQUIRED")
    selected = MountedInputStore(root).select(arm, step, revision)
    c = read(root / "runtime-config.json")
    data = load_assets(c, root / "assets")
    trainer, nursery, rng, _ = runner.load_state(
        root / "arms" / arm, c, arm, data, meta["source_sha"], selector=selected
    )
    new = deepcopy(c)
    new["r3b"].update(
        experiment_id=new_experiment_id, run_id=new_run_id, parent_checkpoint=selected
    )
    new["execution"]["source_sha"] = runner.reused.source_sha()
    new["execution"]["configuration_sha256"] = canonical(
        {k: v for k, v in new.items() if k != "execution"}
    )
    runner.new_run(
        read(root / "candidate-config.json"),
        new,
        root / "assets",
        output,
        release={"passed": False, "scope": "EXPLICIT_FORK_NOT_AUTHORIZED_TO_TRAIN"},
    )
    trainer.model.experiment_config = deepcopy(new)
    for candidate in nursery.candidates:
        candidate.model.experiment_config = deepcopy(new)
    if optimizer_policy == "reset":

        def reset(previous):
            fresh = Trainer(previous.model)
            fresh.rng = previous.rng
            fresh.step, fresh.data_position, fresh.journal = (
                previous.step,
                previous.data_position,
                previous.journal,
            )
            return fresh

        trainer = reset(trainer)
        for candidate in nursery.candidates:
            candidate.trainer = reset(candidate.trainer)
    expected = identity(new, arm, runner.reused.source_sha(), data)
    nursery.events = {
        canonical(
            {
                "identity": expected,
                "step": v["step"],
                "event": v["event"],
                "base_registry": v["base_registry"],
            }
        ): v
        for v in nursery.events.values()
    }
    folder = output / "arms" / arm
    atomic(folder / "identity.json", expected)
    for name in ("frozen-probe-bank", "development-probe-bank"):
        shutil.copyfile(root / "arms" / arm / (name + ".pt"), folder / (name + ".pt"))
        witness = read(root / "arms" / arm / (name + ".json"))
        atomic(folder / (name + ".json"), dict(witness, identity=expected))
    cp.save(folder, trainer, nursery, rng, expected, event_revision=len(nursery.events) or None)
    atomic(
        output / "parent-fork.json",
        {
            "parent_experiment_id": meta["experiment_id"],
            "parent_run_id": meta["run_id"],
            "parent_checkpoint": selected,
            "parent_source_sha": meta["source_sha"],
            "optimizer_policy": optimizer_policy,
            "fork_updates": 0,
            "future_training_requires_new_protocol": True,
        },
    )
    return runner.package(output)
