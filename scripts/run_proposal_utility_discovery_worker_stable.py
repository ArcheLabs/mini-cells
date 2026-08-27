from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "scripts"))

import minicells.language_proposal_utility as proposal_utility  # noqa: E402
from minicells.language_proposal_checkpoints import (  # noqa: E402
    CHECKPOINT_FORMAT,
    atomic_torch_save,
    checkpoint_root,
    cpu_state_dict,
    donor_path,
    load_checkpoint,
    localized_state_from_payload,
    localized_state_payload,
    phase1_path,
)
from minicells.language_recruitment_numerics import stable_gated_replicator_activity  # noqa: E402

# The utility forward imported the gated replicator into module scope. Replace
# only that numerical primitive; all scientific definitions, models, seeds,
# corpora, losses, epsilon probes and feature extraction remain unchanged.
proposal_utility._gated_replicator_activity = stable_gated_replicator_activity

import run_proposal_utility_discovery_worker as worker  # noqa: E402


CHECKPOINT_ROOT = checkpoint_root()
if CHECKPOINT_ROOT is None:
    raise RuntimeError(
        "stable Experiment 019 worker requires MINICELLS_019_CHECKPOINT_DIR; "
        "launch it through run_proposal_utility_discovery_stable.py"
    )

_original_train_phase1 = worker.e017.train_phase1
_original_train_donor = worker.train_donor
_original_random_control = worker.make_random_control


def _freeze_for_measurement(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def checkpointed_train_phase1(*, replicate, train_stream, validation_stream, vocab_size, device):
    path = phase1_path(CHECKPOINT_ROOT, replicate)
    payload = load_checkpoint(path, kind="phase1", replicate=replicate)
    if payload is not None:
        base_state = cpu_state_dict(payload["base_state"])
        model = worker.e017.clone_from_state(vocab_size, base_state, device)
        print({"replicate": replicate, "checkpoint_reuse": "phase1", "path": str(path)})
        return (
            model,
            base_state,
            payload["checkpoints"],
            payload["events"],
            float(payload["wall_seconds"]),
            [tuple(int(v) for v in item) for item in payload["validation_starts"]],
        )

    result = _original_train_phase1(
        replicate=replicate,
        train_stream=train_stream,
        validation_stream=validation_stream,
        vocab_size=vocab_size,
        device=device,
    )
    model, base_state, checkpoints, events, wall_seconds, validation_starts = result
    atomic_torch_save(path, {
        "format": CHECKPOINT_FORMAT,
        "kind": "phase1",
        "replicate": int(replicate),
        "vocab_size": int(vocab_size),
        "base_state": cpu_state_dict(base_state),
        "checkpoints": checkpoints,
        "events": events,
        "wall_seconds": float(wall_seconds),
        "validation_starts": [tuple(int(v) for v in item) for item in validation_starts],
    })
    print({"replicate": replicate, "checkpoint_saved": "phase1", "path": str(path)})
    return result


def checkpointed_train_donor(
    family,
    *,
    replicate,
    vocab_size,
    base_state,
    device,
):
    path = donor_path(CHECKPOINT_ROOT, replicate, family)
    payload = load_checkpoint(path, kind="donor", replicate=replicate, family=family)
    if payload is not None:
        model = worker.e017.clone_from_state(vocab_size, payload["model_state"], device)
        localized_state = localized_state_from_payload(payload["localized_state"])
        _freeze_for_measurement(model)
        print({"replicate": replicate, "checkpoint_reuse": family, "path": str(path)})
        return model, localized_state, payload["summary"], payload["events"]

    result = _original_train_donor(
        family,
        replicate=replicate,
        vocab_size=vocab_size,
        base_state=base_state,
        device=device,
    )
    model, localized_state, summary, events = result
    atomic_torch_save(path, {
        "format": CHECKPOINT_FORMAT,
        "kind": "donor",
        "replicate": int(replicate),
        "family": family,
        "vocab_size": int(vocab_size),
        "model_state": cpu_state_dict(model.state_dict()),
        "localized_state": localized_state_payload(localized_state),
        "summary": summary,
        "events": events,
    })
    print({"replicate": replicate, "checkpoint_saved": family, "path": str(path)})
    return result


def checkpointed_random_control(*, replicate, vocab_size, base_state, device):
    family = worker.RANDOM_CONTROL
    path = donor_path(CHECKPOINT_ROOT, replicate, family)
    payload = load_checkpoint(path, kind="donor", replicate=replicate, family=family)
    if payload is not None:
        model = worker.e017.clone_from_state(vocab_size, payload["model_state"], device)
        localized_state = localized_state_from_payload(payload["localized_state"])
        _freeze_for_measurement(model)
        print({"replicate": replicate, "checkpoint_reuse": family, "path": str(path)})
        return model, localized_state, payload["summary"], payload["events"]

    result = _original_random_control(
        replicate=replicate,
        vocab_size=vocab_size,
        base_state=base_state,
        device=device,
    )
    model, localized_state, summary, events = result
    atomic_torch_save(path, {
        "format": CHECKPOINT_FORMAT,
        "kind": "donor",
        "replicate": int(replicate),
        "family": family,
        "vocab_size": int(vocab_size),
        "model_state": cpu_state_dict(model.state_dict()),
        "localized_state": localized_state_payload(localized_state),
        "summary": summary,
        "events": events,
    })
    print({"replicate": replicate, "checkpoint_saved": family, "path": str(path)})
    return result


worker.e017.train_phase1 = checkpointed_train_phase1
worker.train_donor = checkpointed_train_donor
worker.make_random_control = checkpointed_random_control


if __name__ == "__main__":
    raise SystemExit(worker.main())
