from __future__ import annotations

import torch

from minicells.language_growing_organism import build_cellular_model
from minicells.language_localized_learning import LocalizedLearningState, conservative_fork
from minicells.language_proposal_checkpoints import (
    CHECKPOINT_FORMAT,
    atomic_torch_save,
    cpu_state_dict,
    donor_path,
    load_checkpoint,
    localized_state_from_payload,
    localized_state_payload,
    phase1_path,
)


def _model_and_localized_state():
    torch.manual_seed(19019)
    model = build_cellular_model(128, "G")
    localized = LocalizedLearningState.capture(model)
    direction = torch.zeros(model.dim)
    direction[0] = 1.0
    child = conservative_fork(model, 1, step=0, direction=direction)
    assert child is not None
    return model, localized


def test_checkpoint_paths_are_replicate_and_family_specific(tmp_path) -> None:
    assert phase1_path(tmp_path, 2).name == "r2-phase1.pt"
    assert donor_path(tmp_path, 1, "REVERSE_INC").name == "r1-donor-REVERSE_INC.pt"


def test_donor_checkpoint_roundtrip_preserves_model_and_localized_state(tmp_path) -> None:
    model, localized = _model_and_localized_state()
    path = donor_path(tmp_path, 0, "PARITY")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "kind": "donor",
        "replicate": 0,
        "family": "PARITY",
        "vocab_size": 128,
        "model_state": cpu_state_dict(model.state_dict()),
        "localized_state": localized_state_payload(localized),
        "summary": {"skill_improvement": 1.0},
        "events": [],
    }
    atomic_torch_save(path, payload)
    loaded = load_checkpoint(path, kind="donor", replicate=0, family="PARITY")
    assert loaded is not None
    for key, value in model.state_dict().items():
        assert torch.equal(loaded["model_state"][key], value.cpu()), key
    restored = localized_state_from_payload(loaded["localized_state"])
    assert torch.equal(restored.base_alive, localized.base_alive)
    assert torch.equal(restored.base_adjacency, localized.base_adjacency)
    assert torch.equal(restored.base_memory, localized.base_memory)


def test_checkpoint_identity_mismatch_is_rejected(tmp_path) -> None:
    path = phase1_path(tmp_path, 0)
    atomic_torch_save(path, {
        "format": CHECKPOINT_FORMAT,
        "kind": "phase1",
        "replicate": 0,
        "base_state": {},
    })
    try:
        load_checkpoint(path, kind="phase1", replicate=1)
    except RuntimeError as exc:
        assert "identity mismatch" in str(exc)
    else:
        raise AssertionError("mismatched checkpoint identity must fail")


def test_force_retrain_bypasses_existing_checkpoint(tmp_path, monkeypatch) -> None:
    path = phase1_path(tmp_path, 0)
    atomic_torch_save(path, {
        "format": CHECKPOINT_FORMAT,
        "kind": "phase1",
        "replicate": 0,
        "base_state": {},
    })
    monkeypatch.setenv("MINICELLS_019_FORCE_RETRAIN", "1")
    assert load_checkpoint(path, kind="phase1", replicate=0) is None
