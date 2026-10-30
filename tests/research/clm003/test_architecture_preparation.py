"""Commit/read-only evidence tests. Zero optimizer calls; no scientific training."""

import copy
import json
import os
from pathlib import Path

import interpolation_audit as original
import pytest
import torch
from connection_commit import receipt
from minicells_r3e.common import fingerprint
from test_architecture import Stream, ab, setup


def test_registered_trust_and_tamper_rejection(tmp_path, monkeypatch):
    m = ab.manifest()
    legacy = ab.legacy_configuration()
    assert m["CLM"]["trust_caps"] == legacy["runtime_template"]["connection"]["trust"]
    original_path = ab.BASE.parent / "SCREENING_MANIFEST.json"
    real_read = Path.read_bytes

    def changed(path):
        raw = real_read(path)
        return raw + b" " if path == original_path else raw

    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(RuntimeError, match="SOURCE_HASH"):
        ab.legacy_configuration()


def test_tensor_receipts_reject_wrong_origin_and_wrong_commit():
    old = {"weight": torch.zeros(2, 2)}
    full = {"weight": torch.ones(2, 2)}
    scaled = {"weight": torch.full((2, 2), 0.25)}
    proposal = {"accepted": True, "scale": 0.25, "selected_connection_sha256": fingerprint(scaled)}
    package = {"original_connection": old, "candidate_connection": full}
    assert receipt(old, scaled, proposal, package)["outer_engine_connection_commits"] == 1
    with pytest.raises(RuntimeError, match="COMMIT_MISMATCH"):
        receipt(old, full, proposal, package)
    with pytest.raises(RuntimeError, match="REJECTED_CONNECTION_CHANGED"):
        receipt(old, scaled, {"accepted": False, "scale": None})
    with pytest.raises(RuntimeError, match="ORIGIN_MISMATCH"):
        receipt(full, scaled, proposal, package)


@pytest.mark.parametrize("accepted", [True, False])
def test_original_outer_engine_commit_and_checkpoint_no_optimizer(tmp_path, monkeypatch, accepted):
    monkeypatch.setattr(os, "fsync", lambda fd: None)
    store, _, models, runtime, _ = setup()
    engine = ab.make_clm_trainer(
        models["CLM"], runtime, store.read("calibration"), tmp_path, connection_only_fixture=True
    )
    engine.fast["connection"]["proposal_steps"] = [1]
    before = copy.deepcopy(engine.model.connection.state_dict())
    package = {
        "original_connection": before,
        "candidate_connection": {k: v + 0.02 for k, v in before.items()},
    }
    scale = 0.25
    expected = {k: v + scale * (package["candidate_connection"][k] - v) for k, v in before.items()}
    target = copy.deepcopy(engine.model)
    target.connection.load_state_dict(expected)
    expected = copy.deepcopy(target.connection.state_dict())
    private = engine.stage_directory / "proposals/1"
    private.mkdir(parents=True)
    torch.save(package, private / "candidate.pt")

    def proposal(tr, data):
        return (target if accepted else None), {
            "accepted": accepted,
            "scale": scale if accepted else None,
            "selected_connection_sha256": fingerprint(expected) if accepted else None,
            "proposal_optimizer_final_state": None,
            "counterfactual_states": 0,
            "actions_per_state": 0,
        }

    monkeypatch.setattr(original.stageb_engine, "propose", proposal)
    engine.update(Stream(store.read("train")), tmp_path / "checkpoint.pt", [])
    record = engine.curve[-1]["connection_commit_receipt"]
    assert record["accepted"] == accepted
    assert record["outer_engine_connection_commits"] == int(accepted)
    assert record["after_sha256"] == fingerprint(expected if accepted else before)
    assert engine.journal.summary()["completed_optimizer_calls"] == 0
    saved = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert saved["state"]["curve"][-1]["connection_commit_receipt"] == record
    out = os.environ.get("ARCHITECTURE_PREPARATION_REPORT")
    if out:
        folder = Path(out)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / ("ACCEPTED.json" if accepted else "REJECTED.json")).write_text(
            json.dumps(
                {
                    "test": "REAL_OUTER_ENGINE_WITH_INJECTED_PROPOSAL_NO_OPTIMIZER",
                    "receipt": record,
                    "optimizer_calls": 0,
                    "gpu_used": False,
                    "scope": "CONTROL_FLOW_VERIFICATION_NOT_PROOF_RANDOM_OR_LEARNED_DIRECTION_QUALITY",
                },
                indent=2,
            )
            + "\n"
        )
    for h in engine._work_handles:
        h.remove()
