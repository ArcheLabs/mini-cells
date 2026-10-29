"""Synthetic checks for formal main-loop integration, not real scientific execution."""

import json
from types import SimpleNamespace

import pytest
import torch
from minicells_fast001.data import Stream
from minicells_fast_r1 import publication, stageb, stageb_engine
from minicells_fast_r1.common import ID, read, sha
from minicells_r3e.common import fingerprint
from test_stateful import fixture


def setup(c):
    c["updates"] = 4
    c["connection"]["proposal_steps"] = [2, 4]
    c["diagnostic"]["gate1_execution_authorized"] = True
    c["stage_b"] = stageb.configuration()["stage_b"]
    x = torch.tensor([[7, 8, 9, 10]])
    c["admission"].update(
        manifest_sha256="fixture", ordered_input_hashes=[sha(x.squeeze(0).tolist())]
    )
    return {"role": "admission", "manifest_sha256": "fixture", "inputs": [x]}


@pytest.mark.parametrize("arm", ["B00-H", "B10-H", "B01-H", "B11-H"])
def test_formal_main_loop_masks_real_candidates_and_checkpoint_restore(tmp_path, arm):
    c, state, folder = fixture(tmp_path)
    bundle = setup(c)
    directory = tmp_path / "run/private/B" / ("7001-" + arm)
    directory.mkdir(parents=True)
    identity = {"seed": 7001, "arm": arm, "scope": "SYNTHETIC_CPU_ONLY"}
    parent = {"checkpoint_sha256": "fixture", "identity": identity}
    engine = stageb_engine.make_engine(
        c,
        state,
        7001,
        arm,
        identity,
        directory,
        tmp_path / "run/proposals",
        bundle,
        parent,
        device="cpu",
        unit=True,
    )
    source_public = fingerprint(
        {n: p for n, p in engine.model.named_parameters() if not p.requires_grad}
    )
    stream = Stream(c, folder, 7001, "H", "B")
    engine.save(directory / "latest.pt", stream)
    while engine.step < 4:
        engine.update(stream, directory / "latest.pt", [])
    assert engine.step == stream.step == 4
    assert source_public == fingerprint(
        {n: p for n, p in engine.model.named_parameters() if not p.requires_grad}
    )
    cost = stageb.costs(tmp_path / "run")
    assert cost["main_update_attempts"] == cost["committed_main_updates"] == 4
    assert cost["cell_optimizer_calls"] == (4 if arm[:3] in ("B10", "B11") else 0)
    assert cost["candidate_optimizer_calls"] == (4 if arm[:3] in ("B01", "B11") else 0)
    assert cost["scale_checks"] == (44 if arm[:3] in ("B01", "B11") else 0)
    for row in engine.curve:
        if row["proposal"]:
            assert row["proposal"]["all_scales"] == 22
            assert row["proposal"]["r3fv2_nonempty_path_required"]
    before = fingerprint(engine.model.state_dict())
    restored = stageb_engine.make_engine(
        c,
        state,
        7001,
        arm,
        identity,
        directory,
        tmp_path / "run/proposals",
        bundle,
        parent,
        device="cpu",
        unit=True,
    )
    rs = Stream(c, folder, 7001, "H", "B")
    restored.restore(directory / "latest.pt", rs)
    assert restored.step == 4 and fingerprint(restored.model.state_dict()) == before
    assert fingerprint(restored.cell_optimizer.state_dict()) == fingerprint(
        engine.cell_optimizer.state_dict()
    )
    assert fingerprint(restored.policy_optimizer_state) == fingerprint(
        engine.policy_optimizer_state
    )
    assert fingerprint(restored.cf_rng.get_state()) == fingerprint(engine.cf_rng.get_state())
    assert fingerprint(restored.route_rng.get_state()) == fingerprint(engine.route_rng.get_state())
    assert fingerprint(rs.state()) == fingerprint(stream.state())
    for e in (engine, restored):
        for hook in e._work_handles:
            hook.remove()


def test_formal_selection_requires_both_rules_before_official_commit(tmp_path, monkeypatch):
    c, state, folder = fixture(tmp_path)
    bundle = setup(c)
    directory = tmp_path / "job"
    directory.mkdir()
    identity = {"seed": 7001, "arm": "B11-H"}
    engine = stageb_engine.make_engine(
        c,
        state,
        7001,
        "B11-H",
        identity,
        directory,
        tmp_path / "reports",
        bundle,
        {"checkpoint_sha256": "fixture", "identity": identity},
        device="cpu",
        unit=True,
    )
    original = stageb_engine.scan.run

    def measured_then_controlled(*args, **kwargs):
        decision = original(*args, **kwargs)
        # Test-only admission decision intervention; keep actual scaled fingerprints.
        for r in decision["attempts"]:
            r["passed"] = r["scale"] in (1.0, 0.5)
            r["r3fv2_nonempty_path_pass"] = r["scale"] == 0.5
        decision["chosen_scale"] = 1.0
        return decision

    monkeypatch.setattr(stageb_engine.scan, "run", measured_then_controlled)
    stream = Stream(c, folder, 7001, "H", "B")
    engine.update(stream, directory / "latest.pt", [])
    engine.update(stream, directory / "latest.pt", [])
    proposal = engine.curve[-1]["proposal"]
    assert proposal["accepted"] and proposal["scale"] == 0.5
    assert proposal["primary_chosen_scale"] == 1.0
    assert (
        fingerprint(engine.model.connection.state_dict()) == proposal["selected_connection_sha256"]
    )


def test_worker_queues_and_frozen_budget_no_baseline_reuse():
    c = stageb.configuration()
    jobs = stageb.queue(c, 0, 2) + stageb.queue(c, 1, 2)
    assert len(jobs) == len(set(jobs)) == 40
    assert c["updates"] == 500 and c["stage_b"]["max_main_updates"] == 20000
    assert c["stage_b"]["max_candidate_optimizer_calls"] == 1600
    assert not c["stage_b"]["historical_final_B_weights_reused"]
    assert c["stage_b"]["all_40_paths_fresh_from_original_A500"]


def test_formal_failure_returns_metadata_and_publisher_never_accepts_private_weights(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(stageb, "require_ci", lambda: {"scope": "TEST_ONLY"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = SimpleNamespace(
        output=tmp_path,
        assets=tmp_path / "assets",
        worker=None,
        publish_existing=False,
        publish_github=False,
    )
    with pytest.raises(RuntimeError, match="STAGEB_STOPPED"):
        stageb.main(args)
    root = next(p.parent for p in tmp_path.glob("*/RUN_MANIFEST.json"))
    manifest = read(root / "RUN_MANIFEST.json")
    assert manifest["scientific_optimizer_updates"] == 0
    assert read(root / "FAILURE.json")["code"] == "R1_STAGEB_GPU_REQUIRED_NO_CPU_TRAINING_FALLBACK"
    assert not args.assets.exists()
    assert publication.allowed("jobs/7001-B11-H.json")
    assert publication.allowed("proposals/7001-B11-H-50/ADMISSION_DECISION.json")
    assert not publication.allowed("private/B/7001-B11-H/latest.pt")
    assert not publication.allowed("proposals/7001-B11-H-50/candidate.pt")
    _, files, _, _ = publication.payload(root)
    assert "RUN_MANIFEST.json" in files and manifest["experiment_id"] == ID


def test_stageb_notebook_selects_formal_entry_and_return_only_retry():
    from minicells_fast_r1.common import BASE

    notebook = json.loads((BASE / "notebooks/kaggle-r1-stage-b.ipynb").read_text())
    code = "".join(notebook["cells"][1]["source"])
    compile(code, "stageb-delivery", "exec")
    assert "CODE/BASE_REL/'stage_b.py'" in code
    assert "'--run-diagnostic'" not in code
    assert ".startswith('FORMAL_STAGE_B_COMPLETE_')" in code
    assert "'--publish-existing'" in code
    assert "fast-r1-stageb-output" in code
    assert notebook["metadata"]["kaggle"]["isGpuEnabled"]
