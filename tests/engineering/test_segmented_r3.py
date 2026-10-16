from pathlib import Path
import json
import pytest
import torch
from minicells_runtime import r3
from minicells_runtime.smoke import configuration, prepare
from minicells_runtime.store import read, digest, ContinuationError


def initial(tmp_path, name="run", run_id="same-run"):
    c = configuration()
    assets = tmp_path / (name + "-assets")
    prepare(assets, c)
    root = tmp_path / name
    r3.new_run(c, assets, root, source=r3.source_sha(), run_id=run_id)
    return root


def test_exact_resume_history_fork_and_no_replay(tmp_path, monkeypatch):
    a = initial(tmp_path, "a")
    control = initial(tmp_path, "control")
    for arm in configuration()["arms"]:
        r3.segment_arm(a, arm, 6)
        r3.segment_arm(control, arm, 10)
    r3.package(a)
    b = tmp_path / "b"
    result = r3.resume(a, b, 10, source=r3.source_sha())
    assert result["status"] == "COMPLETE"
    r3.package(control)
    ca = r3.verify(control)
    cb = r3.verify(b)
    for arm in configuration()["arms"]:
        assert ca["arms"][arm]["training_state_sha256"] == cb["arms"][arm]["training_state_sha256"]
        assert ca["arms"][arm]["event_cursor"] == cb["arms"][arm]["event_cursor"]
    before = (b / "continuation.json").read_bytes()
    assert r3.resume(a, b, 10, source=r3.source_sha())["complete"]
    assert (b / "continuation.json").read_bytes() == before
    done = r3.segment_arm(b, "QC-R3", 6)
    assert done["last_committed_step"] == 10
    assert (b / "continuation.json").read_bytes() == before

    def forbidden(*args, **kwargs):
        raise AssertionError("Optimizer.step during evaluation")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    report = r3.evaluate_checkpoint(
        b, tmp_path / "eval", "QC-R3", 6, {"policies": ["greedy"], "seed": 44}
    )
    assert report["step"] == 6
    assert report["checkpoint"]["revision"] > 0
    assert (b / "continuation.json").read_bytes() == before
    c = read(b / "runtime-config.json")
    fork = r3.fork_experiment(
        b, tmp_path / "fork", "QC-R3", 6, c, run_id="new-fork", optimizer_policy="inherit"
    )
    assert fork["run_id"] == "new-fork" and fork["run_id"] != result["run_id"]
    assert read(tmp_path / "fork/parent-fork.json")["parent_step"] == 6


@pytest.mark.parametrize("stop", [2, 3, 7, 8])
def test_event_boundaries_and_uneven_arms(tmp_path, stop):
    a = initial(tmp_path)
    r3.segment_arm(a, "Q-R3", 10)
    r3.segment_arm(a, "QC-R3", stop)
    first = r3.package(a)
    assert first["status"] == "PAUSED" and "QS-R3" not in first["arms"]
    b = tmp_path / "resumed"
    r3.resume(a, b, 10, source=r3.source_sha())
    meta = r3.verify(b)
    assert meta["complete"]
    assert all(v["last_committed_step"] == 10 for v in meta["arms"].values())


def test_resume_refuses_missing_and_disabled(tmp_path):
    with pytest.raises(ContinuationError, match="RESUME_INPUT_MISSING"):
        r3.resume(tmp_path / "none", tmp_path / "out", 6, source=r3.source_sha())
    with pytest.raises(ContinuationError, match="RESUME_NOT_REQUESTED"):
        r3.resume(tmp_path / "none", tmp_path / "out", 6, require_resume=False)


@pytest.mark.parametrize(
    "phase", ["checkpoint-before", "checkpoint-after", "snapshot-before", "snapshot-after"]
)
def test_interruption_resumes_last_committed_state(tmp_path, phase):
    root = initial(tmp_path)
    fired = []

    def crash(at):
        if at == phase and not fired:
            fired.append(at)
            raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        r3.segment_arm(root, "QC-R3", 6, crash=crash)
    r3.segment_arm(root, "QC-R3", 6)
    meta = r3.package(root)
    assert meta["arms"]["QC-R3"]["last_committed_step"] == 6
    r3.verify(root)


def test_new_process_resume_and_publication_failure_preserves_state(tmp_path, monkeypatch):
    import subprocess
    import sys
    from minicells_runtime import publication

    root = initial(tmp_path)
    r3.segment_arm(root, "QC-R3", 6)
    before = r3.package(root)
    out = tmp_path / "child"
    command = [
        sys.executable,
        str(
            Path(__file__).resolve().parents[2]
            / "research/projects/cellular-language-models/engineering/kaggle-continuation/run.py"
        ),
        "--mode",
        "auto",
        "--checkpoint-input",
        str(root),
        "--output",
        str(out),
        "--require-resume",
        "true",
        "--stop-after-step",
        "10",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert r3.verify(out)["complete"]
    digest_before = digest(out / "continuation.json")

    def fail(*args, **kwargs):
        raise RuntimeError("no credential")

    monkeypatch.setattr(publication.existing, "prepare", fail)
    with pytest.raises(RuntimeError, match="no credential"):
        publication.publish(out)
    assert digest(out / "continuation.json") == digest_before
    assert (
        r3.verify(root)["arms"]["QC-R3"]["last_committed_step"]
        == before["arms"]["QC-R3"]["last_committed_step"]
    )
    value = publication.summary(out)
    assert "model" not in json.dumps(value) and "labels" not in json.dumps(value)
    assert value["KAGGLE_CROSS_SESSION_RESUME"] == "NOT_VERIFIED"


def test_last_bundle_survives_new_selector_and_failed_manifest(tmp_path):
    root = initial(tmp_path)
    r3.segment_arm(root, "QC-R3", 3)
    old = r3.package(root)
    r3.segment_arm(root, "QC-R3", 6)

    def fail(stage):
        raise RuntimeError("manifest interrupted")

    with pytest.raises(RuntimeError, match="manifest interrupted"):
        r3.package(root, crash=fail)
    # The older pointer selects archived latest metadata and immutable weights,
    # even though uncommitted new step files are present alongside it.
    assert (
        r3.verify(root)["arms"]["QC-R3"]["last_committed_step"]
        == old["arms"]["QC-R3"]["last_committed_step"]
    )
    out = tmp_path / "next"
    r3.resume(root, out, 6, source=r3.source_sha())
    assert r3.verify(out)["arms"]["QC-R3"]["last_committed_step"] == 6


@pytest.mark.parametrize("point", ["sidecar", "latest"])
def test_checkpoint_write_fault_preserves_previous_commit(tmp_path, monkeypatch, point):
    root = initial(tmp_path)
    r3.segment_arm(root, "QC-R3", 2)
    old = r3.package(root)
    original = r3.cp.write_json

    def fault(path, value):
        name = Path(path).name
        if (point == "sidecar" and name == "checkpoint-step-000003.sha256.json") or (
            point == "latest" and name == "latest.json"
        ):
            raise RuntimeError("write fault")
        return original(path, value)

    monkeypatch.setattr(r3.cp, "write_json", fault)
    with pytest.raises(RuntimeError, match="write fault"):
        r3.segment_arm(root, "QC-R3", 3)
    assert r3.verify(root)["arms"]["QC-R3"]["last_committed_step"] == 2
    monkeypatch.setattr(r3.cp, "write_json", original)
    r3.segment_arm(root, "QC-R3", 3)
    assert r3.package(root)["arms"]["QC-R3"]["last_committed_step"] == 3
    assert old["status"] == "PAUSED"


def test_diagnostic_plan_and_formal_training_closed(tmp_path):
    c = configuration()
    prepare(tmp_path / "assets", c)
    c["evaluation"]["periodic_diagnostics_enabled"] = False
    with pytest.raises(ContinuationError, match="DIAGNOSTIC_PLAN_NOT_FROZEN"):
        r3.new_run(
            c, tmp_path / "assets", tmp_path / "out", source=r3.source_sha(), run_id="invalid"
        )


def test_legacy_import_keeps_original_identity_and_is_evaluation_only(tmp_path):
    import numpy as np
    from clm001_r3.automatic import runtime
    from clm001_r3.data import load_assets
    from clm001_r3.report import seal
    from minicells_runtime.store import atomic

    c = configuration()
    assets = prepare(tmp_path / "legacy-assets", c)
    data = load_assets(c, assets)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    summary = {"gates": {"gradient_pilot_pass": True, "local_alignment_pass": True}}
    active = runtime(c, data, 0.1, r3.canonical_sha256(summary))
    original_source = "b" * 40
    active["execution"]["source_sha"] = original_source
    atomic(legacy / "runtime-config.json", active)
    atomic(tmp_path / "candidate.json", c)
    seal(legacy / "gpu-pilot-summary.json", summary)
    arm = c["arms"][0]
    folder = legacy / "arms" / arm
    folder.mkdir(parents=True)
    trainer = r3.Trainer(r3.Model(active, arm))
    identity = r3.cp.identity(active, arm, original_source, data)
    atomic(folder / "identity.json", identity)
    r3.bank(trainer.model, data, active, folder, identity)
    r3.cp.save(folder, trainer, r3.Nursery(), np.random.default_rng(1), identity)
    adopted = tmp_path / "adopted"
    meta = r3.import_legacy(
        legacy,
        assets,
        adopted,
        expected_source=original_source,
        candidate_config=tmp_path / "candidate.json",
    )
    assert read(adopted / "arms" / arm / "identity.json") == identity
    assert meta["source_sha"] == original_source
    result = r3.evaluate_checkpoint(
        adopted, tmp_path / "retrospective", arm, 0, {"policies": ["greedy"]}
    )
    assert result["retrospective"] and result["source_identity"] == identity
    with pytest.raises(ContinuationError, match="SOURCE_SHA_MISMATCH"):
        r3.resume(adopted, tmp_path / "wrong-source", 6, source=r3.source_sha())
