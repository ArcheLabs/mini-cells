from copy import deepcopy
import random
import numpy as np
import pytest
import torch
from minicells_runtime import r3 as reused
from minicells_runtime.store import read, digest, ContinuationError
from clm001_r3b import runner
from clm001_r3b.common import ARMS, ENGINE, identity
from clm001_r3.model import Model
from clm001_r3.training import Trainer
from clm001_r3.data import load_assets, batches
from clm001_r3.shadows import Nursery
from clm001_r3 import checkpoint as cp


def test_original_training_unchanged_with_observation(case, tmp_path):
    proposed, c, root = case
    runner.segment(root, "QC-R3b", 2)
    actual = read(root / "arms/QC-R3b/latest.json")
    model_state = torch.load(
        root / "arms/QC-R3b" / actual["file"], map_location="cpu", weights_only=False
    )
    # Same R3 engine/config/arm and all original streams. No observer involved.
    random.seed(c["seed"])
    np.random.seed(c["seed"])
    torch.manual_seed(c["seed"])
    original_config = deepcopy(proposed["engine"])
    trainer = Trainer(Model(original_config, "QC-R3"))
    rng = np.random.default_rng(c["seed"] + 10)
    data = load_assets(original_config, root / "assets")
    for _ in range(2):
        trainer.update(batches(data, original_config, rng, "cpu"))
    folder = tmp_path / "control"
    cp.save(folder, trainer, Nursery(), rng, identity(c, "QC-R3b", reused.source_sha(), data))
    expected = folder / read(folder / "latest.json")["file"]
    assert reused.state_fingerprint(expected) == reused.state_fingerprint(
        root / "arms/QC-R3b" / actual["file"]
    )
    assert all(
        torch.equal(v, trainer.model.state_dict()[k]) for k, v in model_state["model"].items()
    )
    assert actual["identity"]["experiment_id"] == "CLM-001-R3b"
    assert actual["identity"]["implementation_arm"] == "QC-R3"


def test_observations_enabled_disabled_same_state(case, tmp_path, monkeypatch):
    proposed, c, root = case
    runner.segment(root, "Q-R3b", 10)
    from clm001_r3b.runner import new_run

    # Same immutable scientific/config identity; disable only the observation
    # callback, including the admission boundary and final high-cost snapshot.
    monkeypatch.setattr(runner, "snapshot", lambda *args, **kwargs: None)
    other = tmp_path / "off"
    new_run(proposed, c, root / "assets", other)
    runner.segment(other, "Q-R3b", 10)
    left = root / "arms/Q-R3b" / read(root / "arms/Q-R3b/latest.json")["file"]
    right = other / "arms/Q-R3b" / read(other / "arms/Q-R3b/latest.json")["file"]
    assert reused.state_fingerprint(left) == reused.state_fingerprint(right)


def test_checkpoint_alias_and_byte_verified_bundle(case):
    _, _, root = case
    for arm in ARMS:
        runner.segment(root, arm, 1)
    meta = runner.package(root)
    assert set(meta["arms"]) == set(ARMS)
    assert runner.verify(root)["experiment_id"] == "CLM-001-R3b"
    for arm in ARMS:
        assert read(root / "arms" / arm / "identity.json")["implementation_arm"] == ENGINE[arm]


def test_handoff_tamper_and_partial_bundle(case, tmp_path):
    from clm001_r3b.coordinator import merge_handoff

    _, c, live = case
    runner.segment(live, "Q-R3b", 1)
    snapshot = runner.handoff(live / "arms/Q-R3b", tmp_path / "handoffs", phase="post_event")
    master = tmp_path / "master"
    runner.new_run(read(live / "candidate-config.json"), c, live / "assets", master)
    merge_handoff(master, "Q-R3b", snapshot)
    meta = runner.package(master)
    assert not meta["complete"] and set(meta["arms"]) == {"Q-R3b"}
    old = digest(master / "continuation.json")
    (snapshot / "latest.json").write_text("{}")
    with pytest.raises(ContinuationError, match="HANDOFF_HASH"):
        merge_handoff(master, "Q-R3b", snapshot)
    assert digest(master / "continuation.json") == old


def test_event_resume_no_repeat_and_distinct_revisions(case, tmp_path):
    _, c, root = case

    paths = []

    def boundary(folder, phase):
        frozen = runner.handoff(folder, tmp_path / "handoff", phase=phase)
        paths.append((phase, frozen))
        if phase == "event-pre-admission":
            raise RuntimeError("simulated remote outage")

    with pytest.raises(RuntimeError):
        runner.segment(root, "QC-R3b", 8, committed=boundary)
    folder = root / "arms/QC-R3b"
    interrupted = read(folder / "latest.json")
    # Only one pre-admission event persisted, remaining events resume in order.
    trainer, nursery, _, _ = runner.load_state(
        folder, c, "QC-R3b", load_assets(c, root / "assets"), reused.source_sha()
    )
    previous_keys = set(nursery.events)
    runner.segment(root, "QC-R3b", 8)
    _, resumed, _, _ = runner.load_state(
        folder, c, "QC-R3b", load_assets(c, root / "assets"), reused.source_sha()
    )
    assert previous_keys <= set(resumed.events)
    assert sum(v["event"] == "pre-admission" for v in resumed.events.values()) == 1
    assert len(resumed.events) == 8
    assert read(folder / "latest.json")["file"] != interrupted["file"]
    assert len(list(folder.glob("checkpoint-step-000008*.pt"))) >= 6
    assert list((folder / "diagnostics").glob("snapshot-000008-pre_event*.json"))
    assert list((folder / "diagnostics").glob("snapshot-000008-post_event*.json"))


def test_storage_stop_between_boundaries_preserves_last_finished_local_update(case, tmp_path):
    proposed, c, root = case
    c = deepcopy(c)
    c["training"]["checkpoint_interval"] = 50
    from minicells_runtime.store import canonical

    c["execution"]["configuration_sha256"] = canonical(
        {k: v for k, v in c.items() if k != "execution"}
    )
    other = tmp_path / "between"
    runner.new_run(proposed, c, root / "assets", other)
    checks = [0]

    def stopped():
        checks[0] += 1
        return checks[0] >= 3

    result = runner.segment(other, "Q-R3b", 10, stop_requested=stopped)
    assert result["step"] == 1
    assert read(other / "arms/Q-R3b/latest.json")["file"] == "checkpoint-step-000001.pt"
    assert not (other / "continuation.json").exists(), "local stop must not claim remote durability"
