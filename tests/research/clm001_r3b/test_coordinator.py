import shutil
import pytest
from minicells_runtime.store import read, ContinuationError
from minicells_artifacts.hf_storage import HFCheckpointStore
from test_hf_storage import FakeHub
from clm001_r3b import runner, coordinator


def test_two_independent_cpu_queues_one_hf_publisher(case, tmp_path):
    _, _, root = case
    live = tmp_path / "live"
    shutil.copytree(root, live)
    hub = FakeHub(tmp_path / "hub")
    index = coordinator.run(root, live, 1, HFCheckpointStore(hub), cpu=True)
    meta = runner.verify(root)
    assert set(meta["arms"]) == {"Q-R3b", "QC-R3b", "QS-R3b"}
    assert all(p["step"] == 1 for p in meta["arms"].values())
    assert index["status"] == "DURABLE_PARTIAL"
    restored = tmp_path / "restored"
    HFCheckpointStore(hub).fetch_verified_checkpoint(index["checkpoint_ref"], restored)
    assert runner.verify(restored)["run_id"] == meta["run_id"]
    ledger = read(tmp_path / "run-evidence/resource-ledger.json")
    assert ledger["uploads"][-1]["full_readback"]
    assert all("transport" in row for row in ledger["uploads"])


def test_outage_releases_workers_without_killing_or_next_updates(case, tmp_path):
    _, _, root = case
    live = tmp_path / "live"
    shutil.copytree(root, live)
    hub = FakeHub(tmp_path / "hub")
    hub.fail = True
    with pytest.raises(ContinuationError, match="STORAGE_BLOCKED"):
        coordinator.run(root, live, 10, HFCheckpointStore(hub), cpu=True)
    status = read(tmp_path / "run-evidence/storage-status.json")
    assert status["status"] == "STORAGE_BLOCKED"
    assert not (tmp_path / "run-evidence/checkpoint-index.json").exists()
    assert all(p["step"] == 0 for p in read(root / "continuation.json")["arms"].values())


def test_stale_handoff_cannot_rewind_progress(case, tmp_path):
    proposed, c, live = case
    root = tmp_path / "master"
    runner.new_run(proposed, c, live / "assets", root)
    runner.segment(live, "Q-R3b", 1)
    old = runner.handoff(live / "arms/Q-R3b", tmp_path / "handoffs", phase="post_event")
    coordinator.merge_handoff(root, "Q-R3b", old)
    runner.segment(live, "Q-R3b", 2)
    newer = runner.handoff(live / "arms/Q-R3b", tmp_path / "handoffs", phase="post_event")
    coordinator.merge_handoff(root, "Q-R3b", newer)
    with pytest.raises(ContinuationError, match="ROLLBACK"):
        coordinator.merge_handoff(root, "Q-R3b", old)
