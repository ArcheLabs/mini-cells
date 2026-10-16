"""Offline HF protocol tests. No credentials, network or GPU."""

from types import SimpleNamespace
import json
import shutil
import pytest
from minicells_artifacts.hf_storage import HFCheckpointStore
from minicells_runtime import r3
from minicells_runtime.store import ContinuationError, digest, read
from test_segmented_r3 import initial


class Conflict(RuntimeError):
    response = SimpleNamespace(status_code=409)


class FakeHub:
    repo = "test/minicells-research-checkpoints"

    def __init__(self, root):
        self.root = root
        root.mkdir()
        self.revisions = {}
        self.head = "0" * 40
        self.revisions[self.head] = {}
        self.fail = False
        self.conflict_once = False
        self.private = True
        self.writes = 0

    def info(self, revision=None):
        files = self.revisions[revision or self.head]
        return SimpleNamespace(
            sha=self.head if revision is None else revision,
            private=self.private,
            siblings=[SimpleNamespace(rfilename=p) for p in files],
        )

    def get(self, path, revision):
        return self.revisions[revision][path]

    def optional(self, path, revision):
        value = self.revisions[revision].get(path)
        return read(value) if value else None

    def check_write(self):
        if self.fail:
            raise PermissionError("secret must never appear")

    def commit(self, files, parent, message):
        if self.fail:
            raise PermissionError("secret must never appear")
        if self.conflict_once:
            self.conflict_once = False
            raise Conflict()
        if parent != self.head:
            raise Conflict()
        self.writes += 1
        revision = "%040x" % self.writes
        folder = self.root / revision
        folder.mkdir()
        new = dict(self.revisions[self.head])
        for i, (key, source) in enumerate(files.items()):
            target = folder / str(i)
            shutil.copyfile(source, target)
            new[key] = target
        self.head = revision
        self.revisions[revision] = new
        return revision


@pytest.fixture
def bundle(tmp_path):
    root = initial(tmp_path)
    r3.segments(root, 6)
    r3.package(root)
    return root


def test_hf_roundtrip_exact_resume_history_and_fork(bundle, tmp_path, monkeypatch):
    import torch

    backend = FakeHub(tmp_path / "hub")
    store = HFCheckpointStore(backend)
    assert store.preflight()["status"] == "PREFLIGHT_PASSED"
    index = store.publish_committed_checkpoint(bundle, readback=True)
    assert index["status"] == "DURABLE_PARTIAL"
    assert {r["event_revision"] for r in index["checkpoints"]} >= {0, 1}
    ref = index["checkpoint_ref"]
    # New store/client, no dependency on the original local package.
    fetched = tmp_path / "fetched"
    HFCheckpointStore(backend).fetch_verified_checkpoint(ref, fetched, r3.source_sha())
    assert digest(fetched / "continuation.json") == digest(bundle / "continuation.json")
    resumed = tmp_path / "resumed"
    r3.resume(fetched, resumed, 10, source=r3.source_sha())
    control = initial(tmp_path, "control")
    r3.segments(control, 10)
    c = r3.package(control)
    b = r3.verify(resumed)
    for arm in c["arms"]:
        assert c["arms"][arm]["training_state_sha256"] == b["arms"][arm]["training_state_sha256"]
        assert c["arms"][arm]["event_cursor"] == b["arms"][arm]["event_cursor"]
    final = store.publish_committed_checkpoint(resumed, readback=True)
    assert final["status"] == "DURABLE_COMPLETE"
    before = digest(fetched / "continuation.json")

    def forbidden(*args, **kwargs):
        raise AssertionError("optimizer called during evaluate")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    r3.evaluate_checkpoint(fetched, tmp_path / "eval", "QC-R3", 6, {"policies": ["greedy"]})
    assert digest(fetched / "continuation.json") == before
    assert store.lookup(ref["run_key"])["hf_revision"] == final["checkpoint_ref"]["hf_revision"]
    # Older pinned snapshots still exist after latest advances.
    store.fetch_verified_checkpoint(ref, tmp_path / "old-again")
    with pytest.raises(ContinuationError, match="ROLLBACK"):
        store.publish_committed_checkpoint(bundle)


def test_idempotence_conflict_permission_and_tamper(bundle, tmp_path):
    hub = FakeHub(tmp_path / "hub")
    store = HFCheckpointStore(hub)
    hub.conflict_once = True
    first = store.publish_committed_checkpoint(bundle, readback=True)
    count = hub.writes
    assert store.publish_committed_checkpoint(bundle)["checkpoint_ref"] == first["checkpoint_ref"]
    assert hub.writes == count
    hub.fail = True
    with pytest.raises(ContinuationError, match="STORAGE_BLOCKED") as exc:
        store.preflight()
    assert "secret" not in str(exc.value)
    with pytest.raises(ContinuationError, match="STORAGE_BLOCKED"):
        fresh = initial(tmp_path, "new", run_id="new")
        r3.segments(fresh, 1)
        r3.package(fresh)
        store.publish_committed_checkpoint(fresh)
    hub.fail = False
    ref = first["checkpoint_ref"]
    row = first["checkpoints"][0]
    hub.revisions[ref["hf_revision"]][row["hf_path"]].write_bytes(b"corrupt")
    with pytest.raises(ContinuationError, match="HASH_MISMATCH"):
        store.fetch_verified_checkpoint(ref, tmp_path / "corrupt")
    assert not (tmp_path / "corrupt").exists()


def test_private_pinned_identity_and_safe_paths(bundle, tmp_path):
    hub = FakeHub(tmp_path / "hub")
    store = HFCheckpointStore(hub)
    hub.private = False
    with pytest.raises(ContinuationError, match="PRIVATE"):
        store.preflight()
    hub.private = True
    ref = store.publish_committed_checkpoint(bundle)["checkpoint_ref"]
    for updates, message in [
        ({"hf_revision": "main"}, "PINNED"),
        ({"hf_repo": "evil/repo"}, "REPO_IDENTITY"),
        ({"bundle_path": "../bad"}, "UNSAFE"),
    ]:
        with pytest.raises(ContinuationError, match=message):
            store.fetch_verified_checkpoint(dict(ref, **updates), tmp_path / "bad")
    with pytest.raises(ContinuationError, match="SOURCE_SHA"):
        store.fetch_verified_checkpoint(ref, tmp_path / "bad", "e" * 40)
    assert "HF_TOKEN" not in json.dumps(store.publish_committed_checkpoint(bundle))


def test_safe_boundary_upload_and_block_before_next_update(tmp_path):
    from minicells_runtime.hf_run import persist

    root = initial(tmp_path)
    hub = FakeHub(tmp_path / "hub")
    store = HFCheckpointStore(hub)
    r3.segment_arm(root, "QC-R3", 6, committed=lambda p: persist(store, p))
    meta = r3.verify(root)
    assert meta["arms"]["QC-R3"]["last_committed_step"] == 6
    durable = read(tmp_path / "run-storage/checkpoint-index.json")
    assert {x["event_revision"] for x in durable["checkpoints"]} >= {0, 1}
    # Next boundary fails closed; local checkpoint remains verifiable.
    hub.fail = True
    with pytest.raises(ContinuationError, match="STORAGE_BLOCKED"):
        r3.segment_arm(root, "QC-R3", 10, committed=lambda p: persist(store, p))
    assert r3.verify(root)["arms"]["QC-R3"]["last_committed_step"] == 6
    assert read(tmp_path / "run-storage/storage-status.json")["status"] == "STORAGE_BLOCKED"
    assert read(tmp_path / "run-storage/checkpoint-index.json") == durable


def test_github_schema_excludes_private_nested_state(bundle, tmp_path):
    from minicells_runtime.publication import storage_summary

    hub = FakeHub(tmp_path / "hub")
    index = HFCheckpointStore(hub).publish_committed_checkpoint(bundle)
    index["raw_oracle"] = "PRIVATE"
    index["checkpoint_ref"]["arms"] = {"PRIVATE": "PRIVATE"}
    index["checkpoints"][0]["rng"] = "PRIVATE"
    assert "PRIVATE" not in json.dumps(storage_summary("checkpoint-index.json", index))
