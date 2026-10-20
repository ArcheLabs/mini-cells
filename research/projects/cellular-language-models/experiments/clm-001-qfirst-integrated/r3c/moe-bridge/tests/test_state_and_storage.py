from pathlib import Path
from types import SimpleNamespace
import shutil
import numpy as np
import pytest
import torch
from minicells_r3c_moe_bridge.model import BridgeModel
from minicells_r3c_moe_bridge.training import Trainer
from minicells_r3c_moe_bridge.data import synthetic_batches
from minicells_r3c_moe_bridge.common import fingerprint, canonical
from minicells_r3c_moe_bridge.checkpoint import Store, publish_segment, restore_segment


class FakeHub:
    repo = "private/test"

    def __init__(self, root):
        self.root = Path(root)
        self.commits = 0
        self.requests = []

    def info(self, revision=None):
        return SimpleNamespace(private=True, sha="revision")

    def check_write(self):
        pass

    def commit(self, files, parent, message):
        self.commits += 1
        for name, path in files.items():
            dst = self.root / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dst)
        return "pinned-revision"

    def get(self, name, revision):
        self.requests.append((name, revision))
        return self.root / name


@pytest.mark.parametrize("arm", list("ABCDE"))
def test_resume_complete_joint_state(config, arm, tmp_path):
    tr = Trainer(BridgeModel(config, arm, 123))
    identity = {"arm": arm, "config": canonical(config)}
    store = Store(tmp_path, identity)
    tr.update(synthetic_batches(config, tr.data_rng))
    row = store.save(tr)
    assert store.save(tr) == row
    batches = synthetic_batches(config, tr.data_rng)
    log = tr.update(batches)
    expected = fingerprint(tr.state())
    new = Trainer(BridgeModel(config, arm, 123))
    store.restore(new, row)
    actual = new.update(synthetic_batches(config, new.data_rng))
    assert fingerprint(new.state()) == expected and actual["ce"] == log["ce"]
    assert actual["stage_cell_counts"] == log["stage_cell_counts"]


def test_corruption_and_identity_precede_deserialization(config, tmp_path, monkeypatch):
    tr = Trainer(BridgeModel(config, "C", 123))
    store = Store(tmp_path, {"arm": "C"})
    row = store.save(tr)

    def forbidden(*a, **k):
        raise AssertionError("pickle reached")

    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(RuntimeError, match="IDENTITY"):
        Store(tmp_path, {"arm": "D"}).verified(row)
    (tmp_path / row["file"]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="HASH"):
        store.verified(row)


def test_private_segment_independent_reader(config, tmp_path):
    ident = {"arm": "C", "config": canonical(config)}
    tr = Trainer(BridgeModel(config, "C", 123))
    store = Store(tmp_path / "session-a", ident)
    row = store.save(tr)
    hub = FakeHub(tmp_path / "hub")
    ref = publish_segment(hub, store.root, ident, "session-A")
    assert hub.commits == 1 and not ref["cross_session_verified"]
    with pytest.raises(RuntimeError, match="INDEPENDENT_SESSION"):
        restore_segment(hub, ref, tmp_path / "bad", ident, "session-A")
    reader = FakeHub(tmp_path / "hub")
    proof = restore_segment(reader, ref, tmp_path / "session-b", ident, "session-B")
    assert proof["cross_session_verified"] and all(
        rev == "pinned-revision" for _, rev in reader.requests
    )
    assert fingerprint(Store(tmp_path / "session-b", ident).verified(row)) == fingerprint(
        store.verified(row)
    )


def test_independent_route_rng(config):
    torch.manual_seed(19)
    np.random.seed(11)
    before = torch.get_rng_state().clone()
    m = BridgeModel(config, "B", 123)
    assert torch.equal(before, torch.get_rng_state())
    a, b = Trainer(m), Trainer(BridgeModel(config, "A", 123))
    for _ in range(3):
        x = synthetic_batches(config, a.data_rng)
        y = synthetic_batches(config, b.data_rng)
        assert all(torch.equal(u, v) for pair, q in zip(x, y) for u, v in zip(pair, q))
        m(x[0][0], route_rng=a.route_rng)
    assert torch.equal(before, torch.get_rng_state())
