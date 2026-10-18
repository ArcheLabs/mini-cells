from copy import deepcopy
from pathlib import Path
import pytest
from minicells_runtime.store import read, ContinuationError
from minicells_artifacts.hf_storage import HFCheckpointStore
from test_hf_storage import FakeHub
from clm001_r3b import runner, analysis
from clm001_r3b.publication import publish_documents


def test_snapshot_hash_matches_remote_index_and_wrong_arm_rejected(case, tmp_path):
    from clm001_r3.report import seal

    _, _, root = case
    for arm in ("Q-R3b", "QC-R3b"):
        runner.segment(root, arm, 1)
    meta = runner.package(root)
    values = analysis.aggregate(root)
    assert values["experiment-manifest.json"]["diagnostic_status"] == "INCOMPLETE"
    # Hash sidecars share the JSON suffix; filesystem iteration order varies.
    file = next(
        path
        for path in sorted((root / "arms/QC-R3b/diagnostics").glob("snapshot-*.json"))
        if not path.name.endswith(".sha256.json")
    )
    value = read(file)
    assert value["runtime_config_sha256"] == meta["runtime_config_sha256"]
    value["identity"]["arm"] = "Q-R3b"
    seal(file, value)
    with pytest.raises(ContinuationError, match="IDENTITY_MISMATCH"):
        analysis.aggregate(root)


def test_checkpoint_index_transport_cost_and_event_cursor_rollback(case, tmp_path):
    from clm001_r3b.coordinator import Publisher

    _, _, root = case
    runner.segment(root, "QC-R3b", 3)
    hub = FakeHub(tmp_path / "hub")
    store = HFCheckpointStore(hub)
    first = Publisher(root, store).publish(readback=True)
    transport = first["transport"]
    assert transport["payload_bytes_submitted"] > 0
    assert transport["new_content_object_bytes"] > 0
    assert transport["readback_seconds"] >= 0
    assert transport["wire_bytes"] is None
    # A newer step carrying a removed prior Event Cursor is rejected even if the
    # checkpoint ranks advance. This simulates a corrupt independent publisher.
    original = store.backend.optional

    def changed(path, revision):
        result = original(path, revision)
        if path.endswith("/latest.json") and result and "arms" in result:
            result = deepcopy(result)
            result["arms"]["QC-R3b"]["event_cursor"]["missing-future-event"] = {
                "step": 3,
                "event": "propose-cell",
            }
        return result

    store.backend.optional = changed
    with pytest.raises(ContinuationError, match="EVENT_CURSOR_ROLLBACK"):
        store.publish_committed_checkpoint(root)


def test_publication_rejects_non_metadata_and_sanitizes_push_failure(case, tmp_path, monkeypatch):
    from clm001_r3b import publication

    _, _, root = case
    fake = tmp_path / "checkout"
    fake.mkdir()
    monkeypatch.setattr(publication, "preflight", lambda *args: (fake, "results/test"))
    from types import SimpleNamespace

    staged = []

    def git(folder, *args):
        if args[0] == "add":
            staged[:] = args[2:]
        return SimpleNamespace(stdout="\n".join(staged) if args[0] == "diff" else "")

    monkeypatch.setattr(publication.existing, "git", git)

    class Helper:
        ROOT = None

        def push_branch(self, *args):
            raise RuntimeError("do not publish secret token value")

    monkeypatch.setattr(publication.existing, "helper", lambda: Helper())
    with pytest.raises(ContinuationError, match="NOT_WHITELISTED"):
        publish_documents(root, {"model.pt": {}})
    with pytest.raises(ContinuationError, match="PUBLICATION_FAILED"):
        publish_documents(root, {"experiment-manifest.json": {"scientific_decision": False}})
    failure = read(tmp_path / "run-evidence/publication.json")
    assert "secret" not in str(failure)
    assert failure["retrain"] is False
    assert all(Path(name).suffix == ".json" for name in staged)


def test_retrospective_publication_has_separate_namespace(case, tmp_path, monkeypatch):
    from clm001_r3b import publication
    from types import SimpleNamespace

    _, _, root = case
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(
        publication,
        "preflight",
        lambda parent, source, run, scope=None: (checkout, "results/" + str(scope)),
    )
    staged = []

    def git(folder, *args):
        if args[0] == "add":
            staged[:] = args[2:]
        return SimpleNamespace(
            stdout="\n".join(staged)
            if args[0] == "diff"
            else "a" * 40
            if args[0] in ("rev-parse", "ls-remote")
            else ""
        )

    monkeypatch.setattr(publication.existing, "git", git)

    class Helper:
        ROOT = None

        def push_branch(self, *args):
            pass

    monkeypatch.setattr(publication.existing, "helper", lambda: Helper())
    publication.publish_documents(root, {"metrics-summary.json": {"scope": "training"}})
    train_file = next(checkout.rglob("metrics-summary.json"))
    publication.publish_documents(
        root, {"metrics-summary.json": {"scope": "retrospective"}}, scope="evaluation-v1-cp-hash"
    )
    assert read(train_file)["scope"] == "training"
    assert len(list(checkout.rglob("metrics-summary.json"))) == 2
    with pytest.raises(ContinuationError, match="PRIVATE_PUBLICATION"):
        publication.publish_documents(root, {"metrics-summary.json": {"raw_oracle": [1, 2]}})


def test_host_package_does_not_restore_device_rng(case, monkeypatch):
    _, _, root = case
    runner.segment(root, "Q-R3b", 1)

    def forbidden(*args, **kwargs):
        raise RuntimeError("device RNG restore must not run in publisher")

    monkeypatch.setattr(runner.cp, "restore", forbidden)
    meta = runner.package(root)
    assert meta["arms"]["Q-R3b"]["step"] == 1
    assert runner.verify(root)["arms"]["Q-R3b"]["step"] == 1
