import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "shared/python"))
from minicells_runtime.store import (  # noqa: E402
    LocalOutputStore,
    MountedInputStore,
    ContinuationError,
    atomic,
    digest,
    safe,
    capacity,
)


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    identity = {"experiment_id": "ENGINEERING", "source_commit": "a" * 40}
    for name, value in {
        "candidate-config.json": {},
        "runtime-config.json": {"execution": {"source_sha": "a" * 40}},
        "assets/manifest.json": {},
        "assets/tokenizer.json": {},
        "arms/A/identity.json": identity,
        "arms/A/frozen-probe-bank.json": {},
    }.items():
        atomic(root / name, value)
    (root / "protocol.md").write_text("engineering only")
    arm = root / "arms/A"
    (arm / "frozen-probe-bank.pt").write_bytes(b"trusted probe")
    atomic(
        arm / "frozen-probe-bank.json",
        {"identity": identity, "sha256": digest(arm / "frozen-probe-bank.pt")},
    )
    checkpoints = []
    for step, revision in [(6, 0), (6, 1), (10, 0)]:
        name = "checkpoint-step-%06d-event-%06d.pt" % (step, revision)
        (arm / name).write_bytes((str(step) + str(revision)).encode())
        atomic(
            (arm / name).with_suffix(".sha256.json"),
            {"identity": identity, "sha256": digest(arm / name)},
        )
        checkpoints.append(
            {
                "step": step,
                "revision": revision,
                "checkpoint": "arms/A/" + name,
                "checkpoint_sha256": digest(arm / name),
                "sidecar": "arms/A/" + Path(name).with_suffix(".sha256.json").name,
            }
        )
    latest = checkpoints[-1]
    atomic(
        arm / "latest.json",
        {
            "file": Path(latest["checkpoint"]).name,
            "sha256": latest["checkpoint_sha256"],
            "identity": identity,
        },
    )
    meta = {
        "experiment_id": "ENGINEERING",
        "run_id": "run-a",
        "source_sha": "a" * 40,
        "status": "PAUSED",
        "complete": False,
        "arms": {
            "A": dict(
                latest,
                last_committed_step=10,
                checkpoints=checkpoints,
                probe_bank_sha256=digest(arm / "frozen-probe-bank.pt"),
            )
        },
    }
    for file, key in [
        ("candidate-config.json", "candidate_config_sha256"),
        ("runtime-config.json", "runtime_config_sha256"),
        ("protocol.md", "protocol_sha256"),
        ("assets/manifest.json", "assets_manifest_sha256"),
        ("assets/tokenizer.json", "tokenizer_sha256"),
    ]:
        meta[key] = digest(root / file)
    atomic(
        root / "experiment.lock.json",
        {k: v for k, v in meta.items() if k not in ("arms", "status", "complete")},
    )
    LocalOutputStore(root).commit(meta)
    return root, meta


def test_verify_history_and_readonly_copy(bundle, tmp_path):
    root, meta = bundle
    before = (root / "continuation.json").read_bytes()
    store = MountedInputStore(root)
    assert store.select("A", 6)["revision"] == 1
    assert store.select("A", 6, 0)["revision"] == 0
    store.materialize(tmp_path / "next", "a" * 40)
    assert store.verify()["run_id"] == meta["run_id"]
    assert (root / "continuation.json").read_bytes() == before
    with pytest.raises(ContinuationError, match="OUTPUT_ALREADY_EXISTS"):
        store.materialize(tmp_path / "next")


@pytest.mark.parametrize(
    "name",
    [
        "candidate-config.json",
        "runtime-config.json",
        "assets/tokenizer.json",
        "assets/manifest.json",
        "arms/A/frozen-probe-bank.pt",
        "arms/A/checkpoint-step-000006-event-000000.pt",
    ],
)
def test_corruption_rejected(bundle, name):
    root, _ = bundle
    (root / name).write_bytes(b"corrupted")
    with pytest.raises(ContinuationError, match="CHECKPOINT_HASH_MISMATCH"):
        MountedInputStore(root).verify()


def test_missing_source_and_paths(bundle, tmp_path):
    root, _ = bundle
    with pytest.raises(ContinuationError, match="SOURCE_SHA_MISMATCH"):
        MountedInputStore(root).verify("b" * 40)
    for name in ("../bad", "/etc/passwd", "arms/../bad", "a\\b"):
        with pytest.raises(ContinuationError, match="UNSAFE_PATH"):
            safe(root, name)
    (root / "link").symlink_to(root / "protocol.md")
    with pytest.raises(ContinuationError, match="UNSAFE_PATH"):
        safe(root, "link")
    with pytest.raises(ContinuationError, match="RESUME_INPUT_MISSING"):
        MountedInputStore(tmp_path / "missing")


def test_atomic_pointer_survives_interruption(bundle):
    root, meta = bundle
    before = (root / "continuation.json").read_bytes()

    def crash(stage):
        raise RuntimeError("injected " + stage)

    with pytest.raises(RuntimeError, match="injected"):
        LocalOutputStore(root).commit(meta, crash=crash)
    assert (root / "continuation.json").read_bytes() == before
    MountedInputStore(root).verify()


def test_capacity_has_breakdown():
    with pytest.raises(ContinuationError, match="OUTPUT_SIZE_BUDGET_EXCEEDED.*largest"):
        capacity({"big.pt": {"bytes": 100}}, 99)
