import shutil
import pytest
from minicells_runtime.store import ContinuationError
from r3b_cf001.storage import Chunks
from r3b_cf001.sampling import sample
from r3b_cf001.statistics import paired, holm, story_summary
from r3b_cf001.reporting import cf1_summary
from r3b_cf001.common import configuration
from r3b_cf001.driver import choose_budget


def test_sampling_inclusion_weights_hash_order_and_strata():
    rows = [
        dict(
            story_id=str(i % 4),
            window_id=str(i % 4),
            token_position=i,
            depth_zero_based=1 + i % 3,
            state_fingerprint=str(i),
            previous_action=0,
            ctx_action=6,
            n_ctx_so_far=0,
            forced_end_path=True,
            stop_legal=True,
            actual_stop=False,
        )
        for i in range(60)
    ]
    a, meta = sample(rows, 12, 19)
    b, _ = sample(list(reversed(rows)), 12, 19)
    assert a == b and len(a) == 12 and len(meta["strata"]) == 3
    assert sum(r["design_weight"] for r in a) == 60
    assert all(r["inclusion_probability"] == 0.2 for r in a)
    with pytest.raises(ContinuationError):
        sample(rows, 1, 19)


def test_story_pairing_itt_and_conditional_separate():
    base = {"a": {"nll": [1, 1], "cost": [1, 1]}, "b": {"nll": [2, 2], "cost": [1, 1]}}
    changed = {
        "a": {"nll": [0, 1], "cost": [0, 1], "touched": [True, False]},
        "b": {"nll": [2, 2], "cost": [1, 1], "touched": [False, False]},
    }
    report = paired(base, changed, 0.01, replicates=100)
    assert report["delta_nll"]["mean"] == -0.25
    assert report["triggered_subset"]["delta_nll"]["mean"] == -1
    assert report["delta_nll"]["stories"] == 2
    with pytest.raises(ContinuationError):
        paired(base, {"a": changed["a"]}, 0.01)
    changed["a"]["nll"][0] = float("nan")
    with pytest.raises(ContinuationError):
        paired(base, changed, 0.01)
    assert story_summary([1, 3, 9], [0, 0, 1]) == {"0": 2.0, "1": 9.0}


def test_holm_declared_family_and_unresolved_never_correct():
    h = holm({"a": 0.01, "b": 0.04, "c": 0.2})
    assert h["a"]["adjusted_exploratory_p"] == 0.03
    c = configuration()
    c["bootstrap_replicates"] = 100
    row = dict(
        story_id="s",
        depth=1,
        previous_action=0,
        n_ctx_so_far=0,
        forced_end_path=True,
        design_weight=1,
        pred_stop_advantage=-1,
        greedy_consistent={"misranking": True, "true_stop_advantage": 1},
        frozen_sampled_oracle={
            "misranking": False,
            "true_stop_advantage": 1,
            "status": "UNRESOLVED",
        },
    )
    report = cf1_summary([row], c, ["s"])["overall"]["frozen_sampled_oracle"]
    assert report["direction_agreement_resolved"]["estimate"] is None
    assert report["unresolved"]["estimate"] == 1


def test_chunk_atomic_idempotent_tamper_and_orphan(tmp_path):
    root = tmp_path / "chunks"
    store = Chunks(root, {"run": "x"})
    store.put(["a"], {"value": 1})
    store.put(["a"], {"value": 1})
    assert len(store.index) == 1
    with pytest.raises(ContinuationError):
        store.put(["a"], {"value": 2})
    for file in (root / "receipts").glob("*.json"):
        file.unlink()
    rebuilt = Chunks(root, {"run": "x"})
    assert rebuilt.get(["a"]) == {"value": 1}
    object_file = next((root / "objects").glob("*.json"))
    object_file.write_text("{}")
    with pytest.raises(ContinuationError):
        Chunks(root, {"run": "x"})


def test_cross_session_restore_new_root_then_source_deleted(tmp_path):
    a = Chunks(tmp_path / "a/chunks", {"run": "x"})
    a.put(["a"], {"value": 1})
    a.snapshot(tmp_path / "a")
    b = Chunks(tmp_path / "b/chunks", {"run": "x"})
    assert b.import_saved(tmp_path / "a", {"version": 99}) == 1
    shutil.rmtree(tmp_path / "a")
    c = Chunks(tmp_path / "b/chunks", {"run": "x"})
    assert c.get(["a"]) == {"value": 1}
    assert all(r["cross_session_persistence_verified"] for r in c.index.values())


def test_parent_symlink_rejected(tmp_path):
    a = Chunks(tmp_path / "a", {"run": "x"})
    a.put(["x"], {"value": 1})
    shutil.move(tmp_path / "a/objects", tmp_path / "elsewhere")
    (tmp_path / "a/objects").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(ContinuationError):
        Chunks(tmp_path / "a", {"run": "x"})


def test_calibration_only_budget_freeze():
    c = configuration()
    assert choose_budget(2, c)["selected_states"] == 1024
    assert choose_budget(100, c)["selected_states"] == 512
    assert choose_budget(0.1, c)["selected_states"] == 2048
    assert choose_budget(100, c)["target_is_estimate_not_termination"]
