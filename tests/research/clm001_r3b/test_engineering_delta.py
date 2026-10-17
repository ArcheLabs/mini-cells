from clm001_r3b import runner
from clm001_r3b.engineering_delta import compare
from minicells_runtime.store import digest


def test_two_bounded_one_update_traces_leave_parent_immutable(case, tmp_path):
    _, _, root = case
    runner.segment(root, "QC-R3b", 1)
    runner.package(root)
    before = digest(root / "continuation.json")
    result = compare(root, tmp_path / "delta", step=1, device="cpu")
    assert result["status"] == "TRAINING_DELTA_PASS"
    assert result["total_optimizer_updates"] == 2
    assert result["production_steps_added"] == 0
    assert result["exact_invariants_pass"] and result["model_optimizer_pass"]
    assert digest(root / "continuation.json") == before
    # Reentry is report reuse, never more updates.
    assert compare(root, tmp_path / "delta", step=1, device="cpu") == result
