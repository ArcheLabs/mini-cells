"""Synthetic CPU checks for the pre-outcome contract (not a complete R3k gate)."""

import copy
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent / "implementation"))
from minicells_r3k.prediction_freeze import scores, freeze, require_persisted
from minicells_r3k.data_boundaries import boundary_proof, REVISION, TOKENIZER


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid input was accepted")


def main():
    result = scores(
        {"OO": 2.0, "NN": 1.9}, {"OO": 2.0, "ON": 1.8}, {"OO": 1.9, "ON": 1.7, "NN": 1.6}
    )
    assert abs(result["S_K10"] - 0.3) < 1e-12
    assert abs(result["S_Cell"] - 0.1) < 1e-12
    assert abs(result["S_Slope"] - 0.1) < 1e-12
    rows = [
        dict(case_id=f"{s}-{a}-{t}", seed=s, arm=a, proposal_at_step=t, scores=result)
        for s in (151001301, 151001302, 151001303)
        for a in ("A-CH", "A-CT", "B-CH", "B-CT")
        for t in (100, 200, 300, 400)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "FROZEN_PREDICTIONS.json"
        frozen = freeze(path, rows, {"probe": "synthetic"}, {"source": "synthetic"})
        assert freeze(path, rows, {"probe": "synthetic"}, {"source": "synthetic"}) == frozen
        require_persisted(frozen, copy.deepcopy(frozen))
        damaged = copy.deepcopy(frozen)
        damaged["rows"][0]["scores"]["S_Cell"] = 3
        rejected(lambda: require_persisted(frozen, damaged))
        changed = copy.deepcopy(rows)
        changed[0]["scores"]["S_Cell"] = 4
        rejected(lambda: freeze(path, changed, {"probe": "synthetic"}, {"source": "synthetic"}))
        rejected(lambda: freeze(path, rows[:-1], {}, {}))
    stories = {i: str(i) for i in range(8192, 8960)}
    stories[8448] = stories[8192]
    proof = boundary_proof(REVISION, TOKENIZER, {"train": (0, 8192)}, [], stories)
    assert proof["roles"]["value_holdout"]["excluded"][0]["story_index"] == 8448
    rejected(
        lambda: boundary_proof(REVISION, TOKENIZER, {"confirmation": (8500, 9000)}, [], stories)
    )
    print(
        {"status": "PRE_OUTCOME_CONTRACT_CPU_PASS", "full_r3k_ready": False, "gpu_started": False}
    )


if __name__ == "__main__":
    main()
