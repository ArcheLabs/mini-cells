"""Changed ABC controls and one tiny end-to-end CPU trial only."""

import copy
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fast002_r1"))
import screening as sc
from minicells_fast001.data import Stream
from test_phase0 import rv
from test_stateful import fixture


def test_screening_rule_and_fixed_C_dates():
    m = sc.manifest()
    seeds = m["seeds"]
    values = {
        "A": {s: 1.0 for s in seeds},
        "B": {s: 1.02 for s in seeds},
        "C": {s: 1.03 for s in seeds},
    }
    assert sc.decision(m, values)["status"] == "PROPOSE_INDEPENDENT_CONFIRMATORY_EVALUATION"
    values["B"] = {s: 0.9 for s in seeds}
    assert sc.decision(m, values)["status"] == "STOP_BEFORE_MOE_DENSE"
    assert m["update_steps"] == list(range(50, 501, 50))
    assert "A_ACCEPTANCE" in m["C_schedule"]


def test_one_short_ABC_end_to_end_trial(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "fsync", lambda fd: None)
    c, parent, folder = fixture(tmp_path)
    m = copy.deepcopy(sc.manifest())
    c["stage_b"] = copy.deepcopy(m["runtime_template"]["stage_b"])
    m.update(steps=2, update_steps=[2], runtime_template=c)
    import torch

    x = torch.tensor([[7, 8, 9, 10]])
    m["runtime_template"]["admission"].update(
        manifest_sha256="fixture", ordered_input_hashes=[rv.sha(x.squeeze(0).tolist())]
    )
    bundle = {"role": "admission", "manifest_sha256": "fixture", "inputs": [x]}
    starts = []
    inputs = []
    for arm in "ABC":
        e = sc.engine(m, parent, 7001, arm, tmp_path / "trial", bundle, unit=True)
        starts.append(rv.fingerprint(e.model.state_dict()))
        stream = Stream(e.fast, folder, 7001, "H", "B")
        latest = e.stage_directory / "latest.pt"
        while e.step < 2:
            e.update(stream, latest, [], failure_hook=lambda tr: sc.fixed_rewire(tr, m, bundle))
        inputs.append([r["input_sha256"] for r in e.curve])
        assert e.journal.summary()["optimizer_calls_by_group"]["cell"] == 2
        assert ("random_rewire" in e.curve[-1]) == (arm == "C")
        if arm == "C":
            assert e.curve[-1]["random_rewire"]["independent_of_A_acceptance"]
        for h in e._work_handles:
            h.remove()
    assert len(set(starts)) == 1
    assert inputs[0] == inputs[1] == inputs[2]
