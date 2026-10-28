import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

BASE = (
    Path(__file__).resolve().parents[3]
    / "research/projects/cellular-language-models/experiments/clm-003-routing-value-benchmark"
)
runpy.run_path(str(BASE / "run.py"), run_name="bootstrap")["bootstrap"]()
import routing_value as rv


def test_native_ABC_common_random_and_parent_immutable():
    model, _ = rv.build(rv.substrate(), 7001, unit=True)
    before = rv.fingerprint(model.state_dict())
    rng = torch.get_rng_state().clone()
    models = rv.paired_initialization(model, 7002)
    assert torch.equal(rng, torch.get_rng_state())
    assert before == rv.fingerprint(model.state_dict())
    assert len({rv.fingerprint(m.state_dict()) for m in models.values()}) == 1
    assert rv.fingerprint(rv.random_connection(model, 7002)) != rv.fingerprint(
        rv.random_connection(model, 7003)
    )
    for arm, m in models.items():
        for n, p in m.named_parameters():
            assert p.requires_grad == (
                n.startswith("cells.") or (arm == "A" and n.startswith("connection."))
            )


def test_C_matches_actual_A_commits_and_rejects_replay():
    schedule = rv.RewiringSchedule()
    assert not schedule.observe(50, False, None, 7003)["C_rewire"]
    assert schedule.observe(100, True, "a" * 64, 7004)["C_rewire"]
    with pytest.raises(RuntimeError):
        schedule.observe(100, True, "a" * 64, 7005)
    with pytest.raises(RuntimeError):
        schedule.observe(150, True, None, 7006)


def test_paired_seed_unit_not_sample_pseudoreplication():
    assert (
        rv.paired_effects({1: 1.0, 2: 2.0}, {1: 2.0, 2: 3.0}, {1: 3.0, 2: 4.0})[0]["A_minus_B"]
        == -1
    )
    with pytest.raises(RuntimeError):
        rv.paired_effects({1: 1.0}, {1: 2.0}, {1: 3.0})


def test_engineering_manifests_and_metadata_only_payload(tmp_path):
    root = rv.engineering(
        SimpleNamespace(output=tmp_path, publish_existing=False, publish_github=False)
    )
    m, files, _, _ = rv.payload(root)
    assert m["scientific_optimizer_updates"] == 0 and not m["gpu_used"]
    assert len(rv.read(root / "REGISTRATION.json")["blockers"]) == 10
    assert not any(n.endswith(".pt") for n in files)
    (root / "CPU_GATE.json").write_text("{}")
    with pytest.raises(RuntimeError):
        rv.payload(root)
