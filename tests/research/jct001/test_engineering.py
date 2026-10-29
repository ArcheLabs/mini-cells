import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from minicells_r3e.common import fingerprint
from minicells_jct001.data import (
    TaskBank,
    signature,
    equivalent_paths,
    reference,
    feature_tensor,
    batches,
    unit_data as raw_unit_data,
)
from minicells_jct001.training import Trainer
from minicells_jct001.analysis import interventions, paired_contrasts, holm
from minicells_jct001.cost import Journal, WorkMeter
from minicells_jct001.gates import require_scientific


def unit_data(fixture):
    data = raw_unit_data(fixture)
    data["reference_features"] = feature_tensor(TaskBank(fixture).records("calibration")[:4])
    return data


def make(config, identity, tmp_path, arm="D"):
    base = Trainer(config, "BASE", identity, tmp_path / "base.jsonl").state()
    return Trainer(config, arm, identity, tmp_path / (arm + ".jsonl"), base), base


def test_canonical_and_confirmation(config):
    bank = TaskBank(config["fixture"])
    groups = []
    for role in ("train", "calibration", "pilot_report"):
        rows = bank.records(role)
        groups.append({r["canonical_id"] for r in rows})
        for row in rows:
            assert reference(row) + 32 == row["target"]
            assert all(
                signature(p, config["fixture"]["primitive_affines"]) == (row["a"], row["b"])
                for p in equivalent_paths(row, config["fixture"])
            )
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1 :])
    with pytest.raises(RuntimeError, match="SEALED"):
        bank.records("confirmation_reserved")
    assert bank.registry()["confirmation"]["records_materialized"] == 0
    assert TaskBank(config["fixture"]).registry() == bank.registry()


def test_report_cannot_train(config):
    with pytest.raises(RuntimeError, match="TRAIN_ROLE"):
        batches(
            dict(role="pilot_report", records=[]),
            dict(training=dict(effective_batch=4)),
            np.random.default_rng(1),
            "cpu",
        )


def test_four_actual_arm_updates(config, identity, tmp_path):
    base = Trainer(config, "BASE", identity, tmp_path / "base.jsonl").state()
    rows = {}
    trainers = {}
    from minicells_r3e import policy_proposal

    original = policy_proposal.collect
    for arm in "ABCD":
        tr = Trainer(config, arm, identity, tmp_path / (arm + ".jsonl"), base)
        rows[arm] = tr.advance(unit_data(config["fixture"]), tmp_path / (arm + ".pt"))
        trainers[arm] = tr
        assert tr.journal.summary()["completed_optimizer_calls"] == dict(A=0, B=1, C=2, D=3)[arm]
        assert tr.data_cursor == 4
        for group in ("cells", "connection"):
            assert (rows[arm]["parameter_delta_l2"][group] > 0) == config["arms"][arm][group]
    assert policy_proposal.collect is original
    assert len({r["input_batch_sha256"] for r in rows.values()}) == 1
    assert len({fingerprint(t.data_rng.bit_generator.state) for t in trainers.values()}) == 1
    assert len({fingerprint(t.route_rng.get_state()) for t in trainers.values()}) == 1
    assert trainers["D"].journal.summary()["counterfactual_target_evaluations"] > 0
    assert trainers["D"].journal.summary()["executed_operator_work"]["cell_rows"] > 12


@pytest.mark.parametrize("arm", ["A", "B", "C", "D"])
def test_restore_next_update(config, identity, tmp_path, arm):
    config["fixture"]["logical_updates"] = 2
    tr, base = make(config, identity, tmp_path, arm)
    data = unit_data(config["fixture"])
    checkpoint = tmp_path / "state.pt"
    tr.advance(data, checkpoint)
    saved = fingerprint(tr.state())
    tr.save(tmp_path / "boundary.pt")
    tr.advance(data, checkpoint)
    expected = fingerprint(tr.state())
    restored = Trainer(config, arm, identity, tmp_path / (arm + ".jsonl"), base)
    restored.restore(tmp_path / "boundary.pt")
    assert fingerprint(restored.state()) == saved
    restored.advance(data, checkpoint)
    assert fingerprint(restored.state()) == expected
    if arm in "CD":
        assert restored.connection_optimizer_state["state"]
    if arm in "BD":
        assert restored.cell_optimizer.state_dict()["state"]


def test_failed_work_retained(config, identity, tmp_path):
    tr, base = make(config, identity, tmp_path, "B")
    checkpoint = tmp_path / "state.pt"
    tr.save(checkpoint)

    def fail(_):
        raise RuntimeError("INJECTED")

    with pytest.raises(RuntimeError, match="INJECTED"):
        tr.advance(unit_data(config["fixture"]), checkpoint, failure_hook=fail)
    assert tr.journal.summary()["completed_optimizer_calls"] == 1
    tr.restore(checkpoint)
    assert tr.step == 0
    tr.advance(unit_data(config["fixture"]), checkpoint)
    summary = tr.journal.summary()
    assert summary["logical_updates"] == 1 and summary["completed_optimizer_calls"] == 2
    assert summary["input_examples_attempted"] == 8


def test_uncertain_ledger_quarantined(config, identity, tmp_path):
    tr, _ = make(config, identity, tmp_path)
    tr.journal.append("OPT_BEGIN", dict(id="unfinished", optimizer="cell"))
    with pytest.raises(RuntimeError, match="QUARANTINE"):
        tr.advance(unit_data(config["fixture"]), tmp_path / "state.pt")
    with (tmp_path / "D.jsonl").open("a") as f:
        f.write("{broken")
    with pytest.raises(RuntimeError, match="QUARANTINE"):
        Journal(tmp_path / "D.jsonl", tr.identity)


def test_meter_copy():
    meter = WorkMeter()
    copy.deepcopy(meter).add("cell_rows", 7)
    assert meter.counts["cell_rows"] == 7


def test_readonly_interventions(config, identity, tmp_path):
    tr, _ = make(config, identity, tmp_path)
    model = tr.model
    before = fingerprint(model.state_dict())
    rows = TaskBank(config["fixture"]).records("calibration")[:2]
    assert feature_tensor(rows).shape == (2, 3)
    result = interventions(
        model, rows, config["fixture"], copy.deepcopy(model.connection.state_dict())
    )
    assert fingerprint(model.state_dict()) == before
    assert result["oracle"]["oracle_used"] and not result["learned"]["oracle_used"]
    assert result["oracle"]["accuracy"] == 1 and not result["scientific_decision"]


def test_contrasts():
    result = paired_contrasts({1: dict(A=9, B=7, C=8, D=5), 2: dict(A=10, B=7, C=9, D=6)})
    assert result["per_seed"][0]["joint_minus_cell"] == -2
    assert result["per_seed"][0]["interaction"] == -1
    assert result["summary"]["joint_minus_cell"]["confidence_interval"] is None
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    with pytest.raises(ValueError):
        paired_contrasts({1: dict(A=1)})


@pytest.mark.parametrize("stage", ["synthetic-pilot", "real-model", "confirmation"])
def test_science_blocked(config, stage):
    with pytest.raises(RuntimeError):
        require_scientific(stage, config)


def test_runner(config, identity, tmp_path, monkeypatch):
    from minicells_jct001 import runner
    from minicells_jct001.common import read

    monkeypatch.setattr(runner, "run_identity", lambda c, registry: identity)
    result = runner.engineering(config, tmp_path / "out")
    root = Path(result["output"])
    assert result["status"] == "CPU_ENGINEERING_PASS_SCIENTIFIC_BLOCKED"
    assert all((root / name).exists() for name in runner.ARTIFACTS)
    manifest = read(root / "RUN_MANIFEST.json")
    assert manifest["cpu_fixture_optimizer_calls"] == 6
    assert runner.engineering(config, tmp_path / "out")["reused_completed_engineering"]
    assert manifest["scientific_optimizer_updates"] == 0


def test_hard_paths_and_self_connection(config, identity, tmp_path):
    trainer, _ = make(config, identity, tmp_path, "A")
    model = trainer.model
    inputs = torch.tensor([[2.0, 1.0, 3.0]])
    out = model(inputs, keep_trace=True)
    assert out.actions.tolist() == [[0, 0, 0]]
    assert out.value.tolist() == [5.0]
    assert [row["previous"].item() for row in out.trace] == [6, 0, 0]
    assert sum(p.numel() for p in model.parameters()) == 126
    with pytest.raises(ValueError, match="ROUTER_INPUT"):
        model(torch.zeros(1, 4))


def test_rejected_connection_proposal_still_counts_cost(config, identity, tmp_path, monkeypatch):
    from minicells_jct001 import connection

    monkeypatch.setattr(connection, "within", lambda metrics, c: False)
    trainer, _ = make(config, identity, tmp_path, "C")
    before = fingerprint(trainer.model.state_dict())
    row = trainer.advance(unit_data(config["fixture"]), tmp_path / "checkpoint.pt")
    assert not row["connection_admitted"]
    assert before == fingerprint(trainer.model.state_dict())
    assert trainer.journal.summary()["completed_optimizer_calls"] == 2
    assert trainer.journal.summary()["gpu_seconds"] is None
