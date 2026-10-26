import copy

import numpy as np
import pytest
import torch
from minicells_clm002.engine import Engine as OriginalEngine
from minicells_clm002.native import build, classify
from minicells_fast001 import analysis, data, publication
from minicells_fast001.common import atomic, read, release, sha
from minicells_fast001.parent import make_engine, substrate
from minicells_r3e.common import fingerprint


def miniature(config, tmp_path):
    config.update(
        updates=2, batch_size=2, sequence_length=4, targets_per_update=8, evaluation_batch_size=2
    )
    config["connection"].update(proposal_steps=[2], states=1, policy_steps=1)
    model, runtime = build(substrate(), 7001, unit=True)
    identity = {"experiment_id": "FAST_SYNTHETIC_TEST", "source_sha": "a" * 40, "run_id": "b" * 32}
    original = OriginalEngine(model, runtime, "BASE", identity, tmp_path / "base.jsonl", unit=True)
    for parameter in model.cells.parameters():
        original.cell_optimizer.state[parameter] = {
            "step": torch.tensor(300.0),
            "exp_avg": torch.full_like(parameter, 0.01),
            "exp_avg_sq": torch.full_like(parameter, 0.1),
        }
    state = original.state()
    state.update(
        step=300, data_cursor=153600, r3h={"temperature": model.temperature, "rho": model.rho}
    )
    folder = tmp_path / "assets"
    folder.mkdir()
    for task in ("Story", "Arithmetic"):
        rows = np.random.default_rng(7001).integers(0, 32, size=(8, 5), dtype=np.int32)
        np.save(folder / (task + "-train.npy"), rows)
        np.save(folder / (task + "-confirm.npy"), rows)
    atomic(folder / "DATASET_MANIFEST.json", {"scope": "SMALL_SYNTHETIC_TEST_ONLY"})
    return state, identity, folder


def test_generator_identity_independence_and_exact_budgets(config, tmp_path):
    manifest = data.generate(config, tmp_path, counts={"train": 6, "confirm": 3})
    assert manifest["engineering_small_fixture"]
    assert all(r["effective_targets"] == r["records"] * 64 for r in manifest["roles"].values())
    text, windows, inputs, targets = set(), set(), set(), set()
    for task in ("Story", "Arithmetic"):
        for role in ("train", "confirm"):
            for row in read(tmp_path / (task + "-" + role + ".json")):
                assert row["raw_text_sha256"] not in text
                assert row["window_sha256"] not in windows
                text.add(row["raw_text_sha256"])
                windows.add(row["window_sha256"])
            array = np.load(tmp_path / (task + "-" + role + ".npy"))
            for value in array:
                assert sha(value[:-1].tolist()) not in inputs
                assert sha(value[1:].tolist()) not in targets
                inputs.add(sha(value[:-1].tolist()))
                targets.add(sha(value[1:].tolist()))
    assert data.generate(config, tmp_path) == manifest
    (tmp_path / "Story-train.npy").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="DATA_FILE_CHANGED"):
        data.generate(config, tmp_path)


@pytest.mark.parametrize("arm", ["B00-H", "B10-H", "B01-H", "B11-H"])
def test_phase_parent_moments_pairing_cadence_and_frozen_groups(config, tmp_path, arm):
    source, ident, folder = miniature(config, tmp_path)
    before = fingerprint(source)
    engine = make_engine(
        config, source, 7001, arm, ident, tmp_path / "engine.jsonl", unit=True, parent_sha="c" * 64
    )
    assert fingerprint(engine.cell_optimizer.state_dict()) == fingerprint(source["cell_optimizer"])
    assert engine.parent_step == 300 and engine.step == 0
    stream = data.Stream(config, folder, 7001, "H", "B")
    references = [
        r["x"] for r in data.evaluation_rows(config, folder, "cpu", role="train", condition="H")
    ]
    values = {
        name: parameter.detach().clone() for name, parameter in engine.model.named_parameters()
    }
    checkpoint = tmp_path / "latest.pt"
    engine.update(stream, checkpoint, references)
    assert engine.proposal_calls == 0
    engine.update(stream, checkpoint, references)
    groups = engine.journal.summary()["optimizer_calls_by_group"]
    assert groups.get("cell", 0) == (2 if arm in ("B10-H", "B11-H") else 0)
    assert groups.get("connection", 0) == (1 if arm in ("B01-H", "B11-H") else 0)
    for name, parameter in engine.model.named_parameters():
        if not parameter.requires_grad:
            assert torch.equal(values[name], parameter), classify(name)
    assert fingerprint(source) == before
    expected = fingerprint((engine.state(), stream.state()))
    engine.restore(checkpoint, stream)
    assert fingerprint((engine.state(), stream.state())) == expected
    child = make_engine(
        config, engine.state(), 7001, "B11-H", ident, tmp_path / "child.jsonl", unit=True
    )
    assert child.parent_step == 302 and child.step == child.tokens == 0
    assert fingerprint(child.cell_optimizer.state_dict()) == fingerprint(
        engine.cell_optimizer.state_dict()
    )


def test_stream_common_rng_and_no_task_id_in_model_tensors(config, tmp_path):
    _, _, folder = miniature(config, tmp_path)
    first = data.Stream(config, folder, 7001, "H", "B")
    second = data.Stream(config, folder, 7001, "H", "B")
    a, b = first.next(), second.next()
    assert fingerprint([(r["x"], r["y"]) for r in a]) == fingerprint([(r["x"], r["y"]) for r in b])
    assert sorted(r["task"] for r in a) == ["Arithmetic", "Story"]
    assert sum(r["y"].numel() for r in a) == 8
    assert all(r["x"].shape == (1, 4) for r in a)


def test_fixed_stage_a_gate_stops_b_without_replacing_seeds(config):
    def measurement(value):
        return {
            "specialization": {"S": value, "bidirectional": True},
            "single_route_collapse": False,
        }

    rows = [
        {
            "seed": seed,
            "S": {str(s): measurement(0) for s in (400, 500)},
            "H": {str(s): measurement(0.02) for s in (400, 500)},
        }
        for seed in config["seeds"]
    ]
    assert analysis.stage_a(config, rows)["stage_b_allowed"]
    rows[0]["H"]["400"] = measurement(0)
    rows[1]["H"]["500"] = measurement(0)
    assert not analysis.stage_a(config, rows)["stage_b_allowed"]
    with pytest.raises(RuntimeError, match="ONE_PROTOCOL_REVIEW_REQUIRED"):
        release(config, "unregistered")


def test_paired_effect_and_secondary_df4_math(config):
    assert analysis.t_one_sided_negative([0, 1, -1, 2, -2]) == pytest.approx(0.5)
    assert analysis.t_one_sided_negative([-2, -1, -3, -2, -2]) < 0.01
    rows = []
    for seed in config["seeds"]:
        values = {}
        for condition in ("S", "H"):
            for arm, loss in [
                ("B00", 2.0),
                ("B10", 1.99),
                ("B01", 1.99),
                ("B11", 1.9 if condition == "H" else 1.98),
            ]:
                values[arm + "-" + condition] = {
                    "L": loss,
                    "tasks": {k: {"nll": loss} for k in ("Story", "Arithmetic")},
                    "specialization": {
                        "S": 0.02 if condition == "H" else 0.0,
                        "bidirectional": condition == "H",
                    },
                    "single_route_collapse": False,
                }
        rows.append({"seed": seed, "arms": values})
    result = analysis.factorial(config, rows)
    assert result["status"] == "JOINT_SYNERGY_SUPPORTED"
    assert result["I_H"]["mean"] == pytest.approx(-0.08)
    damaged = copy.deepcopy(rows)
    damaged[0]["arms"]["B11-H"]["tasks"]["Story"]["nll"] = 3
    assert not analysis.factorial(config, damaged)["scientific_support"]


def test_publication_rejects_weights_and_hash_changes(config, tmp_path):
    atomic(tmp_path / "BUDGET.json", {"scientific_optimizer_updates": 0})
    manifest = {
        "experiment_id": "CLM-002-FAST-001",
        "scope": "CONTROLLED_SYNTHETIC_MECHANISM_STUDY",
        "run_id": "a" * 32,
        "identity": {},
        "artifact_files": {},
    }
    publication.finalize(tmp_path, manifest, publish_github=False)
    _, files, _, _ = publication.payload(tmp_path)
    assert set(files) == {"BUDGET.json", "RUN_MANIFEST.json"}
    assert not publication.allowed("private/latest.pt")
    assert not publication.allowed("Story-train.npy")
    atomic(tmp_path / "BUDGET.json", {"changed": True})
    with pytest.raises(RuntimeError, match="REPORT_HASH_CHANGED"):
        publication.payload(tmp_path)


def test_a_to_b_parent_and_batched_confirm_are_readonly(config, tmp_path):
    from minicells_fast001.evaluation import evaluate

    source, ident, folder = miniature(config, tmp_path)
    first = make_engine(config, source, 7001, "A-H", ident, tmp_path / "a.jsonl", unit=True)
    stream = data.Stream(config, folder, 7001, "H", "A")
    first.update(stream, tmp_path / "a.pt", [])
    first.update(stream, tmp_path / "a.pt", [])
    before = fingerprint(first.model.state_dict())
    result = evaluate(first.model, config, folder, mechanism=True)
    assert fingerprint(first.model.state_dict()) == before
    assert result["tasks"]["Story"]["tokens"] == 8 * 4
    child = make_engine(
        config, first.state(), 7001, "B00-H", ident, tmp_path / "b.jsonl", unit=True
    )
    assert child.parent_step == 302
    assert fingerprint(child.model.state_dict()) == before
    assert fingerprint(child.cell_optimizer.state_dict()) == fingerprint(
        first.cell_optimizer.state_dict()
    )


def test_partial_optimizer_work_is_counted_and_never_automatically_retrained(config, tmp_path):
    from minicells_fast001.runner import budget

    source, ident, folder = miniature(config, tmp_path)
    root = tmp_path / "run"
    directory = root / "private/A/7001-A-H"
    directory.mkdir(parents=True)
    engine = make_engine(config, source, 7001, "A-H", ident, directory / "cost.jsonl", unit=True)
    stream = data.Stream(config, folder, 7001, "H", "A")
    checkpoint = directory / "latest.pt"
    engine.save(checkpoint, stream)
    with engine.journal.phase("confirm_0"):
        pass
    engine.restore(checkpoint, stream)

    def failure(_):
        raise RuntimeError("disposable interruption")

    with pytest.raises(RuntimeError, match="disposable interruption"):
        engine.update(stream, checkpoint, [], failure_hook=failure)
    result = budget(config, root)
    assert result["actual_cell_optimizer_calls"] == 1
    assert result["partial_jobs"][0]["restoration_uncertain"]
    assert result["gpu_used"] is False
    with pytest.raises(RuntimeError, match="INFLIGHT_WORK_AFTER_CHECKPOINT"):
        engine.restore(checkpoint, stream)


def test_zero_update_acceptance_covers_all_ten_paths_per_seed(config, tmp_path, monkeypatch):
    from minicells_fast001 import parent

    source, _ident, _ = miniature(config, tmp_path)
    original = parent.make_engine
    monkeypatch.setattr(
        parent, "inspect_parent", lambda c, p, o: {"scope": "SYNTHETIC_TEST_NOT_REAL_PARENT"}
    )
    monkeypatch.setattr(parent, "original_state", lambda c, p: source)
    monkeypatch.setattr(
        parent, "make_engine", lambda *args, **kwargs: original(*args, **kwargs, unit=True)
    )
    report = parent.acceptance(
        config, tmp_path / "unused", tmp_path / "acceptance", {"source_sha": "a" * 40}
    )
    assert len(report["forks"]) == 50
    assert report["actual_optimizer_calls"] == report["actual_forward_calls"] == 0
    assert all(r["checks"]["sibling_storage_independent"] for r in report["forks"])
