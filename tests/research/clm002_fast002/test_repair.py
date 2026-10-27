import copy

import pytest
import torch
from minicells_clm002.engine import Engine as OriginalEngine
from minicells_clm002.native import build
from minicells_fast001.common import atomic
from minicells_fast001.parent import substrate
from minicells_fast002 import accounting, checkpoints, policy, publication
from minicells_fast002.common import config
from minicells_r3e.common import fingerprint
from minicells_r3e.evaluation import within
from minicells_r3k.reuse import bind


def fixture(tmp_path, arm="B11-H"):
    torch.set_num_threads(1)
    c = config()
    c.update(updates=1, batch_size=2, targets_per_update=8, sequence_length=4)
    c["connection"].update(states=1, policy_steps=1, proposal_steps=[1])
    model, runtime = build(substrate(), 7001, unit=True)
    original = OriginalEngine(
        model, runtime, "BASE", {"test": True}, tmp_path / "base.jsonl", unit=True
    )
    state = original.state()
    state.update(step=300, r3h={"temperature": model.temperature, "rho": model.rho})
    engine = policy.make_engine(
        c, state, 7001, arm, {"test": True}, tmp_path / "new.jsonl", unit=True
    )
    return c, engine


def test_new_protocol_keeps_thresholds_budget_and_historical_identity():
    c = config()
    assert c["connection"]["backtrack"] == [2.0**-i for i in range(11)]
    assert c["connection"]["trust"] == {
        "kl_max": 0.01,
        "action_change_max": 0.1,
        "path_change_max": 0.2,
    }
    assert c["repair"]["main_logical_update_cap"] == 10000
    assert c["repair"]["calibration_main_updates"] == 0
    assert c["historical"]["run_id"] == "75df16bf240cb1998b134e420964d7bc"


def test_zero_baseline_and_tiny_candidate_scale_resolution(tmp_path):
    _, engine = fixture(tmp_path)
    refs = [torch.tensor([[1, 2, 3, 4]])]
    assert policy.zero_check(engine.model, refs) == {
        "kl": 0.0,
        "action_change": 0.0,
        "path_change": 0.0,
    }
    broken = copy.deepcopy(engine.model)
    broken.training_context = (0, 0)
    with pytest.raises(RuntimeError, match="DETERMINISTIC_CONTEXT"):
        policy.zero_check(broken, refs)
    assert not within({"kl": 0, "action_change": 0.11, "path_change": 0}, engine.c)
    assert not within({"kl": 0.011, "action_change": 0, "path_change": 0}, engine.c)


def test_real_candidate_budget_labels_and_no_main_updates(tmp_path):
    _, engine = fixture(tmp_path)
    x, y = torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 5]])
    collector = torch.no_grad()(
        bind(policy.collect.__wrapped__, train_batches=lambda *args: [(x, y)])
    )
    from minicells_jct001.connection import TorchAdapter

    optimize = bind(
        policy.optimize,
        collect=collector,
        reference=lambda *args: [x],
        torch=TorchAdapter(engine.journal),
    )
    before = fingerprint(engine.model.state_dict())
    candidate, report = optimize(engine, {})
    assert before == fingerprint(engine.model.state_dict())
    assert report["policy_optimizer_updates"] == 1
    assert len(report["attempts"]) <= 11
    assert all("rejected_by" in row and "parameter_delta_l2" in row for row in report["attempts"])
    assert (candidate is not None) == report["connection_parameters_changed"]
    budget = accounting.summary(engine, device="cpu")
    assert budget["actual_optimizer_attempts"] == 1 and budget["training_tokens"] == 0
    assert budget["input_unit"] == "LANGUAGE_TARGET_TOKENS"
    assert budget["accounting_status"] != "RECONCILED_CPU_FIXTURE"
    assert engine.step == engine.tokens == 0


def test_checkpoint_exact_schema_and_hash_guard(tmp_path):
    _, engine = fixture(tmp_path)

    class Stream:
        def state(self):
            return {"test": True}

    path = tmp_path / "latest.pt"
    receipt = engine.save(path, Stream())
    from minicells_fast002.common import digest

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["state"]["step"] = 500
    payload["full_state_sha256"] = fingerprint((payload["state"], payload["stream"]))
    torch.save(payload, path)
    result = {"checkpoint_sha256": digest(path), "identity": payload["identity"]}
    assert checkpoints.load(path, result)["step"] == 500
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="HASH_CHANGED"):
        checkpoints.load(path, result)
    assert receipt["step"] == 0


def test_publication_isolated_scope_no_weights(tmp_path):
    assert not publication.allowed("private/repair/latest.pt")
    assert not publication.allowed("../HISTORICAL_AUDIT.json")
    assert publication.allowed("CALIBRATION.json")
    assert "clm-002-fast-002" in str(publication.REPORTS)
    manifest = {
        "experiment_id": "CLM-002-FAST-002",
        "scope": "CONTROLLED_SYNTHETIC_MECHANISM_STUDY",
        "artifact_files": {},
    }
    atomic(tmp_path / "RUN_MANIFEST.json", manifest)
    _, files, _, _ = publication.payload(tmp_path)
    assert set(files) == {"RUN_MANIFEST.json"}


def test_scan_can_reach_1024_without_relaxing_thresholds(tmp_path, monkeypatch):
    _, engine = fixture(tmp_path)
    x, y = torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 5]])
    collector = torch.no_grad()(
        bind(policy.collect.__wrapped__, train_batches=lambda *args: [(x, y)])
    )
    calls = []

    def controlled_trust(old, new, refs):
        if fingerprint(old.connection.state_dict()) == fingerprint(new.connection.state_dict()):
            return {"kl": 0.0, "action_change": 0.0, "path_change": 0.0}
        calls.append(True)
        return {"kl": 0.0, "action_change": 0.2 if len(calls) < 11 else 0.05, "path_change": 0.0}

    monkeypatch.setattr(policy, "trust", controlled_trust)
    candidate, report = bind(policy.optimize, collect=collector, reference=lambda *args: [x])(
        engine, {}
    )
    assert candidate is not None and report["chosen_scale"] == 1 / 1024
    assert len(report["attempts"]) == 11
    assert report["attempts"][0]["rejected_by"] == ["action_change"]


def test_missing_checkpoint_blocks_without_model_or_training(tmp_path):
    c = config()
    reports = tmp_path / "reports"
    for seed in c["seeds"]:
        for arm in c["arms"]:
            if arm.startswith("A-") or arm[:3] in ("B00", "B10"):
                atomic(reports / "jobs" / f"{seed}-{arm}.json", {"checkpoint_sha256": "a" * 64})
    with pytest.raises(RuntimeError, match="ATTACH_ORIGINAL_FAST001_SAVE_VERSION_OUTPUT_MISSING"):
        checkpoints.locate(c, reports, [tmp_path / "empty"])


def test_no_delivered_connection_prevents_scientific_negative_or_positive_claim(
    tmp_path, monkeypatch
):
    from minicells_fast002 import runner

    c = config()
    for seed in c["seeds"]:
        for arm in c["arms"]:
            if arm.startswith("B"):
                atomic(
                    tmp_path / "jobs" / f"{seed}-{arm}.json",
                    {"evaluation": {}, "budget": {"committed_connection_updates": 0}},
                )
    monkeypatch.setattr(
        runner.analysis,
        "factorial",
        lambda *args: {"status": "JOINT_SYNERGY_SUPPORTED", "scientific_support": True},
    )
    result = runner.final_analysis(c, tmp_path)
    assert result["status"] == "INSUFFICIENT_EVIDENCE_CONNECTION_TREATMENT_NOT_DELIVERED"
    assert not result["scientific_support"]


def test_repair_main_update_exact_restore_and_pairing(tmp_path):
    import numpy as np
    from minicells_fast001.data import Stream

    c, engine = fixture(tmp_path)
    folder = tmp_path / "assets"
    folder.mkdir()
    for task in ("Story", "Arithmetic"):
        np.save(
            folder / (task + "-train.npy"),
            np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], dtype=np.int32),
        )
    atomic(folder / "DATASET_MANIFEST.json", {"fixture": True})
    stream = Stream(c, folder, 7001, "H", "B")
    latest = tmp_path / "update.pt"
    engine.update(stream, latest, [torch.tensor([[1, 2, 3, 4]])])
    assert engine.step == 1 and engine.tokens == 8
    assert engine.journal.summary()["optimizer_calls_by_group"] == {"cell": 1, "connection": 1}
    before = fingerprint((engine.state(), stream.state()))
    engine.restore(latest, stream)
    assert fingerprint((engine.state(), stream.state())) == before
    assert accounting.summary(engine, device="cpu")["training_tokens"] == 8
