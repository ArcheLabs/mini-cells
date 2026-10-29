import copy

import numpy as np
import pytest
import torch
from minicells_clm002.engine import Engine as Original
from minicells_clm002.native import build
from minicells_fast001.data import Stream
from minicells_fast001.parent import make_engine as fast_engine
from minicells_fast001.parent import substrate
from minicells_fast_r1 import candidate, replay, scan
from minicells_fast_r1.common import atomic, config, sha
from minicells_jct001.cost import Journal
from minicells_r3e.common import fingerprint


def fixture(tmp_path):
    torch.set_num_threads(1)
    c = config()
    c.update(batch_size=2, sequence_length=4, targets_per_update=8)
    c["connection"].update(states=1, policy_steps=2)
    model, runtime = build(substrate(), 7001, unit=True)
    source = Original(
        model,
        runtime,
        "BASE",
        {"scope": "CPU_SYNTHETIC_ONLY"},
        tmp_path / "origin.jsonl",
        unit=True,
    )
    for p in model.cells.parameters():
        source.cell_optimizer.state[p] = {
            "step": torch.tensor(300.0),
            "exp_avg": torch.full_like(p, 0.01),
            "exp_avg_sq": torch.full_like(p, 0.1),
        }
    state = source.state()
    state.update(step=300, r3h={"temperature": model.temperature, "rho": model.rho})
    parent = fast_engine(
        c, state, 7001, "A-H", {"scope": "CPU_SYNTHETIC_ONLY"}, tmp_path / "a.jsonl", unit=True
    )
    state = parent.state()
    state.update(step=500, tokens=4000)
    state["schedule"]["epsilon"] = parent.c["training"]["epsilon_final"]
    for row in state["cell_optimizer"]["state"].values():
        row["step"] = torch.tensor(800.0)
    folder = tmp_path / "train"
    folder.mkdir()
    for task in ("Story", "Arithmetic"):
        np.save(
            folder / (task + "-train.npy"),
            np.random.default_rng(7001).integers(0, 32, size=(8, 5), dtype=np.int32),
        )
    atomic(folder / "DATASET_MANIFEST.json", {"scope": "CPU_SYNTHETIC_ONLY"})
    return c, state, folder


def context(c, state, folder, root, arm="B11-H"):
    root.mkdir()
    ident = {
        "scope": "CPU_SYNTHETIC_ONLY",
        "source_sha": "a" * 40,
        "config_sha256": sha(c),
        "arm": arm,
        "seed": 7001,
    }
    engine = replay.make_engine(c, state, 7001, arm, ident, root / "cost.jsonl", unit=True)
    stream = Stream(c, folder, 7001, arm[-1], "B")
    proof = replay.run(engine, stream, root, c, ident)
    return engine, stream, proof, ident


@pytest.mark.parametrize("arm", ["B01-H", "B11-H"])
def test_real_order_step50_pause_masks_optimizer_rng_and_restore(tmp_path, arm):
    c, state, folder = fixture(tmp_path)
    before = fingerprint(state["model"])
    engine, stream, proof, ident = context(c, state, folder, tmp_path / "job", arm)
    assert engine.parent_step == 800 and engine.step == 49 and stream.step == 50
    assert proof["executed_main_updates"] == 50 and proof["counterfactual_optimizer_calls"] == 0
    assert proof["formal_connection_commits"] == 0
    assert engine.journal.summary()["optimizer_calls_by_group"].get("cell", 0) == (
        50 if arm.startswith("B11") else 0
    )
    assert (fingerprint(engine.model.state_dict()) != before) == arm.startswith("B11")
    saved = fingerprint(replay.pending_values(engine))
    restored = replay.make_engine(
        c, state, 7001, arm, ident, tmp_path / "job/cost.jsonl", unit=True
    )
    restored_stream = Stream(c, folder, 7001, "H", "B")
    replay.restore(restored, restored_stream, tmp_path / "job", c, ident)
    assert fingerprint(replay.pending_values(restored)) == saved
    assert replay.scientific_fingerprint(restored.state()) == proof["scientific_state_sha256"]


def test_repeat_replay_scientific_state_exact_cpu_and_arm_states_distinct(tmp_path):
    c, state, folder = fixture(tmp_path)
    first, _, a, _ = context(c, state, folder, tmp_path / "one")
    second, _, b, _ = context(c, state, folder, tmp_path / "two")
    assert a["scientific_state_sha256"] == b["scientific_state_sha256"]
    frozen, _, _, _ = context(c, state, folder, tmp_path / "frozen", "B01-H")
    assert fingerprint(first.model.state_dict()) != fingerprint(frozen.model.state_dict())
    assert fingerprint(first.pending_batch) == fingerprint(frozen.pending_batch)
    assert first.cf_rng.get_state().equal(second.cf_rng.get_state())


def test_unknown_optimizer_tail_cannot_trigger_replay(tmp_path):
    c, state, folder = fixture(tmp_path)
    engine, stream, _proof, ident = context(c, state, folder, tmp_path / "job")
    engine.journal.append(
        "OPT_BEGIN", {"id": "injected", "optimizer": "cell", "gradient_norm": 1.0}
    )
    with pytest.raises(RuntimeError):
        replay.run(engine, stream, tmp_path / "job", c, ident)


def test_full_scan_includes4096_no_early_stop_preserves_weights_and_excludes_confirm(tmp_path):
    c, state, folder = fixture(tmp_path)
    engine, _, _proof, ident = context(c, state, folder, tmp_path / "job")
    before = fingerprint(engine.model.state_dict())
    ledger = Journal(tmp_path / "candidate.jsonl", ident)
    ledger.meter = engine.journal.meter
    package = candidate.generate(engine, ledger, tmp_path / "job", ident)
    assert len(package["losses"]) == 2 and ledger.summary()["optimizer_calls_by_group"] == {
        "connection": 2
    }
    assert package["optimizer_initial"]["state"] == {}
    x = torch.tensor([[7, 8, 9, 10]])
    c["admission"].update(
        manifest_sha256="fixture", ordered_input_hashes=[sha(x.squeeze(0).tolist())]
    )
    inputs = {"role": "admission", "manifest_sha256": "fixture", "inputs": [x]}
    decision = scan.run(engine, package, inputs, ledger, tmp_path / "job", c, ident)
    assert len(decision["attempts"]) == 22
    assert decision["attempts"][13]["scale"] == 1 / 4096
    assert decision["attempts"][-1]["scale"] == 2.0**-20
    assert decision["attempts"][0]["passed"] is False
    assert fingerprint(engine.model.state_dict()) == before
    assert decision["formal_connection_commits"] == 0 and not decision["confirm_read"]
    bad = dict(inputs, role="confirm")
    with pytest.raises(RuntimeError, match="ONLY_FROZEN_ADMISSION"):
        scan.run(engine, package, bad, ledger, tmp_path / "job", c, ident)


def test_four_complete_seed_rule_does_not_approve_formal_training():
    c = config()
    rows = []
    for seed in c["seeds"]:
        for arm in ("B01", "B11"):
            for cond in ("S", "H"):
                rows.append(
                    {
                        "seed": seed,
                        "arm": arm + "-" + cond,
                        "attempts": [{}] * 22,
                        "chosen_scale": 1 / 4096 if seed != c["seeds"][-1] else None,
                    }
                )
    result = scan.aggregate(c, rows)
    assert result["proposed_four_seed_prerequisite"]
    assert not result["gate3_authorized"] and not result["gate1_approved"]
    rows.pop()
    assert not scan.aggregate(c, rows)["complete_contexts"]


def test_candidate_math_matches_frozen_r3e_optimizer(tmp_path):
    from minicells_r3e import policy_proposal
    from minicells_r3k.reuse import bind

    c, state, folder = fixture(tmp_path)
    engine, _, _, ident = context(c, state, folder, tmp_path / "job")
    historical = copy.deepcopy(engine)
    historical.c["proposal"]["backtrack"] = [1.0]
    collector = torch.no_grad()(
        bind(
            policy_proposal.collect.__wrapped__,
            train_batches=lambda *args: historical.pending_batch,
        )
    )
    # Synthetic algorithm-equivalence fixture only; no real Admission claim is made.
    old_candidate, old_report = bind(
        policy_proposal.optimize,
        collect=collector,
        reference=lambda *args: [],
        trust=lambda *args: {"kl": 0.0, "action_change": 0.0, "path_change": 0.0},
    )(historical, {})
    journal = Journal(tmp_path / "candidate.jsonl", ident)
    journal.meter = engine.journal.meter
    package = candidate.generate(engine, journal, tmp_path / "job", ident)
    assert fingerprint(old_candidate.connection.state_dict()) == fingerprint(
        package["candidate_connection"]
    )
    assert old_report["losses"] == package["losses"]


def test_frozen_admission_bundle_never_opens_confirm_and_detects_tamper(tmp_path, monkeypatch):
    from minicells_fast_r1 import assets

    c = config()
    original = np.load

    def only_admission(path, *args, **kwargs):
        assert str(path).endswith("-admission.npy")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", only_admission)
    bundle = assets.refs(c, tmp_path, "cpu")
    assert bundle["role"] == "admission" and len(bundle["inputs"]) == 32
    assert [sha(x.squeeze(0).tolist()) for x in bundle["inputs"]] == c["admission"][
        "ordered_input_hashes"
    ]
    (tmp_path / "Story-admission.npy").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="ADMISSION_FILE_CHANGED"):
        assets.admission(c, tmp_path)


def test_report_allowlist_accepts_metadata_csv_md_but_never_weights(tmp_path):
    from minicells_fast_r1 import publication
    from minicells_fast_r1.common import ID

    (tmp_path / "FINAL_REPORT.md").write_text("Engineering only, no actual diagnostic executed.\n")
    (tmp_path / "CONNECTION_SCALE_SCAN.csv").write_text("scale,passed\n0,False\n")
    manifest = {
        "experiment_id": ID,
        "scope": "STATEFUL_CALIBRATION_ENGINEERING",
        "run_id": "a" * 32,
        "artifact_files": {},
    }
    publication.finalize(tmp_path, manifest, False)
    _, files, _, _ = publication.payload(tmp_path)
    assert set(files) == {"FINAL_REPORT.md", "CONNECTION_SCALE_SCAN.csv", "RUN_MANIFEST.json"}
    assert "candidate.pt" not in publication.NAMES


def test_cli_default_stays_engineering_and_authorized_flag_is_explicit(tmp_path, monkeypatch):
    import runpy
    import sys

    from minicells_fast_r1 import diagnostic, runner
    from minicells_fast_r1.common import BASE

    calls = []
    monkeypatch.setattr(runner, "engineering", lambda args: calls.append("engineering"))
    monkeypatch.setattr(diagnostic, "diagnostic", lambda args: calls.append("bounded_gate1"))
    argv = ["run.py", "--output", str(tmp_path / "out"), "--assets", str(tmp_path / "assets")]
    monkeypatch.setattr(sys, "argv", argv)
    ns = runpy.run_path(str(BASE / "run.py"), run_name="r1_launch")
    ns["main"]()
    argv.append("--run-diagnostic")
    ns["main"]()
    assert calls == ["engineering", "bounded_gate1"]
    c, approval = diagnostic.authorized_config()
    assert approval["limits"]["main_updates"] == 1000
    assert c["diagnostic"]["gate1_execution_authorized"]
    assert not c["diagnostic"]["gate3_execution_authorized"]
    assert not config()["diagnostic"]["gate1_execution_authorized"]


def test_zero_candidate_direction_is_recorded_not_made_into_success(tmp_path, monkeypatch):
    from minicells_fast_r1 import candidate as module

    c, state, folder = fixture(tmp_path)
    engine, _, _, identity = context(c, state, folder, tmp_path / "job")
    width = next(engine.model.cells[0].parameters()).shape[-1]

    @torch.no_grad()
    def constant_costs(tr, data):
        return [
            {
                "h": torch.zeros(1, width),
                "previous": torch.zeros(1, dtype=torch.long),
                "stage": 0,
                "values": torch.ones(6),
                "target": torch.zeros(1, dtype=torch.long),
            }
        ]

    monkeypatch.setattr(module.policy_proposal, "collect", constant_costs)
    journal = Journal(tmp_path / "candidate.jsonl", identity)
    journal.meter = engine.journal.meter
    package = module.generate(engine, journal, tmp_path / "job", identity)
    assert package["losses"] == [0.0, 0.0]
    assert journal.summary()["optimizer_calls_by_group"]["connection"] == 2
    assert fingerprint(package["original_connection"]) == fingerprint(
        package["candidate_connection"]
    )


def test_unattested_parent_cannot_initialize_a_real_replay(tmp_path, monkeypatch):
    c, state, _folder = fixture(tmp_path)
    monkeypatch.setattr(
        replay,
        "original_make_engine",
        lambda *args, **kwargs: pytest.fail("Must reject before model construction"),
    )
    with pytest.raises(RuntimeError, match="EXACT_SEED_CONDITION_A500_PARENT_IDENTITY"):
        replay.make_engine(
            c, state, 7001, "B11-H", {"scope": "NOT_REAL"}, tmp_path / "forbidden.jsonl", unit=False
        )


def test_failure_reports_do_not_export_signed_urls_or_tokens():
    from minicells_fast_r1.runner import safe_failure

    private = "https://storage.example/file?X-Goog-Signature=private-value&token=private-token"
    encoded = str(safe_failure(RuntimeError(private)))
    assert (
        "private-value" not in encoded
        and "private-token" not in encoded
        and "https://" not in encoded
    )
    assert (
        safe_failure(RuntimeError("R1_SOURCE_IDENTITY_CHANGED:private-detail"))["code"]
        == "R1_SOURCE_IDENTITY_CHANGED"
    )


def test_authorized_context_full_pipeline_and_completed_resume(tmp_path, monkeypatch):
    from minicells_fast_r1 import diagnostic, publication
    from minicells_fast_r1.common import ID, read

    c, state, folder = fixture(tmp_path)
    first, _, proof, _ = context(c, state, folder, tmp_path / "reference")
    historical = first.curve + [proof["pending_main_curve_row"]]
    x = torch.tensor([[7, 8, 9, 10]])
    c["admission"].update(
        manifest_sha256="fixture", ordered_input_hashes=[sha(x.squeeze(0).tolist())]
    )
    bundle = {"role": "admission", "manifest_sha256": "fixture", "inputs": [x]}
    root = tmp_path / "diagnostic"
    ident = {"scope": "CPU_SYNTHETIC_ONLY", "seed": 7001, "arm": "B11-H"}
    parent = {"checkpoint_sha256": "fixture", "identity": {"scope": "CPU_SYNTHETIC_ONLY"}}
    decision = diagnostic.execute_context(
        c, state, parent, folder, bundle, root, ident, historical, unit=True
    )
    assert len(decision["attempts"]) == 22
    assert diagnostic.totals(root)["main_replay_updates"] == 50
    assert diagnostic.totals(root)["cell_optimizer_calls"] == 50
    assert diagnostic.totals(root)["candidate_optimizer_calls"] == 2
    monkeypatch.setattr(
        replay, "make_engine", lambda *a, **k: pytest.fail("Completed run must not retrain")
    )
    assert (
        diagnostic.execute_context(
            c, state, parent, folder, bundle, root, ident, historical, unit=True
        )
        == decision
    )
    manifest = {"experiment_id": ID, "scope": diagnostic.SCOPE, "run_id": "a" * 32}
    publication.finalize(root, manifest, False)
    _, files, _, _ = publication.payload(root)
    assert "contexts/7001-B11-H/REPLAY_AUDIT.json" in files
    assert not any(name.endswith(".pt") or name.startswith("private/") for name in files)
    assert read(root / "contexts/7001-B11-H/REPLAY_AUDIT.json")["exact_same_environment_restore"]


def test_partial_attempt_accounting_blocks_uncertain_candidate(tmp_path):
    from minicells_fast_r1 import diagnostic

    identity = {"scope": "CPU_SYNTHETIC_ONLY"}
    path = tmp_path / "private/seed-arm/candidate.jsonl"
    journal = Journal(path, identity)
    journal.append("OPT_BEGIN", {"id": "unknown", "optimizer": "connection", "gradient_norm": 1.0})
    with pytest.raises(RuntimeError, match="BUDGET_OR_UNCERTAIN"):
        diagnostic.totals(tmp_path)
