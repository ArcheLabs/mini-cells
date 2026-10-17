"""Synthetic CPU and FakeHub only. No public data, credentials or GPU."""

from copy import deepcopy
import pytest
from minicells_runtime.store import read, canonical, digest, ContinuationError
from minicells_runtime import r3 as reused
from minicells_artifacts.hf_storage import HFCheckpointStore
from test_hf_storage import FakeHub
from clm001_r3b import runner, analysis_runner as ar
from clm001_r3b.diagnostic_schedule import load_plan, validate
from clm001_r3b.analysis_store import AnalysisStore
from clm001_r3b.analysis_aggregate import aggregate_tasks


def frozen_input(case, tmp_path):
    _, c, root = case
    runner.segment(root, "Q-R3b", 1)
    hub = FakeHub(tmp_path / "hub")
    index = (
        HFCheckpointStore(hub).publish_committed_checkpoint(root)
        if (root / "continuation.json").exists()
        else None
    )
    if index is None:
        runner.package(root)
        index = HFCheckpointStore(hub).publish_committed_checkpoint(root)
    return root, c, hub, index["checkpoint_ref"]


def task(root, ref, c, stage="quality", role="development", policies=None):
    return ar.identity(
        root,
        ref,
        load_plan(),
        reused.source_sha(),
        "Q-R3b",
        0,
        0,
        "post_event",
        stage,
        role,
        (["greedy", "uniform"] if policies is None else policies) if stage == "quality" else [],
    )


@pytest.mark.parametrize("arm", ["Q-R3b", "QC-R3b", "QS-R3b"])
def test_deferred_full_state_and_mandatory_events_equal(case, tmp_path, monkeypatch, arm):
    proposed, c, root = case
    import shutil

    assets = tmp_path / "stable-assets"
    shutil.copytree(root / "assets", assets)
    shutil.rmtree(root)
    c["r3b"]["diagnostic_execution"] = load_plan()
    runner.new_run(proposed, c, assets, root)
    runner.segment(root, arm, 10, observation_reference=True)
    before = root / "arms" / arm / read(root / "arms" / arm / "latest.json")["file"]
    other = tmp_path / "deferred"
    runner.new_run(proposed, c, root / "assets", other)
    monkeypatch.setattr(
        runner, "snapshot", lambda *a, **kw: pytest.fail("heavy diagnosis on training path")
    )
    runner.segment(other, arm, 10)
    after = other / "arms" / arm / read(other / "arms" / arm / "latest.json")["file"]
    assert reused.state_fingerprint(before) == reused.state_fingerprint(after)
    assert not list((other / "arms" / arm / "diagnostics").glob("snapshot-*"))
    rows = [read(p) for p in (other / "arms" / arm / "telemetry").glob("*.json")]
    assert rows and all(r["extra_model_forwards"] == 0 for r in rows)
    from torch import load

    events = load(after, map_location="cpu", weights_only=False)["extra"]["events"]
    assert events  # Real original synthetic proposal/admission/promotion operations executed.
    left = sorted(p.name for p in before.parent.glob("checkpoint-step-*.pt"))
    right = sorted(p.name for p in after.parent.glob("checkpoint-step-*.pt"))
    assert left == right
    assert any("-event-" in name for name in left)
    disabled = tmp_path / "disabled"
    runner.new_run(proposed, c, root / "assets", disabled)
    monkeypatch.setattr(runner, "snapshot", lambda *a, **kw: None)
    runner.segment(disabled, arm, 10, observation_reference=True)
    off = disabled / "arms" / arm / read(disabled / "arms" / arm / "latest.json")["file"]
    assert reused.state_fingerprint(before) == reused.state_fingerprint(off)


def test_posthoc_sample_level_numerical_equivalence(case, tmp_path, monkeypatch):
    root, c, hub, ref = frozen_input(case, tmp_path)
    from clm001_r3.report import verified

    old = next(
        verified(p)
        for p in (root / "arms/Q-R3b/diagnostics").glob("snapshot-*post_event*.json")
        if not p.name.endswith(".sha256.json")
    )
    # This test must fail on any optimizer update, including private Shadow updates.
    monkeypatch.setattr(
        ar.Trainer, "update", lambda *a, **kw: pytest.fail("analysis updated optimizer")
    )
    policies = ["greedy", "sampled", "uniform", "constant"]
    value = ar.execute(root, task(root, ref, c, policies=policies), load_plan())
    assert value["result"]["quality"] == old["quality"]
    for role in ("calibration", "development"):
        value = ar.execute(root, task(root, ref, c, "functional", role), load_plan())
        # Includes every private story/state, not only the aggregate mean.
        assert value["result"]["functional"] == old["functional"][role]
    value = ar.execute(root, task(root, ref, c, "action", "calibration"), load_plan())
    assert value["result"]["router"] == old["router"]
    assert value["result"]["action_type_calibration"] == old["action_type_calibration"]
    assert value["training_state_unchanged"] and value["optimizer_updates"] == 0


def test_analysis_remote_reentry_and_cas_does_not_touch_training(case, tmp_path, monkeypatch):
    root, c, hub, ref = frozen_input(case, tmp_path)
    training_pointer = read(hub.get(ref["run_key"] + "/latest.json", hub.head))
    t = task(root, ref, c)
    store = AnalysisStore(hub)
    results, _ = ar.batch(root, tmp_path / "session-A", [t], load_plan(), store, max_tasks=1)
    monkeypatch.setattr(ar, "execute", lambda *a, **kw: pytest.fail("completed task recomputed"))
    recovered, ledger = ar.batch(
        root, tmp_path / "session-B", [t], load_plan(), AnalysisStore(hub), max_tasks=1
    )
    assert recovered == results and ledger[0]["reused"]
    assert read(hub.get(ref["run_key"] + "/latest.json", hub.head)) == training_pointer
    conflict = deepcopy(results[0])
    conflict["elapsed_seconds"] += 1
    with pytest.raises(ContinuationError, match="RESULT_CONFLICT"):
        store.commit(conflict)
    other = deepcopy(t)
    other["analysis_source_sha"] = "a" * 40
    assert store.completed(other) is None


def test_analysis_budget_identity_failure_and_private_index(case, tmp_path):
    root, c, hub, ref = frozen_input(case, tmp_path)
    t = task(root, ref, c)
    store = AnalysisStore(hub)
    results, ledger = ar.batch(root, tmp_path / "none", [t], load_plan(), store, max_tasks=0)
    assert not results and ledger[0]["status"] == "SKIPPED_BUDGET"
    bad = deepcopy(t)
    bad["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ContinuationError, match="IDENTITY_MISMATCH"):
        ar.execute(root, bad, load_plan())
    plan = load_plan()
    plan["posthoc"]["quality_primary_steps"] = [0, 500]
    with pytest.raises(ContinuationError, match="IDENTITY_MISMATCH"):
        ar.execute(root, t, plan)
    with pytest.raises(ContinuationError, match="ROLE_OR_STAGE"):
        ar.identity(
            root,
            ref,
            load_plan(),
            reused.source_sha(),
            "Q-R3b",
            0,
            0,
            "post_event",
            "quality",
            "confirmation",
            ["greedy"],
        )
    hub.private = False
    with pytest.raises(ContinuationError, match="PRIVATE_REPOSITORY"):
        store.completed(t)


def test_partial_missing_observations_and_figures(case, tmp_path):
    root, c, hub, ref = frozen_input(case, tmp_path)
    t = task(root, ref, c)
    result = ar.execute(root, t, load_plan())
    t2 = deepcopy(t)
    t2["arm"] = "QC-R3b"
    values = aggregate_tasks(c, read(root / "continuation.json"), [t, t2], [result], [])
    assert values["experiment-manifest.json"]["analysis_status"] == "PARTIAL"
    assert values["statistical-contrast.json"]["rows"][0]["status"] == "PARTIAL"
    assert values["experiment-manifest.json"]["missing"][0]["status"] == "NOT_EVALUATED"
    from clm001_r3b.publication import reject_private_payload

    for v in values.values():
        reject_private_payload(v)
    from clm001_r3b.plots import render_analysis

    render_analysis(values, tmp_path / "figures", load_plan())
    assert len(list((tmp_path / "figures").glob("*.png"))) == 2


def test_plan_schema_and_step400_selection():
    plan = load_plan()
    validate(plan)
    plan["posthoc"]["quality_event_compare_at_400"] = ["post_event"]
    with pytest.raises(ContinuationError, match="EVENT_COMPARE"):
        validate(plan)


def test_corrupt_result_and_cas_retry(case, tmp_path):
    root, c, hub, ref = frozen_input(case, tmp_path)
    t = task(root, ref, c)
    result = ar.execute(root, t, load_plan())
    store = AnalysisStore(hub)
    hub.conflict_once = True
    entry = store.commit(result)
    assert store.completed(t) == result
    hub.get(entry["hf_path"], entry["hf_revision"]).write_text("{}")
    with pytest.raises(ContinuationError, match="HASH_MISMATCH"):
        store.completed(t)


def test_exact_event_revision_and_bank_tamper(case, tmp_path):
    _, c, root = case
    runner.segment(root, "QC-R3b", 10)
    meta = runner.package(root)
    hub = FakeHub(tmp_path / "hub")
    ref = HFCheckpointStore(hub).publish_committed_checkpoint(root)["checkpoint_ref"]
    event_step = c["shadow"]["admission_step"]  # Synthetic scaling of production Step 400.
    from minicells_runtime.store import MountedInputStore

    pre = MountedInputStore(root).select("QC-R3b", event_step, 0)
    post = MountedInputStore(root).select("QC-R3b", event_step)
    assert pre["revision"] == 0 and post["revision"] > 0
    assert pre["checkpoint_sha256"] != post["checkpoint_sha256"]
    a = ar.identity(
        root,
        ref,
        load_plan(),
        reused.source_sha(),
        "QC-R3b",
        event_step,
        0,
        "pre_event",
        "quality",
        "development",
        ["greedy", "uniform"],
    )
    b = ar.identity(
        root,
        ref,
        load_plan(),
        reused.source_sha(),
        "QC-R3b",
        event_step,
        post["revision"],
        "post_event",
        "quality",
        "development",
        ["greedy", "uniform"],
    )
    assert canonical(a) != canonical(b)
    with pytest.raises(ContinuationError, match="EXACT_EVENT_PHASE"):
        ar.identity(
            root,
            ref,
            load_plan(),
            reused.source_sha(),
            "QC-R3b",
            event_step,
            0,
            "post_event",
            "quality",
            "development",
            ["greedy"],
        )
    assert meta["arms"]["QC-R3b"]["step"] == 10
    (root / "arms/QC-R3b/development-probe-bank.pt").write_bytes(b"corrupt")
    with pytest.raises(ContinuationError, match="HASH_MISMATCH|PROBE_MISMATCH"):
        ar.identity(
            root,
            ref,
            load_plan(),
            reused.source_sha(),
            "QC-R3b",
            0,
            0,
            "post_event",
            "quality",
            "development",
            ["greedy"],
        )


def test_analysis_outage_does_not_downgrade_training(case, tmp_path):
    root, c, hub, ref = frozen_input(case, tmp_path)
    t = task(root, ref, c)
    before = digest(root / "continuation.json")
    hub.fail = True
    with pytest.raises(PermissionError):
        ar.batch(root, tmp_path / "outage", [t], load_plan(), AnalysisStore(hub), max_tasks=1)
    assert read(tmp_path / "outage/analysis-index.json")["status"] == "ANALYSIS_PARTIAL"
    assert digest(root / "continuation.json") == before
    hub.fail = False
    results, _ = ar.batch(
        root, tmp_path / "retry", [t], load_plan(), AnalysisStore(hub), max_tasks=1
    )
    assert len(results) == 1
