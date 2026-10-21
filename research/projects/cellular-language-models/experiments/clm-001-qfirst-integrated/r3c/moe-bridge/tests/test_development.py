from copy import deepcopy
import random
import numpy as np
import pytest
import torch
from minicells_r3c_moe_bridge import development as dev
from minicells_r3c_moe_bridge.common import atomic, fingerprint, read
from minicells_r3c_moe_bridge.data import synthetic_batches
from minicells_r3c_moe_bridge.model import BridgeModel
from minicells_r3c_moe_bridge.training import Trainer
from minicells_r3c_moe_bridge.checkpoint import Store, publish_segment
from minicells_r3c_moe_bridge.evaluation import evaluate
from test_state_and_storage import FakeHub


@pytest.fixture
def tiny(config, monkeypatch):
    c = deepcopy(config)
    real = dev.configuration()
    c["balance"]["coefficient"] = 0.0
    c["development"] = deepcopy(real["development"])
    c["development"].update(
        steps=4,
        seeds=[101, 102, 103],
        checkpoint_steps=[0, 2, 4],
        recovery_interval=2,
        print_interval=2,
        reuse_pilot_seed=False,
    )
    c["statistics"]["bootstrap_replicates"] = 30
    monkeypatch.setattr(
        dev, "train_batches", lambda data, c, rng, device: synthetic_batches(c, rng, device)
    )

    def windows(data, c, role, device):
        assert role in ("calibration", "development")
        for i in range(3):
            x, y = synthetic_batches(c, np.random.default_rng(i + 19), device)[0]
            yield str(i), x, y

    monkeypatch.setattr(dev, "fixed_windows", windows)
    return c


def test_development_release_bound_and_science_not_released():
    c = dev.configuration()
    d = c["development"]
    assert d["steps"] == 500 and len(d["seeds"]) == 3 and d["aliases"] == {"E": "C"}
    assert d["new_update_upper_bound"] == 6000 and c["balance"]["coefficient"] == 0
    assert not c["release"]["formal_development_authorized"] and not c["scientific_decision"]
    assert c["statistics"]["minimum_practical_delta_nll"] is None


@pytest.mark.parametrize("arm", list("ABCD"))
def test_development_full_job_and_complete_reuse(tiny, tmp_path, monkeypatch, arm):
    result = dev.train_job(tiny, {}, tmp_path, "source", 101, arm, "cpu")
    assert result["step"] == 4 and result["train_tokens"] == 32
    assert set(result["quality"]) == {"calibration", "development"}
    monkeypatch.setattr(
        Trainer, "update", lambda *a, **k: pytest.fail("completed job trained again")
    )
    assert dev.train_job(tiny, {}, tmp_path, "source", 101, arm, "cpu") == result


@pytest.mark.parametrize("arm", ["B", "C"])
def test_development_resume_exact_and_no_intermediate_development(tiny, tmp_path, monkeypatch, arm):
    baseline = dev.train_job(tiny, {}, tmp_path / "base", "source", 101, arm, "cpu")
    old = dev.timed_update

    def fail(tr, batches, device):
        if tr.step == 2:
            raise RuntimeError("interrupted")
        return old(tr, batches, device)

    monkeypatch.setattr(dev, "timed_update", fail)
    calls = []
    evaluator = dev.evaluate
    monkeypatch.setattr(dev, "evaluate", lambda *a, **k: calls.append(1) or evaluator(*a, **k))
    with pytest.raises(RuntimeError):
        dev.train_job(tiny, {}, tmp_path / "resume", "source", 101, arm, "cpu")
    assert calls == []
    monkeypatch.setattr(dev, "timed_update", old)
    resumed = dev.train_job(tiny, {}, tmp_path / "resume", "source", 101, arm, "cpu")
    assert resumed["checkpoint"]["state_fingerprint"] == baseline["checkpoint"]["state_fingerprint"]
    assert resumed["quality"] == baseline["quality"] and len(calls) == 2


def test_zero_balance_c_e_exact_training_equivalence(tiny):
    states = []
    for arm in ("C", "E"):
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        tr = Trainer(BridgeModel(tiny, arm, 101))
        for _ in range(4):
            tr.update(synthetic_batches(tiny, tr.data_rng))
        states.append(fingerprint(tr.state()))
    assert states[0] == states[1]


def test_stop_request_pauses_at_checkpoint_without_evaluation(tiny, tmp_path, monkeypatch):
    old = dev.timed_update

    def request(tr, batches, device):
        row = old(tr, batches, device)
        if tr.step == 1:
            atomic(tmp_path / "STOP_AFTER_CHECKPOINT.json", {"reason": "other worker failed"})
        return row

    monkeypatch.setattr(dev, "timed_update", request)
    monkeypatch.setattr(dev, "evaluate", lambda *a, **k: pytest.fail("unfinished model evaluated"))
    result = dev.train_job(tiny, {}, tmp_path, "source", 101, "C", "cpu")
    assert result["status"] == "PAUSED_AT_COMPLETE_CHECKPOINT" and result["step"] == 2


def test_story_windows_aggregate_before_bootstrap(tiny):
    m = BridgeModel(tiny, "C", 101)
    x, y = synthetic_batches(tiny, np.random.default_rng(19))[0]
    row = evaluate(m, [("same", x, y), ("same", x, y)], 1)["same"]
    assert row["tokens"] == 2 * x.numel() and row["windows"] == 2


def test_archive_resume_fetches_latest_only_and_preserves_old_references(tiny, tmp_path):
    dev.train_job(tiny, {}, tmp_path / "original", "source", 101, "C", "cpu")
    private = tmp_path / "original/private-checkpoints"
    export = dev.export_checkpoints(tiny, private, tmp_path / "export")
    assert len(list(export.rglob("*.pt"))) == 3
    hub = FakeHub(tmp_path / "hub")
    ident = {"run": "test"}
    ref = publish_segment(hub, export, ident, "writer", pointer_path="dev/latest.json")
    manifest = dev.restore_latest(
        hub, ref, tmp_path / "restored/private-checkpoints", ident, "reader"
    )
    assert len(list((tmp_path / "restored").rglob("*.pt"))) == 1
    resumed = dev.train_job(tiny, {}, tmp_path / "restored", "source", 101, "C", "cpu")
    export2 = dev.export_checkpoints(
        tiny, tmp_path / "restored/private-checkpoints", tmp_path / "export2"
    )
    ref2 = publish_segment(hub, export2, ident, "reader", prior_manifest=manifest)
    archived = dev.verified_manifest(hub, ref2, ident)
    assert set(manifest["files"]) <= set(archived["files"])
    assert sum(name.endswith(".pt") for name in archived["files"]) == 3
    assert resumed["step"] == 4


def test_three_seed_summary_aliases_e_and_exposes_no_story_rows(tiny, tmp_path):
    for seed in tiny["development"]["seeds"]:
        for arm in tiny["development"]["physical_arms"]:
            dev.train_job(tiny, {}, tmp_path, "source", seed, arm, "cpu")
    report = dev.collect(tiny, tmp_path, "source")
    public = dev.public_summary(report)
    assert report["status"] == "EXPLORATORY_DEVELOPMENT_COMPLETE"
    assert report["committed_physical_updates"] == 48 and report["aliases"]["E"]["source"] == "C"
    assert report["paired_analysis"]["independent_training_seeds"] == 3
    assert "quality" not in str(public["results"])
    assert all(
        r["comparisons"]["C-E"]["status"] == "STRUCTURALLY_IDENTICAL_NOT_AN_ABLATION"
        for r in report["paired_analysis"]["per_seed"]
    )
    assert not public["scientific_decision"]


def test_archive_rejects_overwriting_immutable_checkpoint(tmp_path):
    hub = FakeHub(tmp_path / "hub")
    root = tmp_path / "first"
    root.mkdir()
    atomic(root / "receipt.json", {"immutable": 1})
    first = publish_segment(hub, root, {"run": "same"}, "a")
    prior = dev.verified_manifest(hub, first, {"run": "same"})
    other = tmp_path / "second"
    other.mkdir()
    atomic(other / "receipt.json", {"immutable": 2})
    with pytest.raises(RuntimeError, match="IMMUTABLE_CONFLICT"):
        publish_segment(hub, other, {"run": "same"}, "b", prior_manifest=prior)


def test_imported_pilot_state_counts_toward_budget_and_replays_exactly(tiny, tmp_path, monkeypatch):
    from minicells_r3c_moe_bridge.runner import identity
    from minicells_r3c_moe_bridge.common import BASE

    c = deepcopy(tiny)
    d = c["development"]
    d.update(steps=8, checkpoint_steps=[0, 4, 8], reuse_pilot_seed=True)
    d["parent_pilot"]["seed"] = 101
    pilot = read(BASE / "config/pilot.json")
    source = d["parent_pilot"]["source_sha"]
    ident = {
        **identity(pilot, "C", 101, source),
        "pilot_job": "arm-C",
        "effective_balance_coefficient": 0.0,
    }
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    tr = Trainer(BridgeModel(c, "C", 101))
    logs = []
    for _ in range(5):
        logs.append(tr.update(synthetic_batches(c, tr.data_rng)))
    directory = tmp_path / "parent/jobs/arm-C"
    store = Store(directory, ident)
    row = store.save(tr)
    d["parent_pilot"]["checkpoint_sha256"]["C"] = row["sha256"]
    atomic(directory / "result.json", {"checkpoint": row, "logs": logs})
    baseline = dev.train_job(c, {}, tmp_path / "baseline", "source", 101, "C", "cpu")
    before = []
    old = dev.timed_update
    monkeypatch.setattr(
        dev,
        "timed_update",
        lambda tr, batches, device: (before.append(tr.step), old(tr, batches, device))[1],
    )
    imported = dev.train_job(
        c, {}, tmp_path / "imported", "source", 101, "C", "cpu", tmp_path / "parent"
    )
    assert before == [5, 6, 7] and imported["pilot_steps_imported"] == 5
    assert (
        imported["checkpoint"]["state_fingerprint"] == baseline["checkpoint"]["state_fingerprint"]
    )
    assert len(imported["curve"]) == 8 and imported["quality"] == baseline["quality"]


def test_automatic_seed_blocks_restore_without_extra_updates_or_uploads(
    tiny, tmp_path, monkeypatch
):
    hub = FakeHub(tmp_path / "hub")
    hub.root.mkdir()
    parent = hub.root / "pilot-manifest.json"
    atomic(
        parent,
        {
            "schema": "minicells.r3c.segment.v1",
            "identity": {"pilot": "old"},
            "writer_session_id": "old",
            "files": {},
        },
    )
    from minicells_r3c_moe_bridge.common import digest

    c = deepcopy(tiny)
    c["input"]["asset_bundle_ref"] = {"hf_repo": hub.repo}
    c["development"]["parent_pilot"]["hf_reference"] = {
        "hf_repo": hub.repo,
        "hf_revision": "old",
        "manifest_path": parent.name,
        "manifest_sha256": digest(parent),
    }
    hub.optional = lambda name, rev: read(hub.root / name) if (hub.root / name).exists() else None
    monkeypatch.setattr(dev, "configuration", lambda: c)
    monkeypatch.setattr(dev, "source_sha", lambda: "f" * 40)
    monkeypatch.setattr(dev, "load_secrets", lambda: None)
    monkeypatch.setattr(dev, "HubBackend", lambda repo: hub)
    monkeypatch.setattr(dev, "private_asset_subset", lambda *a: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    from minicells_r3c_moe_bridge import publication

    monkeypatch.setattr(publication, "prepare_pilot_publication", lambda *a: {})
    monkeypatch.setattr(publication, "publish_aggregate", lambda *a: {"branch": "results/dev"})
    phases = []

    def workers(c, output, assets, seed, count):
        phases.append((seed, count))
        for arm in c["development"]["physical_arms"]:
            dev.train_job(c, {}, output, "f" * 40, seed, arm, "cpu")
        return [0, 0]

    monkeypatch.setattr(dev, "seed_workers", workers)
    result = dev.automatic(tmp_path / "a", tmp_path / "assets")
    assert result["status"] == "EXPLORATORY_DEVELOPMENT_COMPLETE" and hub.commits == 3
    assert all(count == 2 for _, count in phases)
    monkeypatch.setattr(
        Trainer, "update", lambda *a, **k: pytest.fail("repeated completed training")
    )
    again = dev.automatic(tmp_path / "b", tmp_path / "assets")
    assert again["committed_physical_updates"] == 48 and hub.commits == 3


def test_receipt_recovers_observations_after_sidecar_interruption(tiny, tmp_path, monkeypatch):
    original = dev.atomic
    interrupted = [False]

    def write(path, value):
        if path.name == "curve.json" and len(value["rows"]) == 2 and not interrupted[0]:
            interrupted[0] = True
            raise RuntimeError("sidecar write interrupted after committed receipt")
        return original(path, value)

    monkeypatch.setattr(dev, "atomic", write)
    with pytest.raises(RuntimeError, match="sidecar"):
        dev.train_job(tiny, {}, tmp_path, "source", 101, "C", "cpu")
    monkeypatch.setattr(dev, "atomic", original)
    result = dev.train_job(tiny, {}, tmp_path, "source", 101, "C", "cpu")
    assert [row["step"] for row in result["curve"]] == [1, 2, 3, 4]


def test_one_launch_development_notebook_is_pinned_and_has_no_test_phase():
    import ast
    import json
    import subprocess
    from minicells_r3c_moe_bridge.common import BASE, ROOT

    notebook = json.loads((BASE / "notebooks/kaggle-development.ipynb").read_text())
    code = "".join(notebook["cells"][1]["source"])
    ast.parse(code)
    tree = ast.parse(code)
    pin = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SOURCE_SHA" for t in n.targets)
    )
    config = subprocess.check_output(
        ["git", "show", pin + ":" + str(BASE.relative_to(ROOT) / "config/development.json")],
        cwd=ROOT,
        text=True,
    )
    assert json.loads(config)["development"]["steps"] == 500
    assert "'--mode','development'" in code and "pytest" not in code
    assert "kwargs.setdefault('timeout',45)" in code
    assert "'--mode','pilot'" not in code and "ACCEPT_" not in code
