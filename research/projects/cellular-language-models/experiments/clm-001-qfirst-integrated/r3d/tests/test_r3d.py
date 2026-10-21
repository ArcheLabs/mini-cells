import random
import numpy as np
import pytest
import torch
from minicells_r3d.common import configuration, canonical, read, BASE, fingerprint, identity
from minicells_r3d.connection import PathModel
from minicells_r3d.training import Trainer
from minicells_r3d.counterfactual import collect, preference_loss
from minicells_r3d import runner
from minicells_r3c_moe_bridge.data import synthetic_batches
from minicells_r3c_moe_bridge.model import BridgeModel
from minicells_r3c_moe_bridge.checkpoint import Store


def parent(c, seed):
    return {"model": BridgeModel(c["engine"], "D", seed).state_dict()}


def test_frozen_config_and_actual_parameter_count():
    c = configuration()
    model = PathModel(c, 151001301)
    assert len(model.connection.pairs) == 15
    assert sum(p.numel() for p in model.connection.parameters()) == 1995
    assert sum(p.numel() for p in model.parameters()) == 1007825
    assert canonical(c) == read(BASE / "config/releases.lock")["config_sha256"]
    assert c["training"]["planned_job_steps"] == 5700 and c["training"]["j_policy_updates"] == 375


def test_zero_correction_exact_d_parity_and_ties(config):
    p = parent(config, 11)
    m = runner.build_model(config, p, 11, "cpu")
    x = synthetic_batches(config["engine"], np.random.default_rng(1))[0][0]
    assert runner.initial_parity(m, config, x, p)["exact_logits"]
    z = torch.zeros(10, 6)
    a, _ = m.connection.choose(torch.zeros(10, 8), 0, z)
    assert torch.equal(m.connection.pairs[a], torch.tensor([[0, 1]]).repeat(10, 1))
    out = m(x)
    for t, s in enumerate(out.stages):
        action = (
            (s.indices.sort(-1).values[:, None, :] == m.connection.pairs).all(-1).long().argmax(-1)
        )
        h = m.advance(s.inputs, t, action)
        expected = out.stages[t + 1].inputs if t < 3 else out.states.reshape(-1, 8)
        torch.testing.assert_close(h, expected, rtol=0, atol=0)


@pytest.mark.parametrize("arm", ["L", "M", "F", "P", "J"])
def test_gradient_boundaries_and_finite_updates(config, arm):
    tr = Trainer(PathModel(config, 11), config, arm)
    before = {n: p.detach().clone() for n, p in tr.model.named_parameters()}
    row = tr.update(synthetic_batches(config["engine"], tr.data_rng))
    assert np.isfinite(row["ce"]) and row["tokens"] == 8
    tr.verify_frozen()
    for n, p in tr.model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(p, before[n])
    assert any(v > 0 for v in row["cell_grad_norms"]) == (arm in ("F", "P", "J"))
    changed = any(
        not torch.equal(p, before[n])
        for n, p in tr.model.named_parameters()
        if n.startswith("connection.")
    )
    assert changed == (arm in ("L", "M", "J"))
    assert (
        row["connection_parameter_update_norm"]
        <= config["training"]["maximum_parameter_update_norm"] + 1e-7
    )
    if arm in ("L", "M", "J"):
        assert row["counterfactual_snapshot"]["cell_step"] == 0


def test_short_and_long_reroute_and_no_parameter_or_rng_changes(config, monkeypatch):
    m = PathModel(config, 11)
    b = synthetic_batches(config["engine"], np.random.default_rng(11))
    x, y = b[0]
    out = m(x)
    before = fingerprint(m.state_dict())
    rng = torch.get_rng_state().clone()
    stages = []
    advance = m.advance

    def tracked(h, t, action=None):
        stages.append((t, action is None))
        return advance(h, t, action)

    monkeypatch.setattr(m, "advance", tracked)
    records, work = collect(m, [(out, y)], torch.Generator().manual_seed(31), 4, "long")
    assert {r["stage"] for r in records} == {0, 1, 2, 3} and any(
        replanned for t, replanned in stages
    )
    assert all(len(r["candidates"]) <= 3 for r in records) and work["counterfactual_cell_rows"] > 0
    short, _ = collect(m, [(out, y)], torch.Generator().manual_seed(31), 4, "short")
    assert any(not torch.equal(a["values"], b["values"]) for a, b in zip(records, short))
    assert fingerprint(m.state_dict()) == before and torch.equal(torch.get_rng_state(), rng)


def test_joint_counterfactual_uses_pre_update_cells(config, monkeypatch):
    from minicells_r3d import training

    tr = Trainer(PathModel(config, 11), config, "J")
    before = fingerprint(tr.model.cells.state_dict())
    collect_fn = training.collect

    def checked(model, *a, **k):
        assert fingerprint(model.cells.state_dict()) == before
        assert fingerprint(tr.model.cells.state_dict()) != before
        return collect_fn(model, *a, **k)

    monkeypatch.setattr(training, "collect", checked)
    tr.update(synthetic_batches(config["engine"], tr.data_rng))


def test_preference_target_is_detached_and_sampled_only(config):
    m = PathModel(config, 11)
    m.set_trainable("L")
    x, y = synthetic_batches(config["engine"], np.random.default_rng(1))[0]
    records, _ = collect(m, [(m(x), y)], torch.Generator().manual_seed(2), 4, "long")
    loss, stats = preference_loss(m, records, 0.1)
    loss.backward()
    assert stats["scored_pairs"] > 0
    assert all(p.grad is None for p in m.cells.parameters())
    assert any(p.grad is not None for p in m.connection.parameters())
    assert all(not r["values"].requires_grad for r in records)


def test_exact_full_state_restart_and_same_data_sequence(config, tmp_path):
    def initialize():
        random.seed(20)
        np.random.seed(20)
        torch.manual_seed(20)
        return Trainer(PathModel(config, 11), config, "J")

    baseline = initialize()
    a = synthetic_batches(config["engine"], baseline.data_rng)
    baseline.update(a)
    store = Store(tmp_path, {"run": "synthetic"})
    row = store.save(baseline, observations={"curve": []})
    next_batch = synthetic_batches(config["engine"], baseline.data_rng)
    expected = baseline.update(next_batch)
    restored = initialize()
    restored.restore(store.verified(row))
    got = synthetic_batches(config["engine"], restored.data_rng)
    assert all(
        torch.equal(a, b) for pair, other in zip(next_batch, got) for a, b in zip(pair, other)
    )
    actual = restored.update(got)
    assert (
        actual["ce"] == expected["ce"] and actual["connection_loss"] == expected["connection_loss"]
    )
    assert fingerprint(restored.state()) == fingerprint(baseline.state())
    with pytest.raises(ValueError):
        restored.update(got, role="development")


def test_causality_and_target_free_connection(config):
    m = PathModel(config, 11)
    x, y = synthetic_batches(config["engine"], np.random.default_rng(10))[0]
    changed = x.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 16
    torch.testing.assert_close(m(x).logits[:, :-1], m(changed).logits[:, :-1], rtol=0, atol=0)
    with pytest.raises(TypeError):
        m(x, targets=y)


def test_runner_completes_all_arms_and_reuses_without_training(config, tmp_path, monkeypatch):
    def windows(data, c, role, device):
        assert role in ("calibration", "development")
        for i in range(3):
            x, y = synthetic_batches(c, np.random.default_rng(i + 1), device)[0]
            yield str(i), x, y

    import minicells_r3d.evaluation as ev

    monkeypatch.setattr(ev, "fixed_windows", windows)
    monkeypatch.setattr(
        runner, "train_batches", lambda data, c, rng, device: synthetic_batches(c, rng, device)
    )
    config["seeds"] = [11]
    results = {}
    for arm in config["arms"]:
        learned = None
        if arm in ("P", "J"):
            learned = {
                k[len("connection.") :]: v
                for k, v in results["L"]["state"]["model"].items()
                if k.startswith("connection.")
            }
        r = runner.job(config, {}, tmp_path, "source", 11, arm, parent(config, 11), "cpu", learned)
        d = tmp_path / "private-checkpoints/jobs" / ("seed-11-arm-" + arm)
        results[arm] = {
            "result": r,
            "state": Store(d, identity(config, "source", 11, arm)).verified(r["checkpoint"]),
        }
    report = runner.gather(config, tmp_path, "source")
    assert (
        report["status"] == "EXPLORATORY_DEVELOPMENT_COMPLETE" and report["actual_job_steps"] == 16
    )
    assert report["confirmation"] == "SEALED"
    monkeypatch.setattr(
        Trainer, "update", lambda *a, **k: pytest.fail("repeated completed training")
    )
    assert (
        runner.job(config, {}, tmp_path, "source", 11, "J", parent(config, 11), "cpu", learned)
        == results["J"]["result"]
    )


def test_automatic_transport_and_fresh_directory_resume_without_retraining(
    config, tmp_path, monkeypatch
):
    from pathlib import Path
    from types import SimpleNamespace
    import shutil
    import json
    import minicells_r3d.evaluation as ev

    class Hub:
        repo = "private/test"
        commits = 0

        def info(self, revision=None):
            return SimpleNamespace(private=True, sha="revision")

        def check_write(self):
            pass

        def get(self, name, revision):
            return tmp_path / "hub" / name

        def optional(self, name, revision):
            path = self.get(name, revision)
            return json.loads(path.read_text()) if path.exists() else None

        def commit(self, files, parent, message):
            self.commits += 1
            for name, path in files.items():
                target = self.get(name, "revision")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            return "revision"

    hub = Hub()
    config["seeds"] = [11, 12]

    def windows(data, c, role, device):
        assert role in ("calibration", "development")
        for i in range(2):
            x, y = synthetic_batches(c, np.random.default_rng(1 + i), device)[0]
            yield str(i), x, y

    monkeypatch.setattr(ev, "fixed_windows", windows)
    monkeypatch.setattr(runner, "load_secrets", lambda: None)
    monkeypatch.setattr(runner, "HubBackend", lambda repo: hub)
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(runner, "prepare_pilot_publication", lambda *a: None)
    monkeypatch.setattr(
        runner, "public_publish", lambda report, prepared: {"branch": "results/synthetic"}
    )
    monkeypatch.setattr(runner, "load_assets", lambda *a: {})
    monkeypatch.setattr(runner, "restore_parents", lambda *a: None)
    monkeypatch.setattr(
        runner, "train_batches", lambda data, c, rng, device: synthetic_batches(c, rng, "cpu")
    )

    def workers(c, path, output, assets, source, callback):
        for seed in c["seeds"]:
            learned = None
            for arm in c["arms"]:
                runner.job(
                    c,
                    {},
                    output,
                    source,
                    seed,
                    arm,
                    parent(c, seed),
                    "cpu",
                    learned if arm in ("P", "J") else None,
                )
                if arm == "L":
                    root = Path(output) / "private-checkpoints/jobs" / ("seed-%s-arm-L" % seed)
                    r = read(root / "result.json")
                    s = Store(root, identity(c, source, seed, "L")).verified(r["checkpoint"])
                    learned = {
                        k[len("connection.") :]: v
                        for k, v in s["model"].items()
                        if k.startswith("connection.")
                    }
            callback(seed)
        return [0, 0]

    monkeypatch.setattr(runner, "run_workers", workers)
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "manifest.json").write_text("{}")
    one = runner.automatic(config, tmp_path / "config", tmp_path / "session-a", assets)
    assert one["status"] == "EXPLORATORY_DEVELOPMENT_COMPLETE" and hub.commits == 2
    monkeypatch.setattr(
        Trainer, "update", lambda *a, **k: pytest.fail("fresh-session restore retrained")
    )
    two = runner.automatic(config, tmp_path / "config", tmp_path / "session-b", assets)
    assert two["status"] == one["status"] and hub.commits == 2
    assert (tmp_path / "session-b/reports/SCIENTIFIC-FINDINGS.md").exists()


def test_parent_identity_rejects_before_pickle(config, tmp_path, monkeypatch):
    from minicells_r3d import storage
    from minicells_r3d.common import atomic

    directory = tmp_path / "jobs/seed-11-arm-D"
    directory.mkdir(parents=True)
    config["parent"]["checkpoint_sha256"]["11"] = "expected"
    atomic(directory / "latest.json", {"step": 499, "sha256": "expected"})
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("unverified parent deserialized")
    )
    with pytest.raises(RuntimeError, match="PARENT_D_STEP500"):
        storage.parent_state(config, tmp_path, 11)
