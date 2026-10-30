import copy
import sys
from pathlib import Path

import pytest
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "implementation"))
from epclm001.common import full_logits, model_hash
from epclm001.events import IMPLEMENTATIONS, Event, Runtime, trace
from epclm001.incremental import IncrementalLM, PreallocatedIncrementalLM
from epclm001.validation import engineering_model as fixture


@pytest.mark.parametrize("adapter_class", [IncrementalLM, PreallocatedIncrementalLM])
@pytest.mark.parametrize("kind", ["X3", "X4", "T1"])
@pytest.mark.parametrize("batch", [1, 2])
def test_exact_prefix_and_cache(kind, batch, adapter_class):
    torch.set_num_threads(1)
    model = fixture(kind)
    before = model_hash(model)
    a = adapter_class(model)
    ids = torch.randint(0, 32, (batch, 128))
    ids[:, :2] = torch.tensor([2, 3])  # BOS/EOS IDs are ordinary reference IDs.
    cache = a.init_cache(batch, "cpu", torch.float32, max_context=128)
    max_error = 0.0
    with torch.inference_mode():
        for pos in range(128):
            logits, cache = a.decode_one(ids[:, pos : pos + 1], cache, absolute_position=pos)
            full = full_logits(model, ids[:, : pos + 1])[:, -1]
            max_error = max(max_error, (logits - full).abs().max().item())
            torch.testing.assert_close(logits, full, atol=2e-5, rtol=2e-4)
        full = full_logits(model, ids)
        changed = ids.clone()
        changed[:, 64:] = torch.randint(0, 32, (batch, 64))
        torch.testing.assert_close(
            full[:, :64], full_logits(model, changed)[:, :64], atol=2e-5, rtol=2e-4
        )
    assert a.iteration_updates == batch * 128 * (2 if kind == "T1" else 12)
    assert max_error < 2e-5
    assert model_hash(model) == before
    with pytest.raises(ValueError, match="CONTEXT"):
        a.decode_one(ids[:, :1], cache, absolute_position=128)
    a.reset(cache)
    out, cache = a.prefill(ids[:, :7], cache)
    snap = a.snapshot(cache)
    out, cache = a.prefill(ids[:, 7:33], cache, start_position=7)
    restored = a.restore(snap)
    out2, restored = a.prefill(ids[:, 7:33], restored, start_position=7)
    torch.testing.assert_close(out, out2, atol=0, rtol=0)
    torch.testing.assert_close(out, full_logits(model, ids[:, :33])[:, -1], atol=2e-5, rtol=2e-4)
    bad = copy.deepcopy(snap)
    bad["identity"] = "wrong"
    with pytest.raises(ValueError, match="IDENTITY"):
        a.restore(bad)
    with pytest.raises(ValueError, match="PADDING"):
        a.prefill(ids[:, :1], a.reset(cache), attention_mask=torch.ones(batch, 1))
    if kind != "T1":
        bad = copy.deepcopy(snap)
        keys = list(bad["layers"])
        bad["layers"][keys[0]], bad["layers"][keys[1]] = (
            bad["layers"][keys[1]],
            bad["layers"][keys[0]],
        )
        with pytest.raises(ValueError, match="TAG"):
            a.decode_one(ids[:, 7:8], bad, absolute_position=7)


@pytest.mark.parametrize(
    "layout",
    ["uniform-random", "clustered", "wavefront", "hotspot", "burst", "all-active", "zero-active"],
)
def test_event_parity(layout):
    torch.manual_seed(7)
    state, weight = torch.randn(32, 8), torch.randn(8, 8) * 0.1
    events = trace(32, 4, 0.125, layout, 821)
    runtimes = [
        Runtime(state, weight, implementation=i, tile=8, max_steps=5) for i in IMPLEMENTATIONS
    ]
    for r in runtimes:
        for e in events:
            r.enqueue(e)
    for _ in range(4):
        rows = [r.microstep() for r in runtimes]
        for r, row in zip(runtimes[1:], rows[1:]):
            torch.testing.assert_close(r.state, runtimes[0].state, atol=2e-6, rtol=2e-5)
            for key in ("consumed", "dropped", "active_ids", "event_hash"):
                assert row[key] == rows[0][key]
        if layout == "zero-active":
            assert all(row["physical_updates"] == 0 for row in rows)
        if layout == "all-active":
            assert all(row["physical_updates"] == 32 for row in rows)
    assert runtimes[2].total_physical == runtimes[2].total_logical
    assert runtimes[3].total_physical >= runtimes[3].total_logical
    assert trace(32, 4, 0.125, layout, 821) == events


def test_event_visibility_version_reset_snapshot():
    state, weight = torch.zeros(4, 2), torch.eye(2)
    for impl in IMPLEMENTATIONS:
        r = Runtime(state, weight, implementation=impl, tile=2, max_steps=8)
        r.enqueue(Event("root", 0, 0, ttl=1))
        with pytest.raises(ValueError, match="DUPLICATE"):
            r.enqueue(Event("root", 0, 0))
        first = r.microstep()
        assert first["active_ids"] == [0]
        assert r.state[1].eq(0).all()  # Send cannot be read in same microstep.
        snap = r.snapshot()
        row = r.microstep()
        assert row["active_ids"] == [1]
        after = r.state.clone()
        r.restore(snap)
        assert r.microstep()["event_hash"] == row["event_hash"]
        torch.testing.assert_close(r.state, after, rtol=0, atol=0)
        r.enqueue(Event("stale", 2, 0, version=0))
        assert r.microstep()["dropped"] == ["stale"]
        r.enqueue(Event("read", 3, 1, version=r.versions[1], kind="READ_QUERY"))
        assert r.microstep()["logical_updates"] == 0
        r.enqueue(Event("reset", 4, 0, version=r.versions[0], kind="RESET"))
        assert r.microstep()["physical_updates"] == 0
        torch.testing.assert_close(r.state, state)


def test_cli_and_publication_payload(tmp_path):
    import json
    import subprocess

    from epclm001.common import digest
    from epclm001.publication import payload

    result = subprocess.run(
        [
            sys.executable,
            str(BASE / "run.py"),
            "engineering",
            "--output",
            str(tmp_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["gpu_started"] is False
    manifest = {"experiment_id": "EPCLM-001", "optimizer_updates": 0, "artifact_files": {}}
    (tmp_path / "RUN_MANIFEST.json").write_text(json.dumps(manifest))
    (tmp_path / "private.pt").write_text("must never publish")
    _, files, _, _ = payload(tmp_path)
    assert set(files) == {"RUN_MANIFEST.json"}
    (tmp_path / "PARITY.json").write_text("{}")
    with pytest.raises(ValueError, match="HASH"):
        payload(tmp_path)
    manifest["artifact_files"]["PARITY.json"] = {
        "bytes": 2,
        "sha256": digest(tmp_path / "PARITY.json"),
    }
    (tmp_path / "RUN_MANIFEST.json").write_text(json.dumps(manifest))
    assert "PARITY.json" in payload(tmp_path)[1]


def test_round1_scope_never_loads_30m(tmp_path, monkeypatch):
    import json
    import subprocess

    from epclm001 import assets

    config = json.loads((BASE / "protocol.json").read_text())
    assert set(config["active_baselines"]) == set(assets.MODELS) == {"E0B_T1", "E0B_X3", "E0B_X4"}
    assert config["deferred_baselines"]["E0A_30M"]["execution_allowed"] is False
    (tmp_path / "minicells-30m-v0-fp16.pt").write_bytes(
        b"not a real checkpoint; must not be opened"
    )
    with pytest.raises(ValueError, match="OUTSIDE_ROUND1"):
        assets.load_native(tmp_path / "minicells-30m-v0-fp16.pt", "E0A_CLM")

    def forbidden(*args, **kwargs):
        raise AssertionError("30M must never reach any model loader")

    monkeypatch.setattr(assets, "load_native", forbidden)
    audit = assets.inventory([tmp_path])
    deferred = next(x for x in audit["assets"] if x["baseline_id"] == "E0A_30M")
    assert deferred["execution_allowed"] is False
    assert deferred["loadable"] is None
    result = subprocess.run(
        [
            sys.executable,
            str(BASE / "run.py"),
            "bench-incremental",
            "--model",
            "E0A_CLM",
            "--output",
            str(tmp_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    notebook = (BASE / "epclm001_cpu.ipynb").read_text()
    assert "007-minicells-30m" not in notebook


@pytest.mark.parametrize("adapter_class", [IncrementalLM, PreallocatedIncrementalLM])
@pytest.mark.parametrize("batch", [1, 2])
def test_t1_parallel_prefill_and_chunk_cache(adapter_class, batch, monkeypatch):
    model = fixture("T1")
    adapter = adapter_class(model)
    ids = torch.arange(128).remainder(32).repeat(batch, 1)
    cache = adapter.init_cache(batch, "cpu", torch.float32, max_context=128)

    # A serial decode fallback must not be used by Transformer prefill.
    def forbidden(*args, **kwargs):
        raise AssertionError("T1 prefill must process whole chunks")

    monkeypatch.setattr(adapter, "decode_one", forbidden)
    with torch.inference_mode():
        out, cache = adapter.prefill(ids[:, :7], cache)
        torch.testing.assert_close(out, full_logits(model, ids[:, :7])[:, -1], atol=2e-5, rtol=2e-4)
        snap = adapter.snapshot(cache)
        out, cache = adapter.prefill(ids[:, 7:], cache, start_position=7)
        torch.testing.assert_close(out, full_logits(model, ids)[:, -1], atol=2e-5, rtol=2e-4)
        replay = adapter.restore(snap)
        out2, replay = adapter.prefill(ids[:, 7:], replay, start_position=7)
        torch.testing.assert_close(out, out2, atol=0, rtol=0)
        other = adapter_class(model)
        sequential = other.init_cache(batch, "cpu", torch.float32, max_context=128)
        for pos in range(128):
            _, sequential = other.decode_one(
                ids[:, pos : pos + 1], sequential, absolute_position=pos
            )
        for key in cache["layers"]:
            for field in ("k", "v"):
                torch.testing.assert_close(
                    cache["layers"][key][field],
                    sequential["layers"][key][field],
                    atol=2e-5,
                    rtol=2e-4,
                )


@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_performance_mode_excludes_diagnostics(impl, monkeypatch):
    from epclm001 import events

    state = torch.zeros(8, 4)
    r = Runtime(state, torch.eye(4), implementation=impl, tile=4, max_steps=2, mode="performance")
    r.enqueue(Event("start", 0, 0, ttl=1))

    def forbidden(*args, **kwargs):
        raise AssertionError("Diagnostic function called in performance mode")

    with monkeypatch.context() as patch:
        patch.setattr(events, "tensor_hash", forbidden)
        patch.setattr(events, "sha", forbidden)
        patch.setattr(events.time, "perf_counter", forbidden)
        patch.setattr(torch.cuda, "synchronize", forbidden)
        assert r.microstep() is None
        assert r.microstep() is None
    assert r.history == []
    assert r.total_logical == 2
    assert r.state[0].ne(0).any() and r.state[1].ne(0).any()


def test_event_measurement_pass_is_separate():
    from epclm001.benchmark import event_matrix

    matrix = {
        "seed": 1,
        "warmup": 0,
        "repeats": 1,
        "event_points": [
            {"cells": 16, "dim": 4, "steps": 2, "tile": 4, "ratio": 0.25, "layout": "clustered"}
        ],
    }
    rows = event_matrix(matrix)
    assert len(rows) == 4
    assert all(r["microsteps_origin"] == "SEPARATE_UNTIMED_CORRECTNESS_PASS" for r in rows)
    assert all(r["measurement_mode"] == "performance" and r["parity_passed"] for r in rows)
    assert all(r["device_stream_elapsed_ms"] is None for r in rows)


def test_asset_header_discovery_and_unknown_source(tmp_path, monkeypatch):
    import json

    from epclm001 import assets

    path = tmp_path / "opaque-name.ckpt"
    torch.save({"model": assets.MODELS["E0B_T1"], "step": 100, "seed": 91002}, path)
    assert assets.checkpoint_models(path) == {assets.MODELS["E0B_T1"]}
    model = fixture("T1")
    monkeypatch.setattr(
        assets,
        "load_native",
        lambda *args: (model, {"seed": 91002, "step": 100, "profile": "baseline"}),
    )
    row = next(r for r in assets.inventory([tmp_path])["assets"] if r["baseline_id"] == "E0B_T1")
    assert row["checkpoint_uri"] == str(path)
    assert row["architecture_source_sha"] is None
    assert row["adapter_source_sha"] is not None
    assert not row["binding_complete"]
    (tmp_path / "opaque-name.ckpt.identity.json").write_text(
        json.dumps({"checkpoint_sha256": "wrong", "training_source_sha": "a" * 40})
    )
    row = next(r for r in assets.inventory([tmp_path])["assets"] if r["baseline_id"] == "E0B_T1")
    assert row["architecture_source_sha"] is None


@pytest.mark.parametrize("kind", ["T1", "X3", "X4"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; not a GPU PASS")
def test_cuda_fixed_prefix_parity_engineering_only(kind):
    from epclm001.validation import parity

    model = fixture(kind).to("cuda")
    ids = torch.arange(128, device="cuda").remainder(32)[None]
    # Fixed engineering criteria declared before this run; no timing/performance claim.
    result = parity(model, ids, atol=2e-5, rtol=2e-4)
    assert result["passed"] and result["model_unchanged"]
    adapter = IncrementalLM(model)
    cache = adapter.init_cache(1, "cuda:0", torch.float32, max_context=128)
    out, cache = adapter.prefill(ids[:, :127], cache)
    out, cache = adapter.decode_one(ids[:, 127:], cache, absolute_position=127)
    torch.testing.assert_close(out, full_logits(model, ids)[:, -1], atol=2e-5, rtol=2e-4)


def test_event_readout_and_mode_restore_contract():
    state = torch.ones(4, 2)
    correct = Runtime(state, torch.eye(2), implementation=IMPLEMENTATIONS[0], max_steps=2)
    performance = Runtime(
        state, torch.eye(2), implementation=IMPLEMENTATIONS[0], max_steps=2, mode="performance"
    )
    for r in (correct, performance):
        r.enqueue(Event("read", 0, 0, kind="READ_QUERY"))
        r.microstep()
    torch.testing.assert_close(
        correct.last_reads["read"], performance.last_reads["read"], atol=0, rtol=0
    )
    with pytest.raises(ValueError, match="IDENTITY"):
        performance.restore(correct.snapshot())
