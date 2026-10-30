"""v0.4 engineering fixtures; never trained-model or formal performance evidence."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "implementation"))
from epclm001 import formal, formal_storage
from epclm001.common import digest, write
from epclm001.formal_analysis import analyze, interval
from epclm001.formal_correctness import contracts
from epclm001.formal_events import samples, semantics, special_semantics
from epclm001.formal_measure import analytic_work, cache_sizes, request
from epclm001.incremental import IncrementalLM, PreallocatedIncrementalLM
from epclm001.validation import engineering_model


def test_frozen_scope():
    c = formal.configuration()
    assert c["training_authorized"] is False
    assert len(c["event_points"]) == 16 and c["repeats"] == 30
    assert c["gpu_worker_budget_seconds"] == 7200
    assert all(w["prefill"] + w["decode"] <= 128 for w in c["workloads"])
    assert len({a["inference_sha256"] for a in c["assets"].values()}) == 3


@pytest.mark.parametrize("kind", ["T1", "X3", "X4"])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("cls", [IncrementalLM, PreallocatedIncrementalLM])
def test_formal_contracts_on_original_class_fixture(kind, batch, cls):
    torch.set_num_threads(1)
    torch.manual_seed(40)
    model = engineering_model(kind).eval().requires_grad_(False)
    ids = torch.randint(0, 32, (batch, 128))
    assert contracts(model, ids, cls, formal.configuration()["tolerances"], "f" * 64)["passed"]


@pytest.mark.parametrize("kind", ["B0", "B1_X3", "B1_X4"])
@pytest.mark.parametrize(
    "layout", ["uniform-random", "clustered", "zero-active", "all-active", "burst", "wavefront"]
)
def test_b0_b1_executor_parity(kind, layout):
    torch.set_num_threads(1)
    point = {
        "id": "fixture",
        "cells": 32,
        "dim": 188,
        "ratio": 0.25,
        "layout": layout,
        "tile": 8,
        "steps": 4,
    }
    assert semantics(point, kind, formal.configuration(), "cpu")["passed"]


def test_event_control_semantics():
    assert all(r["passed"] for r in special_semantics(formal.configuration(), "cpu"))


def test_event_timing_has_no_diagnostic_hash_and_resumes(monkeypatch):
    from epclm001 import events

    c = formal.configuration()
    point = {
        "id": "fixture",
        "cells": 16,
        "dim": 188,
        "ratio": 0.25,
        "layout": "clustered",
        "tile": 4,
        "steps": 2,
    }
    real_hash = events.tensor_hash

    def no_performance_hash(*args, **kwargs):
        # Hashing remains legal in untimed reference construction.
        import inspect

        assert not any(
            f.function == "microstep"
            and f.frame.f_locals.get("self").mode in ("performance", "profile")
            for f in inspect.stack()
            if f.function == "microstep"
        )
        return real_hash(*args, **kwargs)

    monkeypatch.setattr(events, "tensor_hash", no_performance_hash)
    budget = Mock()
    rows = samples(point, "B1_X3", c, "cpu", budget, 2, 0)
    assert len(rows) == 8
    resumed = samples(point, "B1_X3", c, "cpu", budget, 2, 0, existing=rows[:3])
    assert len(resumed) == 8
    for old in rows[:3]:
        assert old in resumed


def test_cache_backing_memory_and_cpu_timing():
    torch.set_num_threads(1)
    m = engineering_model("X3").eval().requires_grad_(False)
    a = PreallocatedIncrementalLM(m)
    cache = a.init_cache(1, "cpu", torch.float32, max_context=128)
    ids = torch.randint(0, 32, (1, 16))
    a.prefill(ids[:, :7], cache)
    sizes = cache_sizes(cache)
    assert sizes["kv_storage_bytes"] > sizes["kv_visible_bytes"] > 0
    row = request(m, ids, 8, "A1")
    assert len(row["token_samples"]) == 8 and row["ttft_status"].startswith("NOT_MEASURED")
    assert row["prefill_host_ms"] > 0 and row["decode_host_ms"] > 0
    for key in ("T1", "X3", "X4"):
        full = analytic_work(key, "full", 16, 16)
        inc = analytic_work(key, "A1", 16, 16)
        assert 0 < inc["decode"] < full["decode"]


def test_uncertainty_complete_paired_only():
    c = formal.configuration()
    ci = interval([2.0] * 30, [1.0] * 30, c, 50)
    assert ci["lower"] == ci["upper"] == 2 and ci["paired_repetitions"] == 30
    for left, right in [
        ([1.0] * 29, [1.0] * 29),
        ([1.0] * 30, [0.0] * 30),
        ([float("nan")] * 30, [1.0] * 30),
    ]:
        with pytest.raises(ValueError):
            interval(left, right, c, 50)


def test_missing_formal_results_stay_incomplete(tmp_path):
    analyze(tmp_path, formal.configuration(), {"stages": {}})
    assert {
        v["status"] for v in json.loads((tmp_path / "DECISIONS.json").read_text()).values()
    } == {"INCOMPLETE"}
    assert not (tmp_path / "figures").exists()


def make_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(formal_storage, "restore", lambda *a: None)
    monkeypatch.setattr(formal_storage, "persist", Mock())
    return formal.Runner(tmp_path, tmp_path / "assets")


def test_completed_stage_idempotent_hash_guard(tmp_path, monkeypatch):
    r = make_runner(tmp_path, monkeypatch)
    calls = []

    def preflight():
        calls.append(1)
        r.output("fixture.json", {"passed": True})

    r.e1_preflight = preflight
    r.execute("e1-preflight")
    r.execute("e1-preflight")
    assert calls == [1]
    write(tmp_path / "fixture.json", {"passed": False})
    with pytest.raises(RuntimeError, match="OUTPUT_CHANGED"):
        r.execute("e1-preflight")


def test_partial_stage_hash_guard_and_failure_record(tmp_path, monkeypatch):
    r = make_runner(tmp_path, monkeypatch)

    def fail():
        r.output("fixture.json", {"done": 1})
        raise RuntimeError("EXPECTED_FIXTURE_STOP")

    r.e1_preflight = fail
    with pytest.raises(RuntimeError, match="EXPECTED_FIXTURE"):
        r.execute("e1-preflight")
    assert r.m["status"] == "INCOMPLETE"
    assert json.loads((tmp_path / "FAILED_CASES.json").read_text())[0]["stage"] == "e1-preflight"
    write(tmp_path / "fixture.json", {"done": 2})
    with pytest.raises(RuntimeError, match="PARTIAL_STAGE_OUTPUT_CHANGED"):
        r.execute("e1-preflight")


def test_budget_exhaustion_and_device_change_before_gpu(tmp_path, monkeypatch):
    r = make_runner(tmp_path, monkeypatch)
    env = {
        "identity": "fixture",
        "uuid": "fixture",
        "torch": "fixture",
        "cuda": "fixture",
        "python": "fixture",
    }
    monkeypatch.setattr(formal, "hardware", lambda: env)
    r.m["gpu_worker_seconds_charged"] = 7200
    with pytest.raises(formal.BudgetStop), r.gpu():
        pytest.fail("must not execute")
    r.m["gpu_worker_seconds_charged"] = 0
    r.m["device_contract"] = {"identity": "another-GPU"}
    with pytest.raises(RuntimeError, match="DEVICE_CHANGED"), r.gpu():
        pytest.fail("must not execute")


def test_missing_credentials_blocks_preflight_without_gpu(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    r = formal.Runner(tmp_path, tmp_path / "assets")
    monkeypatch.setattr(formal, "hardware", lambda: pytest.fail("No GPU work before persistence"))
    with pytest.raises(RuntimeError, match="CONTROLLER_RESTORE"):
        r.execute("e1-preflight")
    assert r.m["gpu_worker_seconds_charged"] == 0 and r.m["optimizer_updates"] == 0


def test_report_publication_never_includes_weights_and_keeps_manifest(tmp_path):
    write(
        tmp_path / "RUN_MANIFEST.json",
        {"scope": "E1_FORMAL_INFERENCE_ONLY", "optimizer_updates": 0, "run_id": "a" * 32},
    )
    (tmp_path / "weight.pt").write_bytes(b"private")
    (tmp_path / "profiling").mkdir()
    write(tmp_path / "profiling/raw.json", {"trace": "large"})
    for i in range(5):
        (tmp_path / f"a{i}.json").write_text("x" * 3000000)
    _, files, _, _ = formal_storage.payload(tmp_path)
    assert (
        "RUN_MANIFEST.json" in files
        and "weight.pt" not in files
        and "profiling/raw.json" not in files
    )
    assert json.loads(files["GIT_COVERAGE.json"])["omitted_from_git"]


def test_wrong_pinned_weight_refused_before_load(tmp_path):
    from epclm001.formal_assets import load_model

    p = tmp_path / "inference.pt"
    p.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="WEIGHTS_CHANGED"):
        load_model(
            {
                "models": {
                    "T1": {"files": {"inference.pt": "inference.pt"}, "inference_sha256": "f" * 64}
                }
            },
            "T1",
            "cpu",
            tmp_path,
        )


def test_profile_mode_keeps_diagnostics_separate(monkeypatch):
    from epclm001 import events
    from epclm001.formal_events import runtime, setup

    c = formal.configuration()
    point = {
        "id": "fixture",
        "cells": 16,
        "steps": 2,
        "tile": 4,
        "ratio": 0.25,
        "layout": "clustered",
    }
    state, weight, inputs, kernel = setup(point, c, "cpu", "B1_X4")
    run = runtime(state, weight, inputs, point, "S0_sparse_cell", "profile", kernel)
    monkeypatch.setattr(
        events, "tensor_hash", lambda *a: pytest.fail("No per-step tensor hashing in profiler")
    )
    run.microstep()
    run.microstep()
    assert len(run.history) == 2 and set(run.history[0]) == {
        "queue",
        "compact",
        "gather",
        "compute",
        "scatter",
        "message",
    }


def test_pristine_local_manifest_cannot_hide_remote_controller(tmp_path, monkeypatch):
    import huggingface_hub

    c = formal.configuration()
    source = "a" * 40
    write(
        tmp_path / "RUN_MANIFEST.json",
        {"stages": {"e1-preflight": {"status": "FAILED"}}, "gpu_worker_seconds_charged": 0},
    )
    remote = tmp_path / "remote"
    remote.mkdir()
    manifest = remote / "manifest.json"
    write(manifest, {"source_sha": source, "run_id": "b" * 32, "gpu_worker_seconds_charged": 12})
    index = remote / "index.json"
    write(
        index,
        {
            "source_sha": source,
            "config_sha256": formal.sha(c),
            "revision": "c" * 40,
            "files": {
                "RUN_MANIFEST.json": {"remote": "test/manifest.json", "sha256": digest(manifest)}
            },
        },
    )
    monkeypatch.setattr(formal_storage, "credential", lambda: "fixture-not-a-secret")
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo, path, **kw: str(index if path.endswith("CONTROLLER.json") else manifest),
    )
    formal_storage.restore(tmp_path, c, source)
    assert (
        json.loads((tmp_path / "RUN_MANIFEST.json").read_text())["gpu_worker_seconds_charged"] == 12
    )


def test_partial_competitive_analysis_does_not_invent_quality(tmp_path):
    c = formal.configuration()
    row = {
        "model": "T1",
        "mode": "A0",
        "workload": "p32-d32",
        "repeat": 0,
        "prefill_host_ms": 2.0,
        "decode_host_ms": 4.0,
        "total_host_ms": 6.0,
        "tpot_p50_ms": 0.1,
        "tpot_p95_ms": 0.2,
        "tpot_p99_ms": 0.3,
        "cache": {"kv_storage_bytes": 100},
    }
    write(tmp_path / "E1C_TIMING.json", {"records": [row]})
    decisions = analyze(tmp_path, c, {"stages": {}})
    assert decisions["H3_COMPETITIVE"]["status"] == "INCOMPLETE"
    assert not (tmp_path / "figures/08-quality-latency.png").exists()
    assert "UNAVAILABLE" in (tmp_path / "COMPETITIVE_TABLE.csv").read_text()


def test_complete_analysis_outputs_nine_figures_from_fixture(tmp_path):
    import copy

    c = copy.deepcopy(formal.configuration())
    c["repeats"] = 2
    c["bootstrap_samples"] = 100
    a = []
    b = []
    cross = []
    for w in c["workloads"]:
        for key in ("T1", "X3", "X4"):
            for mode in ("full", "A0", "A1"):
                for repeat in range(2):
                    row = {
                        "model": key,
                        "mode": mode,
                        "workload": w["id"],
                        "repeat": repeat,
                        "prefill_host_ms": 2.0,
                        "decode_host_ms": 4.0 if mode == "full" else 2.0,
                        "total_host_ms": 6.0,
                        "tpot_p50_ms": 0.1,
                        "tpot_p95_ms": 0.2,
                        "tpot_p99_ms": 0.3,
                        "cache": {"kv_storage_bytes": 100},
                    }
                    if key != "T1":
                        a.append(row)
                    if key != "T1" or mode != "A1":
                        cross.append(row)
    from epclm001.events import IMPLEMENTATIONS

    for point in c["event_points"]:
        for kernel in c["event_kernels"]:
            for impl in IMPLEMENTATIONS:
                for repeat in range(2):
                    b.append(
                        {
                            "point": point["id"],
                            "kernel": kernel,
                            "implementation": impl,
                            "repeat": repeat,
                            "host_ms": 2.0 if impl.startswith("D") else 1.0,
                            "logical_updates": 10,
                            "physical_updates": 10,
                            "partial_matmul_flops": 1000,
                        }
                    )
    write(tmp_path / "E1A_TIMING.json", {"records": a})
    write(tmp_path / "E1B_TIMING.json", {"records": b})
    write(tmp_path / "E1C_TIMING.json", {"records": cross})
    write(
        tmp_path / "E1C_QUALITY.json",
        {k: {"quality": {"validation_ppl": 12.0}} for k in ("T1", "X3", "X4")},
    )
    write(
        tmp_path / "E1A_PARITY.json",
        {
            k: {
                "cases": [
                    {
                        "batch": 1,
                        "input_role": "holdout",
                        "adapter": "fixture",
                        "report": {"rows": [{"prefix_length": 1, "max_abs": 0.0}]},
                    }
                ]
            }
            for k in ("X3", "X4")
        },
    )
    result = analyze(tmp_path, c, {"stages": {k: {"status": "COMPLETE"} for k in formal.STAGES}})
    assert len(json.loads((tmp_path / "FIGURE_INDEX.json").read_text())["files"]) == 9
    assert result["H1_INCREMENTAL"]["status"] == "SUPPORTED"
    assert result["H2_EVENT_RUNTIME"]["status"] == "SUPPORTED"
    assert json.loads((tmp_path / "SAMPLE_STATISTICS.json").read_text())["records"]
