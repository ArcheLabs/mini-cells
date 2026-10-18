"""Full denominator and continuation contracts, CPU synthetic/FakeHub only."""

from copy import deepcopy
import json
import pytest
from minicells_runtime.store import read, canonical, ContinuationError
from clm001_r3b import full_analysis as full
from clm001_r3b.diagnostic_schedule import load_plan, full_matrix
from clm001_r3b.analysis_store import AnalysisStore
from clm001_r3b.posthoc_performance import build
from test_hf_storage import FakeHub


def test_full_matrix_counts_and_sealed_roles():
    rows = full_matrix(load_plan())
    assert len(rows) == len({r["slot_key"] for r in rows}) == 102
    assert {s: sum(r["stage"] == s for r in rows) for s in ("quality", "functional", "action")} == {
        "quality": 39,
        "functional": 42,
        "action": 21,
    }
    assert not any(r["role"] == "confirmation" for r in rows)
    assert len([r for r in rows if r["step"] == 400]) == 30
    assert {r["phase"] for r in rows if r["step"] == 400} == {"pre_event", "post_event"}


def test_target_identity_no_pilot_or_mismatched_revision():
    ix = read(full.REPORTS / "checkpoint-index.json")
    meta = {
        k: ix[k]
        for k in (
            "source_sha",
            "run_id",
            "runtime_config_sha256",
            "assets_manifest_sha256",
            "tokenizer_sha256",
            "protocol_sha256",
        )
    }
    meta["complete"] = True
    meta["arms"] = {
        a: {
            "checkpoints": [
                {
                    "step": r["step"],
                    "revision": r["event_revision"],
                    "checkpoint_sha256": r["sha256"],
                }
                for r in ix["checkpoints"]
                if r["arm"] == a
            ]
        }
        for a in ("Q-R3b", "QC-R3b", "QS-R3b")
    }
    full.validate_input(ix, meta, ix["checkpoint_ref"])
    for key in ("source_sha", "run_id", "runtime_config_sha256", "tokenizer_sha256"):
        bad = dict(meta, **{key: "wrong"})
        with pytest.raises(ContinuationError):
            full.validate_input(ix, bad, ix["checkpoint_ref"])
    bad = dict(ix["checkpoint_ref"], hf_revision="0" * 40)
    with pytest.raises(ContinuationError):
        full.validate_input(ix, meta, bad)


def fake_run(tmp_path, monkeypatch, *, seconds=10000, fail=None, storage=None):
    plan = load_plan()
    meta = {"source_sha": "a" * 40, "run_id": "synthetic"}

    def resolve(root, ref, plan, source, slot, meta):
        if fail and fail(slot):
            raise ContinuationError("PILOT_BANK_IS_NOT_DEVELOPMENT")
        return {
            **slot,
            "training_source_sha": meta["source_sha"],
            "training_run_id": meta["run_id"],
            "analysis_source_sha": source,
            "analysis_protocol_sha256": canonical(plan),
        }

    monkeypatch.setattr(full, "resolve", resolve)
    calls = []

    def execute(root, task, plan, device):
        calls.append(task)
        return {
            "identity": task,
            "task_status": "COMPLETE",
            "sample_count": 10,
            "coverage_status": "OBSERVED",
            "elapsed_seconds": 1.0,
            "training_state_unchanged": True,
            "optimizer_updates": 0,
        }

    store = storage or AnalysisStore(FakeHub(tmp_path / "hub"))
    result = full.run_tasks(
        tmp_path,
        {},
        plan,
        "b" * 40,
        store,
        tmp_path / "output",
        session_seconds=seconds,
        execute=execute,
        meta=meta,
    )
    return store, calls, result


def test_full_complete_remote_reuse_no_new_execution(tmp_path, monkeypatch):
    store, calls, result = fake_run(tmp_path, monkeypatch)
    assert len(calls) == 102
    assert full.counts(result[2]) == {"complete": 102, "unavailable": 0, "blocked": 0, "pending": 0}
    _, second, again = fake_run(tmp_path, monkeypatch, storage=store)
    assert not second
    assert all(r["reused"] for r in again[2])
    assert len(again[3]) == 2


def test_missing_development_preserves_all_102_and_terminal_pending_zero(tmp_path, monkeypatch):
    _, calls, result = fake_run(tmp_path, monkeypatch, fail=lambda s: s["role"] == "development")
    assert len(calls) == 42
    assert full.counts(result[2]) == {"complete": 42, "unavailable": 60, "blocked": 0, "pending": 0}
    assert all(
        r["reason"] == "PILOT_BANK_IS_NOT_DEVELOPMENT"
        for r in result[2]
        if r["status"] == "UNAVAILABLE"
    )


def test_session_boundary_no_kill_and_resume(tmp_path, monkeypatch):
    store, calls, result = fake_run(tmp_path, monkeypatch, seconds=0)
    assert not calls and full.counts(result[2])["pending"] == 102
    _, calls, result = fake_run(tmp_path, monkeypatch, storage=store)
    assert len(calls) == 102 and full.counts(result[2])["pending"] == 0


def test_authoritative_coverage_readback_and_privacy(tmp_path, monkeypatch):
    from clm001_r3b.publication import reject_private_payload

    store, _, result = fake_run(tmp_path, monkeypatch)
    scope = {
        k: result[0][0][k]
        for k in (
            "training_source_sha",
            "training_run_id",
            "analysis_source_sha",
            "analysis_protocol_sha256",
        )
    }
    saved = store.coverage(scope)
    assert saved["counts"]["complete"] == 102
    reject_private_payload({"tasks": result[2]})
    for payload in ({"raw_oracle": []}, {"value": "hf_" + "x" * 30}):
        with pytest.raises(ContinuationError):
            reject_private_payload(payload)


def test_measured_training_not_wall_or_event_total():
    ledger = read(full.REPORTS / "resource-ledger.json")
    value = build(ledger, [], [])
    assert value["two_hour_verdict"] == "NOT_VERIFIED"
    assert value["training_critical_path_seconds"]["classification"] == "NOT_MEASURED"
    assert value["training_cuda_resource_seconds"]["classification"] == "DERIVED"
    assert value["posthoc_gpu_seconds"]["value"] is None
    assert value["mandatory_event_cost"]["value"] is None
    assert value["storage"]["final_logical_bundle_bytes"]["value"] == 1284580587
    assert value["storage"]["payload_bytes_submitted"]["value"] != 1284580587


def test_notebook_one_launch_exact_input_not_old_pilot():
    notebook = json.loads((full.BASE / "analysis-notebook.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    compile(code, "analysis-notebook", "exec")
    assert '"full-analysis"' in code
    assert "baseline-evidence" not in code and '"--max-tasks"' not in code
    assert '"--publish-github"' in code


def test_input_index_checkpoint_hash_is_frozen():
    index = read(full.REPORTS / "checkpoint-index.json")
    bad = deepcopy(index)
    bad["checkpoints"][0]["sha256"] = "f" * 64
    with pytest.raises(ContinuationError):
        full.validate_input(bad, {}, index["checkpoint_ref"])


def test_orphan_result_is_adopted_without_recompute(tmp_path):
    from minicells_runtime.store import atomic

    hub = FakeHub(tmp_path / "hub")
    store = AnalysisStore(hub)
    task = {
        "training_source_sha": "a" * 40,
        "training_run_id": "synthetic",
        "analysis_source_sha": "b" * 40,
        "analysis_protocol_sha256": "c" * 64,
    }
    result = {
        "identity": task,
        "task_status": "COMPLETE",
        "optimizer_updates": 0,
        "training_state_unchanged": True,
    }
    file = tmp_path / "result.json"
    atomic(file, result)
    hub.commit(
        {store.prefix(task) + "/private/" + canonical(task) + "/result.json": file},
        hub.head,
        "crash before index",
    )
    assert store.completed(task) == result
    assert store.index(task)[1]["tasks"][canonical(task)]["task_status"] == "COMPLETE"
    result["training_state_unchanged"] = False
    with pytest.raises(ContinuationError, match="READ_ONLY_PROOF"):
        store.commit(result)


def test_whole_input_failure_is_102_blocked_without_diagnostics(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise ContinuationError("CHECKPOINT_HASH_MISMATCH")

    monkeypatch.setattr(full, "_run", broken)
    hub = FakeHub(tmp_path / "hub")
    monkeypatch.setattr("minicells_artifacts.hf_storage.HubBackend", lambda *args: hub)
    result = full.run(tmp_path / "out", publish=False)
    assert result["counts"] == {"complete": 0, "unavailable": 0, "blocked": 102, "pending": 0}
    assert result["training_updates"] == 0
    assert read(tmp_path / "out/coverage-ledger.json")["private_hf_coverage_verified"]


def test_publication_remote_receipt_has_real_content_commit_and_hash(tmp_path, monkeypatch):
    import subprocess
    from clm001_r3b import publication
    from minicells_runtime.store import atomic, digest

    remote = tmp_path / "remote.git"
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "--bare", "--quiet", str(remote)], check=True)
    checkout.mkdir()

    def git(*args):
        return publication.existing.git(checkout, *args)

    git("init", "--quiet")
    git("config", "user.email", "synthetic@example.invalid")
    git("config", "user.name", "Synthetic CPU")
    git("remote", "add", "origin", str(remote))
    git("checkout", "-b", "results/synthetic")
    root = tmp_path / "identity"
    atomic(root / "experiment.lock.json", {"source_sha": "a" * 40, "run_id": "synthetic"})
    monkeypatch.setattr(publication, "preflight", lambda *args: (checkout, "results/synthetic"))

    class Helper:
        ROOT = checkout

        def push_branch(self, branch, secret):
            git("push", "origin", "HEAD:refs/heads/" + branch)

    monkeypatch.setattr(publication.existing, "helper", lambda: Helper())
    receipt = publication.publish_documents(
        root, {"metrics-summary.json": {"status": "PARTIAL", "scientific_decision": False}}
    )
    saved = read(checkout / receipt["receipt_path"])
    assert saved["status"] == "PUBLISHED"
    assert saved["content_commit_sha"] != receipt["commit_sha"]
    assert (
        git("ls-remote", "origin", "refs/heads/results/synthetic").stdout.split()[0]
        == receipt["commit_sha"]
    )
    assert all(digest(checkout / p) == sha for p, sha in saved["files"].items())
    assert not any(p.endswith(".pt") for p in saved["files"])


def test_mutating_input_during_task_never_commits_complete(case, tmp_path, monkeypatch):
    from test_deferred_diagnostics import frozen_input, task
    from clm001_r3b import analysis_runner as ar

    root, c, hub, ref = frozen_input(case, tmp_path)
    original = ar.obs.evaluate

    def mutate(*args, **kwargs):
        value = original(*args, **kwargs)
        path = root / "protocol.md"
        path.write_bytes(path.read_bytes() + b"\nmodified\n")
        return value

    monkeypatch.setattr(ar.obs, "evaluate", mutate)
    identity = task(root, ref, c)
    store = AnalysisStore(hub)
    with pytest.raises(ContinuationError, match="READ_ONLY_ANALYSIS_MUTATED_INPUT"):
        ar.batch(root, tmp_path / "analysis", [identity], load_plan(), store, max_tasks=1)
    assert store.completed(identity) is None
