"""Synthetic CPU, FakeKaggle and FakeHub only; no credentials or GPU."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import shutil
import pytest
from minicells_runtime.store import atomic, canonical, read, ContinuationError
from clm001_r3b.diagnostic_schedule import full_matrix, load_plan
from clm001_r3b.full_analysis import counts, unavailable
from kaggle_posthoc.local_store import LocalResultStore, proof
from kaggle_posthoc.kaggle_source import inventory, fetch, locate_training, discover
from kaggle_posthoc.recover import ReadOnlyHFRecovery, scan_raw_results
from kaggle_posthoc.execution import run_tasks
from kaggle_posthoc.reports import finalize_snapshot, public_projection, scientific_findings
from test_hf_storage import FakeHub

ENGINE_SHA = "ed2accbc90f541325ca2f6c53229e9c81475768d"


def identity(slot):
    return {
        **slot,
        "analysis_source_sha": ENGINE_SHA,
        "analysis_protocol_sha256": canonical(load_plan()),
        "training_source_sha": "a" * 40,
        "training_run_id": "synthetic",
        "checkpoint_sha256": "b" * 64,
        "event_revision": 0 if slot["phase"] == "pre_event" else 8,
        "event_semantics": [],
        "state_scope": "FROZEN_INITIAL_STATE",
    }


def result(task):
    if task["stage"] == "quality":
        payload = {
            "quality": {
                p: {"nll": [1.0], "cost": [0.2], "groups": [0], "mean_nll": 1.0}
                for p in task["policies"]
            }
        }
        n = len(task["policies"])
    elif task["stage"] == "functional":
        payload = {
            "functional": {"state_records": [], "records": [], "summary": {}, "crossover": {}}
        }
        n = 0
    else:
        payload = {"router": [], "action_type_calibration": []}
        n = 0
    return {
        "identity": task,
        "task_status": "COMPLETE",
        "optimizer_updates": 0,
        "training_state_unchanged": True,
        "sample_count": n,
        "coverage_status": "OBSERVED" if n else "INSUFFICIENT_COVERAGE",
        "result": payload,
        "elapsed_seconds": 1.0,
        "timings": {},
        "device": "cpu",
        "scientific_decision": False,
    }


class Missing(Exception):
    response = SimpleNamespace(status_code=404)


class FakeKaggle:
    def __init__(self, versions, latest=None):
        self.versions = versions
        self.latest = latest or max(versions)
        self.calls = []
        self.private = True

    def metadata(self, version=None):
        if version is not None and version not in self.versions:
            raise Missing()
        return {
            "is_private": self.private,
            "version_number": version or self.latest,
            "script_version_id": 359735394 if version == 65 else None,
        }

    def page(self, version, token=None):
        self.calls.append((version, token))
        rows = [{"name": n, "size": len(b)} for n, b in self.versions[version].items()]
        start = int(token or 0)
        return {
            "files": rows[start : start + 2],
            "nextPageToken": str(start + 2) if start + 2 < len(rows) else "",
        }

    def download(self, version, names, destination):
        for name in names:
            path = Path(destination) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.versions[version][name])


def test_exact_version_pagination_and_size_validation(tmp_path):
    api = FakeKaggle({65: {f"dir/{i}.json": b"{}" for i in range(5)}, 66: {"wrong.json": b"wrong"}})
    inv = inventory(api, 65)
    assert inv["file_count"] == 5 and all(v == 65 for v, _ in api.calls)
    assert len(api.calls) == 3
    transfer = fetch(api, inv, [r["name"] for r in inv["files"]], tmp_path / "files")
    assert transfer["downloaded_bytes"] == 10
    (tmp_path / "files/dir/0.json").write_bytes(b"oversized")
    with pytest.raises(ContinuationError, match="SIZE_CONFLICT"):
        fetch(api, inv, ["dir/0.json"], tmp_path / "files")


def test_discovery_gaps_and_script_id_separate_from_version():
    api = FakeKaggle({63: {"a.json": b"{}"}, 65: {"b.json": b"{}"}, 67: {"c.json": b"{}"}})
    versions, discovery = discover(api)
    assert {v["version_number"] for v in versions} == {63, 65, 67}
    assert discovery["user_reported_script_version_id"] == 359735394
    assert versions[0]["metadata"]["script_version_id"] == 359735394
    api.private = False
    with pytest.raises(ContinuationError, match="PRIVATE_OUTPUT_REQUIRED"):
        discover(api)


@pytest.mark.parametrize(
    "name", ["../escape.pt", "/absolute.pt", "safe/../../escape", "a\\b", "./x"]
)
def test_output_paths_fail_closed(name):
    with pytest.raises(ContinuationError, match="UNSAFE"):
        inventory(FakeKaggle({65: {name: b"x"}}), 65)


def test_duplicate_paths_and_pagination_loops():
    api = FakeKaggle({65: {"a": b"x"}})
    api.page = lambda *a: {"files": [{"name": "a", "size": 1}], "nextPageToken": "same"}
    with pytest.raises(ContinuationError, match="PAGINATION_LOOP"):
        inventory(api, 65)


def test_repeated_listing_entries_across_pages_are_counted_once():
    api = FakeKaggle({65: {"a": b"x", "b": b"yz"}})
    pages = {
        None: {"files": [{"name": "a", "size": 1}] * 2, "nextPageToken": "second"},
        "second": {"files": [{"name": "a", "size": 1}, {"name": "b", "size": 2}]},
    }
    api.page = lambda version, token=None: pages[token]
    inv = inventory(api, 65)
    assert inv["files"] == [{"name": "a", "bytes": 1}, {"name": "b", "bytes": 2}]
    assert inv["file_count"] == 2 and inv["total_bytes"] == 3
    assert inv["listing_pages"] == 2 and inv["repeated_listing_entries"] == 2
    assert inv["inventory_sha256"] == inventory(FakeKaggle(api.versions), 65)["inventory_sha256"]


def test_same_path_with_conflicting_sizes_fails_closed():
    api = FakeKaggle({65: {"a": b"x"}})
    api.page = lambda *args: {"files": [{"name": "a", "size": 1}, {"name": "a", "size": 2}]}
    with pytest.raises(ContinuationError, match="OUTPUT_PATH_SIZE_CONFLICT"):
        inventory(api, 65)


def training_fixture(tmp_path):
    report = (
        Path(__file__).parents[3]
        / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3b/reports/runs/33734fdd1558/checkpoint-index.json"
    )
    spec = read(report)
    blobs, arms, files, index = {}, {}, {}, []
    for r in spec["checkpoints"]:
        name = f"arms/{r['arm']}/checkpoint-step-{r['step']:06d}-revision-{r['event_revision']}.pt"
        data = name.encode()
        sha = hashlib.sha256(data).hexdigest()
        blobs["run/" + name] = data
        files[name] = {"sha256": sha, "bytes": len(data)}
        arms.setdefault(r["arm"], {"checkpoints": []})["checkpoints"].append(
            {"step": r["step"], "revision": r["event_revision"], "checkpoint_sha256": sha}
        )
        index.append(
            {
                "arm": r["arm"],
                "step": r["step"],
                "event_revision": r["event_revision"],
                "sha256": sha,
            }
        )
    for arm in arms:
        for bank in ("frozen-probe-bank.pt", "development-probe-bank.pt"):
            name = "arms/" + arm + "/" + bank
            data = b"fixed-bank"
            blobs["run/" + name] = data
            files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    manifest = json.dumps({"files": files}, sort_keys=True).encode()
    meta = {
        "source_sha": "a" * 40,
        "run_id": "synthetic",
        "complete": True,
        "arms": arms,
        "files_manifest": "manifest.json",
        "files_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    blobs["run/manifest.json"] = manifest
    blobs["run/continuation.json"] = json.dumps(meta).encode()
    reports = tmp_path / "reports"
    reports.mkdir()
    atomic(reports / "checkpoint-index.json", {"checkpoints": index, "checkpoint_ref": {}})
    calls = []

    def verify(root):
        calls.append(root)
        return read(root / "continuation.json")

    engine = SimpleNamespace(
        REPORTS=reports,
        SOURCE="a" * 40,
        INTERNAL_RUN="synthetic",
        verify=verify,
        validate_input=lambda *a: None,
    )
    return blobs, engine, calls


def test_all_57_checkpoints_verified_without_hf(tmp_path, monkeypatch):
    blobs, engine, calls = training_fixture(tmp_path)
    api = FakeKaggle({65: blobs})
    inv = inventory(api, 65)
    selected, root, meta, transfer = locate_training(api, [inv], tmp_path / "cache", engine)
    assert len(calls) == 1 and selected["version_number"] == 65
    assert sum(len(a["checkpoints"]) for a in meta["arms"].values()) == 57
    assert transfer["downloaded_bytes"] > 0
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert (root / "arms/Q-R3b/frozen-probe-bank.pt").exists()


def test_missing_revision_and_wrong_identity_do_not_load_pickle(tmp_path):
    blobs, engine, calls = training_fixture(tmp_path)
    missing = next(n for n in blobs if "revision-8.pt" in n)
    del blobs[missing]
    api = FakeKaggle({65: blobs})
    with pytest.raises(ContinuationError, match="VERSION_UNRESOLVED"):
        locate_training(api, [inventory(api, 65)], tmp_path / "cache", engine)
    assert not calls
    meta = json.loads(blobs["run/continuation.json"])
    meta["source_sha"] = "wrong"
    blobs["run/continuation.json"] = json.dumps(meta).encode()
    with pytest.raises(ContinuationError, match="VERSION_UNRESOLVED"):
        locate_training(api, [inventory(api, 65)], tmp_path / "other", engine)
    assert not calls


def test_local_idempotence_conflict_and_orphan_rebuild(tmp_path):
    task = identity(full_matrix(load_plan())[0])
    data = result(task)
    store = LocalResultStore(tmp_path / "store")
    entry = store.commit(data, {"version": 65})
    assert not entry["cross_session_persistence_verified"]
    assert store.commit(data, {"version": 65}) == entry
    changed = deepcopy(data)
    changed["elapsed_seconds"] = 2
    with pytest.raises(ContinuationError, match="CONFLICT"):
        store.commit(changed, {})
    (store.root / "analysis-index.json").unlink()
    shutil.rmtree(store.root / "receipts")
    fresh = LocalResultStore(store.root)
    assert fresh.completed(task) == data
    fresh.commit(data, {"version": 66}, persistence="KAGGLE_VERSION_PERSISTED")
    assert LocalResultStore(store.root).index["tasks"][canonical(task)][
        "cross_session_persistence_verified"
    ]
    (store.root / entry["file"]).write_text("{}")
    with pytest.raises(ContinuationError, match="HASH_MISMATCH"):
        LocalResultStore(store.root)


def test_42_raw_recovered_coverage_only_ignored_and_new_session(tmp_path):
    tasks = [identity(s) for s in full_matrix(load_plan())]
    saved = tmp_path / "saved"
    store = LocalResultStore(saved / "private-results")
    for t in tasks[:42]:
        store.commit(result(t), {"version": 65})
    finalize_snapshot(
        saved,
        {
            "measurement_engine_source_sha": ENGINE_SHA,
            "execution_orchestrator_source_sha": "c" * 40,
        },
    )
    atomic(saved / "coverage-only.json", {"complete": 102, "tasks": [{"status": "COMPLETE"}] * 102})
    fresh = LocalResultStore(tmp_path / "new/private-results")
    recovered, rejected = scan_raw_results(
        saved, tasks, fresh, {"kernel": "bingjiangli/minicells", "version_number": 66}
    )
    assert len(recovered) == 42 and not rejected
    assert all(e["cross_session_persistence_verified"] for e in fresh.index["tasks"].values())
    shutil.rmtree(saved)  # previous /working is gone; imported cumulative package survives
    assert fresh.completed(tasks[0]) == result(tasks[0])
    bad = deepcopy(result(tasks[42]))
    bad["identity"]["analysis_source_sha"] = "d" * 40
    atomic(tmp_path / "raw/result.json", bad)
    recovered, rejected = scan_raw_results(tmp_path / "raw", tasks, fresh, {"version": 67})
    assert not recovered and rejected[0]["reason"] == "OLD_ENGINE_OR_TASK_IDENTITY_MISMATCH"


def test_read_only_hf_orphan_recovery_no_commits_and_hash_corruption(tmp_path):
    hub = FakeHub(tmp_path / "hub")
    task = identity(full_matrix(load_plan())[0])
    data = result(task)
    scope = {
        k: task[k]
        for k in (
            "analysis_source_sha",
            "analysis_protocol_sha256",
            "training_run_id",
            "training_source_sha",
        )
    }
    prefix = "analyses/" + canonical(scope)
    file = tmp_path / "result.json"
    atomic(file, data)
    path = prefix + "/private/" + canonical(task) + "/result.json"
    revision = hub.commit({path: file}, hub.head, "existing old result")
    writes = hub.writes
    recovered = ReadOnlyHFRecovery(hub, scope).get(task)
    assert recovered[0] == data and recovered[1]["orphan_recovered"]
    assert hub.writes == writes
    entry = {
        "identity": task,
        "task_status": "COMPLETE",
        "hf_path": path,
        "hf_revision": revision,
        "hf_repo": hub.repo,
        "sha256": "f" * 64,
    }
    atomic(tmp_path / "index.json", {"tasks": {canonical(task): entry}})
    hub.commit({prefix + "/index.json": tmp_path / "index.json"}, hub.head, "old index")
    with pytest.raises(ContinuationError, match="HASH_MISMATCH"):
        ReadOnlyHFRecovery(hub, scope).get(task)


def test_hf_429_never_blocks_local_analysis_and_reuses_102(tmp_path):
    calls = []

    class Throttled:
        count = 0

        def get(self, *args):
            self.count += 1
            e = RuntimeError("throttled")
            e.response = SimpleNamespace(status_code=429)
            raise e

    hf = Throttled()
    engine = SimpleNamespace(
        full_matrix=full_matrix,
        resolve=lambda root, ref, plan, sha, slot, meta: identity(slot),
        ENGINE_SHA=ENGINE_SHA,
        execute=lambda root, t, plan, device: calls.append(t) or result(t),
        unavailable=unavailable,
    )
    store = LocalResultStore(tmp_path / "store")
    _, results, rows, timing = run_tasks(engine, tmp_path, {}, load_plan(), {}, store, {}, hf=hf)
    assert counts(rows) == {"complete": 102, "unavailable": 0, "blocked": 0, "pending": 0}
    assert len(calls) == 102 and hf.count == 1 and timing["hf_commits"] == 0
    calls.clear()
    run_tasks(engine, tmp_path, {}, load_plan(), {}, store, {})
    assert not calls


def test_budget_boundary_preserves_matrix_without_training(tmp_path):
    engine = SimpleNamespace(
        full_matrix=full_matrix,
        resolve=lambda root, ref, plan, sha, slot, meta: identity(slot),
        ENGINE_SHA=ENGINE_SHA,
        execute=lambda *a, **kw: pytest.fail("budget should not start a task"),
        unavailable=unavailable,
    )
    _, results, rows, timing = run_tasks(
        engine,
        tmp_path,
        {},
        load_plan(),
        {},
        LocalResultStore(tmp_path / "store"),
        {},
        session_seconds=0,
    )
    assert not results and counts(rows)["pending"] == 102 and timing["training_updates"] == 0


def test_public_projection_does_not_include_private_raw_paths_or_tokens():
    out = public_projection(
        {
            "coverage-ledger.json": {
                "tasks": [
                    {
                        "identity": {"raw": "private"},
                        "status": "COMPLETE",
                        "storage_provenance": {"private_path": "secret"},
                        "slot_key": "x",
                    }
                ]
            },
            "kaggle-version-inventory.json": {
                "versions": [
                    {
                        "version_number": 65,
                        "files": [{"name": "private"}],
                        "metadata": {"is_private": True},
                    }
                ]
            },
        }
    )
    assert "private" not in json.dumps(out)
    with pytest.raises(ContinuationError, match="PRIVATE_PUBLICATION"):
        public_projection({"metrics-summary.json": {"token": "secret"}})


def test_partial_42_no_false_did_or_confirmation():
    from clm001_r3b.analysis_aggregate import aggregate_tasks

    tasks = [identity(s) for s in full_matrix(load_plan())]
    values = aggregate_tasks(
        {"alignment": {"cell_pairs": []}},
        {"complete": True, "status": "COMPLETE"},
        tasks,
        [result(t) for t in tasks[:42]],
        [],
    )
    assert all("DID" not in r for r in values["statistical-contrast.json"]["rows"])
    findings = scientific_findings(
        values, {"complete": 42, "blocked": 60, "pending": 0, "unavailable": 0}
    )
    assert "H1: INCONCLUSIVE" in findings and "Confirmation SEALED" in findings


def test_raw_proof_rejects_missing_arrays_sample_counts_and_updates():
    task = identity(full_matrix(load_plan())[0])
    data = result(task)
    data["sample_count"] = 200
    with pytest.raises(ContinuationError, match="COVERAGE"):
        proof(data, task)
    data = result(task)
    data["optimizer_updates"] = 1
    with pytest.raises(ContinuationError, match="READ_ONLY"):
        proof(data, task)


def test_notebook_single_save_run_all_and_dual_source_identity():
    from clm001_r3b.common import BASE

    n = read(BASE / "kaggle-output-analysis-notebook.ipynb")
    code = "\n".join("".join(c["source"]) for c in n["cells"] if c["cell_type"] == "code")
    compile(code, "output-notebook", "exec")
    assert ENGINE_SHA in code and "output_analysis.py" in code
    assert "--max-tasks" not in code and "Save & Run All" in "".join(n["cells"][0]["source"])
    assert "KAGGLE_API_TOKEN" in code and "/kaggle/temp/" in code


def test_exact_measurement_checkout_bytes_and_head(tmp_path):
    import subprocess
    from kaggle_posthoc.source_guard import verify_engine

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "--quiet")
    git("config", "user.name", "CPU test")
    git("config", "user.email", "cpu@example.invalid")
    (tmp_path / "engine.py").write_text("FROZEN = True\n")
    git("add", "engine.py")
    git("commit", "--quiet", "-m", "frozen synthetic engine")
    sha = git("rev-parse", "HEAD")
    assert verify_engine(tmp_path, sha)["materialized_files_verified"] == 1
    with pytest.raises(RuntimeError, match="HEAD_MISMATCH"):
        verify_engine(tmp_path, "0" * 40)
    (tmp_path / "engine.py").write_text("FROZEN = False\n")
    with pytest.raises(RuntimeError, match="BYTES_MISMATCH"):
        verify_engine(tmp_path, sha)


def test_recovered_42_only_computes_remaining_60(tmp_path):
    slots = full_matrix(load_plan())
    store = LocalResultStore(tmp_path / "cumulative")
    for slot in slots[:42]:
        store.commit(
            result(identity(slot)), {"version": 65}, persistence="KAGGLE_VERSION_PERSISTED"
        )
    calls = []
    engine = SimpleNamespace(
        full_matrix=full_matrix,
        resolve=lambda root, ref, plan, sha, slot, meta: identity(slot),
        ENGINE_SHA=ENGINE_SHA,
        execute=lambda root, task, plan, device: calls.append(task) or result(task),
        unavailable=unavailable,
    )
    _, _, rows, timing = run_tasks(engine, tmp_path, {}, load_plan(), {}, store, {})
    assert len(calls) == timing["new_tasks_computed"] == 60
    assert len(rows) == 102 and counts(rows)["complete"] == 102
    assert sum(r["cross_session_persistence_verified"] for r in rows) == 42


def test_multiple_training_copies_download_only_one(tmp_path):
    blobs, engine, calls = training_fixture(tmp_path)
    api = FakeKaggle({65: blobs, 66: blobs})
    selected, _, _, _ = locate_training(
        api, [inventory(api, 66), inventory(api, 65)], tmp_path / "cache", engine
    )
    assert selected["version_number"] == 65 and len(calls) == 1
    assert not (tmp_path / "cache/version-66").exists()


def test_version_response_cannot_silently_use_latest():
    api = FakeKaggle({65: {"a": b"x"}})
    api.metadata = lambda version=None: {"is_private": True, "version_number": 66}
    with pytest.raises(ContinuationError, match="VERSION_RESPONSE_MISMATCH"):
        inventory(api, 65)


def test_new_public_publisher_has_real_remote_receipt(tmp_path, monkeypatch):
    import subprocess
    from clm001_r3b import publication
    from kaggle_posthoc.reports import publish
    from minicells_runtime.store import digest

    remote, checkout = tmp_path / "remote.git", tmp_path / "checkout"
    subprocess.run(["git", "init", "--bare", "--quiet", str(remote)], check=True)
    checkout.mkdir()

    def git(*args):
        return publication.existing.git(checkout, *args)

    git("init", "--quiet")
    git("config", "user.email", "synthetic@example.invalid")
    git("config", "user.name", "Synthetic CPU")
    git("remote", "add", "origin", str(remote))
    git("checkout", "-b", "results/synthetic-kaggle")
    root, output = tmp_path / "input", tmp_path / "output"
    atomic(root / "experiment.lock.json", {"source_sha": "a" * 40, "run_id": "synthetic"})
    monkeypatch.setattr(
        publication, "preflight", lambda *args: (checkout, "results/synthetic-kaggle")
    )

    class Helper:
        ROOT = checkout

        def push_branch(self, branch, secret):
            git("push", "origin", "HEAD:refs/heads/" + branch)

    monkeypatch.setattr(publication.existing, "helper", lambda: Helper())
    (output / "public").mkdir(parents=True)
    (output / "figures").mkdir()
    for name in ("SCIENTIFIC-FINDINGS.md", "CLM-001-R3b-POSTHOC-PERFORMANCE.md"):
        (output / "public" / name).write_text("Synthetic CPU only\n")
    for name in ("coverage.png", "quality-pre_event.png", "quality-post_event.png"):
        (output / "figures" / name).write_bytes(b"\x89PNG\r\n\x1a\nsynthetic")
    receipt = publish(
        root, {"metrics-summary.json": {"status": "PARTIAL"}}, output, "b" * 40, "c" * 64
    )
    local = read(output / "publication.json")
    assert local["receipt_commit_sha"] != receipt["content_commit_sha"]
    assert (
        git("ls-remote", "origin", "refs/heads/results/synthetic-kaggle").stdout.split()[0]
        == local["receipt_commit_sha"]
    )
    assert all(digest(checkout / name) == sha for name, sha in receipt["files"].items())
    assert not any(name.endswith(".pt") for name in receipt["files"])


def test_authoritative_output_paths_not_display_basenames(tmp_path, monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import ModuleType
    from kaggle_posthoc.kaggle_source import KaggleOutputAPI

    extended = ModuleType("kaggle.api.kaggle_api_extended")
    extended.ApiListKernelSessionOutputRequest = SimpleNamespace
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", extended)
    requested = []
    names = ["arms/Q-R3b/identity.json", "arms/QC-R3b/identity.json"]
    payloads = {names[0]: b"Q", names[1]: b"QC"}

    def output(request):
        requested.append(request)
        return SimpleNamespace(
            files=[
                SimpleNamespace(file_name=n, url="https://storage.googleapis.com/private/" + n)
                for n in names
            ],
            next_page_token="",
        )

    api = KaggleOutputAPI.__new__(KaggleOutputAPI)
    api._output_urls = {}
    api.api = SimpleNamespace(
        _kernel_version_label=lambda version: "v" + version,
        _set_paging=lambda request, size, token: None,
        build_kaggle_client=lambda: nullcontext(
            SimpleNamespace(
                kernels=SimpleNamespace(
                    kernels_api_client=SimpleNamespace(list_kernel_session_output=output)
                )
            )
        ),
    )
    api.metadata = lambda version: {"is_private": True, "version_number": version}
    inv = inventory(api, 65)
    assert requested[0].version_label == "v65"
    assert {r["name"] for r in inv["files"]} == set(names)
    assert inv["total_bytes"] is None
    assert "storage.googleapis.com" not in json.dumps(inv)
    sizes = []
    api.size = lambda version, name: sizes.append((version, name)) or len(payloads[name])

    def download(version, selected, destination):
        for name in selected:
            p = destination / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(payloads[name])

    api.download = download
    digest_before = inv["inventory_sha256"]
    transfer = fetch(api, inv, [names[0]], tmp_path / "download")
    assert sizes == [(65, names[0])] and transfer["downloaded_bytes"] == 1
    assert inv["inventory_sha256"] == digest_before
    assert inv["files"][1]["bytes"] is None


def test_signed_get_headers_measure_size_without_consuming_body(monkeypatch):
    from contextlib import nullcontext
    from kaggle_posthoc.kaggle_source import KaggleOutputAPI
    import requests

    calls = []
    response = SimpleNamespace(headers={"Content-Length": "123"}, raise_for_status=lambda: None)
    monkeypatch.setattr(
        requests, "get", lambda *args, **kw: calls.append((args[0], kw)) or nullcontext(response)
    )
    api = KaggleOutputAPI.__new__(KaggleOutputAPI)
    api._output_urls = {
        (65, "file.pt"): "https://storage.googleapis.com/private/file.pt?signature=not-persisted"
    }
    assert api.size(65, "file.pt") == 123
    assert calls[0][1]["stream"] is True
    for url in (
        "https://www.kaggle.com/api/v1/output/file?signature=example",
        "https://example.kaggleusercontent.com/private/file.pt",
    ):
        api._output_urls[(65, "file.pt")] = url
        assert api.size(65, "file.pt") == 123
        assert calls[-1][0] == url


def test_429_retry_keeps_completed_pages_and_honors_retry_after():
    from kaggle_posthoc.kaggle_source import KaggleOutputAPI

    adapter = KaggleOutputAPI.__new__(KaggleOutputAPI)
    waits, calls = [], []
    adapter._sleep = waits.append
    adapter.api = SimpleNamespace(
        _get_retry_after_delay=lambda response: float(response.headers["Retry-After"])
    )
    api = FakeKaggle({65: {"a": b"1", "b": b"2", "c": b"3"}})
    original = api.page
    failed = False

    def request(version, token):
        nonlocal failed
        calls.append(token)
        if token == "2" and not failed:
            failed = True
            e = RuntimeError("redacted")
            e.response = SimpleNamespace(status_code=429, headers={"Retry-After": "2"})
            raise e
        return original(version, token)

    api.page = lambda version, token=None: adapter._call(
        lambda: request(version, token), "output-page", pace=False
    )
    assert inventory(api, 65)["file_count"] == 3
    assert calls == [None, "2", "2"] and waits == [2]


def test_non_transient_auth_error_is_not_retried():
    from kaggle_posthoc.kaggle_source import KaggleOutputAPI

    adapter = KaggleOutputAPI.__new__(KaggleOutputAPI)
    adapter._sleep = lambda seconds: pytest.fail("403 must not retry")
    e = RuntimeError("auth")
    e.response = SimpleNamespace(status_code=403)

    def forbidden():
        raise e

    with pytest.raises(RuntimeError, match="auth"):
        adapter._call(forbidden, "metadata", pace=False)


def test_default_worker_discovery_does_not_scan_older_versions():
    api = FakeKaggle(
        {63: {"old.pt": b"x"}, 65: {"run/continuation.json": b"{}"}, 66: {"result.json": b"{}"}}
    )
    saved = []
    rows, status = discover(api, minimum_version=65, on_inventory=saved.append)
    assert {r["version_number"] for r in rows} == {65, 66}
    assert all(version >= 65 for version, token in api.calls)
    assert saved == rows and status["older_versions"] == "DEFERRED_UNLESS_INPUT_MISSING"
    assert not status["exhaustive_to_exposed_latest"]


def test_selected_download_reuses_cli_streaming_without_relisting(tmp_path):
    from contextlib import nullcontext
    from kaggle_posthoc.kaggle_source import KaggleOutputAPI

    adapter = KaggleOutputAPI.__new__(KaggleOutputAPI)
    calls = []
    response = SimpleNamespace(raise_for_status=lambda: None)
    adapter._file_response = lambda version, name: nullcontext(response)

    def stream(response, destination, client, **kwargs):
        calls.append((destination, kwargs))

    adapter.api = SimpleNamespace(
        download_file=stream,
        kernels_output=lambda *args, **kwargs: pytest.fail(
            "do not relist 32 pages for a filtered file"
        ),
    )
    adapter.download(65, ["run/continuation.json", "run/manifest.json"], tmp_path)
    assert len(calls) == 2 and all(kwargs["resume"] and kwargs["quiet"] for path, kwargs in calls)


def test_inventory_progress_survives_later_throttle():
    api = FakeKaggle({65: {"ready": b"x"}, 66: {"later": b"y"}})
    original = api.page
    saved = []

    def page(version, token=None):
        if version == 66:
            e = RuntimeError("throttle")
            e.response = SimpleNamespace(status_code=429)
            raise e
        return original(version, token)

    api.page = page
    with pytest.raises(RuntimeError, match="throttle"):
        discover(api, minimum_version=65, on_inventory=saved.append)
    assert [r["version_number"] for r in saved] == [65]
    assert saved[0]["file_count"] == 1


def test_rate_retry_is_bounded_without_real_sleep():
    from kaggle_posthoc.kaggle_source import KaggleOutputAPI

    api = KaggleOutputAPI.__new__(KaggleOutputAPI)
    waits, calls = [], []
    api._sleep = waits.append
    api.api = SimpleNamespace(_get_retry_after_delay=lambda response: 0)

    def throttled():
        calls.append(1)
        e = RuntimeError("throttled")
        e.response = SimpleNamespace(status_code=429, headers={})
        raise e

    with pytest.raises(RuntimeError, match="throttled"):
        api._call(throttled, "page", pace=False)
    assert len(calls) == 5 and waits == [1] * 4
