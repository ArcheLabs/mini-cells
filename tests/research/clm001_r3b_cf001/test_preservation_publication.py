from pathlib import Path
import hashlib
import json
import shutil
from types import SimpleNamespace
import pytest
from minicells_runtime.store import canonical, read, ContinuationError
from r3b_cf001 import preservation
from r3b_cf001.common import ENGINE_SHA, RESULT_RELATIVE


@pytest.fixture
def old59(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    code = tmp_path / "code"
    remote.mkdir()
    ledger = []
    names = []
    for i in range(59):
        identity = {
            "analysis_source_sha": ENGINE_SHA,
            "stage": "quality",
            "policies": ["greedy"],
            "synthetic_fixture": i,
        }
        result = {
            "identity": identity,
            "task_status": "COMPLETE",
            "optimizer_updates": 0,
            "training_state_unchanged": True,
            "sample_count": 1,
            "result": {"quality": {"greedy": {"nll": [1.0], "cost": [0.5], "groups": [i]}}},
        }
        data = (json.dumps(result) + "\n").encode()
        sha = hashlib.sha256(data).hexdigest()
        name = "saved/private-results/objects/sha256/" + sha + ".json"
        file = remote / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
        names.append(name)
        ledger.append(
            {
                "task_key": canonical(identity),
                "result_sha256": sha,
                "persistence": "COMPUTED_LOCAL",
                "arm": "QC-R3b" if i == 58 else "Q-R3b",
                "step": 500,
                "stage": "quality",
                "policies": ["greedy"],
            }
        )
    ledger.extend({"persistence": "HF_VERIFIED"} for _ in range(43))
    file = code / RESULT_RELATIVE / "coverage-ledger.json"
    file.parent.mkdir(parents=True)
    file.write_text(json.dumps({"tasks": ledger}))

    def download(api, inv, selected, destination):
        for name in selected:
            path = Path(destination) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(remote / name, path)

    monkeypatch.setattr(preservation, "fetch", download)
    inv = {"version_number": 99, "files": [{"name": n} for n in names]}
    return remote, code, inv


def test_old59_exact_saved_readback_and_prior43_separate(old59, tmp_path):
    remote, code, inv = old59
    report, reference = preservation.audit_old(None, [inv], tmp_path / "cache", code)
    assert report["status"] == "OLD_ANALYSIS_PERSISTENCE_VERIFIED"
    assert report["verified_saved_results"] == 59 and report["previous_hf_verified_results"] == 43
    assert (
        report["old_diagnostics_recomputed"] == 0
        and reference["identity"]["synthetic_fixture"] == 58
    )


def test_missing59_does_not_become_verified(old59, tmp_path):
    _, code, inv = old59
    inv["files"].pop()
    report, reference = preservation.audit_old(None, [inv], tmp_path / "cache", code)
    assert report["status"] == "NOT_VERIFIED" and report["verified_saved_results"] == 58
    assert len(report["missing_task_keys"]) == 1 and reference is None


def test_tampered_raw_saved_result_fails_closed(old59, tmp_path):
    remote, code, inv = old59
    (remote / inv["files"][0]["name"]).write_text("{}")
    with pytest.raises(ContinuationError):
        preservation.audit_old(None, [inv], tmp_path / "cache", code)


def test_timeout_retains_verified_file_progress_without_false_completion(
    old59, tmp_path, monkeypatch
):
    import requests

    _, code, inv = old59
    original = preservation.fetch
    count = []
    snapshots = []

    def interrupted(api, inventory, names, destination):
        count.append(list(names))
        if len(count) == 2:
            raise requests.exceptions.ReadTimeout("synthetic header timeout")
        return original(api, inventory, names, destination)

    monkeypatch.setattr(preservation, "fetch", interrupted)
    with pytest.raises(requests.exceptions.ReadTimeout):
        preservation.audit_old(None, [inv], tmp_path / "cache", code, on_progress=snapshots.append)
    assert len(count) == 2 and all(len(names) == 1 for names in count)
    assert snapshots[-1]["verified_saved_results"] == 1
    assert snapshots[-1]["current_file"]["status"] == "READING"
    assert snapshots[-1]["status"] == "NOT_VERIFIED"
    assert len(snapshots[-1]["missing_task_keys"]) == 58


def test_publication_only_metadata_no_workflow_changes(tmp_path, monkeypatch):
    import subprocess
    from r3b_cf001 import publication

    def git(cwd, *args):
        return subprocess.run(["git", *args], cwd=cwd, check=True, text=True, capture_output=True)

    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", str(remote))
    checkout = tmp_path / "checkout"
    git(tmp_path, "clone", str(remote), str(checkout))
    git(checkout, "config", "user.name", "Test")
    git(checkout, "config", "user.email", "test@example.com")
    (checkout / "README.md").write_text("original")
    git(checkout, "add", "README.md")
    git(checkout, "commit", "-m", "base")
    base = git(checkout, "rev-parse", "HEAD").stdout.strip()
    branch = "results/test-cf001"
    git(checkout, "checkout", "-b", branch)
    helper = SimpleNamespace(
        ROOT=checkout,
        push_branch=lambda name, secret: git(checkout, "push", "origin", "HEAD:refs/heads/" + name),
    )
    monkeypatch.setattr(publication.existing, "helper", lambda: helper)
    reports = {"experiment-manifest.json": {"status": "SYNTHETIC_TEST", "training_updates": 0}}
    result = publication.publish(
        (checkout, branch, {}), reports, {"run": "test"}, tmp_path / "output"
    )
    assert result["status"] == "PUBLISHED"
    changed = git(checkout, "diff", "--name-only", base, "HEAD").stdout.splitlines()
    assert all("counterfactual-stop-route/reports/" in n for n in changed)
    assert not any(".github" in n or n.endswith(".pt") for n in changed)
    receipt = read(tmp_path / "output/publication.json")
    assert (
        git(checkout, "ls-remote", "origin", "refs/heads/" + branch).stdout.split()[0]
        == receipt["receipt_commit_sha"]
    )
    with pytest.raises(ContinuationError):
        publication.publish(
            (checkout, branch, {}), {"raw-state.json": {}}, {"run": "test"}, tmp_path / "output"
        )


def test_aggregate_reports_private_payload_rejected():
    from clm001_r3b.publication import reject_private_payload
    from r3b_cf001.statistics import paired

    baseline = {"a": {"nll": [1, 2], "cost": [1, 1]}, "b": {"nll": [2, 3], "cost": [1, 1]}}
    changed = {k: {**r, "touched": [True, False]} for k, r in baseline.items()}
    reject_private_payload(paired(baseline, changed, 0.01, replicates=100))
    with pytest.raises(ContinuationError):
        reject_private_payload({"target": [1]})
    with pytest.raises(ContinuationError):
        reject_private_payload({"nll": [1, 2]})
