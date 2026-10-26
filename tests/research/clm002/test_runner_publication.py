import contextlib
import subprocess
from pathlib import Path
import pytest
from minicells_clm002 import engineering, publication
from minicells_clm002.common import read
from minicells_jct001 import publication as transport


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


@pytest.fixture
def completed(config, ident, tmp_path, monkeypatch):
    monkeypatch.setattr(engineering, "identity", lambda c: ident)
    monkeypatch.setattr(
        engineering,
        "reserve_seeds",
        lambda: dict(
            status="UNIT_SEED_METADATA_ONLY",
            development=[1, 2, 3],
            stage_a_confirmation=[4, 5, 6, 7, 8],
            stage_b_confirmation=[9, 10, 11, 12, 13],
        ),
    )
    result = engineering.run(config, tmp_path / "output")
    return Path(result["output"])


def test_gate0_reports_are_complete_and_honest(completed, config, ident, monkeypatch):
    root = completed
    manifest = read(root / "RUN_MANIFEST.json")
    assert manifest["run_status"] == "CPU_ENGINEERING_COMPLETE_GATE0_BLOCKED"
    assert manifest["scientific_optimizer_updates"] == 0 and not manifest["gpu_used"]
    assert publication.TOP <= set(manifest["artifact_files"])
    assert len(list((root / "artifacts").rglob("RUN_MANIFEST.json"))) == 4
    assert read(root / "CPU_ENGINEERING.json")["actual_cpu_unit_optimizer_calls"] == 8
    assert read(root / "DATA_BOUNDARY_PROOF.json")["status"] == "BLOCKED"
    assert (
        read(root / "DATA_BOUNDARY_PROOF.json")["tokenizer"]["status"]
        == "PASS_LOCAL_ORIGINAL_BYTES"
    )
    assert not read(root / "ARCHITECTURE_AUDIT.json")["parent_bytes_verified"]
    assert read(root / "ARCHITECTURE_AUDIT.json")["parameters"]["total"] > 1000000
    assert read(root / "DECISION.json")["status"] == "INCOMPLETE_OR_PROTOCOL_VIOLATION"
    monkeypatch.setattr(engineering, "identity", lambda c: ident)
    assert engineering.run(config, root.parent)["reused_completed"]
    _, files, _, _ = publication.payload(root)
    assert all(not name.endswith((".pt", ".npy", ".jsonl")) for name in files)


def test_safe_remote_publish_and_idempotent_retry(completed, tmp_path, monkeypatch):
    bare = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", str(bare))
    initial = tmp_path / "initial"
    initial.mkdir()
    git(initial, "init")
    git(initial, "symbolic-ref", "HEAD", "refs/heads/main")
    git(initial, "config", "user.name", "Test")
    git(initial, "config", "user.email", "test@example.com")
    (initial / "README.md").write_text("unchanged baseline\n")
    git(initial, "add", ".")
    git(initial, "commit", "-m", "baseline")
    git(initial, "remote", "add", "origin", str(bare))
    git(initial, "push", "origin", "main")
    monkeypatch.setattr(transport, "REPOSITORY_URL", str(bare))

    @contextlib.contextmanager
    def auth():
        yield None

    monkeypatch.setattr(transport, "_auth_env", auth)
    receipt = publication.publish(completed)
    assert receipt["remote_verified"] and receipt["branch"].startswith("results/clm002-gate0-")
    assert (
        git(tmp_path, "ls-remote", str(bare), "refs/heads/" + receipt["branch"]).split()[0]
        == receipt["remote_commit_sha"]
    )
    changes = git(
        completed / "github-metadata", "diff", "--name-only", receipt["parent_main_sha"], "HEAD"
    ).splitlines()
    assert all(name.startswith(publication.REPORTS.as_posix() + "/") for name in changes)
    assert publication.publish_existing(completed)[0]["reused_existing_remote"]


def test_publication_hash_scope_and_symlink_rejection(completed):
    path = completed / "CONFIG.json"
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="HASH_OR_SIZE"):
        publication.payload(completed)
    assert not publication.permitted("private/checkpoint.pt")
    assert not publication.permitted("../CONFIG.json")


def test_independent_readonly_analysis_has_no_model_updates(completed, tmp_path):
    from minicells_clm002.analysis import analyze_existing

    report = analyze_existing(completed, tmp_path / "review")
    assert report["optimizer_updates"] == 0 and not report["model_loaded"]
    assert report["stage_a"] == report["stage_b"] == "NOT_RUN"
    branch = next((completed / "artifacts").rglob("RUN_MANIFEST.json"))
    child = read(branch)
    assert (
        child["artifact_files"]
        and child["tokenizer_identity"] == "SYNTHETIC_RANDOM_TOKEN_IDS_NO_TOKENIZER"
    )


def test_failed_engineering_still_returns_partial_report(config, ident, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(engineering, "identity", lambda c: ident)
    monkeypatch.setattr(publication, "preflight", lambda root: None)

    def fail(c):
        raise RuntimeError("CLM002_INJECTED_AUDIT_FAILURE")

    monkeypatch.setattr(engineering, "historical_audit", fail)
    monkeypatch.setattr(
        publication, "publish", lambda root: seen.append(read(root / "RUN_MANIFEST.json"))
    )
    with pytest.raises(RuntimeError, match="INJECTED"):
        engineering.run(config, tmp_path / "failed", publish_github=True)
    assert len(seen) == 1 and seen[0]["run_status"] == "CPU_ENGINEERING_FAILED_GATE0_BLOCKED"
    assert seen[0]["scientific_optimizer_updates"] == 0
