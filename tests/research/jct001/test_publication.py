import contextlib
import subprocess
from pathlib import Path
import pytest
from minicells_jct001 import publication as p, runner
from minicells_jct001.common import read


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


@pytest.fixture
def remote(tmp_path, monkeypatch):
    bare = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", str(bare))
    checkout = tmp_path / "main"
    checkout.mkdir()
    git(checkout, "init")
    git(checkout, "symbolic-ref", "HEAD", "refs/heads/main")
    git(checkout, "config", "user.email", "test@example.com")
    git(checkout, "config", "user.name", "Test")
    (checkout / "README.md").write_text("baseline\n")
    (checkout / ".github/workflows").mkdir(parents=True)
    (checkout / ".github/workflows/untouched.yml").write_text("name: untouched\n")
    git(checkout, "add", ".")
    git(checkout, "commit", "-m", "baseline")
    git(checkout, "remote", "add", "origin", str(bare))
    git(checkout, "push", "origin", "main")
    monkeypatch.setattr(p, "REPOSITORY_URL", str(bare))

    @contextlib.contextmanager
    def auth():
        yield None

    monkeypatch.setattr(p, "_auth_env", auth)
    return bare


def complete(config, identity, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "run_identity", lambda c, registry: identity)
    return Path(runner.engineering(config, tmp_path / "out")["output"])


def test_result_only_remote_roundtrip(config, identity, tmp_path, monkeypatch, remote):
    root = complete(config, identity, tmp_path, monkeypatch)
    p.preflight(root)
    receipt = p.publish(root)
    assert receipt["status"] == "PUBLISHED" and receipt["remote_verified"]
    assert (
        git(tmp_path, "ls-remote", str(remote), "refs/heads/" + receipt["branch"]).split()[0]
        == receipt["remote_commit_sha"]
    )
    changes = git(
        root / "github-metadata", "diff", "--name-only", receipt["parent_main_sha"], "HEAD"
    ).splitlines()
    assert all(
        name.startswith(p.REPORTS.as_posix() + "/") and not name.endswith(".pt") for name in changes
    )
    assert p.publish(root)["reused_existing_remote"]
    assert read(root / "RUN_MANIFEST.json")["remote_commit_sha"] == receipt["remote_commit_sha"]


def test_corrupted_or_symlink_artifact_rejected(config, identity, tmp_path, monkeypatch):
    root = complete(config, identity, tmp_path, monkeypatch)
    (root / "BUDGET_PROOF.json").write_text("{}")
    with pytest.raises(p.PublicationError, match="HASH_MISMATCH"):
        p.payload(root)
    target = root / "ARM_RESULT.json"
    target.unlink()
    target.symlink_to(root / "RUN_MANIFEST.json")
    with pytest.raises(p.PublicationError, match="NOT_REGULAR"):
        p.payload(root)


def test_failed_push_keeps_local_and_retry_no_updates(
    config, identity, tmp_path, monkeypatch, remote
):
    root = complete(config, identity, tmp_path, monkeypatch)
    original = p._git

    def fail(checkout, *args, **kw):
        if args[0] == "push":
            raise RuntimeError("SIMULATED_NETWORK_FAILURE")
        return original(checkout, *args, **kw)

    monkeypatch.setattr(p, "_git", fail)
    with pytest.raises(p.PublicationError):
        p.publish(root)
    receipt = read(root / "PUBLICATION.json")
    assert receipt["status"] == "FAILED_LOCAL_RETAINED" and not receipt["remote_verified"]
    assert read(root / "RUN_MANIFEST.json")["cpu_fixture_optimizer_calls"] == 6
    monkeypatch.setattr(p, "_git", original)
    assert p.publish_existing(root)[0]["remote_verified"]
    assert read(root / "RUN_MANIFEST.json")["cpu_fixture_optimizer_calls"] == 6
