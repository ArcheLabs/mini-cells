from types import SimpleNamespace
import json
import subprocess
import pytest
from clm001 import publication
from clm001.common import BASE, ROOT


def git(cwd, *args):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def test_metadata_publisher_uses_current_main_and_preserves_previous_results(tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    remote.mkdir()
    git(remote, "init", "--bare")
    base = tmp_path / "main"
    base.mkdir()
    git(base, "init")
    git(base, "checkout", "-b", "main")
    git(base, "config", "user.name", "Test")
    git(base, "config", "user.email", "test@example.com")
    (base / ".github/workflows").mkdir(parents=True)
    (base / ".github/workflows/ci.yml").write_text("name: existing\n")
    (base / "README.md").write_text("main\n")
    git(base, "add", ".")
    git(base, "commit", "-m", "main")
    git(base, "remote", "add", "origin", str(remote))
    git(base, "push", "origin", "main")
    calls = []
    h = SimpleNamespace(ROOT=None)

    def preflight(branch, secret):
        calls.append((branch, secret))
        assert secret == "GITHUB_TOKEN"

    def push(branch, secret):
        git(h.ROOT, "push", "origin", "HEAD:refs/heads/" + branch)

    h.preflight_push = preflight
    h.push_branch = push
    monkeypatch.setattr(publication, "helper", lambda: h)
    original = publication.git

    def local(checkout, *args):
        if args[:3] == ("remote", "add", "origin"):
            args = (*args[:3], str(remote))
        return original(checkout, *args)

    monkeypatch.setattr(publication, "git", local)
    output = tmp_path / "output"
    output.mkdir()
    (output / "report.json").write_text(
        json.dumps({"experiment_id": "CLM-001-QFIRST", "status": "INCOMPLETE"})
    )
    (output / "weights.pt").write_bytes(b"not metadata")
    (output / "train.npy").write_bytes(b"never publish")
    checkout = tmp_path / "result-checkout"
    result = publication.publish(output, checkout, "a" * 40, {"seed": 1})
    first = result["commit"]
    branch = result["branch"]
    assert calls
    names = git(checkout, "diff", "--name-only", "origin/main", "HEAD").splitlines()
    assert names and all(n.startswith(str(BASE.relative_to(ROOT)) + "/artifacts/") for n in names)
    assert not any(n.endswith((".pt", ".npy")) for n in names)
    # A newly changed main workflow is inherited from main, rather than experiment source.
    (base / ".github/workflows/ci.yml").write_text("name: current-main\n")
    git(base, "add", ".")
    git(base, "commit", "-m", "main advanced")
    git(base, "push", "origin", "main")
    (output / "report.json").write_text(
        json.dumps({"experiment_id": "CLM-001-QFIRST", "status": "DEVELOPMENT_COMPLETED"})
    )
    result = publication.publish(output, checkout, "a" * 40, {"seed": 1})
    git(checkout, "merge-base", "--is-ancestor", first, "HEAD")
    assert not git(checkout, "diff", "origin/main", "HEAD", "--", ".github/")
    assert result["branch"] == branch


def test_no_whitelisted_result_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(publication, "prepare", lambda *args: (tmp_path, "fake"))
    output = tmp_path / "out"
    output.mkdir()
    (output / "private.pt").write_bytes(b"private")
    with pytest.raises(ValueError):
        publication.publish(output, tmp_path, "a" * 40, {"seed": 1})
