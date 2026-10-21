import subprocess
from types import SimpleNamespace
from minicells_r3c_moe_bridge import publication


def test_metadata_push_descends_main_and_contains_no_workflow(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()

    def git(*args, cwd=remote):
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()

    git("init", "--quiet")
    git("symbolic-ref", "HEAD", "refs/heads/main")
    git("config", "user.name", "CPU")
    git("config", "user.email", "cpu@example.com")
    (remote / "README.md").write_text("Main unchanged")
    workflow = remote / ".github/workflows"
    workflow.mkdir(parents=True)
    (workflow / "old.yml").write_text("old: unchanged")
    git("add", ".")
    git("commit", "--quiet", "-m", "base")
    parent = git("rev-parse", "HEAD")
    actual = publication.git

    def local(checkout, *args):
        if args[:3] == ("remote", "add", "origin"):
            args = (*args[:3], str(remote))
        return actual(checkout, *args)

    helper = SimpleNamespace(ROOT=remote, push_branch=lambda *a: None)
    monkeypatch.setattr(publication, "git", local)
    monkeypatch.setattr(publication, "helper", lambda: helper)
    summary = {
        "schema": "minicells.r3c.public-engineering.v1",
        "confirmation": "SEALED",
        "scientific_decision": False,
    }
    result = publication.publish(summary, tmp_path / "checkout")
    assert result["parent_main_sha"] == parent
    changed = git("diff", "--name-only", parent, "HEAD", cwd=tmp_path / "checkout").splitlines()
    assert (
        len(changed) == 1 and changed[0].endswith(".json") and "/reports/published/" in changed[0]
    )
    assert helper.ROOT == remote


def test_pilot_publication_preflight_and_retry_are_idempotent(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()

    def git(*args, cwd=remote):
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()

    git("init", "--quiet")
    git("symbolic-ref", "HEAD", "refs/heads/main")
    git("config", "user.name", "CPU")
    git("config", "user.email", "cpu@example.com")
    (remote / "README.md").write_text("main")
    git("add", ".")
    git("commit", "--quiet", "-m", "base")
    actual = publication.git
    events = []

    def local(checkout, *args):
        if args[:3] == ("remote", "add", "origin"):
            args = (*args[:3], str(remote))
        return actual(checkout, *args)

    helper = SimpleNamespace(ROOT=remote)
    helper.preflight_push = lambda *a: events.append("preflight")

    def push(branch, secret):
        events.append("push")
        git("push", str(remote), "HEAD:refs/heads/" + branch, cwd=helper.ROOT)

    helper.push_branch = push
    monkeypatch.setattr(publication, "git", local)
    monkeypatch.setattr(publication, "helper", lambda: helper)
    monkeypatch.setattr(
        publication,
        "pilot_public_summary",
        lambda r: {
            "schema": "minicells.r3c.public-pilot.v1",
            "confirmation": "SEALED",
            "scientific_decision": False,
        },
    )
    first = publication.prepare_pilot_publication(tmp_path / "first", "a" * 64)
    written = publication.publish_pilot_result({}, first)
    second = publication.prepare_pilot_publication(tmp_path / "second", "a" * 64)
    retried = publication.publish_pilot_result({}, second)
    assert written["branch"] == retried["branch"] and retried["status"] == "ALREADY_PUBLISHED"
    assert events == ["preflight", "push", "preflight"] and helper.ROOT == remote
