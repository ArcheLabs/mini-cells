"""CPU controller integration, mocking only remote gates/device launch."""

from minicells_runtime.store import atomic, read, digest
from test_hf_storage import FakeHub


def test_explicit_train_controller_without_github_is_defined_and_cpu_only(
    case, tmp_path, monkeypatch, capsys
):
    import sys
    from clm001_r3b import cli, preflight, coordinator, plots, runner
    from clm001_r3.common import source_sha

    candidate, _, original = case
    candidate_file = tmp_path / "candidate.json"
    atomic(candidate_file, candidate)
    approval = tmp_path / "approval.json"
    atomic(approval, {})
    pilot_file = tmp_path / "pilot.json"
    atomic(
        pilot_file,
        {"beta": 0.1, "assets_manifest_sha256": digest(original / "assets/manifest.json")},
    )
    hub = FakeHub(tmp_path / "hub")
    monkeypatch.setattr(cli, "HubBackend", lambda repo: hub)
    monkeypatch.setattr(preflight, "engineering_evidence", lambda path: {})
    workspace = tmp_path / "work"

    def prepared(*args, **kwargs):
        import shutil

        workspace.mkdir(exist_ok=True)
        shutil.copytree(original / "assets", workspace / "assets")
        return {
            "assets_manifest_sha256": digest(original / "assets/manifest.json"),
            "acceptance": {},
        }

    monkeypatch.setattr(preflight, "run", prepared)

    def cpu_controller(root, live, end, store, **kwargs):
        for arm in ("Q-R3b", "QC-R3b", "QS-R3b"):
            runner.segment(root, arm, 1)
        runner.package(root)
        return store.publish_committed_checkpoint(root)

    monkeypatch.setattr(coordinator, "run", cpu_controller)
    monkeypatch.setattr(plots, "render", lambda root: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "r3b",
            "--mode",
            "train",
            "--config",
            str(candidate_file),
            "--output",
            str(workspace),
            "--authorization",
            str(approval),
            "--pilot-report",
            str(pilot_file),
        ],
    )
    cli.main()
    assert read(workspace / "run/continuation.json")["source_sha"] == source_sha()
    assert '"GITHUB_RESULTS_BRANCH": null' in capsys.readouterr().out
