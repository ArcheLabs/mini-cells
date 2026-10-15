from fast001.io import write_json
import json
import copy
import subprocess
import threading
import time
import pytest
from fast001.config import BASE, ROOT, ARMS, validate
from fast001.schedule import matrix
from fast001.report import report
from fast001.benchmark import benchmark, validate_pilot


def test_dynamic_scheduler_starts_clms_and_free_gpu_takes_moe(tiny, tmp_path, monkeypatch):
    import fast001.schedule as schedule

    events = []
    lock = threading.Lock()

    class Worker:
        def __init__(self, cmd, **kwargs):
            self.arm = cmd[cmd.index("--arm") + 1]
            self.device = cmd[cmd.index("--device") + 1]
            self.stdout = iter([])
            with lock:
                events.append((self.arm, self.device))

        def wait(self):
            time.sleep(0.06 if self.arm == "CLM-S16" else 0.005)
            return 0

    monkeypatch.setattr(schedule.subprocess, "Popen", Worker)
    monkeypatch.setattr(schedule, "authorize", lambda *args: None)
    result = matrix(
        tiny, tmp_path / "config.json", tmp_path / "assets", tmp_path / "work", ["cpu", "cpu:1"]
    )
    assert events[:2] == [("CLM-S16", "cpu"), ("CLM-S8", "cpu:1")] or set(events[:2]) == {
        ("CLM-S16", "cpu"),
        ("CLM-S8", "cpu:1"),
    }
    assert events[-1] == ("M0-TINY", "cpu:1")
    assert result["workers_joined"] and result["status"] == "DEVELOPMENT_COMPLETED"


def test_failed_worker_preserves_other_progress(tiny, tmp_path, monkeypatch):
    import fast001.schedule as schedule

    class Worker:
        def __init__(self, cmd, **kwargs):
            self.arm = cmd[cmd.index("--arm") + 1]
            self.stdout = iter([])

        def wait(self):
            return 1 if self.arm == "CLM-S16" else 0

    monkeypatch.setattr(schedule.subprocess, "Popen", Worker)
    monkeypatch.setattr(schedule, "authorize", lambda *args: None)
    result = matrix(
        tiny, tmp_path / "config.json", tmp_path / "assets", tmp_path / "work", ["cpu", "cpu:1"]
    )
    assert result["status"] == "INCOMPLETE"
    assert any(
        r["arm"] == "M0-TINY" and r["status"] == "DEVELOPMENT_COMPLETED" for r in result["workers"]
    )


def test_incomplete_report_has_no_success_or_contrasts(proposed, tmp_path):
    row = report(proposed, tmp_path)
    assert row["status"] == "INCOMPLETE" and row["decision"] == "INCOMPLETE"
    assert all(v is None for v in row["contrasts"].values())
    assert row["scientific_decision"] is False


def test_exact_three_arms_and_budgets_only(proposed):
    for change in ("extra", "J1", "cost", "budget", "confirmation"):
        c = copy.deepcopy(proposed)
        if change == "extra":
            c["arms"]["fourth"] = {}
        elif change == "J1":
            c["arms"]["CLM-S8"]["mechanism"] = "J1"
        elif change == "cost":
            c["training"]["cost_lambda"] = 0.1
        elif change == "budget":
            c["training"]["tokens"] += 1
        else:
            c["approval"]["confirmation"] = True
        with pytest.raises(ValueError):
            validate(c)


def test_cpu_pilot_cannot_authorize_gpu_training(tiny, tmp_path):
    # Fixture architecture, single short CPU optimizer step per arm, no real data.
    result = benchmark(
        tiny, tmp_path / "pilot", "cpu", microbatches=(1,), warmup=0, steps=1, fixture=True
    )
    assert result["status"] == "CPU_ENGINEERING_ONLY"
    assert len(result["rows"]) == 3
    with pytest.raises(RuntimeError):
        validate_pilot(tiny, result)


def test_history_diff_is_append_only():
    paths = subprocess.check_output(
        ["git", "diff", "--name-only", "91b1340"], cwd=ROOT, text=True
    ).splitlines()
    allowed = (
        str(BASE.relative_to(ROOT)) + "/",
        "tests/research/path_first_fast001/",
        ".github/workflows/fast001-cpu.yml",
    )
    assert all(any(p.startswith(prefix) for prefix in allowed) for p in paths)


def test_notebook_code_and_default_authorization():
    import ast

    notebook = json.loads((BASE / "notebook.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    tree = ast.parse(code)
    settings = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert settings["MODE"] == "preflight"
    assert settings["ALLOW_GPU_PILOT"] is False and settings["APPROVE_DEVELOPMENT"] is False


def test_pause_checkpoint_config_cannot_be_changed_after_approval(proposed, tmp_path, monkeypatch):
    from fast001.config import freeze, authorize
    import fast001.benchmark as bench

    monkeypatch.setattr(
        bench, "validate_pilot", lambda *a: None
    )  # unit-freeze mechanics only, no GPU claim
    frozen = freeze(
        proposed, {"projected_total_seconds": 1}, tmp_path / "frozen.json", approve=True
    )
    frozen["training"]["learning_rate"] *= 2
    with pytest.raises(RuntimeError, match="configuration was changed"):
        authorize(frozen, "cpu", "train")


def test_complete_synthetic_pipeline_report_has_provenance_and_no_scientific_win(
    tiny, assets, tmp_path
):
    from fast001.train import train
    from fast001.evaluate import profile
    from fast001.interventions import suite

    work = tmp_path / "matrix"
    for arm in ARMS:
        folder = work / "runs" / arm
        train(tiny, arm, assets, folder)
        profile(tiny, arm, assets, folder, "cpu")
        if arm != "M0-TINY":
            suite(tiny, arm, assets, folder, "cpu")
    result = report(tiny, work)
    assert result["status"] == "DEVELOPMENT_COMPLETED"
    assert result["decision"] == "ENGINEERING_SMOKE_ONLY"
    assert result["research_claim"] == "ENGINEERING_SMOKE_ONLY" and result["synthetic_data"]
    assert set(result["cost_and_latency"]) == set(ARMS)
    assert all(result["arms"][a]["final_dev_nll"] is not None for a in ARMS)
    assert result["contrasts"]["paired_story_bootstrap"]["depth"]["replicates"] == 40
    result["arms"]["CLM-S8"]["identity"]["asset_sha256"] = "f" * 64
    write_json(work / "runs/CLM-S8/metrics.json", result["arms"]["CLM-S8"])
    with pytest.raises(ValueError, match="asset_sha"):
        report(tiny, work)
