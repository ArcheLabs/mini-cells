import pytest
from clm001_r2.model import Model
from clm001_r2.training import Trainer
from clm001_r2.shadows import Nursery
from clm001_r2.pilot import private_fault
from clm001_r2.cli import collect_pilot
from clm001_r2.report import seal, verified, aggregate
from clm001_r2.common import hash_accepted_state
from test_relative import batch


@pytest.mark.parametrize("arm", ["QF-R2", "J0-R2"])
@pytest.mark.parametrize("kind", ["cell", "connection", "path"])
def test_real_private_injection_all_arms_and_candidates(config, arm, kind):
    trainer = Trainer(Model(config, arm))
    before = hash_accepted_state(trainer.model)
    result = private_fault(trainer.model, trainer, batch(config), kind)
    assert result["backward_hook_fired"] and result["accepted_witness_ok"]
    assert result["failure_code"] == "CANDIDATE_NUMERICAL_FAILURE"
    assert hash_accepted_state(trainer.model) == before and not trainer.opt.state
    if arm == "J0-R2" and kind == "path":
        assert any(name.startswith("controller.critic.") for name in result["injected_parameters"])
        assert not any(
            name.startswith("controller.scorer.") for name in result["injected_parameters"]
        )


def test_fixed_j0_path_does_not_run_scorer_backward(config):
    trainer = Trainer(Model(config, "J0-R2"))
    nursery = Nursery()
    candidate = nursery.propose(trainer.model, "path", trainer.opt)
    fired = []
    param = next(
        p for name, p in candidate.model.named_parameters() if name.startswith("controller.scorer.")
    )
    handle = param.register_hook(lambda gradient: fired.append(True) or gradient)
    try:
        assert nursery.probation(
            candidate, trainer.model, batch(config), 1, accepted_trainer=trainer
        )
    finally:
        handle.remove()
    assert not fired  # Reproduces why the original Pilot never injected its intended fault.


@pytest.mark.parametrize("codes", [[0, 1], [1, 0], [1, 1]])
def test_missing_pilot_success_report_preserves_sibling(config, tmp_path, codes):
    for arm, code in zip(config["arms"], codes):
        folder = tmp_path / "preflight" / arm
        if not code:
            seal(
                folder / "gpu-pilot.json",
                {"arm": arm, "status": "GPU_PILOT_PASS", "p50_step_seconds": 1.0},
            )
        else:
            seal(
                folder / "gpu-pilot-failure.json",
                {"arm": arm, "error_type": "RuntimeError", "message": "original failure"},
            )
    result = collect_pilot(config, tmp_path, codes)
    assert not result["passed"] and len(result["failures"]) == sum(bool(code) for code in codes)
    assert len(result["arms"]) == sum(code == 0 for code in codes)
    report = aggregate(config, tmp_path, "a" * 40)
    assert report["gpu_pilot"] == verified(tmp_path / "preflight-summary.json")
    assert all(f["message"] == "original failure" for f in report["gpu_pilot"]["failures"])


def test_absent_failure_report_has_actionable_fallback(config, tmp_path):
    result = collect_pilot(config, tmp_path, [1, 1])
    assert not result["passed"]
    assert all("worker.log" in v["message"] for v in result["failures"])
