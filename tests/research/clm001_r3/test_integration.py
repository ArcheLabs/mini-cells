import numpy as np
import pytest
from clm001_r3.model import Model
from clm001_r3.training import Trainer
from clm001_r3.shadows import Nursery
from clm001_r3.checkpoint import save, restore, identity
from clm001_r3.events import process
from clm001_r3.common import load, authorize, hash_accepted_state, source_sha
from clm001_r3.automatic import select_beta, runtime, device_queues
from clm001_r3.report import primary
from test_credit import batch


@pytest.mark.parametrize("arm", ["Q-R3", "QC-R3", "QS-R3"])
@pytest.mark.parametrize("kind", ["cell", "connection", "path"])
def test_real_private_probation_and_failure_isolation(config, arm, kind):
    t = Trainer(Model(config, arm))
    n = Nursery()
    candidate = n.propose(t.model, kind, t.opt)
    assert n.probation(candidate, t.model, batch(), 1, accepted_trainer=t)
    assert candidate.state == "PROBATION"
    # Exact R3 callback, including Connection-only post-gradient observation, not a stub.
    from clm001_r3.pilot import private_fault

    before = hash_accepted_state(t.model)
    out = private_fault(t.model, t, batch(), kind)
    assert out["accepted_witness_ok"] and out["backward_hook_fired"]
    assert hash_accepted_state(t.model) == before and t.step == 0


def test_exact_restart_preserves_competition_stream_and_parameters(config, tmp_path):
    t = Trainer(Model(config, "QC-R3"))
    t.update(batch())
    rng = np.random.default_rng(3)
    n = Nursery()
    save(tmp_path, t, n, rng, {"experiment_id": "CLM-001-R3"})
    expected = t.update(batch())[0]
    sha = hash_accepted_state(t.model)
    other = Trainer(Model(config, "QC-R3"))
    restore(tmp_path, other, rng, {"experiment_id": "CLM-001-R3"})
    actual = other.update(batch())[0]
    assert hash_accepted_state(other.model) == sha
    assert (
        actual["lcg_loss"] == expected["lcg_loss"]
        and actual["lcg_q_cosine"] == expected["lcg_q_cosine"]
    )
    with pytest.raises(ValueError):
        restore(tmp_path, other, rng, {"experiment_id": "CLM-001-R2"})


def test_committed_step400_events_resume_without_main_update(config, assets, tmp_path):
    folder, data = assets
    t = Trainer(Model(config, "QC-R3"))
    t.update(batch())
    t.step = 400
    n = Nursery()
    rng = np.random.default_rng(2)
    for kind in ("cell", "connection", "path"):
        n.propose(t.model, kind, t.opt)
    run_id = identity(config, "QC-R3", source_sha(), data)
    out = tmp_path / "run"
    save(out, t, n, rng, run_id)

    def crash(kind, stage):
        if kind == "admit-cell" and stage == "after":
            raise RuntimeError("power loss")

    with pytest.raises(RuntimeError):
        process(config, t, n, data, rng, out, run_id, crash=crash)
    other = Trainer(Model(config, "QC-R3"))
    nn = restore(out, other, rng, run_id)
    process(config, other, nn, data, rng, out, run_id)
    before = hash_accepted_state(other.model)
    count = len(nn.events)
    process(config, other, nn, data, rng, out, run_id)
    assert other.step == 400 and other.data_position == 4 and len(nn.events) == count == 5
    assert hash_accepted_state(other.model) == before
    assert (out / "checkpoint-step-000400.pt").exists()
    assert any("policies" in value for value in nn.events.values())


def test_beta_freeze_and_automatic_identity():
    c = load()
    assert c["competition"]["beta"] is None
    pilot = {"gradient_rows": [{"lcg_grad_norm": 2.0, "q_grad_norm": 4.0}] * 3}
    beta, evidence = select_beta(c, pilot)
    assert beta == 1.0 and evidence["scaled_ratio_median"] == 0.5
    frozen = runtime(c, {"asset_sha256": "a", "tokenizer_sha256": "b"}, beta, "pilot-hash")
    authorize(frozen, "train-dev", "cuda:0")
    with pytest.raises(RuntimeError):
        authorize(c, "train-dev", "cuda:0")
    frozen["training"]["learning_rate"] *= 2
    with pytest.raises(RuntimeError):
        authorize(frozen, "train-dev", "cuda:0")
    with pytest.raises(RuntimeError):
        select_beta(c, {"gradient_rows": []})


def test_three_arms_use_only_two_devices():
    queues = device_queues(load())
    assert queues == [["Q-R3", "QS-R3"], ["QC-R3"]]
    assert len(set(a for queue in queues for a in queue)) == 3


def test_primary_is_paired_difference_in_differences(config):
    def policies(greedy, uniform):
        return {
            kind: {
                "nll": [value] * 4,
                "cost": [0.5] * 4,
                "groups": [0, 0, 1, 1],
                "paired_identity": "matched",
            }
            for kind, value in [("greedy", greedy), ("uniform", uniform)]
        }

    values = {
        arm: {"quality": policies(2.0 if arm == "QC-R3" else 3.0, 2.5)} for arm in config["arms"]
    }
    value = primary(config, values, "final")
    assert value["status"] == "EXPLORATORY_DEV_SIGNAL"
    assert value["contrasts"]["Q-R3"]["nll"]["mean"] == -1.0
    assert value["scientific_decision"] is False


@pytest.fixture
def short_assets(config, request):
    config["shadow"].update(cell_proposal_step=1, connection_path_step=1, admission_step=2)
    config["evaluation"]["snapshots"] = [0, 1, 2]
    config["evaluation"]["periodic_diagnostics_enabled"] = True
    config["training"].update(checkpoint_interval=1, evaluation_interval=1)
    return request.getfixturevalue("assets")


def test_complete_three_arm_cpu_lifecycle_and_publication_aggregation(
    config, short_assets, tmp_path
):
    from clm001_r3.runner import development
    from clm001_r3.report import aggregate, verified
    from clm001_r3.checkpoint import verify_portable

    folder, data = short_assets
    output = tmp_path / "full"
    for arm in config["arms"]:
        result = development(config, arm, output / "arms" / arm, folder, "cpu")
        assert result["training_status"] == "COMPLETE"
        assert result["resource_ledger"]["live_training_tokens"] == 8
        assert result["specialization"]["snapshots"] == [0, 1, 2]
        assert verify_portable(output / "arms" / arm / "portable", result["identity"])
        assert (
            development(config, arm, output / "arms" / arm, folder, "cpu")["identity"]
            == result["identity"]
        )
    report = aggregate(config, output, source_sha())
    assert report["training_status"] == "COMPLETE" and not report["scientific_decision"]
    assert verified(output / "pre-admission-contrast.json")["endpoint"].startswith("DID_")


def test_lean_run_skips_all_periodic_diagnostics(config, short_assets, tmp_path, monkeypatch):
    from clm001_r3 import runner

    folder, _ = short_assets
    config["evaluation"]["periodic_diagnostics_enabled"] = False

    def forbidden(*args, **kwargs):
        raise AssertionError("periodic snapshot executed in lean mode")

    monkeypatch.setattr(runner, "record_snapshot", forbidden)
    result = runner.development(config, "Q-R3", tmp_path / "lean", folder, "cpu")
    assert result["training_status"] == "COMPLETE"
    assert result["specialization"]["snapshots"] == []
    assert not list((tmp_path / "lean").glob("development-step-*.json"))
    assert result["quality"]
