import numpy as np
import pytest
import torch
from clm001_r2.model import Model
from clm001_r2.training import Trainer, NumericalFailure
from clm001_r2.shadows import Nursery
from clm001_r2.checkpoint import save, restore, identity
from clm001_r2.common import hash_accepted_state, source_sha, validate, load, authorize
from clm001_r2.oracle import OracleUntrusted
from clm001_r2.admission import assess, promote
from clm001_r2.events import process
from clm001_r2.report import aggregate, seal, verified
from clm001_r2.runner import development
from test_relative import batch


@pytest.mark.parametrize("kind", ["cell", "connection", "path"])
@pytest.mark.parametrize("arm", ["QF-R2", "J0-R2"])
def test_private_success_and_frozen(config, kind, arm):
    m = Model(config, arm)
    t = Trainer(m)
    n = Nursery()
    v = n.propose(m, kind, t.opt)
    before = {k: x.clone() for k, x in v.model.state_dict().items()}
    parent = hash_accepted_state(m)
    assert n.probation(v, m, batch(config), 1)
    assert v.state == "PROBATION" and hash_accepted_state(m) == parent
    for name, value in v.model.state_dict().items():
        if not name.startswith("controller.") and not (
            kind == "cell" and name.startswith("cells.2.")
        ):
            assert torch.equal(value, before[name])
    assert not t.opt.state


@pytest.mark.parametrize("kind", ["cell", "connection", "path"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_private_fault_isolated(config, kind, value):
    t = Trainer(Model(config, "QF-R2"))
    n = Nursery()
    v = n.propose(t.model, kind, t.opt)
    witness = hash_accepted_state(t.model)
    rng = {k: g.get_state().clone() for k, g in t.rng.items()}
    name, p = next((name, p) for name, p in v.model.named_parameters() if p.requires_grad)
    hook = p.register_hook(lambda g: torch.full_like(g, value))
    assert not n.probation(v, t.model, batch(config), 1)
    hook.remove()
    assert v.state == "FAILED" and v.metrics["failure"]["details"]["parameter_name"] == name
    assert hash_accepted_state(t.model) == witness
    assert all(torch.equal(g.get_state(), rng[k]) for k, g in t.rng.items())
    t.update(batch(config))
    assert t.step == 1


def test_oracle_corruption_not_swallowed(config):
    t = Trainer(Model(config, "QF-R2"))
    n = Nursery()
    v = n.propose(t.model, "connection")

    def fail(*args, **kwargs):
        raise OracleUntrusted("snapshot changed")

    v.trainer.update = fail
    with pytest.raises(OracleUntrusted):
        n.probation(v, t.model, batch(config), 1)


def test_rebase_unique_seeds_and_moments(config):
    t = Trainer(Model(config, "QF-R2"))
    t.update(batch(config))
    n = Nursery()
    candidates = [n.propose(t.model, k, t.opt) for k in ("cell", "connection", "path")]
    for v in candidates:
        n.rebase(v, t.model, t.opt)
    assert len({v.trainer.rng["live"].initial_seed() for v in candidates}) == 3
    assert all(not v.trainer.opt.state for v in candidates)


def test_restart_exact(config, tmp_path):
    t = Trainer(Model(config, "QF-R2"))
    n = Nursery()
    rng = np.random.default_rng(1)
    t.update(batch(config))
    v = n.propose(t.model, "cell", t.opt)
    n.probation(v, t.model, batch(config), 1)
    save(tmp_path, t, n, rng, {"id": "r2"})
    t.update(batch(config))
    expected = hash_accepted_state(t.model)
    other = Trainer(Model(config, "QF-R2"))
    restored = restore(tmp_path, other, np.random.default_rng(2), {"id": "r2"})
    other.update(batch(config))
    assert hash_accepted_state(other.model) == expected
    assert hash_accepted_state(restored.candidates[0].model) == hash_accepted_state(v.model)
    with pytest.raises(ValueError):
        restore(tmp_path, other, rng, {"id": "r1"})


@pytest.mark.parametrize(
    "point",
    [
        ("pre-admission", "before"),
        ("pre-admission", "after"),
        ("admit-cell", "after"),
        ("promote", "after"),
    ],
)
def test_400_crash_restart_exactly_once(config, assets, tmp_path, point):
    assetfolder, data = assets
    m = Model(config, "QF-R2")
    t = Trainer(m)
    n = Nursery()
    rng = np.random.default_rng(9)
    t.update(batch(config))
    t.step = 400
    for kind in ("cell", "connection", "path"):
        n.propose(m, kind, t.opt)
    run_id = identity(config, m.arm, source_sha(), data)
    out = tmp_path / "event"
    save(out, t, n, rng, run_id)

    def crash(kind, stage):
        if (kind, stage) == point:
            raise RuntimeError("power loss")

    with pytest.raises(RuntimeError):
        process(config, t, n, data, rng, out, run_id, crash=crash)
    other = Trainer(Model(config, "QF-R2"))
    other_rng = np.random.default_rng(4)
    nn = restore(out, other, other_rng, run_id)
    assert other.step == 400
    process(config, other, nn, data, other_rng, out, run_id)
    count = len(nn.events)
    births = nn.births
    groups = nn.groups
    sha = hash_accepted_state(other.model)
    process(config, other, nn, data, other_rng, out, run_id)
    assert (
        count == 5
        and len(nn.events) == count
        and nn.births == births == 1
        and nn.groups == groups == 1
    )
    assert hash_accepted_state(other.model) == sha
    assert (out / "checkpoint-step-000400.pt").is_file()
    assert other.data_position == 4  # main token update never replayed


def test_admission_promotion_rollback(config):
    m = Model(config, "QF-R2")
    t = Trainer(m)
    n = Nursery()
    v = n.propose(m, "connection")
    n.probation(v, m, batch(config), 1)
    a = {"paired_identity": "paired", "groups": [0, 1], "nll": [3.0, 3.0], "cost": [0.5, 0.5]}
    b = {**a, "nll": [2.0, 2.0]}
    assert assess(v, m, a, b, a["groups"], config)
    before = hash_accepted_state(m)

    def fail():
        raise RuntimeError("promotion interruption")

    with pytest.raises(RuntimeError):
        promote(v, m, t, failure=fail)
    assert hash_accepted_state(m) == before and v.state == "EVALUATED"
    promote(v, m, t)
    assert v.state == "PROMOTED"


def test_full_synthetic_pipeline_and_partial_aggregate(config, assets, tmp_path):
    assetfolder, data = assets
    config["shadow"].update(cell_proposal_step=1, connection_path_step=1, admission_step=2)
    config["training"].update(checkpoint_interval=1, evaluation_interval=1)
    output = tmp_path / "run"
    result = development(config, "J0-R2", output / "arms/J0-R2", assetfolder, "cpu")
    assert (
        result["training_status"] == "COMPLETE"
        and result["resource_ledger"]["live_training_tokens"] == 8
    )
    seal(
        output / "arms/QF-R2/failure.json",
        {"type": "NumericalFailure", "message": "synthetic", "last_committed_step": 0},
    )
    report = aggregate(config, output, source_sha())
    assert report["training_status"] == "PARTIAL"
    assert report["arms"]["J0-R2"]["training_status"] == "COMPLETE"
    assert verified(output / "arm-J0-R2-metrics-summary.json")["complete"]


@pytest.mark.parametrize(
    "case", ["confirmation", "overlap", "extra_arm", "target_tokens", "summary", "regression"]
)
def test_schema_failclosed(config, case):
    if case == "confirmation":
        config["formal_confirmation"] = True
    if case == "overlap":
        config["data"]["admission_range"] = config["data"]["train_range"]
    if case == "extra_arm":
        config["arms"].append("M0")
    if case == "target_tokens":
        config["training"]["target_tokens"] += 1
    if case == "summary":
        config["model"]["connection_extra_context_summary"] = True
    if case == "regression":
        config["shadow"]["max_upper95_nll_regression"] = 99
    with pytest.raises(ValueError):
        validate(config)


def test_explicitly_disabled_development_is_blocked():
    c = load()
    c["authorization"]["development_training"] = False
    with pytest.raises(RuntimeError):
        authorize(c, "train-dev", "cuda:0")


def test_candidate_optimizer_parent_corruption_is_fatal(config):
    t = Trainer(Model(config, "QF-R2"))
    n = Nursery()
    v = n.propose(t.model, "connection")

    def corrupt(*args, **kwargs):
        t.opt.param_groups[0]["lr"] = 99
        raise NumericalFailure("gradient", {"parameter_name": "private"})

    v.trainer.update = corrupt
    with pytest.raises(RuntimeError, match="INTEGRITY_FAILURE"):
        n.probation(v, t.model, batch(config), 1, accepted_trainer=t)


def test_checksum_rejects_corrupt_result(tmp_path):
    p = tmp_path / "metrics.json"
    seal(p, {"status": "COMPLETE"})
    p.write_text('{"status":"made-up"}')
    with pytest.raises(ValueError):
        verified(p)


def test_checkpoint_restores_scheduled_policy_before_event(config, tmp_path):
    t = Trainer(Model(config, "QF-R2"))
    t.step = 399
    t.update(batch(config))
    assert t.model.experiment_config["policy"]["epsilon"] == 0.15
    n = Nursery()
    rng = np.random.default_rng(1)
    save(tmp_path, t, n, rng, {"id": "schedule"})
    other = Trainer(Model(config, "QF-R2"))
    restore(tmp_path, other, rng, {"id": "schedule"})
    assert other.step == 400 and other.model.experiment_config["policy"]["epsilon"] == 0.15
