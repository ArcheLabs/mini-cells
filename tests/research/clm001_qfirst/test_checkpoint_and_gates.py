import numpy as np
import pytest
import torch
from clm001.common import authorize, load, hash_accepted_state, validate
from clm001.model import Model
from clm001.training import Trainer
from clm001.shadows import Nursery
from clm001.checkpoint import save, restore
from test_core import batch


def test_exact_restart(config, tmp_path):
    m = Model(config, "QF-FULL")
    trainer = Trainer(m)
    nursery = Nursery()
    rng = np.random.default_rng(18)
    trainer.update([batch()])
    cand = nursery.propose(m, "cell", trainer.opt)
    nursery.probation(cand, m, [batch()], 1)
    identity = {"experiment_id": "CLM-001-QFIRST", "source": "synthetic", "arm": m.arm}
    digest = save(tmp_path, trainer, nursery, rng, identity)
    global_state = torch.get_rng_state().clone()
    expected, paths, _ = trainer.update([batch()])
    expected_hash = hash_accepted_state(m)
    other = Trainer(Model(config, "QF-FULL"))
    other_rng = np.random.default_rng(19)
    restored = restore(tmp_path, other, other_rng, identity)
    assert torch.equal(torch.get_rng_state(), global_state)
    actual, actual_paths, _ = other.update([batch()])
    assert hash_accepted_state(other.model) == expected_hash and actual_paths == paths
    assert (
        actual["live_ce"] == expected["live_ce"]
        and actual["label_record_hashes"] == expected["label_record_hashes"]
    )
    assert restored.candidates[0].state == "PROBATION"
    assert hash_accepted_state(restored.candidates[0].model) == hash_accepted_state(cand.model)
    assert len(digest) == 64
    with pytest.raises(ValueError):
        restore(tmp_path, other, other_rng, {**identity, "source": "different"})


def test_checksum_and_interrupted_save(config, tmp_path):
    t = Trainer(Model(config, "J0-FULL"))
    rng = np.random.default_rng(1)
    n = Nursery()
    i = {"id": "test"}
    save(tmp_path, t, n, rng, i)
    (tmp_path / "checkpoint-step-000001.pt.tmp").write_bytes(b"interrupted")
    restore(tmp_path, t, rng, i)
    assert t.step == 0
    path = tmp_path / "checkpoint-step-000000.pt"
    path.write_bytes(path.read_bytes() + b"bad")
    with pytest.raises(ValueError):
        restore(tmp_path, t, rng, i)


@pytest.mark.parametrize("phase", ["gpu-pilot", "train-smoke", "train-dev", "evaluate"])
def test_default_gates_closed(phase):
    with pytest.raises(RuntimeError):
        authorize(load(), phase, "cuda:0")


def test_no_synthetic_gpu_escape(config):
    with pytest.raises(RuntimeError):
        authorize(config, "train-dev", "cuda:0")


@pytest.mark.parametrize(
    "mutation", ["confirmation", "overlap", "budget", "arm", "reserved", "suffix"]
)
def test_bad_config_rejected(config, mutation):
    if mutation == "confirmation":
        config["gates"]["confirmation"] = True
    elif mutation == "overlap":
        config["data"]["admission_range"] = config["data"]["train_range"]
    elif mutation == "budget":
        config["training"]["target_tokens"] += 1
    elif mutation == "arm":
        config["arms"].append("M0")
    elif mutation == "reserved":
        config["data"]["development_range"] = [59999, 60001]
    else:
        config["policy"]["suffix_kind"] = "replay_old_actions"
    with pytest.raises((ValueError, RuntimeError)):
        validate(config)


def test_optimizer_failure_rolls_back(config):
    t = Trainer(Model(config, "QF-FULL"))
    before = hash_accepted_state(t.model)

    def fail():
        raise RuntimeError("inject optimizer transaction failure")

    with pytest.raises(RuntimeError):
        t.update([batch()], failure=fail)
    assert before == hash_accepted_state(t.model) and t.step == 0 and not t.opt.state


def test_expired_rejected_shadow_cannot_promote(config):
    from clm001.admission import promote

    m = Model(config, "QF-FULL")
    t = Trainer(m)
    n = Nursery()
    c = n.propose(m, "cell")
    t.update([batch()])
    n.expire_stale(m)
    assert c.state == "EXPIRED"
    with pytest.raises(ValueError):
        promote(c, m, t)


def test_private_updates_only_candidate_and_connection(config):
    m = Model(config, "QF-FULL")
    t = Trainer(m)
    n = Nursery()
    c = n.propose(m, "cell", t.opt)
    before = {k: v.clone() for k, v in c.model.state_dict().items()}
    n.probation(c, m, [batch()], 1)
    for name, value in c.model.state_dict().items():
        if not name.startswith(("cells.2.", "controller.")):
            assert torch.equal(value, before[name])
