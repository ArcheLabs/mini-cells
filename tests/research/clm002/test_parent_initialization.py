import copy

import pytest
import torch
from minicells_clm002.data import PairedStream, unit_pools
from minicells_clm002.engine import Engine
from minicells_clm002.initialization import fork_for_cpu_audit
from minicells_clm002.native import build
from minicells_r3e.common import fingerprint


def historical_fixture(config, ident, tmp_path):
    model, runtime = build(config, 7001, unit=True)
    engine = Engine(model, runtime, "BASE", ident, tmp_path / "baseline.jsonl", unit=True)
    # Nonzero moments represent an inherited optimizer. No optimizer is executed.
    for parameter in model.cells.parameters():
        engine.cell_optimizer.state[parameter] = {
            "step": torch.tensor(300.0),
            "exp_avg": torch.full_like(parameter, 0.125),
            "exp_avg_sq": torch.full_like(parameter, 0.25),
        }
    snapshot = engine.state()
    return {
        "model": copy.deepcopy(snapshot["model"]),
        "cell_optimizer": copy.deepcopy(snapshot["cell_optimizer"]),
        "step": 300,
        "data_cursor": 153600,
        "r3h": {"routing_mode": "H", "temperature": model.temperature, "rho": model.rho},
        "data_rng": snapshot["data_rng"],
        "route_rng": snapshot["route_rng"],
        "cf_rng": snapshot["cf_rng"],
        "global_cpu_rng": snapshot["cpu_rng"],
        "global_cuda_rng": [torch.tensor([1, 2, 3], dtype=torch.uint8)],
        "python_rng": snapshot["python_rng"],
        "numpy_rng": snapshot["numpy_rng"],
        "work": snapshot["work"],
    }


def test_all_arm_forks_preserve_parent_and_moments_without_updates(config, ident, tmp_path):
    source = historical_fixture(config, ident, tmp_path)
    original = fingerprint(source)
    values = []
    forks = []
    for arm in config["stage_a"]["arms"] + config["stage_b"]["arms"]:
        engine, proof = fork_for_cpu_audit(
            config, source, 7001, arm, ident, tmp_path / (arm + ".jsonl"), "a" * 64, unit=True
        )
        assert all(proof.values())
        assert engine.step == 0 and engine.parent_step == 300 and not engine.unit
        assert not engine.parent_origin["scientific_policy_registered"]
        assert fingerprint(engine.cell_optimizer.state_dict()) == fingerprint(
            source["cell_optimizer"]
        )
        values.append(
            fingerprint(
                (
                    engine.model.state_dict(),
                    engine.cell_optimizer.state_dict(),
                    engine.data_rng.bit_generator.state,
                    engine.route_rng.get_state(),
                    engine.cf_rng.get_state(),
                )
            )
        )
        with pytest.raises(RuntimeError, match="SCIENTIFIC_REGISTRATION_BLOCKED"):
            engine.update(None, None, None)
        forks.append(engine)
    assert len(set(values)) == 1
    assert fingerprint(source) == original
    with torch.no_grad():
        next(forks[0].model.cells.parameters()).add_(1)
    assert fingerprint(forks[1].model.state_dict()) == fingerprint(source["model"])
    first_state = next(iter(forks[0].cell_optimizer.state.values()))
    first_state["exp_avg"].add_(1)
    assert fingerprint(forks[1].cell_optimizer.state_dict()) == fingerprint(
        source["cell_optimizer"]
    )
    assert fingerprint(source) == original


def test_parent_counters_and_provenance_survive_checkpoint_roundtrip(config, ident, tmp_path):
    source = historical_fixture(config, ident, tmp_path)
    engine, _ = fork_for_cpu_audit(
        config, source, 7001, "A-H", ident, tmp_path / "audit.jsonl", "a" * 64, unit=True
    )
    pools = unit_pools(config)
    stream = PairedStream(pools, 7001, "H", engine.c["training"]["tokens_per_step"])
    before = fingerprint((engine.state(), stream.state()))
    engine.save(tmp_path / "fork.pt", stream)
    engine.parent_origin["historical_data_cursor"] = -1
    engine.restore(tmp_path / "fork.pt", stream)
    assert fingerprint((engine.state(), stream.state())) == before
    assert engine.parent_origin["historical_data_cursor"] == 153600


def test_zero_update_audit_returns_only_publishable_report(config, ident, tmp_path, monkeypatch):
    from minicells_clm002 import audit, initialization, publication
    from minicells_clm002.common import digest

    source = historical_fixture(config, ident, tmp_path)
    checkpoint = tmp_path / "synthetic-parent.pt"
    torch.save({"state": source}, checkpoint)
    monkeypatch.setattr(initialization, "identity", lambda c: dict(ident))
    monkeypatch.setattr(
        initialization, "build", lambda c, seed, unit=False: build(c, seed, unit=True)
    )
    monkeypatch.setattr(
        audit,
        "inspect_parent",
        lambda c, path, output: {
            "parent_sha256": digest(path),
            "scope": "MOCK_SYNTHETIC_TEST_ONLY",
        },
    )
    root, report = initialization.inspect_initialization(config, checkpoint, tmp_path / "output")
    assert report["actual_optimizer_calls"] == report["model_forward_calls"] == 0
    assert len(report["arms"]) == 10
    assert not report["scientific_policy_registered"]
    manifest, files, _, _ = publication.payload(root)
    assert manifest["scope"] == "GATE0_CPU_PARENT_FORK_ONLY"
    assert set(files) == {"PARENT_FORK_AUDIT.json", "RUN_MANIFEST.json"}
    assert manifest["source_sha"] == ident["source_sha"]
