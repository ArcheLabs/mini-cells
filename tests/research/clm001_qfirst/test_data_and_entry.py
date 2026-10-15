from pathlib import Path
import json
import numpy as np
import pytest
import torch
from clm001.common import BASE, hash_accepted_state
from clm001.data import select, build, load_assets, batches
from clm001.model import Model
from clm001.training import Trainer
from clm001.evaluation import evaluate, q_diagnostics
from clm001.accounting import profile
from clm001.checkpoint import identity, save, export_portable, verify_portable
from clm001.shadows import Nursery


class Tokenizer:
    def get_vocab_size(self):
        return 16

    def token_to_id(self, x):
        return 1

    def encode(self, text):
        class Encoded:
            pass

        e = Encoded()
        e.ids = [2 + ord(v) % 14 for v in text]
        return e

    def save(self, path):
        Path(path).write_text('{"synthetic":true}')


@pytest.fixture
def assets(config, tmp_path):
    for i, role in enumerate(("train", "calibration", "admission", "development")):
        config["data"][role + "_range"] = [2 * i, 2 * i + 2]
    config["evaluation"].update(
        calibration_windows=2,
        admission_windows=2,
        development_windows=2,
        oracle_states=1,
        oracle_samples=2,
    )
    texts = [
        "violet velvet valley variation",
        "brass bridge bears balloons",
        "cyan circus cloud castle",
        "amber autumn acorn animal",
        "silver snow silent swan",
        "golden garden geese green",
        "purple planet people painting",
        "scarlet sailing salty sea",
    ]

    def source():
        for t in texts:
            yield {"text": t}
        raise AssertionError("read outside selected nonconfirmation data")

    selected, audit = select(source(), config)
    folder = tmp_path / "assets"
    build(config, selected, audit, folder, "synthetic-shard", Tokenizer())
    return folder, load_assets(config, folder)


def test_four_roles_and_no_boundary_read(config, assets):
    folder, data = assets
    assert set(data["windows"]) == {"train", "calibration", "admission", "development"}
    ids = [i for meta in data["windows"].values() for i in meta["raw_story_ids"]]
    assert len(ids) == len(set(ids)) == 8 and max(ids) < 8
    assert not data["manifest"]["confirmation_materialized"]
    assert data["manifest"]["tokenizer_training"] == "train only"


def test_data_corruption(config, assets):
    folder, data = assets
    p = folder / "train.npy"
    p.write_bytes(p.read_bytes() + b"tamper")
    with pytest.raises(ValueError):
        load_assets(config, folder)


def test_old_experiment_cannot_resume(config, assets):
    folder, _ = assets
    p = folder / "manifest.json"
    meta = json.loads(p.read_text())
    meta["experiment_id"] = "FAST-001"
    p.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_assets(config, folder)


def test_full_synthetic_engine_and_portable(config, assets, tmp_path):
    folder, data = assets
    model = Model(config, "QF-FULL")
    trainer = Trainer(model)
    rng = np.random.default_rng(123)
    n = Nursery()
    trainer.update(batches(data, config, rng, "cpu"))
    run_id = identity(config, model.arm, "a" * 40, data)
    run = tmp_path / "run"
    save(run, trainer, n, rng, run_id)
    portable = tmp_path / "saved-session-output"
    export_portable(run, folder, portable, run_id, config)
    assert verify_portable(portable, run_id)["actual_kaggle_persistence_verified"] is False
    before = hash_accepted_state(model)
    state = trainer.rng["live"].get_state().clone()
    result = evaluate(model, data, config, "admission", policy="sampled")
    assert len(result["groups"]) == 2
    q = q_diagnostics(model, data, config)
    assert q["role"] == "calibration" and q["states"] == 1
    assert hash_accepted_state(model) == before and torch.equal(
        state, trainer.rng["live"].get_state()
    )
    assert (
        profile(model, batches(data, config, rng, "cpu")[0][0], repeats=1)[
            "executed_parameter_work_per_output_token"
        ]
        > 0
    )
    assert not trainer.cache


def test_label_free_forward_signature(config):
    import inspect

    parameters = inspect.signature(Model.forward).parameters
    assert not {"target", "targets", "labels", "y"}.intersection(parameters)


def test_no_hot_dispatch():
    source = (BASE / "implementation/clm001/training.py").read_text()
    assert "TorchDispatch" not in source and "ObservedGemmFlops" not in source


def test_result_publisher_is_wired_and_metadata_only():
    source = (BASE / "implementation/clm001/publication.py").read_text()
    assert "preflight_push" in source and "push_branch" in source
    assert "refs/remotes/origin/main" in source
    assert "workflow" not in __import__("clm001.publication", fromlist=["ALLOWED"]).ALLOWED


@pytest.mark.parametrize("arm", ["QF-FULL", "J0-FULL"])
def test_complete_two_step_development_and_shadow(config, assets, tmp_path, arm):
    from clm001.cli import development
    from clm001.report import summarize

    folder, data = assets
    config["training"].update(checkpoint_interval=1, evaluation_interval=1)
    config["shadow"].update(cell_proposal_step=1, connection_path_step=1, admission_step=2)
    config["shadow"]["minimum_admission_stories"] = 2
    output = tmp_path / arm
    result = development(config, arm, output, folder, "cpu", "train-dev")
    assert (
        result["status"] == "DEVELOPMENT_COMPLETE"
        and result["shadow_counts"]["shadow_proposed"] == 3
    )
    assert result["live_tokens"] == 8
    assert (output / "portable/portable.json").exists()
    # A completed run loads existing metadata, without retraining or overwriting immutable Checkpoints.
    again = development(config, arm, output, folder, "cpu", "train-dev")
    assert again == result
    report = summarize(config, {arm: result}, oracle_pass=True)
    assert report["status"] == "INCOMPLETE" and not report["scientific_decision"]
    assert set(report["arms"]["QF-FULL"]) == set(report["arms"]["J0-FULL"])


def test_historical_reuse_hashes_unchanged():
    from clm001.common import ROOT, file_sha256

    ledger = json.loads((BASE / "reports/read-only-source-inputs.json").read_text())
    for name, expected in ledger["files"].items():
        assert file_sha256(ROOT / name) == expected, name


def test_notebook_default_and_source_compilation():
    notebook = json.loads((BASE / "notebook.ipynb").read_text())
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert "APPROVAL_CONFIG" not in "".join(cells[0]["source"])
    for c in cells:
        assert c["outputs"] == [] and c["execution_count"] is None
        compile("".join(c["source"]), "notebook-cell", "exec")
    code = "".join(cells[-1]["source"])
    assert '"auto"' in code and "APPROVAL_CONFIG" not in code
    assert "terminate(" not in code and "kill(" not in code


def test_delivered_source_manifest_matches():
    from clm001.common import ROOT, file_sha256

    manifest = json.loads((BASE / "reports/source-manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert file_sha256(ROOT / name) == digest, name
