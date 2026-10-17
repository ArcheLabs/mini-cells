import json
import pytest
import torch
from minicells_runtime.store import canonical, read, digest, ContinuationError
from clm001_r3b import common, runner, analysis
from clm001_r3b.evaluation import evaluate_checkpoint, fork_checkpoint
from clm001_r3.model import Model


@pytest.mark.parametrize(
    "section,key",
    [
        ("model", "width"),
        ("training", "learning_rate"),
        ("policy", "lambda_cost"),
        ("oracle", "suffix_samples"),
        ("competition", "beta"),
        ("shadow", "private_train_steps"),
        ("evaluation", "development_windows"),
    ],
)
def test_frozen_real_r3_science_cannot_change(section, key):
    candidate = common.load()
    candidate["engine"][section][key] = 0.5
    with pytest.raises((ContinuationError, ValueError)):
        common.validate(candidate)


def test_proposed_release_is_not_formal_authorization():
    candidate = common.load()
    with pytest.raises(ContinuationError, match="GATE_D"):
        common.require_training_release(candidate, "a" * 40, {}, {}, {})
    accepted = {
        "cross_session_status": "PASS",
        "sessions": {
            a: {
                "kaggle_version": "https://www.kaggle.com/code/user/notebook/output?scriptVersionId="
                + str(i)
            }
            for i, a in enumerate("ABC")
        },
        "exact_control_pass": True,
    }
    with pytest.raises(ContinuationError, match="GATE_D"):
        common.require_training_release(candidate, "a" * 40, accepted, {}, {})
    pilot = {
        "status": "GPU_PILOT_PASS",
        "source_sha": "a" * 40,
        "candidate_sha256": canonical(candidate),
        "gradient_and_alignment_pass": True,
        "two_t4_hardware": True,
        "resource_target_met": True,
        "snapshot_and_event_budget_reviewed": True,
        "diagnostic_plan_sha256": canonical(candidate["diagnostics"]),
    }
    with pytest.raises(ContinuationError, match="GATE_E"):
        common.require_training_release(candidate, "a" * 40, accepted, {}, pilot)


def test_three_arms_use_original_treatment_and_initial_weights():
    candidate = common.load()
    models = [Model(candidate["engine"], common.ENGINE[a]) for a in common.ARMS]
    first = models[0].state_dict()
    assert all(
        all(torch.equal(v, model.state_dict()[k]) for k, v in first.items()) for model in models
    )
    assert models[1].arm == "QC-R3" and models[2].arm == "QS-R3"
    assert len({sum(p.numel() for p in m.parameters()) for m in models}) == 1


def test_exact_historical_evaluation_has_no_optimizer_update(case, tmp_path, monkeypatch):
    _, _, root = case
    runner.segment(root, "QC-R3b", 3)
    runner.package(root)
    before = digest(root / "continuation.json")

    def forbidden(*args, **kwargs):
        raise AssertionError("optimizer.step in historical evaluator")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    result = evaluate_checkpoint(root, tmp_path / "eval", "QC-R3b", 3, 0)
    assert result["checkpoint"]["revision"] == 0
    assert result["optimizer_updates"] == 0
    assert digest(root / "continuation.json") == before
    with pytest.raises(ContinuationError, match="CHECKPOINT_STEP_NOT_FOUND"):
        evaluate_checkpoint(root, tmp_path / "missing", "QC-R3b", 99)


def test_explicit_fork_has_new_identity_and_parent_evidence(case, tmp_path):
    _, _, root = case
    runner.segment(root, "QC-R3b", 3)
    runner.package(root)
    before = digest(root / "continuation.json")
    output = tmp_path / "child"
    result = fork_checkpoint(
        root,
        output,
        "QC-R3b",
        3,
        revision=0,
        new_run_id="child",
        new_experiment_id="CLM-001-R3c-PROPOSAL",
        optimizer_policy="reset",
    )
    assert result["experiment_id"] == "CLM-001-R3c-PROPOSAL"
    assert result["run_id"] == "child"
    assert read(output / "parent-fork.json")["parent_run_id"] == "r3b-cpu"
    assert digest(root / "continuation.json") == before
    assert read(output / "preflight.json")["passed"] is False


def test_primary_DID_is_story_paired_and_not_pseudo_replication():
    c = common.load()["engine"]
    c["evaluation"]["bootstrap_replicates"] = 100
    groups = [0, 0, 1, 1]

    def quality(g, u):
        def row(values):
            return {
                "nll": values,
                "cost": [0.5] * 4,
                "groups": groups,
                "paired_identity": "same-frozen-grid",
            }

        return {"greedy": row(g), "uniform": row(u)}

    policies = {
        "Q-R3b": quality([3.0] * 4, [4.0] * 4),
        "QC-R3b": quality([2.0] * 4, [4.0] * 4),
        "QS-R3b": quality([3.0] * 4, [4.0] * 4),
    }
    result = analysis.primary(c, policies)
    assert result["DID"]["nll"]["mean"] == -1
    assert result["DID"]["stories"] == 2
    assert result["independent_seeds"] == 1
    assert result["Q_QS_are_not_independent_replications"]
    assert result["scientific_decision"] is False
    policies["QC-R3b"]["uniform"]["groups"] = [1, 0, 1, 1]
    with pytest.raises(ValueError, match="unpaired"):
        analysis.primary(c, policies)


def test_snapshots_complete_only_after_all_required_points(case):
    _, _, root = case
    for arm in common.ARMS:
        runner.segment(root, arm, 1)
    runner.package(root)
    values = analysis.aggregate(root)
    manifest = values["experiment-manifest.json"]
    assert manifest["diagnostic_status"] == "INCOMPLETE"
    assert manifest["missing"]
    assert manifest["CAUSAL_FEEDBACK"] == "CAUSAL_FEEDBACK_NOT_TESTED"
    assert manifest["MULTISEED"] == "MULTISEED_NOT_TESTED"
    assert all(not row["scientific_decision"] for row in values.values())
    # Public aggregates must not contain private per-story arrays or state tensors.
    payload = json.dumps(values)
    assert '"nll": [' not in payload

    def check(value):
        if isinstance(value, dict):
            for key, row in value.items():
                if key == "groups" and isinstance(row, list):
                    assert not any(isinstance(v, (int, float)) for v in row), "raw story IDs leaked"
                check(row)
        elif isinstance(value, list):
            for row in value:
                check(row)

    check(values)
    from clm001_r3b.plots import render

    plots = render(root)
    assert len(plots) >= 4 and all(path.stat().st_size > 0 for path in plots)


def test_shared_package_protocol_source_and_probe_tamper_rejected(case):
    _, _, root = case
    runner.segment(root, "Q-R3b", 1)
    runner.package(root)
    (root / "arms/Q-R3b/development-probe-bank.pt").write_bytes(b"tamper")
    with pytest.raises(ContinuationError, match="HASH_MISMATCH"):
        runner.verify(root)


def test_historical_acceptance_is_informational(tmp_path):
    from clm001_r3b.preflight import engineering_evidence
    from minicells_runtime.store import atomic

    evidence = tmp_path / "acceptance.json"
    for status in ("NOT_VERIFIED", "USER_VERIFIED", "DURABLE_COMPLETE"):
        value = {"cross_session_status": status, "sessions": {}}
        atomic(evidence, value)
        assert engineering_evidence(evidence) == value


def test_hf_index_loader_rejects_arbitrary_urls():
    from clm001_r3b.cli import checkpoint_ref

    with pytest.raises(ContinuationError, match="GITHUB_INDEX_URL_REJECTED"):
        checkpoint_ref("https://evil.test/weights.json")
    from urllib.error import HTTPError

    error = HTTPError("https://private.invalid/?token=secret", 404, "not found", {}, None)
    projected = common.safe_error(error)
    assert projected["failure_code"] == "HTTPError_404"
    assert "secret" not in str(projected)
    assert str(ContinuationError(projected["failure_code"])) == "HTTPError_404"
