import ast
import hashlib
import json
import pytest
from clm001_r2.common import BASE, ROOT, load, validate, canonical_sha256
from clm001_r2.publication import ALLOWED
from clm001_r2.report import aggregate, seal, contrast


def test_source_and_readonly_manifests():
    for name in ("source-manifest.json", "read-only-inputs.json"):
        manifest = json.loads((BASE / "reports" / name).read_text())
        for relative, digest in manifest["files"].items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative


def test_notebook_entry_and_sparse_coverage():
    notebook = json.loads((BASE / "notebook.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    ast.parse(code)
    assert "codex/clm001-r2" in code and "clm-001-qfirst-integrated/r2" in code
    assert "/tests/research/clm001_r2/" in code and "/.github/workflows/clm001-r2-cpu.yml" in code
    assert (
        "GITHUB_TOKEN" not in code
    )  # existing authenticated publisher reads this secret in Kaggle
    manifest = json.loads((BASE / "reports/source-manifest.json").read_text())
    for path in manifest["files"]:
        assert any(
            path.startswith(prefix)
            for prefix in [
                "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r2/",
                "tests/research/clm001_r2/",
                ".github/workflows/clm001-r2-cpu.yml",
            ]
        )
    assert not any(c.get("outputs") for c in notebook["cells"])


def test_publish_allowlist_is_aggregate_only():
    assert "arm-QF-R2-metrics-summary.json" in ALLOWED
    assert "arm-J0-R2-metrics-summary.json" in ALLOWED
    assert not ALLOWED.intersection(
        {
            "checkpoint.pt",
            "tokenizer.json",
            "oracle-labels.json",
            "metrics.json",
            "worker.log",
            "runtime-config.json",
        }
    )


@pytest.mark.parametrize(
    "combination", [(False, False), (True, False), (False, True), (True, True)]
)
def test_every_worker_completion_combination(config, tmp_path, combination):
    source = "a" * 40
    for arm, done in zip(config["arms"], combination):
        folder = tmp_path / "arms" / arm
        if done:
            quality = {
                "nll": [2.0, 3.0],
                "cost": [0.5, 0.6],
                "groups": [0, 1],
                "paired_identity": "same",
            }
            seal(
                folder / "metrics.json",
                {
                    "identity": {
                        "source_commit": source,
                        "experiment_id": config["experiment_id"],
                        "arm": arm,
                        "seed": config["seed"],
                        "config_sha256": canonical_sha256(config),
                    },
                    "quality": {"greedy": quality},
                    "pre_admission": {"quality": quality},
                },
            )
        else:
            seal(folder / "failure.json", {"type": "SyntheticFailure", "scope": "FAULT_INJECTION"})
    value = aggregate(config, tmp_path, source)
    assert sum(v["complete"] for v in value["arms"].values()) == sum(combination)
    assert value["training_status"] == (
        "COMPLETE" if all(combination) else ("PARTIAL" if any(combination) else "INCOMPLETE")
    )


def test_bootstrap_story_pairing_and_sign(config):
    a = {
        "nll": [1.0, 1.0, 1.0],
        "cost": [0.4, 0.4, 0.4],
        "groups": [0, 0, 1],
        "paired_identity": "x",
    }
    b = {**a, "nll": [2.0, 2.0, 2.0]}
    result = contrast(a, b, config)
    assert result["stories"] == 2 and result["utility"]["ci95"][1] < 0
    with pytest.raises(ValueError):
        contrast(a, {**b, "paired_identity": "y"}, config)


def test_real_candidate_budget_and_inactive_confirmation():
    c = load()
    validate(c)
    assert c["training"]["target_tokens"] == 500 * 8 * 64
    assert c["authorization"]["development_training"] and not c["authorization"]["confirmation"]
    assert c["data"]["reserved_boundary"] == 60000


def test_analysis_never_makes_confirmation_claim(config):
    from clm001_r2.analysis import decide

    record = {
        "coverage_missing": False,
        "identifiable": True,
        "regret": 0.0,
        "uniform_regret": 0.1,
        "group": 1,
        "spearman": 1.0,
    }
    q = {
        "q_calibration": {"records": [record], "stop_error_rate": 0.0},
        "quality": {"greedy": {"actual_stop_rate": 0.5, "forced_end_rate": 0.2}},
    }
    result = decide(
        config,
        {"QF-R2": q, "J0-R2": {}},
        {"utility": {"ci95": [-0.2, -0.1]}},
        {"utility": {"ci95": [-0.2, -0.1]}},
    )
    assert result["decision"] == "QF_DEV_SIGNAL" and not result["scientific_decision"]
    assert not result["expansion_budget_approved"]
