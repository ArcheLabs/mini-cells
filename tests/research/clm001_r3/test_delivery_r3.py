import ast
import hashlib
import json
import inspect
import pytest
import torch
from clm001_r3.common import BASE, ROOT, validate
from clm001_r3.model import Model
from clm001_r3.publication import ALLOWED
from clm001_r3.probing import EventSampler
from clm001_r3.competitive_credit import candidates
from clm001_r3.training_errors import CreditFailure
from clm001_r3.report import aggregate, seal
from test_credit import batch


def test_source_and_readonly_ledgers():
    for name in ("source-manifest.json", "read-only-inputs.json"):
        value = json.loads((BASE / "reports" / name).read_text())
        for path, digest in value["files"].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path


def test_notebook_is_one_launch_and_sparse_inputs_present():
    n = json.loads((BASE / "notebook.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in n["cells"] if c["cell_type"] == "code")
    ast.parse(code)
    assert "codex/clm001-r3-competitive-credit" in code and "clm-001-qfirst-integrated/r3" in code
    assert "/tests/research/clm001_r3/" in code and "/.github/workflows/clm001-r3-cpu.yml" in code
    assert all(not c.get("outputs") for c in n["cells"])
    assert "targets" not in inspect.signature(Model.forward).parameters


@pytest.mark.parametrize(
    "mutation", ["extra_arm", "confirmation", "seed", "beta", "summary", "sampler", "sham"]
)
def test_schema_rejects_research_drift(config, mutation):
    if mutation == "extra_arm":
        config["arms"].append("M0")
    elif mutation == "confirmation":
        config["authorization"]["confirmation"] = True
    elif mutation == "seed":
        config["synthetic"] = False
        config["seed"] = 151001302
    elif mutation == "beta":
        config["competition"]["beta"] = -1.0
    elif mutation == "summary":
        config["model"]["connection_extra_context_summary"] = True
    elif mutation == "sampler":
        config["competition"]["candidate_cells"] = 1
    else:
        config["competition"]["sham"] = "large_noise"
    with pytest.raises(ValueError):
        validate(config)


def test_candidate_legality_and_inclusion(config):
    model = Model(config, "QC-R3")
    x, y = batch()[0]
    sampler = EventSampler(4, torch.Generator().manual_seed(4), 1)
    model(x, replay=torch.tensor([[[0, 0, 4]] * 4]), competition_sampler=sampler)
    event = sampler.events[0]
    ids, inclusion = candidates(model, event, torch.Generator().manual_seed(8))
    assert ids[0] == 0 and set(ids) == {0, 1} and inclusion == [1.0, 1.0]
    event.state.chosen = model.registry.ctx
    with pytest.raises(CreditFailure):
        candidates(model, event, torch.Generator().manual_seed(8))
    sampler.close()


def test_three_arm_partial_report_never_discards_completed_sibling(config, tmp_path):
    arm = "Q-R3"
    source = "a" * 40
    from clm001_r3.common import canonical_sha256

    result = {
        "identity": {
            "source_commit": source,
            "config_sha256": canonical_sha256(config),
            "experiment_id": config["experiment_id"],
            "arm": arm,
        },
        "quality": {},
        "resource_ledger": {},
        "specialization": {},
    }
    seal(tmp_path / "arms" / arm / "metrics.json", result)
    seal(tmp_path / "arms/QC-R3/failure.json", {"type": "NumericalFailure", "message": "synthetic"})
    report = aggregate(config, tmp_path, source)
    assert report["training_status"] == "PARTIAL" and report["arms"][arm]["complete"]
    assert not report["scientific_decision"] and report["H2"] == "CAUSAL_FEEDBACK_NOT_TESTED"


def test_allowlist_excludes_private_labels_and_weights():
    assert (
        "local-vs-suffix-agreement.json" in ALLOWED and "arm-QS-R3-metrics-summary.json" in ALLOWED
    )
    assert not ALLOWED.intersection(
        {
            "frozen-probe-bank.pt",
            "oracle-labels.json",
            "metrics.json",
            "runtime-config.json",
            "worker.log",
        }
    )
