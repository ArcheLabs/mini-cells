"""Changed ABC controls and one tiny end-to-end CPU trial only."""

import copy
import os
import sys
from pathlib import Path

import pytest
from minicells_r3f.evaluation import trust

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fast002_r1"))
import screening as sc
from minicells_fast001.data import Stream
from test_phase0 import rv
from test_stateful import fixture


def test_screening_rule_and_fixed_C_dates():
    m = sc.manifest()
    seeds = m["seeds"]
    values = {
        "A": {s: 1.0 for s in seeds},
        "B": {s: 1.02 for s in seeds},
        "C": {s: 1.03 for s in seeds},
    }
    assert sc.decision(m, values)["status"] == "PROPOSE_INDEPENDENT_CONFIRMATORY_EVALUATION"
    values["B"] = {s: 0.9 for s in seeds}
    assert sc.decision(m, values)["status"] == "STOP_BEFORE_MOE_DENSE"
    assert m["update_steps"] == list(range(50, 501, 50))
    assert "A_ACCEPTANCE" in m["C_schedule"]


def test_one_short_ABC_end_to_end_trial(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "fsync", lambda fd: None)
    c, parent, folder = fixture(tmp_path)
    m = copy.deepcopy(sc.manifest())
    c["stage_b"] = copy.deepcopy(m["runtime_template"]["stage_b"])
    m.update(steps=2, update_steps=[2], runtime_template=c)
    import torch

    x = torch.tensor([[7, 8, 9, 10]])
    m["runtime_template"]["admission"].update(
        manifest_sha256="fixture", ordered_input_hashes=[rv.sha(x.squeeze(0).tolist())]
    )
    bundle = {"role": "admission", "manifest_sha256": "fixture", "inputs": [x]}
    starts = []
    inputs = []
    for arm in "ABC":
        e = sc.engine(m, parent, 7001, arm, tmp_path / "trial", bundle, unit=True)
        starts.append(rv.fingerprint(e.model.state_dict()))
        stream = Stream(e.fast, folder, 7001, "H", "B")
        latest = e.stage_directory / "latest.pt"
        while e.step < 2:
            e.update(stream, latest, [], failure_hook=lambda tr: sc.fixed_rewire(tr, m, bundle))
        inputs.append([r["input_sha256"] for r in e.curve])
        assert e.journal.summary()["optimizer_calls_by_group"]["cell"] == 2
        assert ("random_rewire" in e.curve[-1]) == (arm == "C")
        if arm == "A":
            reports = list((tmp_path / "trial/proposals").glob("*/*/ADMISSION_DECISION.json"))
            assert len(reports) == 1
            attempts = rv.read(reports[0])["attempts"]
            assert len(attempts) == 22
            assert all(
                r.get("interpolation_check_scope")
                == "FP64_AFFINE_LAW_AND_EXACT_DEPLOYED_FP32_PARAMETERS"
                for r in attempts
            )
        if arm == "C":
            assert e.curve[-1]["random_rewire"]["independent_of_A_acceptance"]
        for h in e._work_handles:
            h.remove()
    assert len(set(starts)) == 1
    assert inputs[0] == inputs[1] == inputs[2]


def test_actual_screening_notebook_commands_parse(tmp_path, monkeypatch):
    import ast
    import json
    import runpy

    from test_phase0 import BASE

    notebook = json.loads((BASE / "notebooks/kaggle-abc-screening.ipynb").read_text())
    commands = [
        n.args[0]
        for n in ast.walk(ast.parse("".join(notebook["cells"][1]["source"])))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "run"
        and n.args
        and isinstance(n.args[0], ast.List)
        and any(isinstance(e, ast.Constant) and e.value == "--output" for e in n.args[0].elts)
    ]
    assert len(commands) == 2
    calls = []
    monkeypatch.setattr(sc, "run", lambda args: calls.append(args))
    monkeypatch.syspath_prepend(str(BASE))
    for node in commands:
        values = eval(
            compile(ast.Expression(node), "<screen-command>", "eval"),
            {
                "sys": sys,
                "CODE": BASE.parent,
                "BASE_REL": BASE.name,
                "OUTPUT": tmp_path / "output",
                "ASSETS": tmp_path / "assets",
            },
        )
        monkeypatch.setattr(sys, "argv", list(map(str, values[1:])))
        runpy.run_path(str(BASE / "screen.py"), run_name="screen_parser")["main"]()
    assert len(calls) == 2
    assert any(x.publish_existing for x in calls) and any(x.publish_github for x in calls)


def test_retained_publication_supplies_required_config_hash_without_training(tmp_path, monkeypatch):
    import hashlib
    import json

    from minicells_r3k.reuse import bind as real_bind

    m = sc.manifest()
    rv.atomic(tmp_path / "REGISTRATION.json", {"manifest": m})
    raw = (json.dumps(m, ensure_ascii=False, indent=2) + "\n").encode()
    rv.atomic(
        tmp_path / "RUN_MANIFEST.json",
        {
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "scope": "CLM003_ABC_SCREENING",
            "artifact_files": {},
            "run_id": "a" * 32,
            "source_sha": "b" * 40,
        },
    )

    def intercept(function, **kwargs):
        if function is rv.transport.previous.old.transport.publish:

            def transport(root):
                manifest, _, _, _ = kwargs["payload"](root)
                assert manifest["config_sha256"] == rv.sha(m)
                return {"status": "TEST_PUBLICATION_CONTRACT_PASS"}

            return transport
        return real_bind(function, **kwargs)

    monkeypatch.setattr(sc, "bind", intercept)
    assert sc.publish(tmp_path)["status"] == "TEST_PUBLICATION_CONTRACT_PASS"


def test_affine_audit_handles_fp32_rounding_and_rejects_wrong_parameters(monkeypatch):

    import interpolation_audit as audit
    import torch
    from minicells_r3e.graph import GraphConnection

    torch.manual_seed(414)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.connection = GraphConnection(128)

        def scores(self, h, t, p):
            z = torch.zeros(h.shape[0], 6)
            q, d = self.connection(h, t, z, p)
            return q, d, z

    old = Model()
    for p in old.connection.parameters():
        p.data.normal_()
    full = copy.deepcopy(old)
    for p in full.connection.parameters():
        p.data.add_(torch.randn_like(p) * 0.01)
    new = copy.deepcopy(old)
    for name, p in new.connection.named_parameters():
        a = old.connection.state_dict()[name]
        b = full.connection.state_dict()[name]
        p.data.copy_(a + 0.5 * (b - a))
    h = torch.randn(4096, 128)
    previous = torch.zeros(4096, dtype=torch.long)
    rows = [{"state": h, "stage": 0, "previous": previous}]
    package = {
        "original_connection": old.connection.state_dict(),
        "candidate_connection": full.connection.state_dict(),
    }
    sentinel = {"kl": 0.123, "action_change": 0.04, "path_change": 0.17}
    monkeypatch.setattr(audit, "trust", lambda *a, **k: dict(sentinel))

    def run_scan(*args, **kwargs):
        return trust(
            old, new, [], fixed=rows, expected_delta=[full.scores(h, 0, previous)[1]], scale=0.5
        )

    monkeypatch.setattr(audit.scan, "run", run_scan)
    metrics = audit.audited_scan(None, package, None, None, None, None, None)
    assert not metrics["legacy_fp32_interpolation_pass"]
    assert metrics["interpolation_max_abs_error"] < 1e-10
    assert all(metrics[k] == v for k, v in sentinel.items())
    next(new.connection.parameters()).data.add_(0.01)
    with pytest.raises(RuntimeError, match="SCALED_PARAMETER_IDENTITY"):
        audit.audited_scan(None, package, None, None, None, None, None)
