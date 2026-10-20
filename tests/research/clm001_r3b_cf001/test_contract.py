import ast
import json
from r3b_cf001.common import BASE, configuration, protocol_identity, ENGINE_SHA


def test_no_training_calls_and_frozen_contract():
    c = configuration()
    assert c["engine_sha"] == ENGINE_SHA and c["confirmation"] == "SEALED"
    assert (
        c["checkpoint_sha256"]["QC-R3b"]
        == "57b9daaaa54b0a67c6bc43c2f216ac375d0bb127ffbda132ac00f4cd42caffbb"
    )
    assert len(protocol_identity(c)) == 64
    for path in (BASE / "implementation").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.get_source_segment(path.read_text(), node.func)
                assert not name.endswith(".backward") and name not in (
                    "Trainer",
                    "Trainer.update",
                    "optimizer.step",
                )
    notebook = json.loads((BASE / "notebook.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "--publish-github" in code and "--engine-root" in code
    assert "pytest" not in code and "train_pair" not in code
    assert "KAGGLE_API_TOKEN" in code and "GITHUB_TOKEN" in code


def test_full_native_vs_adapter_statistics_and_reporting(routed, tmp_path):
    import torch
    from r3b_cf001.inference import census, rollout, statistics
    from r3b_cf001.reporting import interventions
    from r3b_cf001.plots import render
    from clm001_r3b.publication import reject_private_payload

    m, x, y = routed
    c = configuration()
    c["bootstrap_replicates"] = 100
    base = {}
    policies = {
        p: {}
        for p in ("FORCE_STOP_D1", "CTX_ONCE", "NO_ADJACENT_SAME_CELL", "CTX_ONCE_AND_NO_REPEAT")
    }
    with torch.no_grad():
        for i in range(3):
            native, records = census(m, x, y, str(i), str(i))
            base[str(i)] = statistics(native, y, m.registry)
            for p in policies:
                policies[p][str(i)] = statistics(rollout(m, x, p), y, m.registry)
    report = interventions(base, policies, c)
    reject_private_payload(report)
    assert report["comparisons"]["FORCE_STOP_D1"]["delta_cost"]["mean"] < 0
    render({"cf2-forced-stop.json": report}, tmp_path)
    assert (tmp_path / "cf2-paired-effects.png").stat().st_size > 0
