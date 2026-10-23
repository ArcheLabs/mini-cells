"""Synthetic CPU checks only. Never executed by the Kaggle Notebook."""

# ruff: noqa: E402
from pathlib import Path
import copy
import runpy
import tempfile
from unittest.mock import patch
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
runpy.run_path(str(BASE / "run.py"), run_name="r3l_tests")
from minicells_r3l import runner
from minicells_r3l.selection import finalists
from minicells_r3l.candidate_directions import directions
from minicells_r3l.continuation import restore
from minicells_r3l.evaluation import measured
from minicells_r3l.path_oracle import run as oracle
from minicells_r3l.data_boundaries import boundary_proof, REVISION, TOKENIZER, RANGES
from minicells_r3e.common import read, fingerprint, atomic
from minicells_r3h.routing import Model
from minicells_r3i.trainer import Trainer
from minicells_r3j.storage import Store, latest
from minicells_r3c_moe_bridge.common import synthetic_configuration
from minicells_r3c_moe_bridge.data import synthetic_batches


def main():
    torch.set_num_threads(1)
    c = read(BASE / "config/frozen.json")
    runtime = copy.deepcopy(runner.runtime(c))
    runtime["engine"] = synthetic_configuration()
    runtime["engine"]["routing"]["cells"] = 6
    runtime["training"].update(tokens_per_step=8, steps=500)
    data = {
        r: np.array([2, 3, 4, 5, 6, 7, 8, 9] * 20, dtype=np.int32)
        for r in ("calibration", "admission", "development")
    }
    data["windows"] = {r: {"evaluation": {"starts": [0, 4], "groups": [0, 1]}} for r in data}
    for arch in "AB":
        model = Model(runtime, 77, arch + "-H")
        model.route_switch_margin.fill_(1e-6)
        model.route_numeric_tolerance.fill_(1e-7)
        original = Trainer(model, runtime, arch + "-CH")
        original.parent_identity = {"synthetic": True}
        original.step = 300
        original.data_cursor = 2400
        state = copy.deepcopy(original.state())
        connection = copy.deepcopy(model.connection.state_dict())
        restored = restore(runtime, state, 77, arch + "-CH", "cpu")
        assert fingerprint(restored.cell_opt.state_dict()) == fingerprint(
            original.cell_opt.state_dict()
        )
        from minicells_r3i.adaptation import observe_update

        def paired_batches(data, engine, rng, device):
            return synthetic_batches(engine, rng, device)

        with patch("minicells_r3e.training.train_batches", paired_batches):
            continuous = restore(runtime, state, 77, arch + "-CH", "cpu")
            observe_update(continuous, data)
            restart = restore(runtime, continuous.state(), 77, arch + "-CH", "cpu")
            observe_update(continuous, data)
            observe_update(restart, data)
            assert fingerprint(continuous.model.state_dict()) == fingerprint(
                restart.model.state_dict()
            )
            assert fingerprint(continuous.cell_opt.state_dict()) == fingerprint(
                restart.cell_opt.state_dict()
            )
            assert fingerprint(continuous.data_rng.bit_generator.state) == fingerprint(
                restart.data_rng.bit_generator.state
            )
        t_model = Model(runtime, 77, arch + "-T")
        t_model.load_state_dict(model.state_dict())
        t_model.temperature = 0.7
        t_original = Trainer(t_model, runtime, arch + "-CT")
        t_original.parent_identity = {"synthetic": True}
        t_original.step = 300
        t_original.data_cursor = 2400
        t_state = t_original.state()
        with patch("minicells_r3e.training.train_batches", paired_batches):
            a = restore(runtime, t_state, 77, arch + "-CT", "cpu")
            observe_update(a, data)
            b = restore(runtime, a.state(), 77, arch + "-CT", "cpu")
            observe_update(a, data)
            observe_update(b, data)
            assert fingerprint(a.model.state_dict()) == fingerprint(b.model.state_dict())
            assert fingerprint(a.cell_opt.state_dict()) == fingerprint(b.cell_opt.state_dict())
        dirs = directions(model.connection, arch, 42, 0.01)
        assert len(dirs) == 12 and [x["family"] for x in dirs].count("RANDOM") == 8
        assert fingerprint(dirs) == fingerprint(directions(model.connection, arch, 42, 0.01))
        for row in dirs:
            assert set(row["direction"]) == set(connection)
        x = torch.tensor([[2, 3, 4, 5]])
        out = model(x)
        assert int(out.operator_rows.sum()) == 8 * x.numel()
        before = fingerprint(model.state_dict())
        summary, paths = measured(model, runtime, data)
        assert fingerprint(model.state_dict()) == before
        same, _ = measured(model, runtime, data, paths)
        assert same["path_change"] == 0
        diagnostic = oracle(model, runtime, data, count=2, width=2)
        assert (
            diagnostic["scope"] == "TARGET_AWARE_NON_DEPLOYABLE_DIAGNOSTIC"
            and not diagnostic["global_upper_bound"]
        )
        assert all(len(r["beam_path"]) == (8 if arch == "B" else 4) for r in diagnostic["records"])
        from minicells_r3h.evaluation import windows

        _, oracle_x, _ = next(windows(data, runtime, "development", "cpu"))
        actual = model(oracle_x).actions
        for position, record in enumerate(diagnostic["records"]):
            assert record["baseline_path"] == actual[position].tolist()
            assert all(0 <= a < (6 if arch == "B" else 15) for a in record["beam_path"])
        package = dict(
            official=state,
            original_connection=connection,
            original_fingerprint=fingerprint(connection),
            parameter_delta={k: torch.zeros_like(v) for k, v in connection.items()},
            numeric_tolerance=1e-7,
        )

        def source(*args):
            return (
                restore(runtime, state, 77, arch + "-CH", "cpu"),
                copy.deepcopy(package),
                {"sha256": "synthetic"},
            )

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            arm = arch + "-CH"
            scan_calls = []

            def scanned(tr, data, payload, root, ident):
                scan_calls.append(ident)
                return {
                    "chosen_scale": 0.5,
                    "attempts": [{"scale": 0.5, "passed": True}],
                    "work": {},
                    "seconds": 0.0,
                }

            history = [dict(seed=77, arm=arm, at_step=300, chosen_scale=0.5)]
            with patch.object(runner, "anchor", side_effect=source), patch.object(
                runner, "runtime", return_value=runtime
            ), patch.object(runner, "HubBackend", return_value=None), patch.object(
                runner, "assets", return_value=data
            ), patch.object(runner, "scan", side_effect=scanned), patch.object(
                runner, "public_history", return_value=history
            ):
                runner.generate(c, output, 77, arm, output, "cpu")
                assert len(scan_calls) == 14
                root = runner.root_for(output, 77, arm)
                registry = read(root / "CANDIDATES.json")
                assert len(registry) == 15
                roles = []

                def role_data(c, folder, role, *args):
                    roles.append(role)
                    assert role in ("search", "select")
                    return data

                with patch.object(runner.data, "load", side_effect=role_data):
                    runner.selection(c, output, 77, arm, output, "cpu")
                final = read(root / "FINALISTS.json")
                assert roles == ["search", "select"]
                random = sorted(
                    r["candidate_id"] for r in registry if r["family"] == "RANDOM" and r["valid"]
                )
                assert final["roles"]["RANDOM_CONTROL"] == random[0]

                def batches(data, engine, rng, device):
                    return synthetic_batches(engine, rng, device)

                with patch("minicells_r3e.training.train_batches", batches), patch(
                    "minicells_r3i.adaptation.generate", side_effect=AssertionError("new Proposal")
                ):
                    runner.train(c, output, 77, arm, output, "cpu")
                costs = read(root / "TRAINED.json")
                assert costs["physical_branches"] <= 4
                for sha in final["physical_groups"]:
                    directory = root / "branches" / sha
                    ident = runner.branch_identity(c, 77, arm, sha)
                    done = latest(directory, ident)
                    assert done["step"] == 400 and done["data_cursor"] == 3200
                    assert "r3i" not in done
                    for h in (0, 10, 50, 100):
                        assert Store(directory, ident).receipt_file(h).exists()
                before = fingerprint(costs)
                runner.train(c, output, 77, arm, output, "cpu")
                assert fingerprint(read(root / "TRAINED.json")) == before
                with patch.object(runner.data, "load", return_value=data), patch.object(
                    runner, "evaluation_barrier"
                ):
                    runner.evaluate_anchor(c, output, 77, arm, output, "cpu")
                results = read(root / "EVALUATION.json")
                assert len(results) == 16
                assert all(abs(r["V"] - r["R"] - r["C"]) < 1e-12 for r in results)
    # Interrupted mutable pointer recovers the highest fully verified receipt, not an older update.
    from minicells_r3l.storage import recover_latest
    from minicells_r3j.storage import save

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        ident = {"scope": "SYNTHETIC_ONLY"}
        save(root, ident, {"cursor": 1}, 1)
        old = read(root / "latest.json")
        save(root, ident, {"cursor": 2}, 2)
        atomic(root / "latest.json", old)
        assert recover_latest(root, ident) == {"cursor": 2}
        assert read(root / "latest.json")["step"] == 2
    # Complete twelve-anchor report fixture, including logical/physical budget separation.
    from minicells_r3l.reports import report

    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder)
        test = copy.deepcopy(c)
        test["bootstrap_replicates"] = 20
        atomic(output / "DATA_BOUNDARY_PROOF.json", {"scope": "SYNTHETIC_ONLY"})
        for seed in c["seeds"]:
            for arm in c["arms"]:
                root = runner.root_for(output, seed, arm)
                atomic(
                    root / "TRAINED.json",
                    dict(physical_branches=4, new_updates=400, new_tokens=204800),
                )
                atomic(
                    root / "CANDIDATES.json",
                    [
                        dict(
                            candidate_id="synthetic",
                            family="BASELINE",
                            valid=True,
                            status="VALID",
                            trust={},
                            parameter_sha256="synthetic",
                            anchor_receipt={"scope": "SYNTHETIC_ONLY"},
                        )
                    ],
                )
                atomic(root / "FINALISTS.json", {"scope": "SYNTHETIC_ONLY"})
                rows = []
                for horizon in (0, 10, 50, 100):
                    for role in ("BASELINE", "HISTORICAL", "SEARCH_BEST", "RANDOM_CONTROL"):
                        value = copy.deepcopy(results[0])
                        value.update(role=role, horizon=horizon)
                        rows.append(value)
                atomic(root / "EVALUATION.json", rows)
        result = report(test, output, [], None)
        assert result["status"] == "COMPLETED_EXPLORATORY"
        assert result["oracle"] == "DIAGNOSTIC_NOT_COMPLETED"
        costs = read(output / "reports/COMPUTE_COST.json")
        assert costs["new_updates"] == 4800 and costs["new_tokens"] == 2457600
        assert len(read(output / "reports/FROZEN_CELL_OPPORTUNITY.json")["comparisons"]) == 6
    # Admission helper remains the original scan implementation with only its save callback bound.
    from minicells_r3f_v2.scale_scan import scan as original_scan

    assert runner.scan.__code__ is original_scan.__code__

    stories = {i: str(i) for a, b in RANGES.values() for i in range(a, b)}
    proof = boundary_proof(REVISION, TOKENIZER, {"r3k": (8192, 8960)}, [], stories)
    assert len(proof["roles"]["report"]["accepted"]) == 512
    rows = [
        dict(candidate_id=k, family=f, valid=True, parameter_sha256=k)
        for k, f in [("base", "BASELINE"), ("hist", "HISTORICAL"), ("a", "RANDOM"), ("z", "RANDOM")]
    ]
    chosen = finalists(rows, {"hist": 1.0, "a": 2.0, "z": 3.0}, {"hist": 2.0, "a": 1.0, "z": 3.0})
    assert chosen["roles"]["RANDOM_CONTROL"] == "a"
    print(
        {"status": "R3L_SYNTHETIC_CPU_PASS", "gpu_started": False, "scientific_confirmation": False}
    )


if __name__ == "__main__":
    main()
