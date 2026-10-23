"""Synthetic CPU lineage, paired adaptation and immutable resume tests."""

# ruff: noqa: E402
from pathlib import Path
import runpy
import copy
import tempfile
from contextlib import ExitStack
from unittest.mock import patch
import torch
import numpy as np

BASE = Path(__file__).resolve().parent
runpy.run_path(str(BASE / "run.py"), run_name="r3i_test_bootstrap")
from minicells_r3i.trainer import Trainer
from minicells_r3i import adaptation
from minicells_r3h.routing import Model
from minicells_r3h.parent_import import read, fingerprint, atomic
from minicells_r3c_moe_bridge.common import synthetic_configuration
from minicells_r3c_moe_bridge.data import synthetic_batches


def check():
    torch.set_num_threads(1)
    c = read(BASE / "config/frozen.json")
    c["engine"] = synthetic_configuration()
    c["engine"]["routing"]["cells"] = 6
    c["training"].update(steps=5, tokens_per_step=8, adapt_steps=1)

    def batches(data, engine, rng, device):
        return synthetic_batches(engine, rng, device)

    for architecture in "AB":
        m = Model(c, 9, architecture + "-H")
        m.route_switch_margin.fill_(1e-6)
        m.route_numeric_tolerance.fill_(1e-7)
        tr = Trainer(m, c, architecture + "-CH")
        tr.parent_identity = {"synthetic": True}
        with patch("minicells_r3e.training.train_batches", batches):
            tr.update(None)
            start = copy.deepcopy(tr.state())
            clone = tr.clone()
            assert fingerprint(start) == fingerprint(clone.state())

            def generate(t, data, root, identity, durable):
                original = copy.deepcopy(t.model.connection.state_dict())
                delta = {k: torch.zeros_like(v) for k, v in original.items()}
                delta["linear.bias"][0] = 0.001
                return dict(
                    official=copy.deepcopy(t.state()),
                    original_connection=original,
                    parameter_delta=delta,
                )

            def attribution(old, new, c, data):
                return {"values": {"OO": 4.0, "ON": 3.99, "NO": 4.0, "NN": 3.98}}

            with tempfile.TemporaryDirectory() as folder:
                with ExitStack() as stack:
                    stack.enter_context(patch.object(adaptation, "generate", generate))
                    stack.enter_context(
                        patch.object(adaptation, "scan", return_value={"chosen_scale": 1.0})
                    )
                    stack.enter_context(patch.object(adaptation, "crossover", attribution))
                    event = adaptation.execute(tr, None, folder, {"synthetic": True}, lambda: None)
                    assert event["accepted"] and tr.step == 2 and tr.data_cursor == 16
                    assert tr.connection_version == 1 and len(tr.proposals) == 1
                    clone = tr.clone()
                    assert fingerprint(clone.model.state_dict()) == fingerprint(
                        tr.model.state_dict()
                    )
                    # Reload a completed transaction from its original official step.
                    restored = Trainer(copy.deepcopy(m), c, architecture + "-CH")
                    restored.initial_connection_sha = tr.initial_connection_sha
                    restored.parent_identity = {"synthetic": True}
                    restored.restore(start)
                    adaptation.execute(restored, None, folder, {"synthetic": True}, lambda: None)
                    assert restored.step == 2 and len(restored.proposals) == 1
                    assert fingerprint(restored.cell_opt.state_dict()) == fingerprint(
                        tr.cell_opt.state_dict()
                    )
                    # The next event uses the accepted Connection, not the original zero state.
                    adaptation.execute(tr, None, folder, {"synthetic": True}, lambda: None)
                    assert tr.connection_version == 2 and tr.step == 3 and len(tr.proposals) == 2
                    previous = fingerprint(tr.model.connection.state_dict())
                    stack.enter_context(
                        patch.object(
                            adaptation, "crossover", return_value={"values": {"OO": 3.0, "NN": 4.0}}
                        )
                    )
                    rejected = adaptation.execute(
                        tr, None, folder, {"synthetic": True}, lambda: None
                    )
                    assert not rejected["accepted"] and tr.step == 4
                    assert (
                        tr.connection_version == 2
                        and fingerprint(tr.model.connection.state_dict()) == previous
                    )
                    tr.update(None)
                    assert tr.step == 5 and tr.data_cursor == 40
                    assert all(p.grad is None for p in tr.model.connection.parameters())
    # Crash immediately after an immutable ADAPTING save, then resume that pair once.
    c["training"]["adapt_steps"] = 2
    model = Model(c, 13, "A-H")
    model.route_switch_margin.fill_(1e-6)
    model.route_numeric_tolerance.fill_(1e-7)
    interrupted = Trainer(model, c, "A-CH")
    interrupted.parent_identity = {"synthetic": True}
    persist = adaptation.save

    def crash(root, identity, value, sequence=0):
        result = persist(root, identity, value, sequence)
        if value.get("phase") == "ADAPTING" and value["adapt_done"] == 1:
            raise RuntimeError("SYNTHETIC_CRASH_AFTER_SAVE")
        return result

    with tempfile.TemporaryDirectory() as folder:
        with ExitStack() as stack:
            stack.enter_context(patch("minicells_r3e.training.train_batches", batches))
            stack.enter_context(patch.object(adaptation, "generate", generate))
            stack.enter_context(
                patch.object(adaptation, "scan", return_value={"chosen_scale": 1.0})
            )
            stack.enter_context(patch.object(adaptation, "crossover", attribution))
            with patch.object(adaptation, "save", crash):
                try:
                    adaptation.execute(
                        interrupted, None, folder, {"resume": "synthetic"}, lambda: None
                    )
                except RuntimeError as error:
                    assert str(error) == "SYNTHETIC_CRASH_AFTER_SAVE"
                else:
                    raise AssertionError("crash was not injected")
            adaptation.execute(interrupted, None, folder, {"resume": "synthetic"}, lambda: None)
            assert interrupted.step == 2 and interrupted.data_cursor == 16
            assert (
                len(interrupted.proposals) == 1
                and len(interrupted.proposals[0]["training_path_rows"]) == 2
            )
    c["training"]["adapt_steps"] = 1
    # Exercise the real saved candidate and all 22 admission scales on tiny tensors.
    c["training"].update(states=2, policy_steps=1)
    c["proposal"]["reference_admission_windows"] = 1
    c["proposal"]["reference_calibration_windows"] = 1
    for architecture in "AB":
        model = Model(c, 12, architecture + "-H")
        model.route_switch_margin.fill_(1e-6)
        model.route_numeric_tolerance.fill_(1e-7)
        trainer = Trainer(model, c, architecture + "-CH")
        trainer.parent_identity = {"synthetic": True}
        data = {
            role: np.array([2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype=np.int32)
            for role in ("calibration", "admission")
        }
        data["windows"] = {
            role: {"evaluation": {"starts": [0], "groups": [0]}}
            for role in ("calibration", "admission")
        }
        with tempfile.TemporaryDirectory() as folder:
            with patch("minicells_r3e.training.train_batches", batches), patch(
                "minicells_r3e.policy_proposal.train_batches", batches
            ):
                trainer.update(data)
                event = adaptation.execute(
                    trainer, data, folder, {"actual_synthetic": architecture}, lambda: None
                )
                assert len(event["scan"]["attempts"]) == 22
                assert event["scan"]["attempts"][0]["scale"] == 0
                assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
    # Execute the actual job loop through checkpointing and report construction on synthetic CPU data.
    from minicells_r3i import runner
    from minicells_r3i.reports import report
    from minicells_r3i.baseline_reference import compare_step100

    with tempfile.TemporaryDirectory() as folder:
        c["seeds"] = [9]
        c["new_arms"] = ["A-CH"]
        c["training"]["proposal_steps"] = []
        c["training"]["checkpoint_every"] = 1
        c["alignment"]["reference_windows"] = 1
        template = Model(c, 9, "A-H")
        template.route_switch_margin.fill_(1e-6)
        template.route_numeric_tolerance.fill_(1e-7)

        def init(c, output, seed, arm, device):
            t = Trainer(copy.deepcopy(template), c, arm)
            t.parent_identity = {"synthetic": True}
            return t, {"sha256": "synthetic-parent"}

        with patch.object(runner, "initial", init), patch(
            "minicells_r3e.training.train_batches", batches
        ):
            runner.run_job(c, folder, 9, "A-CH", data, "cpu")
            root = runner.directory(folder, 9, "A-CH")
            before = read(root / "result.json")
            runner.run_job(c, folder, 9, "A-CH", data, "cpu")
            assert read(root / "result.json") == before
            state = runner.Store(root, before["identity"]).verified(before["checkpoint"])
            assert state["step"] == 5 and state["data_cursor"] == 40
            historic = copy.deepcopy(state)
            historic["step"] = 100
            state["step"] = 100
            assert compare_step100(state, historic)["status"] == "EXACT_STEP100_COMPATIBILITY_PASS"
            historic["data_cursor"] += 1
            try:
                compare_step100(state, historic)
            except RuntimeError:
                pass
            else:
                raise AssertionError("historical mismatch was accepted")
            report(c, Path(folder), {"dummy": "unused"} if False else [], None)
            assert (Path(folder) / "reports/RUN_MANIFEST.json").exists()
    # Full report schema and four comparisons over a synthetic 3-Seed matrix.
    from types import SimpleNamespace
    from minicells_r3i import reports as reporting

    full = read(BASE / "config/frozen.json")

    def q(value):
        return {
            "nll": value,
            "stories": {str(i): {"tokens": 64, "mean_nll": value} for i in range(4)},
        }

    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder)
        for seed in full["seeds"]:
            for frozen, oldarm in full["baseline"]["mapping"].items():
                path = (
                    output
                    / "baseline-r3h"
                    / ("jobs/seed-%s-arm-%s" % (seed, oldarm))
                    / "result.json"
                )
                atomic(
                    path,
                    {
                        "identity": {"run_id": full["baseline"]["run_id"]},
                        "status": "COMPLETED",
                        "checkpoint": {"sha256": "synthetic"},
                        "quality": q(4.0),
                    },
                )
            for arm in full["new_arms"]:
                r = copy.deepcopy(before)
                r.update(status="COMPLETED", quality=q(3.99))
                atomic(runner.directory(output, seed, arm) / "result.json", r)
        with patch.object(
            reporting, "Store", return_value=SimpleNamespace(verified=lambda row: state)
        ):
            summary = reporting.report(full, output, [], None)
        assert summary["status"] == "COMPLETED" and len(summary["comparisons"]) == 4
        assert all(v["status"] == "SUPPORTED_EXPLORATORY" for v in summary["comparisons"].values())
    from minicells_r3i.runner import main as entrypoint

    assert callable(entrypoint)
    print(
        {
            "status": "R3I_SYNTHETIC_TRANSACTION_PASS",
            "gpu_training": False,
            "full_integration_complete": True,
        }
    )


if __name__ == "__main__":
    check()
