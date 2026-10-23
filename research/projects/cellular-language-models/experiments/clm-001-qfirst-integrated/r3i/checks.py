"""Synthetic CPU lineage, paired adaptation and immutable resume tests."""

# ruff: noqa: E402
from pathlib import Path
import runpy
import copy
import tempfile
from unittest.mock import patch
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
runpy.run_path(str(BASE / "run.py"), run_name="r3i_test_bootstrap")
from minicells_r3i.trainer import Trainer
from minicells_r3i import adaptation
from minicells_r3h.routing import Model
from minicells_r3h.parent_import import read, fingerprint
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
                with (
                    patch.object(adaptation, "generate", generate),
                    patch.object(adaptation, "scan", return_value={"chosen_scale": 1.0}),
                    patch.object(adaptation, "crossover", attribution),
                ):
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
                    tr.update(None)
                    assert tr.step == 4 and tr.data_cursor == 32
                    assert all(p.grad is None for p in tr.model.connection.parameters())
    print(
        {
            "status": "R3I_SYNTHETIC_TRANSACTION_PASS",
            "gpu_training": False,
            "full_integration_complete": False,
        }
    )


if __name__ == "__main__":
    check()
