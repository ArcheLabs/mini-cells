"""Only R3e scientific-correctness checks on tiny synthetic CPU state; never run in Kaggle."""

from pathlib import Path
import copy
import json
import runpy
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F

BASE = Path(__file__).resolve().parent
runpy.run_path(str(BASE / "run.py"), run_name="r3e_checks_bootstrap")
from minicells_r3e.common import configuration, fingerprint, identity  # noqa:E402
from minicells_r3e.stable_connection import build  # noqa:E402
from minicells_r3e.training import Trainer  # noqa:E402
from minicells_r3e.policy_proposal import collect, optimize  # noqa:E402
from minicells_r3e.adaptation import execute  # noqa:E402
from minicells_r3e.evaluation import terminal, trust  # noqa:E402
from minicells_r3e.runner import parity, job  # noqa:E402
from minicells_r3e.storage import Store, save, export  # noqa:E402
from minicells_r3c_moe_bridge.common import synthetic_configuration  # noqa:E402
from minicells_r3c_moe_bridge.model import BridgeModel  # noqa:E402


def comparable(s):
    s = copy.deepcopy(s)
    s["cost"].pop("elapsed_update_seconds", None)
    return s


def main():
    torch.set_num_threads(1)
    c = copy.deepcopy(configuration())
    c["synthetic"] = True
    c["engine"] = synthetic_configuration()
    c["engine"]["routing"]["cells"] = 6
    c["engine"]["balance"]["coefficient"] = 0.0
    c["training"].update(
        steps=5,
        tokens_per_step=8,
        proposal_steps=[1],
        policy_steps=2,
        states=4,
        adapt_steps=2,
        checkpoint_every=1,
        print_every=5,
    )
    c["proposal"].update(
        reference_calibration_windows=2,
        kl_max=100,
        action_change_max=1,
        path_change_max=1,
        minimum_calibration_improvement=-100.0,
    )
    parent = {"model": BridgeModel(c["engine"], "D", 7).state_dict()}
    token = np.arange(65, dtype="int64") % 14 + 2
    data = {r: token.copy() for r in ("train", "calibration", "development")}
    data["windows"] = {
        r: {"starts": [0, 8, 16, 24], "evaluation": {"starts": [0, 8], "groups": [0, 1]}}
        for r in data
    }
    x = torch.tensor(token[:4][None])
    for arm in c["arms"]:
        m = build(c, parent, 7, arm, "cpu")
        parity(m, parent, x)
        out = m(x)
        assert out.operator_rows.sum() == 8 * x.numel()
        if arm == "B-R":
            assert torch.equal(
                out.actions, m(x, epsilon=0.15, route_rng=torch.Generator().manual_seed(1)).actions
            )
        if arm == "B-C":
            # Force an actual self edge on every step without changing Cell structure.
            with torch.no_grad():
                m.connection.linear.bias[0] = 1e4
            assert (m(x).actions == 0).all()
    m = build(c, parent, 7, "A-C", "cpu")
    tr = Trainer(m, c, "A-C")
    tr.update(data)
    tr.verify_frozen()
    state = copy.deepcopy(tr.state())
    clone = tr.clone()
    a = tr.update(data)
    b = clone.update(data)
    assert a == b and fingerprint(comparable(tr.state())) == fingerprint(comparable(clone.state()))
    tr.restore(state)
    for arm in ("A-C", "B-C"):
        m = build(c, parent, 7, arm, "cpu")
        tr = Trainer(m, c, arm)
        tr.update(data)
        before = copy.deepcopy(tr.state())
        records = collect(tr, data)
        # Exact single-state forced rerouted suffix, including graph predecessor, agrees with vector Q.
        r = records[0]
        action = 0
        h, p = m.transition(r["h"], r["stage"], r["previous"], torch.tensor([action]))
        for t in range(r["stage"] + 1, m.steps):
            aa, _ = m.select(h, t, p)
            h, p = m.transition(h, t, p, aa)
        assert len(r["values"]) == (6 if m.graph else 15)
        single = F.cross_entropy(m.head(h), r["target"])
        torch.testing.assert_close(single, r["values"][0], rtol=1e-5, atol=1e-6)
        # Policy objective must not alter official Cells or connection during proposal construction.
        original_weights = fingerprint(m.state_dict())
        candidate, info = optimize(tr, data)
        assert fingerprint(m.state_dict()) == original_weights
        assert candidate is not None and len(info["losses"]) == 2
        assert trust(m, m, [x]) == {"kl": 0.0, "action_change": 0.0, "path_change": 0.0}
        tr.restore(before)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ident = identity(c, "synthetic", 7, arm)
            # Interrupt after first paired adaptation; restore exactly and finish transaction.
            import minicells_r3e.adaptation as adapter

            real_save = adapter.save_event

            def interrupt(store, package, seq):
                real_save(store, package, seq)
                if seq == 1:
                    raise RuntimeError("SYNTHETIC_INTERRUPTION")

            try:
                with patch.object(adapter, "save_event", interrupt):
                    execute(tr, data, root, ident)
            except RuntimeError as e:
                assert str(e) == "SYNTHETIC_INTERRUPTION"
            resumed = Trainer(build(c, parent, 7, arm, "cpu"), c, arm)
            resumed.restore(before)
            execute(resumed, data, root, ident)
            with tempfile.TemporaryDirectory() as clean:
                uninterrupted = Trainer(build(c, parent, 7, arm, "cpu"), c, arm)
                uninterrupted.restore(before)
                execute(uninterrupted, data, clean, ident)

            # Wall timing is telemetry, excluded from scientific/restart state equivalence.
            assert fingerprint(comparable(resumed.state())) == fingerprint(
                comparable(uninterrupted.state())
            )
            assert (
                resumed.step == 3
                and resumed.data_cursor == 24
                and resumed.proposals[-1]["accepted"]
            )
            assert resumed.cost["discarded_adaptation_updates"] == 2
            chosen = fingerprint(resumed.state())
            # A restart from the pre-event checkpoint applies the committed transaction exactly once.
            repeat = Trainer(build(c, parent, 7, arm, "cpu"), c, arm)
            repeat.restore(before)
            execute(repeat, data, root, ident)
            assert fingerprint(repeat.state()) == chosen
            store = Store(root / "main", ident)
            row = save(resumed, store)
            assert fingerprint(store.verified(row)) == chosen
            q = terminal(resumed.model, c, data)
            assert np.isfinite(q["nll"]) and q["inference_cost"]["cell_rows"] == 8 * x.numel()
            if arm == "B-C":
                assert sum(map(sum, q["graph"]["edge_counts"])) == 7 * 8
        # Reject all scales: official parameters and optimizer untouched, cf RNG legitimately advances.
        rejected = Trainer(build(c, parent, 7, arm, "cpu"), copy.deepcopy(c), arm)
        rejected.update(data)
        rejected.c["proposal"].update(kl_max=-1.0)
        old_weights = fingerprint(rejected.model.state_dict())
        old_opt = fingerprint(rejected.cell_opt.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            execute(rejected, data, tmp, identity(c, "reject", 7, arm))
        assert rejected.step == 1 and not rejected.proposals[-1]["accepted"]
        assert (
            fingerprint(rejected.model.state_dict()) == old_weights
            and fingerprint(rejected.cell_opt.state_dict()) == old_opt
        )
    # Tiny complete six-arm run, including final-only Development, receipts and coordinator export.
    with tempfile.TemporaryDirectory() as tmp:
        for arm in c["arms"]:
            r = job(c, data, Path(tmp), "synthetic", 7, arm, parent, "cpu")
            assert r["checkpoint"]["step"] == 5 and r["training_cost"]["main_tokens"] == 40
        export(Path(tmp) / "private-checkpoints", Path(tmp) / "export")
        assert len(list((Path(tmp) / "export").glob("jobs/*/latest.json"))) == 6
    # Exact production parameter totals, random fixture only; no private weights or real data.
    production = configuration()
    p = BridgeModel(production["engine"], "D", 151001301)
    totals = {}
    for arm in production["arms"]:
        m = build(production, {"model": p.state_dict()}, 151001301, arm, "cpu")
        totals[arm] = {
            "total": sum(v.numel() for v in m.parameters()),
            "cells": sum(v.numel() for v in m.cells.parameters()),
            "connection": sum(v.numel() for v in m.connection.parameters()),
        }
    evidence = {
        "status": "R3E_FOCUSED_CPU_PASS",
        "scope": "SYNTHETIC_ONLY_NO_REAL_DATA_NO_GPU_TRAINING",
        "parameters": totals,
        "checks": [
            "D parity",
            "eight true Cell calls",
            "deterministic random and self edges",
            "joint finite Cell gradients",
            "exact next update restore",
            "full-action replanned suffix",
            "official snapshot isolation",
            "three trust constraints",
            "paired adaptation interruption/recovery",
            "atomic Cell and optimizer branch selection",
            "all-scale rejection",
            "complete six-arm tiny fixture",
            "existing Store and export",
        ],
    }
    print(json.dumps(evidence, indent=2))
    return evidence


if __name__ == "__main__":
    main()
