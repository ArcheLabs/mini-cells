"""Focused synthetic CPU correctness only; never imported by the Kaggle launcher."""

# ruff: noqa: E402
from pathlib import Path
import copy
import runpy
import tempfile
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
runpy.run_path(str(BASE / "run.py"), run_name="r3h_checks_bootstrap")
from minicells_r3h.parent_import import config, fingerprint
from minicells_r3h.routing import Model
from minicells_r3h.sampler import random_inputs, sample, probabilities
from minicells_r3h.trainer import Trainer
from minicells_r3h.storage import Store, save
from minicells_r3h.alignment import temperature, measure
from minicells_r3h.statistics import compare
from minicells_r3h.runner import barrier
from minicells_r3c_moe_bridge.common import synthetic_configuration
from minicells_r3c_moe_bridge.data import synthetic_batches
from minicells_r3f.stable_connection import Model as OriginalModel, hard_action


def check():
    torch.set_num_threads(1)
    c = config(BASE / "config/frozen.json")
    assert len(c["arms"]) * len(c["seeds"]) * 500 == 12000
    q = torch.tensor([[0.02, 0.03, 0.01]])
    hard = torch.tensor([0])
    assert hard_action(q, hard, torch.tensor(0.02)).item() == 0
    for mode in "FHTM":
        p = probabilities(q, hard, mode, 0.15, 0.1, 0.75)
        assert torch.allclose(p.sum(-1), torch.ones(1, dtype=torch.double))
        n = 50000
        u = random_inputs(9, 1, 0, 0, n, 3, "cpu")
        a = sample(q.expand(n, -1), hard.expand(n), mode, 0.15, 0.1, 0.75, u)
        observed = torch.bincount(a, minlength=3).double() / n
        assert (observed - p[0]).abs().max() < 0.012, (mode, observed, p)
    u = random_inputs(9, 1, 0, 0, 4, 3, "cpu")
    v = random_inputs(9, 1, 0, 0, 4, 3, "cpu")
    assert all(torch.equal(u[k], v[k]) for k in u)
    assert not torch.equal(u["gumbel"], random_inputs(9, 2, 0, 0, 4, 3, "cpu")["gumbel"])
    c = copy.deepcopy(c)
    c["engine"] = synthetic_configuration()
    c["engine"]["routing"]["cells"] = 6
    c["training"]["tokens_per_step"] = 8
    c["training"]["steps"] = 2
    c["training"]["checkpoint_every"] = 1
    c["alignment"]["reference_windows"] = 1
    c["proposal"]["reference_calibration_windows"] = 1
    data = {
        "admission": np.array([2, 3, 4, 5, 6, 7, 8, 9] * 4, dtype=np.int32),
        "windows": {"admission": {"evaluation": {"starts": [0], "groups": [0]}}},
    }
    for arch in "AB":
        torch.manual_seed(123)
        m = Model(c, 11, arch + "-F")
        m.route_switch_margin.fill_(1e-6)
        m.route_numeric_tolerance.fill_(1e-7)
        x = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
        original = OriginalModel(c, 11, arch + "-C")
        original.load_state_dict(m.state_dict())
        with torch.no_grad():
            expected = original(x)
            actual = m(x)
            assert torch.equal(actual.logits, expected.logits) and torch.equal(
                actual.actions, expected.actions
            )
        # Exact original F marginal and common Gumbel/epsilon mechanism with supplied draws.
        rng = torch.Generator().manual_seed(10)
        with torch.no_grad():
            h, _ = m.initial(x)
            prev = torch.full((len(h),), 6, dtype=torch.long)
            raw, _, _ = m.scores(h, 0, prev)
            g = torch.rand(raw.shape, generator=rng).clamp_(1e-7, 1 - 1e-7)
            e = torch.rand(len(h), generator=rng)
            pick = torch.randint(raw.shape[-1], (len(h),), generator=rng)
            expected = torch.where(e < 0.15, pick, (raw - (-g.log()).log()).argmax(-1))
            action = sample(
                raw,
                actual.actions[:, 0],
                "F",
                0.15,
                1.0,
                0.75,
                {"gumbel": g, "epsilon": e, "exploration": pick, "mixture": e},
            )
            assert torch.equal(action, expected)
        fit = temperature(m, c, data)
        assert fit["temperature"] in c["alignment"]["temperatures"]
        assert fit["uses_targets"] is False
        logs = []
        for mode in "FHTM":
            model = copy.deepcopy(m)
            model.routing_mode = mode
            model.temperature = fit["temperature"] if mode == "T" else 1.0
            with torch.no_grad():
                assert torch.equal(model(x).logits, actual.logits)
                met = measure(model, c, data, 0.15)
            assert 0 <= met["hard_path_match_probability"]["mean"] <= 1
            if mode == "H":
                k = 6 if arch == "B" else 15
                p = (1 - 0.15) + 0.15 / k
                assert abs(met["hard_path_match_probability"]["mean"] - p**model.steps) < 1e-10, (
                    arch,
                    met["hard_path_match_probability"]["mean"],
                    p**model.steps,
                    model.steps,
                )
            tr = Trainer(model, c, arch + "-" + mode)
            tr.parent_identity = {"synthetic": True}
            batches = synthetic_batches(c["engine"], np.random.default_rng(1))
            tr.update(None, batches)
            assert tr.step == 1 and tr.data_cursor == 8
            assert all(p.grad is None for p in model.connection.parameters())
            assert int(np.array(tr.curve[-1]["exposure"]["stage_cells"]).sum()) == 8 * 8
            saved = copy.deepcopy(tr.state())
            with tempfile.TemporaryDirectory() as tmp:
                store = Store(tmp, {"synthetic": mode})
                receipt = save(tr, store)
                restored = Trainer(copy.deepcopy(m), c, arch + "-" + mode) if mode == "F" else None
                # Reconstruct router metadata before restore, never infer from current defaults.
                clone = copy.deepcopy(model)
                restored = Trainer(clone, c, arch + "-" + mode)
                restored.parent_identity = {"synthetic": True}
                restored.restore(store.verified(receipt))
                tr.update(None, batches)
                restored.update(None, batches)
                assert fingerprint(tr.model.state_dict()) == fingerprint(
                    restored.model.state_dict()
                )
                assert fingerprint(tr.cell_opt.state_dict()) == fingerprint(
                    restored.cell_opt.state_dict()
                )
                assert tr.data_cursor == restored.data_cursor == 16
                bad = copy.deepcopy(saved)
                bad["r3h"]["temperature"] *= 2
                try:
                    restored.restore(bad)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("router identity accepted")
                broken = copy.deepcopy(saved)
                broken["data_cursor"] += 1
                try:
                    restored.restore(broken)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("cursor mismatch accepted")
            logs.append(tr.curve[0]["exposure"])
        # All models see identical external data; their selected actions are allowed to differ.
        assert len(logs) == 4
    with tempfile.TemporaryDirectory() as tmp:
        try:
            barrier(c, tmp)
        except Exception as error:
            assert "MANIFEST_INVALID" in str(error)
        else:
            raise AssertionError("Development barrier bypassed")
    # Synthetic full matrix statistics: primary family=4, T secondary only.
    full = config(BASE / "config/frozen.json")
    results = {}
    for seed in full["seeds"]:
        per = {}
        for arm in full["arms"]:
            loss = 4.0 if arm[-1] == "F" else 3.99
            stories = {
                str(i): {"tokens": 64, "mean_nll": loss, "nll_sum": loss * 64} for i in range(8)
            }
            per[arm] = {"quality": {"nll": loss, "stories": stories}}
        results[str(seed)] = per
    stats = compare(full, results)
    assert len(stats) == 6 and sum("holm_exploratory_p" in v for v in stats.values()) == 4
    assert all(v["status"] == "SUPPORTED_EXPLORATORY" for v in stats.values())
    print(
        {
            "status": "R3H_SYNTHETIC_CPU_PASS",
            "real_training_started": False,
            "confirmation": "SEALED",
        },
        flush=True,
    )


if __name__ == "__main__":
    check()
