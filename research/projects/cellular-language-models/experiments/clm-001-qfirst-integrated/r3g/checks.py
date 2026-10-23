"""Synthetic CPU attribution checks. No historical data, optimizer or network required."""
# ruff: noqa: E402

from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
import copy
import json
import runpy
import tempfile
from unittest.mock import patch
import torch
import numpy as np

BASE = Path(__file__).resolve().parent
runpy.run_path(str(BASE / "run.py"), run_name="r3g_checks_bootstrap")
from minicells_r3g.common import configuration, parent_identity, atomic, digest  # noqa:E402
from minicells_r3g import (
    branch_restore,
    parameter_diff,
    crossover,
    routing_distribution,
    trajectory_audit,
)  # noqa:E402
from minicells_r3g.parent_restore import Inputs  # noqa:E402
from minicells_r3g.runner import models, arm_run  # noqa:E402
from minicells_r3f.stable_connection import Model  # noqa:E402
from minicells_r3c_moe_bridge.common import synthetic_configuration  # noqa:E402
from minicells_r3c_moe_bridge.checkpoint import Store  # noqa:E402


class Data(dict):
    def __getitem__(self, key):
        if key in ("train", "admission", "development", "confirmation"):
            raise AssertionError("forbidden data role")
        return super().__getitem__(key)


def fixture():
    c = copy.deepcopy(configuration())
    c["synthetic"] = True
    runtime = json.loads((BASE.parent / "r3f-v2/config/frozen.json").read_text())
    runtime["engine"] = synthetic_configuration()
    runtime["engine"]["routing"]["cells"] = 6
    runtime["training"]["tokens_per_step"] = 8
    runtime["proposal"]["reference_calibration_windows"] = 2
    c["parent_runtime"] = runtime
    data = Data(calibration=np.array([2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype=np.int32))
    data["windows"] = {"calibration": {"evaluation": {"starts": [0, 8], "groups": [0, 0]}}}
    return c, data


def states(c, arm):
    torch.manual_seed(3)
    model = Model(c["parent_runtime"], 151001301, arm)
    model.route_switch_margin.fill_(1e-6)
    model.route_numeric_tolerance.fill_(1e-7)
    initial = copy.deepcopy(model.state_dict())
    original = copy.deepcopy(model.connection.state_dict())
    full = copy.deepcopy(original)
    full["linear.bias"][0] += 4.0
    delta = {k: full[k] - original[k] for k in original}
    chosen = {
        k: original[k] + c["chosen_scale"] * delta[k]
        if original[k].is_floating_point()
        else original[k]
        for k in original
    }
    common = {"data_rng": {"same": 1}, "route_rng": torch.Generator().manual_seed(1).get_state()}
    official = {
        "step": 100,
        "data_cursor": 800,
        "model": initial,
        "cell_optimizer": {"initial": 1},
        **copy.deepcopy(common),
    }
    old = copy.deepcopy(official)
    old.update(step=110, data_cursor=880)
    for k in old["model"]:
        if k.startswith("cells.") and old["model"][k].is_floating_point():
            old["model"][k] += 1e-3
    new = copy.deepcopy(old)
    if arm == "A-C":
        key = next(
            k
            for k in new["model"]
            if k.startswith("cells.0.") and new["model"][k].is_floating_point()
        )
        new["model"][key].reshape(-1)[0] += 1e-4
    for k, v in chosen.items():
        new["model"]["connection." + k] = v.clone()
    candidate_model = copy.deepcopy(initial)
    for k, v in chosen.items():
        candidate_model["connection." + k] = v.clone()
    initial_pair = {
        "phase": "PROPOSED",
        "official": copy.deepcopy(official),
        "candidate": candidate_model,
        "old": None,
        "new": None,
    }
    proposal = {
        "official": official,
        "original_connection": original,
        "candidate_connection": full,
        "parameter_delta": delta,
    }
    callshape = 4 if arm == "A-C" else 8
    calls = [[0] * 6 for _ in range(callshape)]
    for row in calls:
        row[0] = 8 * (2 if arm == "A-C" else 1)
    rows = [
        {
            "step": step,
            "input_hashes": ["synthetic"],
            "action_change": 1 / 64 if arm == "A-C" and step == 108 else 0.0,
            "path_change": 1 / 8 if arm == "A-C" and step == 108 else 0.0,
            "old_cell_counts": copy.deepcopy(calls),
            "new_cell_counts": copy.deepcopy(calls),
            "old_cell_gradient_norms": [1.0] * 6,
            "new_cell_gradient_norms": [1.0] * 6,
            "old_state_training_mixture_kl": 1e-10,
            "new_state_training_mixture_kl": 1e-10,
        }
        for step in range(101, 111)
    ]
    pair = {
        "phase": "ADAPTING",
        "adapt_done": 10,
        "old": old,
        "new": new,
        "training_route_rows": rows,
    }
    selected = copy.deepcopy(new if c["selected_branches"][arm] == "new" else old)
    decision = {
        "chosen_scale": c["chosen_scale"],
        "trust_region_passed": True,
        "paired_route_streams_verified": True,
        "adapt_steps": 10,
        "final_quality_accepted": c["selected_branches"][arm] == "new",
    }
    committed = {
        "phase": "COMMITTED",
        "selected": {**copy.deepcopy(selected), "proposals": [decision]},
    }
    return {
        "official": official,
        "old": old,
        "new": new,
        "selected": selected,
        "proposal": proposal,
        "pair": pair,
        "initial": initial_pair,
        "committed": committed,
        "routes": rows,
    }


def validate(c, arm, s):
    return branch_restore.validate(
        c,
        arm,
        s["official"],
        s["old"],
        s["new"],
        s["selected"],
        s["proposal"],
        s["pair"],
        s["initial"],
        s["committed"],
    )


def write_parent(root, c, arm, s, nn):
    ident = {
        **parent_identity(c),
        "arm": arm,
        "seed": c["seed"],
        "parent_checkpoint_sha256": "synthetic",
    }

    def store(name, value, sequence=0):
        st = Store(root / "private-checkpoints" / name, ident)
        row = st.save(SimpleNamespace(step=sequence, state=lambda: value))
        atomic(st.root / "latest.json", row)

    store("proposals/" + arm, s["proposal"])
    for sequence in range(11):
        st = Store(
            root / "private-checkpoints/jobs" / arm / "transactions/proposal-0100",
            {**ident, "proposal_at_step": 100},
        )
        item = (
            s["initial"] if sequence == 0 else {**copy.deepcopy(s["pair"]), "adapt_done": sequence}
        )
        if sequence > 0:
            for side in ("old", "new"):
                item[side]["step"] = 100 + sequence
                item[side]["data_cursor"] = (100 + sequence) * c["parent_runtime"]["training"][
                    "tokens_per_step"
                ]
        row = st.save(SimpleNamespace(step=sequence, state=lambda: item))
        atomic(st.root / "latest.json", row)
    st = Store(
        root / "private-checkpoints/jobs" / arm / "transactions/proposal-0100",
        {**ident, "proposal_at_step": 100},
    )
    row = st.save(SimpleNamespace(step=12, state=lambda: s["committed"]))
    atomic(st.root / "latest.json", row)
    store("jobs/" + arm + "/selected", s["selected"], 110)
    store("jobs/" + arm + "/training-observations", {"rows": s["routes"]}, 10)
    atomic(
        root / "private-checkpoints/jobs" / arm / "result.json",
        {
            "identity": ident,
            "trust": {"chosen_scale": c["chosen_scale"]},
            "training_routes": s["routes"],
            "adaptation": {"old_calibration_nll": nn["L_OO"], "new_calibration_nll": nn["L_NN"]},
        },
    )
    atomic(root / "identity.json", parent_identity(c))


def main():
    torch.set_num_threads(1)
    c, data = fixture()
    with ExitStack() as guard:
        guard.enter_context(
            patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("optimizer step"))
        )
        guard.enter_context(
            patch.object(
                torch.optim.AdamW, "__init__", side_effect=AssertionError("optimizer created")
            )
        )
        guard.enter_context(
            patch.object(torch.Tensor, "backward", side_effect=AssertionError("backward called"))
        )
        for arm in c["arms"]:
            s = states(c, arm)
            verification = validate(c, arm, s)
            assert verification["selected_branch_verified"]
            ms = models(c, arm, s, "cpu")
            diff = parameter_diff.cells(s["old"]["model"], s["new"]["model"])
            assert diff["all_cells_bitwise_equal"] == (arm == "B-C")
            assert len(diff["cells"]) == 6
            trial = crossover.evaluate(c, arm, ms, data)
            assert not trial["historical_nll_reproduced"]
            c["historical_nll"][arm] = {k: trial["L_" + k] for k in ("OO", "NN")}
            cross = crossover.evaluate(c, arm, ms, data)
            assert cross["historical_nll_reproduced"]
            if arm == "B-C":
                assert cross["L_NO"] == cross["L_OO"] and cross["L_NN"] == cross["L_ON"]
                assert cross["interaction"] == 0
            trajectory = trajectory_audit.review(
                c, arm, {"routes": s["routes"], "parent_result": {"training_routes": s["routes"]}}
            )
            assert trajectory["matches_protocol_described_steps"]
            attributed = crossover.attribution(cross, diff, trajectory)
            assert attributed["cell_parameters_changed"] == (arm == "A-C")
            routing = routing_distribution.analyze(c, ms, data)
            for v in routing["connections"].values():
                assert len(v["epsilon_schedule"]) == 10
                entropy = v["raw_softmax"]["normalized_entropy"]["median"]
                assert 0 <= entropy <= 1 + 1e-12
                assert 0 <= v["raw_softmax"]["expected_hard_sample_agreement"] <= 1
            bad = copy.deepcopy(s)
            key = next(
                k
                for k in bad["new"]["model"]
                if not k.startswith(("cells.", "connection."))
                and bad["new"]["model"][k].is_floating_point()
            )
            bad["new"]["model"][key].reshape(-1)[0] += 1
            try:
                validate(c, arm, bad)
            except RuntimeError as e:
                assert str(e).startswith("FROZEN_COMPUTATION_WEIGHTS_DIFFER")
            else:
                raise AssertionError("frozen mismatch accepted")
            bad = copy.deepcopy(s)
            bad["selected"]["cell_optimizer"] = {"wrong": 1}
            try:
                validate(c, arm, bad)
            except RuntimeError as e:
                assert str(e) == "SELECTED_BRANCH_STATE_MISMATCH"
            else:
                raise AssertionError("selected mismatch accepted")
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                parent = root / "parent"
                write_parent(parent, c, arm, s, cross)
                inp = Inputs(c, root / "audit", local=parent)
                restored = branch_restore.restore(inp, c, arm)
                assert restored["verification"]["selected_branch_verified"]
                with patch("minicells_r3g.runner.ensure", return_value=data):
                    result = arm_run(c, arm, inp, None, root / "assets", "cpu")
                assert result["valid"] and result["crossover"]["historical_nll_reproduced"]
                assert inp.unchanged()
                # Pin a changing remote head once and reuse local paths without repeated HF calls.
                files = {
                    p.relative_to(parent / "private-checkpoints").as_posix(): {
                        "sha256": digest(p),
                        "bytes": p.stat().st_size,
                        "object": str(p),
                    }
                    for p in (parent / "private-checkpoints").rglob("*")
                    if p.is_file()
                }
                fetched = []

                def get_file(name, revision):
                    assert revision == "pinned-revision"
                    fetched.append(name)
                    return Path(name)

                backend = SimpleNamespace(
                    repo=c["parent"]["hf_repo"],
                    info=lambda revision=None: SimpleNamespace(sha="pinned-revision", private=True),
                    optional=lambda path, revision: {
                        "identity": parent_identity(c),
                        "manifest_path": "manifest",
                        "manifest_sha256": "hash",
                    },
                    get=get_file,
                )
                with patch(
                    "minicells_r3g.parent_restore.verified_manifest", return_value={"files": files}
                ):
                    remote = Inputs(c, root / "remote-audit", backend=backend)
                    remote.checkpoint("jobs/" + arm + "/transactions/proposal-0100", 10)
                    count = len(fetched)
                    remote.checkpoint("jobs/" + arm + "/transactions/proposal-0100", 10)
                    assert len(fetched) == count
                    assert remote.reference["hf_revision"] == "pinned-revision"
                    assert remote.unchanged()
                rowpath = parent / "private-checkpoints/jobs" / arm / "transactions/proposal-0100"
                receipt = Store(
                    rowpath,
                    {
                        **parent_identity(c),
                        "arm": arm,
                        "seed": c["seed"],
                        "parent_checkpoint_sha256": "synthetic",
                        "proposal_at_step": 100,
                    },
                ).receipt_file(10)
                receipt.unlink()
                try:
                    branch_restore.restore(inp, c, arm)
                except RuntimeError as e:
                    assert str(e) == "PAIRED_BRANCH_STATE_NOT_RETAINED"
                else:
                    raise AssertionError("selected-only substitute accepted")
    values = {"0O": 5.0, "0N": 4.9, "OO": 4.0, "ON": 3.9, "NO": 3.8, "NN": 3.5}
    effects = crossover.effects(values)
    assert abs(effects["interaction"] + 0.2) < 1e-12
    print(
        json.dumps(
            {
                "status": "R3G_SYNTHETIC_CPU_PASS",
                "optimizer_updates": 0,
                "real_audit_executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
