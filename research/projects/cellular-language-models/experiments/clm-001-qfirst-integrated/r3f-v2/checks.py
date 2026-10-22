"""Focused synthetic CPU checks only; no assets, network or real training."""

from pathlib import Path
import copy
import json
import runpy
import tempfile
from contextlib import ExitStack
from unittest.mock import patch
import torch
import numpy as np

BASE = Path(__file__).resolve().parent
runpy.run_path(str(BASE / "run.py"), run_name="r3fv2_checks_bootstrap")
from minicells_r3f_v2.common import configuration, fingerprint, canonical, digest  # noqa:E402
from minicells_r3f_v2 import proposal, scale_scan, adaptation, admission, parent_import, storage  # noqa:E402
from minicells_r3f_v2.runner import hypotheses  # noqa:E402
from minicells_r3f.stable_connection import Model  # noqa:E402
from minicells_r3e.training import Trainer  # noqa:E402
from minicells_r3c_moe_bridge.common import synthetic_configuration  # noqa:E402
from minicells_r3f_v2.evaluation import admission_windows  # noqa:E402


class Data(dict):
    def __getitem__(self, key):
        if key in ("development", "confirmation"):
            raise AssertionError("forbidden split")
        return super().__getitem__(key)


def fixture():
    c = copy.deepcopy(configuration())
    c["synthetic"] = True
    c["engine"] = synthetic_configuration()
    c["engine"]["routing"]["cells"] = 6
    c["engine"]["balance"]["coefficient"] = 0.0
    c["training"].update(tokens_per_step=8, states=12, policy_steps=8, adapt_steps=10)
    c["proposal"]["reference_admission_windows"] = 2
    c["proposal"]["reference_calibration_windows"] = 2
    c["engine"]["input"] = {
        "data_config": {"admission_range": [100, 200]},
        "assets_manifest_sha256": "synthetic",
    }
    data = Data(
        {
            r: np.array([2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype=np.int32)
            for r in ("train", "calibration", "admission")
        }
    )
    data["windows"] = {
        r: {
            "starts": [0, 8],
            "groups": [0, 0],
            "story_offsets": [0, 80],
            "evaluation": {"starts": [0, 8], "groups": [0, 0]},
        }
        for r in ("train", "calibration", "admission")
    }
    return c, data


def trainer(c, arm):
    torch.manual_seed(9)
    m = Model(c, 151001301, arm)
    m.route_switch_margin.fill_(1e-6)
    m.route_numeric_tolerance.fill_(1e-7)
    t = Trainer(m, c, arm)
    t.step = 100
    t.data_cursor = 100 * c["training"]["tokens_per_step"]
    return t


def parent_restore_regression(c):
    from types import SimpleNamespace
    from minicells_r3c_moe_bridge.checkpoint import Store
    from minicells_r3f.common import identity as old_identity

    old = copy.deepcopy(c)
    old["experiment_id"] = "r3f-connection-admission"
    test = copy.deepcopy(c)
    test["local_experiment"]["parent_config_sha256"] = canonical(old)
    arm = "A-C"
    ident = old_identity(old, test["local_experiment"]["parent_source_sha"], 151001301, arm)
    state = trainer(old, arm).state()
    state["proposals"] = [
        {
            "at_step": 100,
            "trust_region_passed": False,
            "adapt_steps": 0,
            "experiment_identity": ident,
            "snapshot_fingerprint": fingerprint(state["model"]),
        }
    ]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prefix = "jobs/seed-151001301-arm-A-C/"
        store = Store(root / prefix, ident)
        row = store.save(SimpleNamespace(step=100, state=lambda: state))
        (store.root / "latest.json").write_text(json.dumps(row))
        files = {
            p.relative_to(root).as_posix(): {"sha256": digest(p), "object": str(p)}
            for p in store.root.rglob("*")
            if p.is_file()
        }
        backend = SimpleNamespace(get=lambda name, revision: Path(name))
        with ExitStack() as stack:
            stack.enter_context(patch.object(parent_import, "historical_config", return_value=old))
            stack.enter_context(
                patch.object(parent_import, "verified_manifest", return_value={"files": files})
            )
            restored, evidence = parent_import.restore(test, backend, root / "imported", arm)
            assert evidence["post_rejection_verified"] and fingerprint(restored) == fingerprint(
                state
            )
        # A different/missing Step 100 is never silently replaced with Step 110.
        wrong = copy.deepcopy(state)
        wrong["step"] = 110
        wrong["data_cursor"] = 110 * c["training"]["tokens_per_step"]
        row = store.save(SimpleNamespace(step=110, state=lambda: wrong))
        (store.root / "latest.json").write_text(json.dumps(row))
        files = {
            p.relative_to(root).as_posix(): {"sha256": digest(p), "object": str(p)}
            for p in store.root.rglob("*")
            if p.is_file()
        }
        with ExitStack() as stack:
            stack.enter_context(patch.object(parent_import, "historical_config", return_value=old))
            stack.enter_context(
                patch.object(parent_import, "verified_manifest", return_value={"files": files})
            )
            try:
                parent_import.restore(test, backend, root / "missing100", arm)
            except RuntimeError as e:
                assert str(e) == "OFFICIAL_POST_REJECTION_STEP100_NOT_AVAILABLE_NO_RETRAIN"
            else:
                raise AssertionError("Step 110 used as Step 100")


def main():
    torch.set_num_threads(1)
    c, data = fixture()
    parent_restore_regression(c)
    for arm in c["arms"]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tr = trainer(c, arm)
            ident = {"test": arm}
            before = fingerprint(tr.state())
            saved = []

            def durable():
                package = storage.latest(root / "proposal", ident)
                assert package["phase"] == "FULL_CANDIDATE_SAVED_BEFORE_TRUST"
                assert len(package["losses"]) == 8 and len(package["sampled_states"]) == 12
                saved.append(package["candidate_fingerprint"])

            package = proposal.generate(tr, data, root / "proposal", ident, durable)
            assert saved and package["candidate_fingerprint"] != package["original_fingerprint"]
            with patch.object(
                proposal, "collect", side_effect=AssertionError("candidate reoptimized")
            ):
                restored = proposal.generate(tr, data, root / "proposal", ident, durable)
            assert restored["candidate_fingerprint"] == package["candidate_fingerprint"]
            assert (
                fingerprint(tr.state()) != before
            )  # cf_rng advanced once, other streams are preserved.
            assert fingerprint(package["official"]["model"]) == fingerprint(tr.model.state_dict())
            refs = list(admission_windows(tr.model, c, data))
            assert all(
                torch.equal(x, torch.tensor(data["admission"][i : i + 4])[None])
                for x, i in zip(refs, [0, 8])
            )
            scan = scale_scan.scan(tr, data, package, root / "scan", ident)
            assert len(scan["attempts"]) == 22 and scan["attempts"][0]["scale"] == 0
            zero = scan["attempts"][0]
            assert zero["kl"] == zero["action_changed_count"] == zero["path_changed_count"] == 0
            assert not zero["passed"]
            assert scan["data_role"] == "admission"
            replay = scale_scan.scan(tr, data, package, root / "scan", ident)
            assert replay["attempts"] == scan["attempts"]
            assert replay["chosen_scale"] == scan["chosen_scale"]
            # Real paired-update transaction check, independent of an incidental synthetic admission outcome.
            controlled = copy.deepcopy(package)
            controlled["candidate_connection"] = copy.deepcopy(package["original_connection"])
            controlled["candidate_connection"]["linear.bias"][0] += 0.001
            controlled["parameter_delta"] = {
                k: v - package["original_connection"][k]
                for k, v in controlled["candidate_connection"].items()
            }
            controlled_scan = copy.deepcopy(scan)
            controlled_scan.update(chosen_scale=1.0, rejection_reason=None)
            step0 = tr.step
            event, routes = adaptation.execute(
                tr, data, controlled, controlled_scan, root / "job", ident
            )
            assert tr.step == step0 + 10 and event["cell_adaptation_executed"]
            assert len(routes) == 10 and event["paired_route_streams_verified"]
            fingerprint0 = fingerprint(tr.state())
            event2, routes2 = adaptation.execute(
                tr, data, controlled, controlled_scan, root / "job", ident
            )
            assert fingerprint(tr.state()) == fingerprint0 and routes2 == routes
            assert event2["adapt_steps"] == 10
            # Exact mid-adaptation restart: no candidate optimization and no repeated paired steps.
            for interrupted_step in (1, 5, 10):
                retry = trainer(c, arm)
                original_save = adaptation.existing.save_event

                def stop_after_pair(store, state, sequence):
                    original_save(store, state, sequence)
                    if state.get("phase") == "ADAPTING" and state["adapt_done"] == interrupted_step:
                        raise RuntimeError("CONTROLLED_PAUSE")

                retry_root = root / ("interrupt-" + str(interrupted_step))
                with patch.object(adaptation.existing, "save_event", stop_after_pair):
                    try:
                        adaptation.execute(
                            retry, data, controlled, controlled_scan, retry_root, ident
                        )
                    except RuntimeError as error:
                        assert str(error) == "CONTROLLED_PAUSE"
                    else:
                        raise AssertionError("interruption missing")
                _, resumed_routes = adaptation.execute(
                    retry, data, controlled, controlled_scan, retry_root, ident
                )
                assert len(resumed_routes) == 10 and retry.step == 110
                a, b = copy.deepcopy(retry.state()), copy.deepcopy(tr.state())
                a["cost"].pop("elapsed_update_seconds", None)
                b["cost"].pop("elapsed_update_seconds", None)
                assert fingerprint(a) == fingerprint(b)
            # Explicit import keeps all fields except the documented optimizer identity rebind.
            old = copy.deepcopy(package["official"])
            old["optimizer_identity"] = "historical-identity"
            imported, evidence = parent_import.import_state(
                c, old, {"step": 100}, arm, "cpu", root / "import", ident
            )
            assert evidence["explicit_parent_import"] and imported.step == 100
            assert fingerprint(old["cell_optimizer"]) == fingerprint(imported.cell_opt.state_dict())
            assert fingerprint(old["route_rng"]) == fingerprint(imported.route_rng.get_state())
    metrics = {
        "kl": 0.0,
        "action_change": 0.0,
        "path_change": 0.0,
        "score_delta_max_abs": 0.1,
        "centered_score_delta_max_abs": 0.1,
        "parameter_delta_norm": 0.1,
        "action_changed_count": 0,
        "path_changed_count": 0,
    }
    assert not admission.decide(metrics, c, 1e-7)["passed"]
    metrics.update(
        action_change=0.02, path_change=0.02, action_changed_count=1, path_changed_count=1
    )
    assert admission.decide(metrics, c, 1e-7)["passed"]
    rows = [
        {"scale": 1.0, **admission.decide(metrics, c, 1e-7)},
        {"scale": 0.5, **admission.decide(metrics, c, 1e-7)},
    ]
    assert admission.select(rows)[0] == 1.0
    metrics["path_change"] = 0.3
    assert not admission.decide(metrics, c, 1e-7)["passed"]
    h = hypotheses(
        {
            "A-C": {
                "valid": True,
                "trust": {"chosen_scale": None},
                "adaptation": {"cell_adaptation_executed": False},
            }
        }
    )
    assert h["H1"] == "NOT_SUPPORTED" and h["H3"] == "NOT_TESTED"
    print(
        json.dumps(
            {
                "status": "R3FV2_CPU_PASS",
                "scope": "SYNTHETIC_ONLY",
                "gpu_experiment_executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
