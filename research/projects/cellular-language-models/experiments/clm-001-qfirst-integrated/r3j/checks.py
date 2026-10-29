"""Synthetic CPU contract tests. Not run by the Kaggle notebook."""

# ruff: noqa: E402
from pathlib import Path
import copy
import runpy
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
runpy.run_path(str(BASE / "run.py"), run_name="r3j_tests")
from minicells_r3j.case_selection import select
from minicells_r3j.parent_restore import Parent
from minicells_r3j.continuation import trainer, advance
from minicells_r3j.evaluation import evaluate
from minicells_r3j.storage import save, latest, Store, Incremental
from minicells_r3j import reports, runner
from minicells_r3i.trainer import Trainer
from minicells_r3h.routing import Model
from minicells_r3e.common import read, fingerprint, atomic
from minicells_r3c_moe_bridge.common import synthetic_configuration
from minicells_r3c_moe_bridge.data import synthetic_batches


def records():
    rows = []
    for seed in (1, 2, 3):
        for arm in ("A-CH", "A-CT", "B-CH", "B-CT"):
            for step in (100, 200, 300, 400):
                rows.append(
                    dict(
                        case_id="%s-%s-%s" % (seed, arm, step),
                        seed=seed,
                        arm=arm,
                        proposal_at_step=step,
                        historical_accepted=(seed == 2 and step in (100, 200)),
                        trust_region_passed=True,
                        historical_event={"attribution": {"values": {"OO": 4.0, "NN": 3.99}}},
                    )
                )
    return rows


def check():
    torch.set_num_threads(1)
    c = read(BASE / "config/frozen.json")
    runtime = runner.runtime(c)
    historical = runner.public_history(c)
    assert len(historical) == 48 and sum(bool(r["accepted"]) for r in historical) == 8
    selection = select(records())
    assert len(selection["cases"]) == 12
    assert all(
        r["seed"] == 1 and r["proposal_at_step"] == 100
        for r in selection["cases"]
        if not r["historical_accepted"]
    )
    runtime = copy.deepcopy(runtime)
    runtime["engine"] = synthetic_configuration()
    runtime["engine"]["routing"]["cells"] = 6
    runtime["training"].update(tokens_per_step=8, steps=500)
    data = {
        role: np.array([2, 3, 4, 5, 6, 7, 8, 9] * 20, dtype=np.int32)
        for role in ("calibration", "development", "admission")
    }
    data["windows"] = {role: {"evaluation": {"starts": [0, 4], "groups": [0, 1]}} for role in data}
    for arch in "AB":
        m = Model(runtime, 77, arch + "-H")
        m.route_switch_margin.fill_(1e-6)
        m.route_numeric_tolerance.fill_(1e-7)
        original = Trainer(m, runtime, arch + "-CH")
        original.parent_identity = {"synthetic": True}
        original.step = 110
        original.data_cursor = 880
        source = copy.deepcopy(original.state())
        old = trainer(runtime, source, 77, arch + "-CH", "cpu")
        new = trainer(runtime, source, 77, arch + "-CH", "cpu")
        with torch.no_grad():
            connection = copy.deepcopy(new.model.connection.state_dict())
            connection["linear.bias"][0] += 0.001
        new.authorize_connection(
            connection,
            transaction="historical-trial",
            accepted=True,
            expected_previous=new.connection_sha,
        )
        connection_hash = (old.connection_sha, new.connection_sha)

        def batches(data, engine, rng, device):
            return synthetic_batches(engine, rng, device)

        with tempfile.TemporaryDirectory() as folder:
            ident = {"synthetic": arch, "historical_accepted": False, "branch_pair": True}
            save(folder, ident, {"old": old.state(), "new": new.state(), "rows": []}, 0)

            def checkpoint(a, b, row):
                save(
                    folder,
                    ident,
                    {
                        "old": copy.deepcopy(a.state()),
                        "new": copy.deepcopy(b.state()),
                        "rows": [row],
                    },
                    a.step - 110,
                )

            with patch("minicells_r3e.training.train_batches", batches):
                with patch(
                    "minicells_r3i.adaptation.generate",
                    side_effect=AssertionError("NEW_PROPOSAL_FORBIDDEN"),
                ):
                    advance(old, new, data, 112, checkpoint)
                    state = latest(folder, ident)
                    restored_old = trainer(runtime, state["old"], 77, arch + "-CH", "cpu")
                    restored_new = trainer(runtime, state["new"], 77, arch + "-CH", "cpu")
                    advance(old, new, data, 114, checkpoint)
                    advance(restored_old, restored_new, data, 114, lambda a, b, r: None)
                    assert fingerprint(old.model.state_dict()) == fingerprint(
                        restored_old.model.state_dict()
                    )
                    assert fingerprint(new.cell_opt.state_dict()) == fingerprint(
                        restored_new.cell_opt.state_dict()
                    )
            assert (
                old.data_cursor == new.data_cursor == 912
                and (old.connection_sha, new.connection_sha) == connection_hash
            )
            assert old.epsilon() < 0.15
            result = evaluate(old, new, runtime, data, "development", 2)
            evaluation_result = copy.deepcopy(result)
            assert abs(result["V"] - result["R"] - result["C"]) < 1e-12
            assert result["measured_repeat_error"] == 0
            assert set(result["values"]) == {"OO", "ON", "NO", "NN"}
            assert reports.interval(result, 2026102302, 2000)["stories"] == 2
            payload = read(Store(folder, ident).receipt_file(4))
            payload["identity"] = {"swapped": True}
            try:
                Store(folder, ident).verified(payload)
            except Exception:
                pass
            else:
                raise AssertionError("swapped identity accepted")
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder)
        (out / "private-checkpoints").mkdir()
        atomic(out / "EXECUTION.json", {"execution_id": "synthetic-new-session"})
        result = reports.report(c, out, selection, [], None)
        assert result["status"] == "PARTIAL_ENGINEERING_FAILURE"
        assert (out / "reports/RUN_MANIFEST.json").exists()
    # Fixed-revision receipt restoration and no repeated backend gets for an existing object.
    from minicells_r3e.common import digest

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        remote = root / "remote"
        cache = root / "cache"
        expected = {
            "experiment_id": "R3I-SYNTHETIC",
            "source_sha": "source",
            "config_sha256": "config",
            "run_id": "run",
        }
        rel = Path("jobs/seed-1-arm-A-CH/transactions/proposal-0100/candidate")
        ident = dict(expected, seed=1, arm="A-CH", proposal_at_step=100)
        save(remote / rel, ident, {"historical": "actual-saved-object"}, 0)
        files = {}
        lookup = {}
        for file in (remote / rel).rglob("*"):
            if file.is_file():
                name = file.relative_to(remote).as_posix()
                key = "objects/" + digest(file)
                files[name] = {"object": key, "sha256": digest(file), "bytes": file.stat().st_size}
                lookup[key] = file
        manifest = remote / "manifest.json"
        atomic(manifest, {"identity": expected, "files": files})
        lookup["manifest.json"] = manifest
        ref = {
            "hf_repo": "synthetic/private",
            "hf_revision": "fixed-revision",
            "manifest_path": "manifest.json",
            "manifest_sha256": digest(manifest),
        }

        class Backend:
            repo = "synthetic/private"

            def __init__(self):
                self.calls = []

            def get(self, name, revision):
                assert revision == "fixed-revision"
                self.calls.append(name)
                return lookup[name]

            def info(self, revision=None):
                return SimpleNamespace(private=True, sha="fixed-revision")

        backend = Backend()
        parent = Parent({"parent": dict(expected, hf_reference=ref)}, backend, cache)
        value, receipt = parent.checkpoint(1, "A-CH", 100, 0, "candidate")
        assert value["historical"] == "actual-saved-object"
        count = len(backend.calls)
        parent.checkpoint(1, "A-CH", 100, 0, "candidate")
        assert len(backend.calls) == count
        try:
            parent.checkpoint(1, "A-CH", 100, 11)
        except RuntimeError as error:
            assert "PAIRED_BRANCH_NOT_RECOVERABLE" in str(error)
        else:
            raise AssertionError("missing paired state substituted")
        parent.unchanged()
    # Only new immutable objects are uploaded; a second unchanged publish does nothing.
    import shutil

    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder)
        private = output / "private"
        private.mkdir()

        class Hub:
            repo = "synthetic/private"

            def __init__(self):
                self.revision = "0"
                self.versions = {"0": {}}
                self.commits = []

            def info(self, revision=None):
                return SimpleNamespace(private=True, sha=self.revision)

            def check_write(self):
                pass

            def get(self, name, revision):
                return self.versions[revision][name]

            def commit(self, additions, previous, message):
                assert previous == self.revision
                revision = str(int(self.revision) + 1)
                mapping = dict(self.versions[self.revision])
                for name, source in additions.items():
                    target = output / "remote" / revision / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    mapping[name] = target
                self.revision = revision
                self.versions[revision] = mapping
                self.commits.append(set(additions))
                return revision

        hub = Hub()
        ident = {"synthetic": "incremental", "run_id": "run"}
        save(private / "cases/one", ident, {"old": "one", "new": "one"}, 0)
        publisher = Incremental(hub, private, output, ident, "synthetic-session")
        publisher.publish()
        assert len(hub.commits) == 1
        publisher.publish()
        assert len(hub.commits) == 1
        previous = set(publisher.known)
        save(private / "cases/one", ident, {"old": "two", "new": "two"}, 1)
        publisher.publish()
        assert len(hub.commits) == 2 and previous < set(publisher.known)
        assert len(publisher.verified) == 2
    # Actual import -> 90 paired updates -> horizon receipts -> final evaluation -> no replay.
    model = Model(runtime, 77, "A-H")
    model.route_switch_margin.fill_(1e-6)
    model.route_numeric_tolerance.fill_(1e-7)
    imported_old = Trainer(model, runtime, "A-CH")
    imported_old.parent_identity = {"synthetic": True}
    imported_old.step = 110
    imported_old.data_cursor = 880
    imported_new = trainer(runtime, copy.deepcopy(imported_old.state()), 77, "A-CH", "cpu")
    case = {
        "case_id": "synthetic-integration",
        "seed": 77,
        "arm": "A-CH",
        "proposal_at_step": 100,
        "historical_accepted": False,
    }
    case["historical_event"] = {
        "attribution": {
            "values": evaluate(imported_old, imported_new, runtime, data, "calibration", 2)[
                "values"
            ]
        }
    }
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder)
        atomic(
            out / "private-checkpoints/CASE_SELECTION_MANIFEST.json",
            {"sha256": "synthetic-frozen-case-set"},
        )
        with patch.object(runner, "runtime", return_value=runtime), patch.object(
            runner, "HubBackend", return_value=None
        ), patch.object(runner, "assets", return_value=data), patch.object(
            runner, "load_pair", return_value=(imported_old, imported_new, {})
        ):
            runner.preflight_case(c, out, case, "synthetic-assets", "cpu")
            with patch("minicells_r3e.training.train_batches", batches), patch(
                "minicells_r3i.adaptation.generate", side_effect=AssertionError("new proposal")
            ):
                runner.continuation_case(c, out, case, "synthetic-assets", "cpu")
            root = runner.case_root(out, case)
            ident = runner.pair_identity(c, case, out)
            for sequence in (0, 40, 90):
                receipt = read(Store(root, ident).receipt_file(sequence))
                saved = Store(root, ident).verified(receipt)
                assert (
                    saved["old"]["step"] == 110 + sequence
                    and saved["new"]["data_cursor"] == (110 + sequence) * 8
                )
            assert len(list((root / "receipts").glob("*.json"))) == 3
            original_cost = read(root / "TRAINED.json")
            runner.continuation_case(c, out, case, "synthetic-assets", "cpu")
            assert read(root / "TRAINED.json") == original_cost
            runner.evaluate_case(c, out, case, "synthetic-assets", "cpu")
            assert all((root / ("EVALUATION-%s.json" % h)).exists() for h in (10, 50, 100))
    # All 12 selected cases, three horizons and public report schema.
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder)
        atomic(out / "EXECUTION.json", {"execution_id": "synthetic-independent-execution"})
        for case in selection["cases"]:
            root = runner.case_root(out, case)
            atomic(root / "READY.json", {"status": "synthetic-restored"})
            atomic(
                root / "TRAINED.json",
                {"case_id": case["case_id"], "updates_per_branch": 90, "new_tokens": 92160},
            )
            for horizon in (10, 50, 100):
                value = copy.deepcopy(evaluation_result)
                value.update(
                    case_id=case["case_id"],
                    horizon=horizon,
                    old_alignment={},
                    new_alignment={},
                    training_path_rows=[],
                    checkpoint={"sha256": "synthetic"},
                )
                atomic(root / ("EVALUATION-%s.json" % horizon), value)
        summary = reports.report(c, out, selection, [], None)
        assert (
            summary["status"] == "COMPLETED_EXPLORATORY"
            and summary["valid_cases"] == 12
            and summary["accepted_complete"]
        )
    # Epoch replay at an already reached horizon must not call update at all.
    with patch.object(old, "update", side_effect=AssertionError("completed update replayed")):
        assert advance(old, new, data, old.step, lambda *args: None) == []
    assert reports.correlation([1, 2, 3], [3, 2, 1]) == -1.0
    print(
        {
            "status": "R3J_SYNTHETIC_CPU_PASS",
            "historical_K10_replayed": False,
            "new_proposals": 0,
            "gpu_training": False,
        }
    )


if __name__ == "__main__":
    check()
