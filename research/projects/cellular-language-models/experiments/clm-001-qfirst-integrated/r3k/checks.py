"""Synthetic CPU checks. No real data or GPU experiment is executed."""
# ruff: noqa: E402

import copy
from pathlib import Path
import runpy
import tempfile

runpy.run_path(str(Path(__file__).parent / "run.py"), run_name="r3k_tests")
from minicells_r3k.prediction_freeze import scores, freeze, require_persisted
from minicells_r3k.data_boundaries import boundary_proof, REVISION, TOKENIZER


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid input was accepted")


def main():
    result = scores(
        {"OO": 2.0, "NN": 1.9}, {"OO": 2.0, "ON": 1.8}, {"OO": 1.9, "ON": 1.7, "NN": 1.6}
    )
    assert abs(result["S_K10"] - 0.3) < 1e-12
    assert abs(result["S_Cell"] - 0.1) < 1e-12
    assert abs(result["S_Slope"] - 0.1) < 1e-12
    rows = [
        dict(case_id=f"{s}-{a}-{t}", seed=s, arm=a, proposal_at_step=t, scores=result)
        for s in (151001301, 151001302, 151001303)
        for a in ("A-CH", "A-CT", "B-CH", "B-CT")
        for t in (100, 200, 300, 400)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "FROZEN_PREDICTIONS.json"
        frozen = freeze(path, rows, {"probe": "synthetic"}, {"source": "synthetic"})
        assert freeze(path, rows, {"probe": "synthetic"}, {"source": "synthetic"}) == frozen
        require_persisted(frozen, copy.deepcopy(frozen))
        damaged = copy.deepcopy(frozen)
        damaged["rows"][0]["scores"]["S_Cell"] = 3
        rejected(lambda: require_persisted(frozen, damaged))
        changed = copy.deepcopy(rows)
        changed[0]["scores"]["S_Cell"] = 4
        rejected(lambda: freeze(path, changed, {"probe": "synthetic"}, {"source": "synthetic"}))
        rejected(lambda: freeze(path, rows[:-1], {}, {}))
    stories = {i: str(i) for i in range(8192, 8960)}
    stories[8448] = stories[8192]
    proof = boundary_proof(REVISION, TOKENIZER, {"train": (0, 8192)}, [], stories)
    assert proof["roles"]["value_holdout"]["excluded"][0]["story_index"] == 8448
    rejected(
        lambda: boundary_proof(REVISION, TOKENIZER, {"confirmation": (8500, 9000)}, [], stories)
    )
    integration()
    print(
        {
            "status": "PRE_OUTCOME_CONTRACT_CPU_PASS",
            "engineering_cpu_pass": True,
            "gpu_started": False,
        }
    )


def integration():
    from unittest.mock import patch
    import numpy as np
    import torch
    from minicells_r3k import runner, reports
    from minicells_r3k.reuse import bind
    from minicells_r3j.continuation import trainer
    from minicells_r3j.storage import save, latest, Store
    from minicells_r3i.trainer import Trainer
    from minicells_r3h.routing import Model
    from minicells_r3e.common import read, atomic, fingerprint
    from minicells_r3c_moe_bridge.common import synthetic_configuration
    from minicells_r3c_moe_bridge.data import synthetic_batches
    from minicells_r3k.evaluation import evaluate

    torch.set_num_threads(1)
    c = read(Path(__file__).parent / "config/frozen.json")
    runtime = copy.deepcopy(runner.runtime(c))
    runtime["engine"] = synthetic_configuration()
    runtime["engine"]["routing"]["cells"] = 6
    runtime["training"].update(tokens_per_step=8, steps=500)
    data = {
        r: np.array([2, 3, 4, 5, 6, 7, 8, 9] * 20, dtype=np.int32)
        for r in ("calibration", "admission", "development")
    }
    data["windows"] = {r: {"evaluation": {"starts": [0, 4], "groups": [0, 1]}} for r in data}
    data["manifest"] = {"synthetic": True}
    model = Model(runtime, 77, "A-H")
    model.route_switch_margin.fill_(1e-6)
    model.route_numeric_tolerance.fill_(1e-7)
    original = Trainer(model, runtime, "A-CH")
    original.parent_identity = {"synthetic": True}
    original.step = 110
    original.data_cursor = 880
    state = copy.deepcopy(original.state())
    old = trainer(runtime, state, 77, "A-CH", "cpu")
    new = trainer(runtime, state, 77, "A-CH", "cpu")
    case = dict(
        case_id="synthetic",
        seed=77,
        arm="A-CH",
        proposal_at_step=100,
        historical_accepted=False,
        chosen_scale=0.001,
        reused_r3j=False,
        historical_event={"attribution": {"values": {"OO": 2.0, "NN": 1.9}}},
    )

    def batches(data, engine, rng, device):
        return synthetic_batches(engine, rng, device)

    # Actual inherited 90-update continuation on the R3k identity and exact restart, no proposals.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        atomic(out / "private-checkpoints/CASE_SELECTION_MANIFEST.json", {"sha256": "synthetic"})
        ident = runner.pair_identity(c, case, out)
        root = runner.case_root(out, case)
        save(
            root,
            ident,
            dict(old=old.state(), new=new.state(), rows=[], new_updates_per_branch=0),
            0,
        )
        operation = bind(
            runner.continuation_case,
            runtime=lambda c: runtime,
            HubBackend=lambda *x: None,
            assets=lambda *x: data,
        )
        with patch("minicells_r3e.training.train_batches", batches), patch(
            "minicells_r3i.adaptation.generate", side_effect=AssertionError("new proposal")
        ):
            operation(c, out, case, "assets", "cpu")
        done = latest(root, ident)
        assert done["old"]["step"] == 200 and done["new"]["data_cursor"] == 1600
        assert fingerprint(done["old"]["model"]) == fingerprint(done["new"]["model"])
        before = fingerprint(done)
        operation(c, out, case, "assets", "cpu")
        assert fingerprint(latest(root, ident)) == before
        assert len(list((root / "receipts").glob("*.json"))) == 3
        assert read(root / "TRAINED.json")["new_tokens"] == 1440
        # Holdout evaluation cannot run without a successful immutable remote barrier.
        rejected_or_blocked = False
        try:
            runner.evaluate_case(c, out, case, Path(tmp), "cpu")
        except Exception as error:
            assert type(error).__name__ == "ContinuationError"
            rejected_or_blocked = True
        assert rejected_or_blocked
        assert not (root / "EVALUATION-100.json").exists()
    # Predictor uses only Probe and does not invoke update or look at any long-horizon checkpoint.
    from minicells_r3k.parent_restore import R3jParent
    from types import SimpleNamespace

    class CandidateParent:
        def __init__(self, *args):
            pass

        def checkpoint(self, *args):
            assert args[-1] == "candidate"
            initial = copy.deepcopy(state)
            initial["step"] = 100
            initial["data_cursor"] = 800
            return {"official": initial}, {"sha256": "synthetic"}

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        atomic(out / "private-checkpoints/CASE_SELECTION_MANIFEST.json", {"sha256": "synthetic"})

        def probe_only(c, folder, role):
            assert role == "predictor_probe"
            return data

        with patch.object(runner.previous, "load_pair", return_value=(old, new, {})), patch.object(
            runner.previous, "Parent", CandidateParent
        ), patch.object(runner, "runtime", return_value=runtime), patch.object(
            runner, "HubBackend", return_value=None
        ), patch.object(runner.research, "load", side_effect=probe_only), patch.object(
            Trainer, "update", side_effect=AssertionError("predictor training")
        ), patch.object(
            runner, "R3jParent", side_effect=AssertionError("future checkpoint access")
        ):
            runner.predict_case(c, out, case, Path(tmp), "cpu")
        saved = read(runner.case_root(out, case) / "PREDICTION.json")
        assert set(saved["scores"]) == set(
            scores({"OO": 2, "NN": 1}, {"OO": 2, "ON": 1}, {"OO": 2, "ON": 1.5, "NN": 1})
        )
    # Fixed R3j horizon Store receipt identity, hash and absolute cursor are checked.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        branch = copy.deepcopy(state)
        branch["step"] = 200
        branch["data_cursor"] = 200 * 512
        remote = root / "remote"
        rel = Path("cases") / case["case_id"]
        expected = dict(
            c["r3j_identity"],
            case_id=case["case_id"],
            case_selection_manifest_sha256="02f1318b72e86463fdaee7f5e2d68915c9b665ec52f450e30e087640f4240745",
            historical_accepted=False,
            counterfactual_new_branch=True,
        )
        save(remote / rel, expected, dict(old=branch, new=branch), 90)
        from minicells_r3e.common import digest

        inventory = {
            x.relative_to(remote).as_posix(): dict(
                object=x.relative_to(remote).as_posix(), sha256=digest(x), bytes=x.stat().st_size
            )
            for x in remote.rglob("*")
            if x.is_file() and x.name != "latest.json"
        }
        backend = SimpleNamespace(
            info=lambda revision: SimpleNamespace(private=True),
            get=lambda name, revision: remote / name,
        )
        with patch(
            "minicells_r3k.parent_restore.verified_manifest",
            return_value={"identity": c["r3j_identity"], "files": inventory},
        ), patch.object(
            R3jParent,
            "selection",
            return_value={"sha256": expected["case_selection_manifest_sha256"]},
        ):
            parent = R3jParent(c, backend, root / "cache")
            restored, receipt = parent.checkpoint(case, 90)
            assert restored["old"]["step"] == 200 and receipt["step"] == 90
    # Reused cases cannot call the NEW36 optimizer path.
    reused = dict(case, reused_r3j=True)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        atomic(out / "private-checkpoints/CASE_SELECTION_MANIFEST.json", {"cases": [reused]})
        with patch.object(runner, "require_barrier"), patch.object(
            runner, "import_r3j_case"
        ) as imported, patch.object(
            runner, "continuation_case", side_effect=AssertionError("retrained historical case")
        ):
            runner.worker(c, out, Path(tmp), 0, 1, "train")
            assert imported.call_count == 1

    # Actual predictor model evaluation leaves training states and optimizer unchanged.
    before = fingerprint([old.state(), new.state()])
    value = evaluate(old, new, runtime, data, "development", 2)
    assert fingerprint([old.state(), new.state()]) == before
    assert abs(value["V"] - value["R"] - value["C"]) < 1e-12
    # Full48 report, including NEW36 and frozen score comparisons.
    cases = []
    predictions = []
    for seed in (151001301, 151001302, 151001303):
        for arm in ("A-CH", "A-CT", "B-CH", "B-CT"):
            for step in (100, 200, 300, 400):
                accepted = seed == 151001302 and step in (100, 200)
                reused = accepted or (seed == 151001301 and step == 100)
                row = dict(
                    case_id=f"{seed}-{arm}-{step}",
                    seed=seed,
                    arm=arm,
                    proposal_at_step=step,
                    historical_accepted=accepted,
                    reused_r3j=reused,
                )
                cases.append(row)
                predictions.append(
                    dict(
                        row,
                        scores={
                            k: float(step) / 10000
                            for k in ("S_Cal", "S_Direct", "S_K10", "S_Cell", "S_Slope")
                        },
                        numerical_error_bound=1e-6,
                        seconds=0.0,
                    )
                )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        selection = {"cases": cases, "sha256": "synthetic"}
        frozen = freeze(out / "private-checkpoints/FROZEN_PREDICTIONS.json", predictions, {}, {})
        for case in cases:
            root = runner.case_root(out, case)
            atomic(
                root / "TRAINED.json",
                dict(
                    case_id=case["case_id"],
                    updates_per_branch=0 if case["reused_r3j"] else 90,
                    new_tokens=0 if case["reused_r3j"] else 92160,
                ),
            )
            for h in (10, 50, 100):
                r = dict(
                    value,
                    checkpoint={"sha256": "synthetic"},
                    V=case["proposal_at_step"] / 100000,
                    R=0.0,
                    C=case["proposal_at_step"] / 100000,
                )
                atomic(root / f"EVALUATION-{h}.json", r)
        test = copy.deepcopy(c)
        test["bootstrap_replicates"] = 20
        result = reports.report(test, out, selection, [], None)
        assert result["new_cell_updates"] == 6480 and result["new_tokens"] == 3317760
        assert result["status"] == "COMPLETED_EXPLORATORY"
        stats = read(out / "reports/PREDICTIVE_SIGNAL_ANALYSIS.json")
        assert stats["groups"]["R3K_NEW36"]["S_K10"]["n"] == 36
        assert len(stats["primary_comparisons"]) == 3
        assert all(0 <= x["holm_adjusted_exploratory_p"] <= 1 for x in stats["primary_comparisons"])
        assert len(frozen["rows"]) == 48
    # Incremental transport, immutable prediction readback and real Store restore through a fresh session.
    import shutil
    from minicells_r3k.storage import Incremental
    from minicells_r3c_moe_bridge.checkpoint import restore_segment

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        private = out / "private"
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
                revision = str(int(previous) + 1)
                mapping = dict(self.versions[previous])
                for name, source in additions.items():
                    target = out / "remote" / revision / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    mapping[name] = target
                self.revision = revision
                self.versions[revision] = mapping
                self.commits.append(set(additions))
                return revision

        hub = Hub()
        ident = {"run_id": "synthetic"}
        save(private / "cases/one", ident, {"state": "historical"}, 0)
        atomic(private / "FROZEN_PREDICTIONS.json", {"synthetic": "immutable"})
        pub = Incremental(hub, private, out, ident, "session-A")
        ref = pub.publish()
        assert "r3k/synthetic/latest.json" in hub.commits[0]
        assert "r3j/synthetic/latest.json" not in hub.commits[0]
        pub.publish()
        assert len(hub.commits) == 1
        restored = out / "session-B"
        restore_segment(hub, ref, restored, ident, "session-B")
        assert latest(restored / "cases/one", ident) == {"state": "historical"}
        assert read(restored / "FROZEN_PREDICTIONS.json") == {"synthetic": "immutable"}

    # Selective recovery excludes superseded objects and keeps a newer local cursor.
    from minicells_r3k.storage import restore_active

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        private = out / "source"
        case = dict(case_id="one", historical_accepted=False)
        ident = {"run_id": "selective"}
        selection = {"sha256": "fixed", "cases": [case]}
        atomic(private / "CASE_SELECTION_MANIFEST.json", selection)
        pair = dict(
            ident,
            case_id="one",
            case_selection_manifest_sha256="fixed",
            historical_accepted=False,
            counterfactual_new_branch=True,
        )
        for step in (0, 1, 40):
            save(private / "cases/one", pair, {"cursor": step}, step)
        hub = Hub()
        pub = Incremental(hub, private, out, ident, "session-A")
        ref = pub.publish()
        local = out / "restored"
        save(local / "cases/one", pair, {"cursor": 41}, 41)
        restore_active(hub, ref, local, out, ident, "session-B")
        assert latest(local / "cases/one", pair) == {"cursor": 41}
        assert not Store(local / "cases/one", pair).receipt_file(1).exists()
        assert Store(local / "cases/one", pair).receipt_file(40).exists()
        assert not (local / "readback.json").exists()
        restore_active(hub, ref, local, out, ident, "session-C")
        assert latest(local / "cases/one", pair) == {"cursor": 41}

    # Coordinator oracle: freeze/readback must occur before training; all models before evaluation.
    import minicells_r3k.publication as publication

    oracle_source = """def workers(c,path,output,folder,phase,callback):
    cases=read(Path(output)/'private-checkpoints/CASE_SELECTION_MANIFEST.json')['cases']
    root=Path(output)/'private-checkpoints'
    if phase=='predict':
        for case in cases:
            atomic(root/'cases'/case['case_id']/'PREDICTION.json',dict(case,scores={k:.01 for k in ('S_Cal','S_Direct','S_K10','S_Cell','S_Slope')},numerical_error_bound=1e-6,seconds=0.))
    elif phase=='train':
        assert (root/'PREDICTION_PERSISTENCE.json').exists()
        for case in cases:
            atomic(root/'cases'/case['case_id']/'TRAINED.json',dict(case_id=case['case_id'],updates_per_branch=0 if case['reused_r3j'] else 90,new_tokens=0 if case['reused_r3j'] else 92160))
    elif phase=='evaluate':
        assert all((root/'cases'/case['case_id']/'TRAINED.json').exists() for case in cases)
        template=read(Path(output)/'SYNTHETIC_EVAL_TEMPLATE.json')
        for case in cases:
            for horizon in (10,50,100):
                atomic(root/'cases'/case['case_id']/('EVALUATION-%s.json'%horizon),dict(template,checkpoint={'sha256':'synthetic'}))
    callback(True)
"""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        assets = Path(tmp) / "assets"
        hub = Hub()
        hub.optional = lambda name, revision: None
        atomic(out / "SYNTHETIC_EVAL_TEMPLATE.json", value)

        def fixed_registry(c, output, backend):
            result = {"sha256": "synthetic-registry", "cases": cases}
            atomic(output / "private-checkpoints/CASE_SELECTION_MANIFEST.json", result)
            return result

        def prepared(c, runtime, backend, folder):
            atomic(folder / "research/manifest.json", {"synthetic": True})
            atomic(folder / "research/DATA_BOUNDARY_PROOF.json", {"synthetic": True})

        with patch.object(runner, "registry", side_effect=fixed_registry), patch.object(
            runner, "secrets"
        ), patch.object(runner, "HubBackend", return_value=hub), patch.object(
            runner.research, "prepare", side_effect=prepared
        ), patch.object(runner.torch.cuda, "device_count", return_value=2), patch.object(
            runner.inspect, "getsource", return_value=oracle_source
        ), patch.object(
            publication, "publish", return_value={"status": "SYNTHETIC_PUBLICATION_ONLY"}
        ):
            oracle_config = copy.deepcopy(c)
            oracle_config["bootstrap_replicates"] = 20
            result = runner.automatic(oracle_config, Path("synthetic-config"), out, assets)
        assert result["new_cell_updates"] == 6480
        assert (out / "private-checkpoints/PREDICTION_PERSISTENCE.json").exists()
        assert (
            read(out / "private-checkpoints/FROZEN_PREDICTIONS.json")["protocol"] == oracle_config
        )

    # CLI worker indices match the reused coordinator's interface, including worker 0.
    with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "secrets"), patch.object(
        runner, "worker"
    ) as work, patch(
        "sys.argv",
        [
            "run",
            "--output",
            tmp,
            "--assets",
            tmp,
            "--worker",
            "0",
            "--workers",
            "2",
            "--phase",
            "predict",
        ],
    ):
        runner.main()
        assert work.call_args[0][3:] == (0, 2, "predict")
    # Binding must never alter the historical function namespace.
    assert (
        runner.previous.continuation_case.__globals__["pair_identity"]
        is runner.previous.pair_identity
    )
    print({"integration": "R3K_CPU_PASS", "gpu_started": False})


if __name__ == "__main__":
    main()
