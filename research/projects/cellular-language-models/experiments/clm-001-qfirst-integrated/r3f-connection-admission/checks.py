"""R3f-specific mechanism tests. Synthetic admission is never a real single-seed release."""

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
runpy.run_path(str(BASE / "run.py"), run_name="r3f_checks_bootstrap")
from minicells_r3f.common import configuration, identity, fingerprint  # noqa:E402
from minicells_r3f.stable_connection import build, baseline_action, hard_action  # noqa:E402
from minicells_r3f.evaluation import trust, reference, admission  # noqa:E402
from minicells_r3f.diagnostics import diagnostic  # noqa:E402
from minicells_r3f.policy_proposal import optimize  # noqa:E402
from minicells_r3f.adaptation import execute, paired  # noqa:E402
from minicells_r3f.runner import trial_gate  # noqa:E402
from minicells_r3f.training import Trainer  # noqa:E402
from minicells_r3f.jobs import train_job  # noqa:E402
from minicells_r3c_moe_bridge.common import synthetic_configuration  # noqa:E402
from minicells_r3c_moe_bridge.model import BridgeModel  # noqa:E402
from minicells_r3e.storage import Store, save  # noqa:E402
from minicells_r3e.stable_connection import Model as OriginalModel  # noqa:E402


class TrainingData(dict):
    def __getitem__(self, key):
        if key in ("development", "confirmation", "admission"):
            raise AssertionError("Forbidden role accessed during admission")
        return super().__getitem__(key)


def comparable(state):
    result = copy.deepcopy(state)
    result["cost"].pop("elapsed_update_seconds", None)
    return result


def publication_regression():
    from types import SimpleNamespace
    from minicells_r3f.publication import push_results

    calls = []
    original = Path("/historical")
    auth = SimpleNamespace(ROOT=original)

    def supported_git(root, *args, check=True, env=None):
        calls.append((root, args, check, env))

    def push(branch, secret):
        supported_git(auth.ROOT, "push", branch, check=False, env={"secret_label": secret})

    auth.push_branch = push
    with patch("minicells_r3c_moe_bridge.publication.helper", return_value=auth):
        push_results(Path("/retained"), "results/test")
        assert calls[0][0] == Path("/retained")
        assert auth.ROOT == original
        auth.push_branch = lambda *args: (_ for _ in ()).throw(RuntimeError("offline"))
        try:
            push_results(Path("/retained"), "results/test")
        except RuntimeError:
            pass
        else:
            raise AssertionError("push failure hidden")
        assert auth.ROOT == original


def evidence_regression():
    from minicells_r3f.evidence import execution_directory, seal
    from minicells_r3f.common import canonical
    from minicells_r3f.readonly import inspect

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        first = execution_directory(root / "runs")
        second = execution_directory(root / "runs")
        assert first != second
        (first / "reports").mkdir()
        (first / "reports/SINGLE_SEED_TRIAL.json").write_text('{"status":"STOPPED"}')
        ident = {
            "experiment_id": "r3f-connection-admission",
            "source_sha": "test",
            "config_sha256": "config",
            "run_id": "run",
        }
        result = seal(
            first,
            ident,
            status="BLOCKED",
            exit_code=1,
            publication={"status": "LOCAL_RETAINED"},
            started_at="test",
        )
        assert result["exit_code"] == 1 and "remote_commit_sha" not in result
        assert result["artifact_files"][0]["bytes"] > 0
        assert len(result["artifact_files"][0]["sha256"]) == 64
        c = configuration()
        ident["config_sha256"] = canonical(c)
        ident["run_id"] = canonical({"source": "test", "config": canonical(c)})
        (first / "identity.json").write_text(json.dumps(ident))
        (first / "reports/RUN_MANIFEST.json").write_text(json.dumps(ident))
        before = {str(f): f.read_bytes() for f in first.rglob("*") if f.is_file()}
        with patch("minicells_r3f.readonly.configuration", return_value=c):
            report = inspect(first, root / "audits")
        assert report["training_updates"] == report["policy_optimizer_updates"] == 0
        assert all(
            v["scan_status"] == "CANDIDATE_NOT_RETAINED_NO_REOPTIMIZATION" for v in report["arms"].values()
        )
        assert before == {str(f): f.read_bytes() for f in first.rglob("*") if f.is_file()}
        try:
            inspect(first, first / "nested")
        except RuntimeError:
            pass
        else:
            raise AssertionError("readonly output inside input accepted")


def main():
    publication_regression()
    evidence_regression()
    torch.set_num_threads(1)
    c = copy.deepcopy(configuration())
    c["synthetic"] = True
    c["engine"] = synthetic_configuration()
    c["engine"]["routing"]["cells"] = 6
    c["engine"]["balance"]["coefficient"] = 0.0
    c["training"].update(
        steps=6,
        tokens_per_step=8,
        proposal_steps=[1],
        states=4,
        policy_steps=2,
        adapt_steps=3,
        checkpoint_every=1,
        print_every=6,
    )
    c["proposal"]["reference_calibration_windows"] = 2
    parent = {"model": BridgeModel(c["engine"], "D", 7).state_dict()}
    # Deliberately tied Gate permits a controlled, nonempty score change with unchanged hard actions.
    for k in parent["model"]:
        if k.startswith("gate."):
            parent["model"][k].zero_()
    tokens = np.arange(65, dtype="int64") % 14 + 2
    data = TrainingData({r: tokens.copy() for r in ("train", "calibration")})
    data["windows"] = {
        r: {"starts": [0, 8, 16, 24], "evaluation": {"starts": [0, 8], "groups": [0, 1]}}
        for r in ("train", "calibration")
    }
    checks = []

    def controlled(tr, data):
        old = tr.model
        new = copy.deepcopy(old)
        with torch.no_grad():
            new.connection.linear.bias[0].add_(1e-4)
        metrics = trust(old, new, reference(old, c, data))
        status = admission(metrics, c, float(old.route_numeric_tolerance))
        assert status["trust_region_passed"] and status["effective_update"]
        return new, {
            "_readonly_candidate_connection": copy.deepcopy(new.connection.state_dict()),
            "route_switch_margin": float(old.route_switch_margin),
            "numeric_tolerance": float(old.route_numeric_tolerance),
            "attempts": [{"scale": 1.0, **metrics, **status}],
            "chosen_scale": 1.0,
            "trust_region_passed": True,
            "rejection_reason": None,
            "work": {},
            "policy_optimizer_updates": 0,
            "proposal_optimizer_final_state": None,
        }

    for arm in ("A-T", "A-C", "B-C"):
        model = build(c, parent, 7, arm, "cpu", data)
        refs = reference(model, c, data)
        assert sum(p.numel() for p in model.parameters()) == sum(
            p.numel() for p in OriginalModel(c, 7, arm).parameters()
        )
        same = trust(model, copy.deepcopy(model), refs)
        assert same["kl"] == same["action_change"] == same["path_change"] == 0
        tr = Trainer(model, c, arm)
        diag = diagnostic(model, c, data)
        assert len(diag["scale_scan"]) == 10
        assert diag["scale_scan"][0]["parameter_delta_norm"] == 0
        assert not diag["scale_scan"][0]["effective_update"]
        x = refs[0]
        out = model(x, keep_trace=True)
        assert out.operator_rows.sum() == 8 * x.numel()
        if not model.graph:
            eps = torch.finfo(torch.float32).eps
            z = torch.tensor([[1 - eps / 2, 0.5, 0.6, 0.7, 1.0, 1 + eps]])
            base = baseline_action(model, z)
            raw = z[:, model.connection.pairs].sum(-1)
            assert not torch.equal(raw.argmax(-1), base)  # actual floating-point pair-sum mismatch
            assert torch.equal(hard_action(raw, base, 8 * eps * raw.abs().max()), base)
            tiny = raw.clone()
            tiny[:, 0] += 1e-10
            assert torch.equal(hard_action(tiny, base, 8 * eps * raw.abs().max()), base)
        for rec in out.trace:
            raw, _, z = model.scores(rec["state"], rec["stage"], rec["previous"])
            base = baseline_action(model, z)
            assert torch.equal(base, hard_action(raw, base, model.route_switch_margin))
            assert torch.equal(
                base,
                hard_action(raw + 1e-9 * torch.arange(raw.shape[-1]), base, model.route_switch_margin),
            )
            for scale in (0.0, 0.125, 1.0):
                raw2 = raw.clone()
                raw2[:, 0] += scale * 1e-4
                assert (
                    (hard_action(raw2, base, model.route_switch_margin) >= 0)
                    & (hard_action(raw2, base, model.route_switch_margin) < raw.shape[-1])
                ).all()
        bad = dict(same, kl=float("nan"))
        try:
            admission(bad, c, float(model.route_numeric_tolerance))
        except FloatingPointError:
            pass
        else:
            raise AssertionError("Nonfinite KL accepted")
        for metric, limit in [
            ("kl", "kl_max"),
            ("action_change", "action_change_max"),
            ("path_change", "path_change_max"),
        ]:
            tested = dict(same)
            tested.update(centered_score_delta_max_abs=0.01, score_delta_max_abs=0.01)
            tested[metric] = c["proposal"][limit] + 0.1
            assert not admission(tested, c, float(model.route_numeric_tolerance))["trust_region_passed"]
        zero = admission(same, c, float(model.route_numeric_tolerance))
        assert zero["reject_reasons"] == ["NO_EFFECTIVE_UPDATE"]
        real_adam = torch.optim.AdamW

        def invalid_gradient_adam(params, **kwargs):
            params = list(params)
            for param in params:
                param.register_hook(lambda grad: torch.full_like(grad, float("nan")))
            return real_adam(params, **kwargs)

        with patch("minicells_r3f.policy_proposal.torch.optim.AdamW", invalid_gradient_adam):
            try:
                optimize(tr, data)
            except FloatingPointError:
                pass
            else:
                raise AssertionError("Nonfinite proposal gradient accepted")
        candidate, info = optimize(tr, data)
        assert len(info["attempts"]) == 11
        assert all(
            "interpolation_max_abs_error" in row and "reject_reasons" in row for row in info["attempts"]
        )
        # A zero-learning-rate optimizer must produce only empty updates, not a falsely successful candidate.
        empty = Trainer(build(c, parent, 7, arm, "cpu", data), copy.deepcopy(c), arm)
        empty.c["training"]["connection_lr"] = 0.0
        empty_candidate, empty_info = optimize(empty, data)
        assert empty_candidate is None and empty_info["rejection_reason"] == "NO_EFFECTIVE_UPDATE"
        with torch.no_grad():
            model.connection.linear.bias[0] = float("nan")
        try:
            model(x)
        except FloatingPointError:
            pass
        else:
            raise AssertionError("Nonfinite logits accepted")
        checks.append(
            arm
            + ": baseline/tiny delta/legal actions, identical trust, all 11 scales, empty update and finite guards"
        )
    for arm in ("A-C", "B-C"):
        tr = Trainer(build(c, parent, 7, arm, "cpu", data), c, arm)
        tr.update(data)
        start = copy.deepcopy(tr.state())
        # Equal parameters, data and route stream imply exactly equal Cell and optimizer updates.
        same = tr.clone()
        tr.update(data)
        same.update(data)
        assert fingerprint(comparable(tr.state())) == fingerprint(comparable(same.state()))
        tr.restore(start)
        import minicells_r3f.adaptation as adapter

        with tempfile.TemporaryDirectory() as clean:
            with patch.object(adapter, "optimize", controlled):
                execute(tr, data, clean, identity(c, "fixture", 7, arm))
            archive = Path(clean) / "readonly-proposals/transactions/proposal-0001"
            event_ident = {**identity(c, "fixture", 7, arm), "proposal_at_step": 1}
            package = Store(archive, event_ident).verified(json.loads((archive / "latest.json").read_text()))
            from minicells_r3f.readonly import scan

            with patch("minicells_r3f.data.load", return_value=data):
                scanned = scan(c, identity(c, "fixture", 7, arm), package, "unused", "cpu")
            assert scanned["scan_status"] == "SCANNED" and len(scanned["attempts"]) == 22
            assert scanned["attempts"][0]["action_change"] == 0

        expected = copy.deepcopy(tr.state())
        event = tr.proposals[-1]
        assert (
            event["trust_region_passed"] and event["cell_adaptation_executed"] and event["adapt_steps"] == 3
        )
        assert tr.step == 4 and tr.data_cursor == 32 and tr.cost["discarded_adaptation_updates"] == 3
        assert isinstance(event["final_quality_accepted"], bool)
        for stop in (0, 1, 2, 4):
            interrupted = Trainer(build(c, parent, 7, arm, "cpu", data), c, arm)
            interrupted.restore(start)
            real = adapter.save_event

            def interrupt(store, state, seq):
                real(store, state, seq)
                if seq == stop:
                    raise RuntimeError("CONTROLLED_INTERRUPT")

            with tempfile.TemporaryDirectory() as tmp:
                with ExitStack() as stack:
                    stack.enter_context(patch.object(adapter, "optimize", controlled))
                    stack.enter_context(patch.object(adapter, "save_event", interrupt))
                    try:
                        execute(interrupted, data, tmp, identity(c, "fixture", 7, arm))
                    except RuntimeError as e:
                        assert str(e) == "CONTROLLED_INTERRUPT"
                    else:
                        raise AssertionError("Interruption not reached")
                resumed = Trainer(build(c, parent, 7, arm, "cpu", data), c, arm)
                resumed.restore(start)
                if stop == 4:
                    with patch.object(
                        adapter,
                        "calibration",
                        side_effect=AssertionError("Calibration repeated after EVALUATED"),
                    ):
                        execute(resumed, data, tmp, identity(c, "fixture", 7, arm))
                else:
                    execute(resumed, data, tmp, identity(c, "fixture", 7, arm))
                assert fingerprint(comparable(resumed.state())) == fingerprint(comparable(expected))
                committed = fingerprint(resumed.state())
                again = Trainer(build(c, parent, 7, arm, "cpu", data), c, arm)
                again.restore(start)
                execute(again, data, tmp, identity(c, "fixture", 7, arm))
                assert fingerprint(again.state()) == committed
        # Admission and mechanism execution are separate from final quality acceptance.
        for accept, values in [(False, [1.0, 1.0]), (True, [1.0, 0.9])]:
            selected = Trainer(build(c, parent, 7, arm, "cpu", data), c, arm)
            selected.restore(start)
            expected_branch = selected.clone()
            if accept:
                proposed, _ = controlled(expected_branch, data)
                expected_branch.model.load_state_dict(proposed.state_dict(), strict=True)
            for _ in range(c["training"]["adapt_steps"]):
                expected_branch.update(data)
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                stack.enter_context(patch.object(adapter, "optimize", controlled))
                stack.enter_context(patch.object(adapter, "calibration", side_effect=values))
                execute(selected, data, tmp, identity(c, "quality-control", 7, arm))
            assert selected.proposals[-1]["final_quality_accepted"] == accept
            assert selected.proposals[-1]["cell_adaptation_executed"]
            assert fingerprint(selected.model.state_dict()) == fingerprint(expected_branch.model.state_dict())
            assert fingerprint(selected.cell_opt.state_dict()) == fingerprint(
                expected_branch.cell_opt.state_dict()
            )
        altered = tr.clone()
        altered.route_rng.manual_seed(2)
        try:
            paired(tr, altered)
        except RuntimeError as e:
            assert str(e) == "ADAPTATION_ROUTE_STREAMS_DIFFER"
        else:
            raise AssertionError("Different random streams accepted")
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp, identity(c, "fixture", 7, arm))
            row = save(tr, store)
            restored = Trainer(build(c, parent, 7, arm, "cpu", data), c, arm)
            restored.restore(store.verified(row))
            assert fingerprint(restored.state()) == fingerprint(tr.state())
        checks.append(
            arm
            + ": actual paired adaptation, branch/cost commit, proposal/first/mid/post-eval interruptions and exact restart"
        )
    with tempfile.TemporaryDirectory() as tmp:
        for arm in ("A-C", "B-C"):
            with patch("minicells_r3f.adaptation.optimize", controlled):
                result = train_job(c, data, tmp, "fixture", 7, arm, parent, "cpu", target=4)
            assert result["checkpoint"]["step"] == 4 and result["proposals"][-1]["cell_adaptation_executed"]
        assert not trial_gate(c, {arm: result})["passed"]  # synthetic admission cannot release real training.
    import minicells_r3f.runner as runtime

    tasks = []
    with ExitStack() as stack:
        stack.enter_context(patch.object(runtime, "load", return_value=data))
        stack.enter_context(patch.object(runtime, "parent_state", return_value=parent))
        stack.enter_context(
            patch.object(runtime, "train_job", side_effect=lambda *args: tasks.append((args[5], args[7])))
        )
        runtime.worker(c, "unused", "unused", "fixture", 0, 2, "trial")
        runtime.worker(c, "unused", "unused", "fixture", 1, 2, "trial")
    assert tasks == [("A-C", "cuda:0"), ("B-C", "cuda:1")]
    checks.append("Train/Calibration-only trial entry; synthetic evidence cannot release real matrix")
    print(
        json.dumps(
            {
                "status": "R3F_SYNTHETIC_MECHANISM_TESTS_PASS",
                "scope": "CPU_ONLY_NOT_REAL_SINGLE_SEED_EVIDENCE",
                "checks": checks,
                "formal_matrix_allowed": False,
                "real_diagnostics": "PENDING_PRIVATE_CHECKPOINTS",
                "real_trial": "NOT_RUN",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
