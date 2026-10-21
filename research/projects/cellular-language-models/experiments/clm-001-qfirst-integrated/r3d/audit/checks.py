"""Small CPU arithmetic/adapter checks only; no training, assets or historical test suite."""

from pathlib import Path
import runpy
import copy
import json
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
runpy.run_path(str(BASE / "run.py"), run_name="readonly_checks_bootstrap")
from minicells_r3d_audit.common import configuration, fingerprint  # noqa: E402
from minicells_r3d_audit.statistics import ranking, clustered, calibrate_bins, summaries  # noqa: E402
from minicells_r3d_audit.analysis import Cost, consistency, counterfactual, interpolation  # noqa: E402
from minicells_r3d_audit.sampling import plan, batches  # noqa: E402
from minicells_r3c_moe_bridge.common import synthetic_configuration  # noqa: E402
from minicells_r3d.connection import PathModel  # noqa: E402


def main():
    torch.set_num_threads(1)
    c, original, _ = configuration()
    c = copy.deepcopy(c)
    original = copy.deepcopy(original)
    original["engine"] = synthetic_configuration()
    original["engine"]["routing"]["cells"] = 6
    original["engine"]["balance"]["coefficient"] = 0.0
    c["sampling"].update(roles={"calibration": 2, "development": 2}, evaluation_batch_windows=2)
    c["statistics"]["bootstrap_replicates"] = 40
    check = ranking(np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0]), c)
    assert (
        check["accuracy"] == 1.0
        and abs(check["spearman"] - 1) < 1e-12
        and check["kendall_tau_b"] == 1.0
    )
    assert ranking(np.ones(3), np.arange(3.0), c)["accuracy"] == 0.5
    assert np.isnan(ranking(np.ones(3), np.ones(3), c)["accuracy"])
    assert clustered([1.0, 3.0, 5.0, 7.0], ["a", "a", "b", "b"], 1, 40)["stories"] == 2
    data = {role: np.arange(32, dtype="int64") % 14 + 2 for role in ("calibration", "development")}
    data["windows"] = {
        role: {"evaluation": {"starts": [0, 8, 16], "groups": [0, 1, 2]}}
        for role in ("calibration", "development")
    }
    model = PathModel(original, 7)
    # Define a fixed synthetic fixture before the read-only boundary, not a trained replacement.
    state = model.state_dict()
    state["connection.linear.bias"] = torch.linspace(-0.1, 0.1, 15)
    model.load_state_dict(state, strict=True)
    model.eval()
    selected = plan(c, original, data, 7, "calibration")
    before = fingerprint(model.state_dict())
    with torch.inference_mode():
        rows, x, y = next(iter(batches(data, original, "calibration", "cpu", 2, selected)))
        assert consistency(model, x, c)["status"] == "PASS"
        cost = Cost()
        values = counterfactual(model, c, original, data, "calibration", selected, "cpu", cost)
        assert (
            values["long"].shape == (2, 4, 15)
            and cost.candidate_actions == 120
            and cost.candidate_cell_rows == 600
        )
        assert np.array_equal(values["short"][:, 3], values["long"][:, 3])
        full = model(x)
        for t in range(4):
            pos = rows[0]["token_positions"][t]
            for action in (0, 14):

                def intervene(stage, h, z, ids, weights):
                    if stage < t:
                        return full.stages[stage].indices, full.stages[stage].weights
                    if stage == t:
                        ids, weights = ids.clone(), weights.clone()
                        pair, mix = model.indices_weights(z[pos : pos + 1], torch.tensor([action]))
                        ids[pos], weights[pos] = pair[0], mix[0]
                    return ids, weights

                forced = model(x, intervention=intervene)
                value = torch.nn.functional.cross_entropy(
                    forced.logits[0, pos : pos + 1], y[0, pos : pos + 1]
                )
                np.testing.assert_allclose(
                    float(value), values["long"][0, t, action], rtol=1e-5, atol=1e-6
                )
        thresholds = calibrate_bins(values, c)
        cf, rank = summaries(values, selected, c, 7, thresholds)
        assert len(cf["stages"]) == 4 and rank["overall"]["q_accuracy"]["stories"] == 2
        closed = interpolation(model, c, original, data, "cpu", Cost())
        assert set(closed) == {"0.0", "0.1", "0.25", "0.5", "0.75", "1.0"}
        assert fingerprint(model.state_dict()) == before
    # Exercise one read-only result/cache/figure path, not a training or historical regression suite.
    import tempfile
    from types import SimpleNamespace
    from minicells_r3d_audit import job as audit_job
    from minicells_r3d_audit.reporting import aggregate, plots, markdown
    from minicells_r3d_audit.inputs import Inputs, restore_object
    from minicells_r3d_audit.common import atomic, digest, run_identity

    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        # Reproduce simultaneous worker publication at the replace boundary.
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        from unittest.mock import patch
        import os

        barrier = Barrier(2)
        replace = os.replace

        def simultaneous(source, destination):
            barrier.wait(timeout=10)
            return replace(source, destination)

        with patch("os.replace", simultaneous), ThreadPoolExecutor(max_workers=2) as pool:
            list(
                pool.map(
                    lambda _: atomic(temp / "shared-reference.json", {"revision": "fixed"}),
                    range(2),
                )
            )
        assert json.loads((temp / "shared-reference.json").read_text()) == {"revision": "fixed"}
        assert not list(temp.glob("shared-reference.json.*.tmp"))
        # An absent cache destination must be created before the existing file verifier runs.
        from minicells_r3c_moe_bridge.checkpoint import Store
        from minicells_runtime.store import ContinuationError

        payload_state = {"model": {"weight": torch.tensor([1.0])}, "step": 0}
        ident = {"fixture": "readonly-restore"}
        source = temp / "source.pt"
        torch.save(
            {
                "identity": ident,
                "state": payload_state,
                "state_fingerprint": fingerprint(payload_state),
            },
            source,
        )
        row = {
            "schema": "minicells.r3c.checkpoint.v1",
            "identity": ident,
            "file": "objects/fixture.pt",
            "sha256": digest(source),
            "bytes": source.stat().st_size,
            "state_fingerprint": fingerprint(payload_state),
        }
        cache = temp / "fresh-checkpoint-cache"
        restore_object(source, cache, row)
        assert fingerprint(Store(cache, ident).verified(row)) == fingerprint(payload_state)
        restore_object(source, cache, row)  # verified cache reuse
        for unsafe in ("../escape.pt", "/absolute.pt"):
            try:
                restore_object(source, cache, dict(row, file=unsafe))
            except ContinuationError:
                pass
            else:
                raise AssertionError("unsafe destination accepted")
        (cache / "linked").symlink_to(temp, target_is_directory=True)
        try:
            restore_object(source, cache, dict(row, file="linked/escape.pt"))
        except ContinuationError:
            pass
        else:
            raise AssertionError("symlink destination accepted")
        h = copy.deepcopy(configuration()[2])
        h["nll"] = {"7": {"L": closed["1.0"]["mean_nll"]}}
        h["candidate_learning"] = {"7": {"L": {"scope": "SYNTHETIC_CPU_FIXTURE"}}}

        class Fixture:
            output = temp

            def descriptor(self, seed, arm):
                return {
                    "row": {
                        "sha256": "0" * 64,
                        "identity": {"scope": "SYNTHETIC_ONLY"},
                        "step": 200,
                        "bytes": 0,
                    }
                }

            def model(self, *args):
                return model, {"scope": "SYNTHETIC_ONLY_NOT_HISTORICAL_RESTORE"}

        fixture = Fixture()
        old_optimizer = torch.optim.Optimizer.__init__

        def forbidden(*args, **kwargs):
            raise AssertionError("optimizer created during read-only audit")

        torch.optim.Optimizer.__init__ = forbidden
        try:
            result = audit_job.run(fixture, c, original, h, data, 7, "L", "cpu", "synthetic")
            fixture.model = lambda *a: (_ for _ in ()).throw(
                AssertionError("completed read-only work repeated")
            )
            assert (
                audit_job.run(fixture, c, original, h, data, 7, "L", "cpu", "synthetic") == result
            )
        finally:
            torch.optim.Optimizer.__init__ = old_optimizer
        small = copy.deepcopy(c)
        small["seeds"] = [7]
        summary = aggregate(small, {"7": {"L": result}}, [], {"seeds": [7]}, {})
        plots(summary, temp / "figures")
        assert len(list((temp / "figures").glob("*.png"))) == 3
        assert "INCONCLUSIVE" in markdown(summary)
        # Source identity, not path prefix, selects the correct immutable manifest.
        expected = run_identity(c, original)
        wrong = {"identity": {"experiment_id": "R3C_PARENT"}, "files": {}}
        right = {"identity": expected, "files": {}}
        atomic(temp / "wrong.json", wrong)
        atomic(temp / "right.json", right)

        class ReadHub:
            repo = original["parent"]["hf_reference"]["hf_repo"]

            def info(self):
                return SimpleNamespace(sha="fixed")

            def optional(self, name, revision):
                assert name == "r3d/" + expected["run_id"] + "/latest.json"
                return {
                    "manifest_path": "right.json",
                    "manifest_sha256": digest(temp / "right.json"),
                }

            def get(self, name, revision):
                return temp / name

            def check_write(self):
                raise AssertionError("write gate used")

            def commit(self, *a):
                raise AssertionError("HF write attempted")

        h["hf_reference"] = {
            "hf_repo": ReadHub.repo,
            "hf_revision": "old",
            "manifest_path": "wrong.json",
            "manifest_sha256": digest(temp / "wrong.json"),
        }
        reader = Inputs(c, original, h, temp / "read-output", local=temp / "no-training-output")
        reader.backend = ReadHub()
        reader.resolve_manifest()
        assert reader.manifest["identity"] == expected and reader.rejected
    print(
        json.dumps(
            {
                "status": "READ_ONLY_CPU_CHECKS_PASS",
                "training_updates": 0,
                "optimizer_instances": 0,
                "checks": [
                    "ranking/ties",
                    "Story clustering",
                    "source-forward/advance parity",
                    "all-15 batched CF against full forced paths",
                    "last-Stage identity",
                    "fixed lambda rollouts",
                    "parameter immutability",
                    "single-job cache and three figures",
                    "read-only HF manifest identity fallback",
                    "fresh checkpoint cache restoration and existing Store verification",
                    "simultaneous shared-reference writers with synchronized replace",
                ],
                "scope": "SYNTHETIC_CPU_ONLY_NOT_CHECKPOINT_EVIDENCE",
            }
        )
    )


if __name__ == "__main__":
    main()
