"""Synthetic CPU gates, never invoked by the Kaggle scientific launcher."""

# ruff: noqa: E402
import runpy
import tempfile
import copy
from pathlib import Path
from unittest.mock import patch
import unittest
import torch

BASE = Path(__file__).resolve().parent
runpy.run_path(str(BASE / "run.py"), run_name="r3m_tests")
from minicells_r3m.protocol import task_for, schedule, main_gate, pilot_decision
from minicells_r3m.data import boundaries
from minicells_r3m.arithmetic import generate, isolation
from minicells_r3m.anchor import runtime
from minicells_r3m.pressure import Detector
from minicells_r3m.training import advance, restore, initial_aux, semantic
from minicells_r3m.recovery import reference, verify
from minicells_r3e.common import read, fingerprint
from minicells_r3h.routing import Model
from minicells_r3h.trainer import Trainer
from minicells_r3c_moe_bridge.common import synthetic_configuration
from minicells_r3c_moe_bridge.data import synthetic_batches


def fixture():
    c = read(BASE / "config/pilot.json")
    r = runtime(c)
    r["engine"] = synthetic_configuration()
    r["engine"]["routing"]["cells"] = 6
    r["training"].update(tokens_per_step=8)
    c["tokens_per_step"] = 8
    model = Model(r, 77, "B-H")
    model.route_switch_margin.fill_(1e-6)
    model.route_numeric_tolerance.fill_(1e-7)
    tr = Trainer(model, r, "B-H")
    tr.parent_identity = {"synthetic": True}
    tr.step = 300
    tr.data_cursor = 2400
    return c, r, tr


class Gate(unittest.TestCase):
    def test_curriculum(self):
        self.assertEqual(sum(task_for(i, "C") == "arithmetic" for i in range(600)), 240)
        self.assertEqual(sum(task_for(i, "N") == "arithmetic" for i in range(600)), 0)
        for a, b, k in [(0, 100, 10), (100, 200, 30), (200, 400, 100), (400, 600, 100)]:
            self.assertEqual(sum(task_for(i, "C") == "arithmetic" for i in range(a, b)), k)

    def test_schedule(self):
        c = read(BASE / "config/pilot.json")
        self.assertEqual(schedule(0.003, 0.05, 0, c), (0.003, 0.05))
        self.assertAlmostEqual(schedule(0.003, 0.05, 600, c)[0], 0.0006)
        self.assertAlmostEqual(schedule(0.003, 0.05, 600, c)[1], 0.02)

    def test_main_is_sealed(self):
        c = read(BASE / "config/pilot.json")
        with self.assertRaises(RuntimeError):
            main_gate(c, {"status": "REORGANIZATION_OPPORTUNITY_OBSERVED"}, "hash")
        c["main_release"].update(reviewed=True, frozen=True, pilot_manifest_sha256="hash")
        with self.assertRaises(RuntimeError):
            main_gate(c, {"status": "REORGANIZATION_OPPORTUNITY_OBSERVED"}, "hash")

    def test_boundaries(self):
        c = read(BASE / "config/pilot.json")
        with self.assertRaisesRegex(RuntimeError, "STORY_RANGE_CONFLICT"):
            boundaries(c)
        c.update(story_isolation_scope="R3H_LINEAGE", data_registration_reviewed=True)
        boundaries(c)
        c["story_ranges"]["trigger"] = [8190, 8256]
        with self.assertRaises(RuntimeError):
            boundaries(c)

    def test_arithmetic_isolation_and_masks(self):
        class Tokenizer:
            def encode(self, text):
                return type(
                    "Encoded",
                    (),
                    {
                        "ids": [ord(x) for x in text],
                        "offsets": [(i, i + 1) for i in range(len(text))],
                    },
                )()

        roles = {}
        for role in ("train", "trigger", "pilot_report", "main_report"):
            rows, meta = generate(role, Tokenizer(), 128)
            roles[role] = rows
            self.assertTrue(rows)
            self.assertEqual(meta["valid_count"], len(rows))
            self.assertEqual({r["carry"] for r in rows}, {True, False})
            for row in rows:
                self.assertEqual(row["answer_tokens"], [ord(x) for x in str(row["answer"])])
        isolation(roles)
        roles["trigger"].append(roles["train"][0])
        with self.assertRaises(RuntimeError):
            isolation(roles)

    def test_pressure_has_no_task_or_future(self):
        c = read(BASE / "config/pilot.json")["pressure"]
        d = Detector(c)
        self.assertEqual(d.observe(1.0, [1.0, 0.0], 1)["loss_surprise"], 0.0)
        snap = copy.deepcopy(d.state)
        other = Detector(c, snap)
        self.assertEqual(d.observe(2.0, [0.0, 1.0], 2), other.observe(2.0, [0.0, 1.0], 2))
        self.assertFalse(d.observe(1000.0, [-1.0, 0.0], 20)["triggered"])

    def test_short_restart_and_next_update(self):
        c, r, tr = fixture()
        b = synthetic_batches(r["engine"], tr.data_rng)
        aux = initial_aux(tr, c, {"seed": 77})
        saved = copy.deepcopy(tr.state())
        row = advance(tr, b, c, aux["schedule"])
        self.assertEqual(sum(row["cell_counts"]), 8 * 8)
        restarted = restore(r, copy.deepcopy(tr.state()), 77, "cpu")
        advance(tr, b, c, aux["schedule"])
        advance(restarted, b, c, aux["schedule"])
        self.assertEqual(
            fingerprint(semantic(tr.state())), fingerprint(semantic(restarted.state()))
        )
        start = restore(r, saved, 77, "cpu")
        ref = reference(start, b, c, {"seed": 77})
        proof = verify(ref, r, c, "cpu", reader_session="independent-test-container")
        self.assertEqual(proof["status"], "CROSS_SESSION_RESTORE_PASS")
        with self.assertRaises(RuntimeError):
            verify(ref, r, c, "cpu", reader_session=ref["writer_session"])
        ref["expected"]["data_cursor"] += 1
        with self.assertRaises(RuntimeError):
            verify(ref, r, c, "cpu", reader_session="independent-test-container")

    def test_shadow_is_within_budget_and_resumes_mid_transaction(self):
        from minicells_r3m import engine

        c, r, tr = fixture()
        c.update(continuation_steps=12, proposal_at=2, shadow_steps=3, barriers=[0, 2, 5, 12])
        ident = {"seed": 77, "run_id": "synthetic", "arm": "P1"}

        def supplied(tr, *args):
            return synthetic_batches(tr.c["engine"], tr.data_rng)

        def proposal(tr, *args):
            return copy.deepcopy(tr.model.connection.state_dict()), dict(
                effective=False, action_change=0, attempts=[], chosen_scale=None
            )

        with tempfile.TemporaryDirectory() as temp:
            calls = [0]

            def interrupt():
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError("synthetic interruption")

            root = Path(temp) / "private-checkpoints/shard/P1"
            with patch.object(engine, "batches", supplied), patch.object(
                engine, "propose", proposal
            ):
                with self.assertRaises(RuntimeError):
                    engine.run_arm(c, r, tr, ident, root, None, None, "N", "P1", "cpu", interrupt)
                resumed, aux = engine.run_arm(c, r, tr, ident, root, None, None, "N", "P1", "cpu")
                self.assertEqual(resumed.step, 312)
                self.assertEqual(aux["main_updates"], 12)
                self.assertEqual(aux["shadow_updates"], 3)
                self.assertEqual(aux["discarded_shadow_updates"], 3)
                self.assertEqual(len(aux["transactions"]), 1)
                self.assertFalse(aux["transactions"][0]["accepted"])
                self.assertEqual(resumed.data_cursor, 312 * 8)
                again, aux2 = engine.run_arm(c, r, tr, ident, root, None, None, "N", "P1", "cpu")
                self.assertEqual(
                    fingerprint(semantic(again.state())), fingerprint(semantic(resumed.state()))
                )
                self.assertEqual(aux2["main_updates"], 12)

    def test_real_reused_qfirst_proposal_and_trust(self):
        from minicells_r3m.intervention import propose

        c, r, tr = fixture()
        batch = synthetic_batches(r["engine"], tr.data_rng)
        seen = [[(x.cpu(), y.cpu()) for x, y in batch]]
        before = fingerprint(tr.model.cells.state_dict())
        with tempfile.TemporaryDirectory() as temp:
            connection, trace = propose(
                tr, seen, Path(temp), {"seed": 77, "run_id": "cpu"}, lambda: None
            )
        self.assertEqual(len(trace["attempts"]), 22)
        self.assertEqual(trace["data_role"], "ALREADY_OBSERVED_TRAIN")
        self.assertEqual(before, fingerprint(tr.model.cells.state_dict()))
        self.assertEqual(set(connection), set(tr.model.connection.state_dict()))
        for row in trace["attempts"]:
            if row["passed"]:
                self.assertLessEqual(row["kl"], 0.01)
                self.assertLessEqual(row["action_change"], 0.1)
                self.assertLessEqual(row["path_change"], 0.2)

    def test_evaluation_and_four_grid_are_read_only(self):
        from minicells_r3m.evaluation import repeated, attribution

        c, r, tr = fixture()
        research = {
            "story": {"pilot_report": [{"group": 1, "windows": [[2, 3, 4, 5, 6]]}]},
            "arithmetic": {
                "pilot_report": [
                    {
                        "ids": [2, 3, 4, 5],
                        "prompt_ids": [2, 3],
                        "answer_positions": [2, 3],
                        "answer_tokens": [4, 5],
                        "answer": 23,
                        "carry": False,
                        "group": "11+12",
                        "family": "synthetic",
                    }
                ]
            },
        }
        before = fingerprint(tr.model.state_dict())
        value = repeated(tr.model, research, "pilot_report")
        self.assertEqual(value["repeat_error"], 0)
        grid = attribution(tr.model, tr.model, research, "pilot_report")
        self.assertEqual(grid["answer_nll"]["V"], 0)
        self.assertEqual(before, fingerprint(tr.model.state_dict()))
        self.assertEqual(sum(value["teacher_forced_cell_usage"]["arithmetic"]), 3 * 8)

    def test_complete_report_matrix_and_negative_findings(self):
        from minicells_r3m.reports import write, FILES
        from minicells_r3m.runner import location
        from minicells_r3e.common import atomic, digest

        c = read(BASE / "config/pilot.json")
        quality = {
            "story_nll": 2.0,
            "answer_nll": 3.0,
            "exact_match": 0.0,
            "rows": [
                {"task": "story", "group": "s", "nll": 2.0},
                {"task": "arithmetic", "group": "a", "answer_nll": 3.0},
            ],
        }
        evals = {
            arm: {str(t): copy.deepcopy(quality) for t in c["barriers"]} for arm in c["pilot_arms"]
        }
        transactions = {
            a: [{"accepted": False, "trace": {"effective": False, "action_change": 0.0}}]
            for a in c["pilot_arms"]
        }
        grid = {
            str(t): {name: {"V": 0.0, "R": 0.0, "C": 0.0} for name in ("story_nll", "answer_nll")}
            for t in (400, 600)
        }
        with tempfile.TemporaryDirectory() as temp:
            for seed in c["pilot_seeds"]:
                for env in c["environments"]:
                    atomic(
                        location(temp, seed, env) / "EVALUATION.json",
                        dict(
                            seed=seed,
                            environment=env,
                            evaluations=evals,
                            attribution=grid,
                            transactions=transactions,
                            model_identity={},
                            curves={},
                            cost={},
                            specialization={},
                            answer_frequency_baseline=0.0,
                        ),
                    )
            result = write(c, Path(temp), {"run_id": "test"}, [], None)
            self.assertEqual(result["decision"]["status"], "TASK_LEARNABILITY_INSUFFICIENT")
            self.assertEqual(result["decision"]["Q4"], "NOT_TESTED")
            self.assertEqual(set(p.name for p in (Path(temp) / "reports").iterdir()), set(FILES))
            for row in result["artifacts"]:
                self.assertEqual(digest(Path(temp) / "reports" / row["path"]), row["sha256"])

    def test_main_statistics_unit_and_holm(self):
        from minicells_r3m.main_study import statistics, random_schedule

        rows = statistics({"one": [1.0, 1.0, 1.0, 1.0, 1.0], "two": [0.0, 0.0, 0.0, 0.0, 0.0]})
        self.assertEqual(rows[0]["unit"], "INDEPENDENT_SEED_NOT_STORY")
        self.assertGreaterEqual(rows[0]["holm_p_value"], rows[0]["p_value"])
        self.assertEqual(random_schedule(1), random_schedule(1))
        a, b = random_schedule(1)
        self.assertGreaterEqual(b - a, 100)

    def test_main_shared_phase_i_is_reused_not_retrained(self):
        from minicells_r3m import main_study

        c, r, initial = fixture()
        c["anchor_step"] = 3
        c["main_initialization"] = {"77": {"parent_seed": 151001301, "step": 0}}

        def synthetic(data, engine, rng, device):
            return synthetic_batches(engine, rng, device)

        with tempfile.TemporaryDirectory() as temp:
            with patch.object(
                main_study.anchor, "runtime", lambda c: copy.deepcopy(r)
            ), patch.object(
                main_study.anchor, "load", lambda *args, **kw: (initial, {"source": "synthetic"})
            ), patch("minicells_r3e.training.train_batches", synthetic):
                first = main_study.phase_i(c, None, temp, 77, "cpu", None, {"run_id": "main-cpu"})
                again = main_study.phase_i(c, None, temp, 77, "cpu", None, {"run_id": "main-cpu"})
            self.assertEqual(first.step, 3)
            self.assertEqual(again.step, 3)
            self.assertEqual(
                fingerprint(semantic(first.state())), fingerprint(semantic(again.state()))
            )

    def test_accepted_shadow_retains_selected_curve_and_cf_stream(self):
        from minicells_r3m import engine

        c, r, tr = fixture()
        c.update(continuation_steps=5, proposal_at=2, shadow_steps=2, barriers=[0, 2, 4, 5])

        def supplied(tr, *args):
            return synthetic_batches(tr.c["engine"], tr.data_rng)

        cf_after = []

        def proposal(tr, *args):
            torch.rand(3, generator=tr.cf_rng)
            cf_after.append(tr.cf_rng.get_state().clone())
            result = copy.deepcopy(tr.model.connection.state_dict())
            return result, dict(effective=True, action_change=0.06, attempts=[], chosen_scale=0.1)

        with tempfile.TemporaryDirectory() as temp:
            with patch.object(engine, "batches", supplied), patch.object(
                engine, "propose", proposal
            ), patch.object(engine, "train_quality", side_effect=[2.0, 1.0]):
                value, aux = engine.run_arm(
                    c,
                    r,
                    tr,
                    {"seed": 77, "run_id": "accepted"},
                    Path(temp) / "private-checkpoints/P1",
                    None,
                    None,
                    "N",
                    "P1",
                    "cpu",
                )
            self.assertTrue(aux["transactions"][0]["accepted"])
            self.assertTrue(torch.equal(value.cf_rng.get_state(), cf_after[0]))
            self.assertEqual(len(aux["curve"]), 5)
            self.assertEqual(len(aux["pressure_events"]), 5)
            self.assertEqual(aux["shadow_updates"], 2)

    def test_registration_amendment_remains_blocked_by_exact_historical_overlap(self):
        proof = read(BASE / "registration/DATA_BOUNDARY_PROOF.json")
        self.assertEqual(proof["status"], "REGISTRATION_BLOCKED_HISTORICAL_RANGE_CONFLICT")
        replay = proof["historical_stream_verification"]
        self.assertEqual(replay["expected_sha256"], replay["actual_sha256"])
        self.assertEqual(replay["actual_stories"], 476847)
        self.assertEqual(replay["last_raw_index"], 476900)
        self.assertEqual(sum(len(rows) for rows in proof["per_story_hashes"].values()), 1024)
        self.assertFalse(proof["formal_config_updated"])
        c = read(BASE / "config/pilot.json")
        self.assertEqual(c["story_ranges"]["trigger"], [10048, 10304])

    def test_scientific_missing_is_inconclusive(self):
        c = read(BASE / "config/pilot.json")
        self.assertEqual(
            pilot_decision([], c["pilot_thresholds"], c["pilot_seeds"])["status"], "INCONCLUSIVE"
        )


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
