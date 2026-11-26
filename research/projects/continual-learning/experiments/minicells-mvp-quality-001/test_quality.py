from contextlib import nullcontext, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import run
from dataset import DOMAINS, build
from evaluation import metrics, nll
from reporting import calculate, decision, routing
from test_sdk import Fixture, Reader, fixture_patch
from minicells_native.sdk import MiniCells


class QualityTest(unittest.TestCase):
    def test_runner_starts_as_a_clean_subprocess_from_another_directory(self):
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        with TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, str(Path(run.__file__).resolve()), "--help"],
                                    cwd=temp, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model-revision", result.stdout)
        self.assertIn("--split", result.stdout)

    def test_dataset_is_disjoint_and_has_real_reference_chains(self):
        records, data = build()
        self.assertEqual(build(), (records, data))
        self.assertEqual([len(s) for s in data.values()], [9, 56, 128])
        seen = set()
        for split in data.values():
            for row in split:
                self.assertNotIn(row["prompt"], seen)
                seen.add(row["prompt"])
                self.assertFalse(set(row["relevant_cell_ids"]) & set(row["irrelevant_cell_ids"]))
                self.assertEqual(len(row["irrelevant_cell_ids"]), 2)
                if row["task_type"] == "cross_cell":
                    a, b = row["relevant_cell_ids"]
                    self.assertNotIn(row["gold_answer"], "\n".join(r["text"] for r in records[a]))
                    for domain, key in zip((a, b), row["source_record_ids"], strict=True):
                        self.assertIn(key, [r["record_id"] for r in records[domain]])

    def test_answer_only_uses_last_prompt_logit_then_previous_answer_token(self):
        calls = []
        model = SimpleNamespace(model=SimpleNamespace(lm_head=lambda h: h))
        def step(ids):
            calls.append(ids)
            return torch.tensor([[[10., 0.], [0., 10.]]]) if len(ids) == 2 else torch.tensor([[[10., 0.]]])
        targets = [1, 0]
        a = nll(model, step, [9, 8], targets)
        b = nll(model, step, [9, 8], targets)
        self.assertEqual(a, b)
        self.assertEqual(calls, [[9, 8], [1], [9, 8], [1]])
        self.assertLess(a["mean_target_nll"], .001)
        self.assertEqual(a["target_token_count"], 2)
        for context, target in (([1], []), ([], [1]), ([1] * 2048, [1])):
            with self.assertRaises(ValueError):
                nll(model, step, context, target)

    def test_routing_and_answer_evaluator(self):
        self.assertEqual(routing({"selected": ["x", "y"], "ranked": ["x", "y", "z"]}, ["y", "z"])["recall_at_2"], .5)
        self.assertEqual(routing({"selected": ["x", "y"], "ranked": ["x", "y"]}, [])["false_selection_count"], 2)
        self.assertTrue(metrics(" 123 ", "123")["exact_match"])
        self.assertTrue(metrics("The answer is 123.", "123")["parse_failure"])

    def test_real_sdk_routes_and_storage_in_cpu_fixture(self):
        records, data = build()
        row = data["smoke"][3]
        f = Fixture()
        f.tokenizer = SimpleNamespace(encode=lambda text, **kw: [1] * len(text), decode=lambda seq, **kw: "123",
                                      eos_token_id=0, all_special_ids=[0])
        args = SimpleNamespace(max_new_tokens=8, evaluation="both", timeout=180)
        with TemporaryDirectory() as tmp, fixture_patch(), patch(
                "minicells_native.query.FusionReader", Reader), patch(
                "evaluation.FusionReader", Reader), patch("run.deadline", lambda _: nullcontext()):
            root = Path(tmp)
            # Domain-specific fixture writes are distinct by construction.
            adjusted = {key: [{"record_id": str(i), "text": "a" * (i + 3)}]
                        for i, key in enumerate(DOMAINS)}
            with (
                MiniCells._from_backbone(f, root / "registry") as mc,
                MiniCells._from_backbone(f, root / "base") as base,
            ):
                self.assertFalse(base.cells.list())
                cells = run.prepare_cells(mc, root / "work", adjusted)
                ids = {domain: item["id"] for domain, item in cells.items()}
                original = {c.id: run.sha(mc.cells._store.path(c.id)) for c in mc.cells.list()}
                for arm, selected in run.requests(row, ids, list("ABCD")):
                    f.calls.clear()
                    runtime = base if arm == "A" else mc
                    result = run.trial(runtime, {**row, "gold_answer": "123"}, arm, selected, ids, args)
                    self.assertTrue(result["exact_match"])
                    self.assertEqual(result["selection"]["executed"], result["selection"]["selected"])
                    scored = [call for call in f.calls if call[0] == "score"]
                    if arm == "D":
                        self.assertEqual(set(result["selection"]["scores"]), set(ids.values()))
                        self.assertTrue(all(len(ids) > 1 for _, ids, _ in scored))
                    else:
                        self.assertFalse(scored)
                    if arm == "A":
                        self.assertEqual(f.calls[0][2], [0.])
                after = {c.id: run.sha(mc.cells._store.path(c.id)) for c in mc.cells.list()}
                self.assertEqual(original, after)

    def test_cpu_smoke_runner_persists_and_resumes_all_attempts(self):
        f = Fixture()
        f.tokenizer.all_special_ids = [0]
        f.identity = {"model_revision": run.REVISION_13B, "tokenizer_revision": run.REVISION_13B}
        f.model.lm_head = lambda h: torch.nn.functional.pad(h, (0, 8))
        def open_fixture(_model, storage, **options):
            return MiniCells._from_backbone(f, storage)
        with TemporaryDirectory() as temp, fixture_patch(), patch(
                "minicells_native.query.FusionReader", Reader), patch(
                "evaluation.FusionReader", Reader), patch("run.deadline", lambda _: nullcontext()), patch(
                "run.MiniCells.open", side_effect=open_fixture), patch("run.git", return_value="fixture"), patch(
                "run.version", return_value="fixture"), patch("run.environment", return_value={}), patch(
                "torch.cuda.device_count", return_value=1), patch("torch.cuda.get_device_name",
                return_value="Tesla T4"), patch("torch.cuda.get_device_capability", return_value=(7, 5)), patch(
                "torch.cuda.set_device"), patch("torch.cuda.get_device_properties"), patch(
                "torch.set_num_threads"), redirect_stdout(StringIO()):
            output = Path(temp) / "run"
            argv = ["quality", "--output", str(output), "--split", "smoke"]
            with patch("sys.argv", argv):
                run.main()
            before = (output / "sample_results.jsonl").read_bytes()
            self.assertEqual(len(before.splitlines()), 42)
            self.assertEqual(run.read(output / "decision.json")["engineering_status"], "PASS")
            cell_manifest = run.read(output / "cell_manifest.json")
            self.assertEqual(len({x["state_sha256"] for x in cell_manifest.values()}), 8)
            with patch("sys.argv", [*argv, "--resume"]):
                run.main()
            self.assertEqual((output / "sample_results.jsonl").read_bytes(), before)

    def test_smoke_artifacts_are_recomputable_from_paired_rows(self):
        _, data = build()
        sample = data["smoke"][0]
        records = {}
        for arm in "ABCD":
            records[sample["sample_id"], arm] = dict(
                sample_id=sample["sample_id"], split="smoke", task_type="single_cell", arm=arm,
                status="OK", fallback=False, execution_mismatch=False,
                relevant_cell_ids=["a"], exact_match=arm == "D",
                target_token_nll=[1.0], mean_target_nll=1.0, target_perplexity=2.718,
                selection=dict(selected=["a"], ranked=["a"]))
        with TemporaryDirectory() as temp:
            output = Path(temp)
            run.report_run(output, data, records, list(records.values()), list("ABCD"),
                           SimpleNamespace(split="smoke", evaluation="both"))
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "INCONCLUSIVE")
            contrast = summary["splits"]["smoke"]["single_cell"]["contrasts"]["D-A"]["exact_match"]
            self.assertEqual((contrast["n"], contrast["mean"], contrast["ci95"]), (1, 1.0, [1.0, 1.0]))
            self.assertEqual(len((output / "teacher_forced_metrics.jsonl").read_text().splitlines()), 4)
            self.assertTrue((output / "report.md").exists())

    def test_paired_statistics_fail_frozen_threshold_when_effect_is_zero(self):
        sample = dict(sample_id="q", task_type="single_cell", relevant_cell_ids=["a"])
        rows = [dict(sample_id="q", arm=arm, status="OK", fallback=False, exact_match=False,
                     relevant_cell_ids=["a"], selection={"selected": ["a"], "ranked": ["a"]})
                for arm in "ABCD"]
        summary, _ = calculate([sample], rows, list("ABCD"), "both")
        self.assertEqual(decision("held_out", [sample], rows, summary, list("ABCD"), "both")[0], "FAIL")
