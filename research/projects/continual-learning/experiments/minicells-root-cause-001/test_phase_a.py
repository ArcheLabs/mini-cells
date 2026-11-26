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
from fixtures import prompts
from test_sdk import Fixture, fixture_patch
from minicells_native.sdk import MiniCells


class PhaseATest(unittest.TestCase):
    def test_token_and_schedule_contract(self):
        def tokens(text, add_special_tokens):
            return ([1] if add_special_tokens else []) + [3] * len(text)
        tokenizer = SimpleNamespace(encode=tokens, bos_token_id=1, eos_token_id=2, all_special_ids=[0, 1, 2])
        for row in prompts():
            bare = run.encode(tokenizer, row["text"], False)
            with_bos = run.encode(tokenizer, row["text"], True)
            self.assertEqual(with_bos, [1, *bare])
            for schedule in ("whole", "prefill_decode", "tokenwise"):
                chunks = run.split_input(with_bos, schedule)
                self.assertEqual(sum(chunks, []), with_bos)
                self.assertEqual(sum(1 in chunk for chunk in chunks), 1)
        self.assertTrue(run.known_completion(" Paris, France", "Paris"))
        self.assertFalse(run.known_completion("Parisian", "Paris"))

    def test_comparison_detects_nonfinite_and_schedule_difference(self):
        ref = torch.tensor([[[1., 2.]]])
        self.assertTrue(run.differences(ref, ref + 0.03, run.PARITY)["pass_"])
        self.assertFalse(run.differences(ref, ref + 1., run.PARITY)["pass_"])
        self.assertFalse(run.differences(ref, ref * float("nan"), run.PARITY)["pass_"])

    def test_phase_a_runs_both_paths_and_all_conditions_with_distinct_caches(self):
        f = Fixture()
        f.tokenizer.bos_token_id = 1
        f.tokenizer.eos_token_id = 0
        f.tokenizer.all_special_ids = [0, 1]
        f.tokenizer.encode = lambda text, add_special_tokens: ([1] if add_special_tokens else []) + [2] * 8
        f.tokenizer.decode = lambda ids, **kw: "Paris"
        f.model.config = SimpleNamespace(bos_token_id=1, eos_token_id=0)

        def upstream(input_ids, past_key_values, **_):
            cache = past_key_values if past_key_values is not None else f.stream({})
            hidden = f.step(input_ids[0].tolist(), [cache], [1.0])
            return SimpleNamespace(last_hidden_state=hidden, past_key_values=cache)

        f.model.model = upstream
        with TemporaryDirectory() as temp, fixture_patch(), patch(
                "run.official_wrapper", return_value={"status": "FAIL"}):
            root = Path(temp)
            with MiniCells._from_backbone(f, root / "base") as mc:
                result = run.phase_a(mc, root)
                self.assertFalse(mc.cells.list())
            self.assertEqual(len((root / "phase_a_rows.jsonl").read_text().splitlines()), 36)
        self.assertEqual([result["gates"][k] for k in ("native_reference", "upstream_split", "greedy_reference")],
                         ["PASS"] * 3)
        self.assertEqual(result["gates"]["base_completion"], "FAIL")
        self.assertEqual(result["official_wrapper"]["status"], "FAIL")

    def test_script_starts_without_external_pythonpath(self):
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        with TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, str(Path(run.__file__).resolve()), "--help"],
                                    cwd=temp, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output", result.stdout)
