import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import torch

import run_a1


class ComponentTest(unittest.TestCase):
    def test_replay_separates_state_reset_from_state_preserving_kernel(self):
        x = torch.ones(1, 12, 2)

        def step(part, state):
            y = x[:, part].cumsum(1)
            if state is not None:
                y = y + state
            return y, y[:, -1:]
        self.assertTrue(all(check["pass_"] for values in run_a1.compare_schedules(step).values()
                            for check in values.values()))

        def reset(part, state):
            return step(part, None)
        self.assertFalse(run_a1.compare_schedules(reset)["two_chunks"]["outputs"]["pass_"])

    def test_a1_script_starts_without_external_pythonpath(self):
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        with TemporaryDirectory() as temp:
            result = subprocess.run([sys.executable, str(Path(run_a1.__file__).resolve()), "--help"],
                                    cwd=temp, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output", result.stdout)
