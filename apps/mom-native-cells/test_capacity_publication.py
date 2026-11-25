"""Exercise the real repository packager without a GPU or network access."""
import json
from pathlib import Path
import runpy
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from run_t4_1b_capacity import main, write_decision


class CapacityPublicationTest(unittest.TestCase):
    def test_real_publisher_accepts_blocked_audit_and_runner_failure(self):
        repo = Path(__file__).resolve().parents[2]
        publisher = runpy.run_path(str(repo / "scripts/research/publish_experiment_results.py"))
        with TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-c", "user.name=Capacity Fixture", "-c", "user.email=test@example.invalid",
                            "commit", "-q", "--allow-empty", "-m", "fixture"], cwd=root, check=True)
            audit = root / "results/audit"
            with patch("sys.argv", ["capacity", "--output", str(audit), "--stage", "audit"]), patch(
                    "run_t4_1b_capacity.environment", return_value={"hardware_status": "NON_TARGET"}):
                main()
            failure = root / "results/failure"
            failure.mkdir(parents=True)
            write_decision(failure, "BLOCKED", "Capacity runner did not finish")
            for entry in (audit, failure):
                with self.subTest(receipt=entry.name):
                    spec = publisher["ExperimentSpec"](
                        entry.relative_to(root).as_posix(), f"artifacts/{entry.name}",
                        tuple(sorted(file.name for file in entry.iterdir())), "kaggle/fixture",
                        "minicells.t4.1b.capacity.001.v1",
                    )
                    bundle = publisher["prepare_artifacts"](root, "fixture", spec, "123")
                    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
                    self.assertEqual(metadata["status"], "BLOCKED")
                    self.assertEqual(metadata["experiment_format"], spec.expected_format)
                    self.assertTrue((bundle / "RESULTS.md").is_file())
