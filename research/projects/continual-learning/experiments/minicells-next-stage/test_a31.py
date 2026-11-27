"""Regression checks for exact Saved Output recovery and normalization replay inputs."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

from a31_norm import Boundary, FIELDS, describe, materialize, metrics, restore, tensor_bytes
import m1_l2norm_isolation as experiment


class L2NormIsolationTest(unittest.TestCase):
    def test_mismatched_bundle_cannot_be_substituted(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "work").mkdir()
            fake = root / "prefix.cell"
            fake.write_bytes(b"other experiment")
            with self.assertRaises(experiment.BundleUnavailable):
                experiment.recover_bundle(root, fake, None)
            digest = sha256(fake.read_bytes()).hexdigest()
            self.assertEqual(experiment.select_bundle([fake], expected=digest), fake)

    def test_exact_capture_and_replay_preserves_fp16_bits_and_offset(self):
        base = torch.arange(30, dtype=torch.float16).reshape(5, 6)
        original = base[1:4, 1:5].T
        cpu, receipt = materialize(original, describe(original))
        replayed = restore(cpu, receipt, "cpu")
        self.assertEqual(replayed.dtype, torch.float16)
        self.assertEqual(replayed.stride(), original.stride())
        self.assertEqual(replayed.storage_offset(), original.storage_offset())
        self.assertEqual(tensor_bytes(replayed), tensor_bytes(original))
        changed = cpu.clone()
        changed[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            restore(changed, receipt, "cpu")

    def test_bitwise_metric_detects_signed_zero(self):
        result = metrics(torch.tensor([0.], dtype=torch.float16), torch.tensor([-0.], dtype=torch.float16))
        self.assertFalse(result["bitwise_equal"])
        self.assertEqual(result["max_absolute"], 0.0)
        self.assertEqual(result["first_index"], [0])

    def test_records_actual_first_recurrent_slice_without_copying_to_host_early(self):
        kernel = SimpleNamespace(fn=lambda: None, best_config=None)
        module = SimpleNamespace(l2norm_fwd_kernel=kernel, l2norm_fwd_kernel1=kernel)

        def norm(x, eps=1e-6, output_dtype=None):
            return x.clone()

        observer = Boundary(norm, module)
        observer.active = True
        x = torch.arange(48, dtype=torch.float16).reshape(2, 1, 3, 2, 4)
        with patch("a31_norm.materialize") as transfer:
            q, k = observer.norm(x), observer.norm(x + 1)
            observer.recur(lambda **args: args["q"], q=q[1], k=k[1])
            transfer.assert_not_called()
        tensors, _ = observer.export()
        self.assertEqual(set(tensors), set(FIELDS))
        self.assertEqual(observer.calls["K"]["first_recurrent_routed_slot"], 1)
        self.assertEqual(observer.calls["K"]["arguments"]["eps"], 1e-6)
        torch.testing.assert_close(tensors["K_RECURRENT"], k[1])

    def test_fixed_version_saved_output_is_downloaded_and_hash_checked(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "work").mkdir()
            payload = b"exact historical bundle"
            expected = sha256(payload).hexdigest()

            token = "fake-kaggle-secret"

            def download(cmd, **kwargs):
                self.assertNotIn(token, cmd)
                if "pip" in cmd:
                    self.assertNotIn("KAGGLE_API_TOKEN", kwargs["env"])
                    self.assertIn("kaggle==2.2.5", cmd)
                    return
                self.assertEqual(cmd[3:6], ["kernels", "output", "owner/notebook/7"])
                self.assertEqual(cmd[7], r"(?:.*/)?prefix[.]cell$")
                self.assertEqual(kwargs["env"]["KAGGLE_API_TOKEN"], token)
                self.assertTrue(kwargs["capture_output"])
                (Path(cmd[-1]) / "prefix.cell").write_bytes(payload)

            def select(paths):
                return choose(paths, expected=expected)

            choose = experiment.select_bundle
            with (patch.object(experiment, "select_bundle", side_effect=select),
                  patch.object(experiment, "BUNDLE_SHA256", expected),
                  patch.object(experiment, "kaggle_download_token", return_value=token),
                  patch.object(experiment.subprocess, "run", side_effect=download)):
                experiment.recover_bundle(root, None, "owner/notebook/7",
                                          mounted_root=root / "read-only")
            proof = json.loads((root / "bundle_provenance.json").read_text())
            self.assertEqual(proof["sha256"], expected)
            self.assertEqual(proof["recovery_method"], "versioned_cli_output")

    def test_frozen_cli_requires_version_number(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "work").mkdir()
            with patch.object(experiment, "download_historical_output") as cli:
                with self.assertRaisesRegex(experiment.BundleUnavailable, "fixed Kaggle"):
                    experiment.recover_bundle(root, None, "owner/notebook", mounted_root=root)
                cli.assert_not_called()

    def test_reads_kaggle_secret_without_exporting_in_parent(self):
        token = "fake-secret"
        secrets = SimpleNamespace(UserSecretsClient=lambda: SimpleNamespace(
            get_secret=lambda name: token if name == "KAGGLE_API_TOKEN" else None))
        with (patch.dict(sys.modules, {"kaggle_secrets": secrets}),
              patch.dict(experiment.os.environ, {}, clear=True)):
            self.assertEqual(experiment.kaggle_download_token(), token)
            self.assertNotIn("KAGGLE_API_TOKEN", experiment.os.environ)

    def test_cli_failure_reports_status_without_publishing_response(self):
        with TemporaryDirectory() as tmp:
            def commands(cmd, **kwargs):
                if "kernels" in cmd:
                    raise subprocess.CalledProcessError(1, cmd, stderr="403 Forbidden; signed-url-secret")

            with (patch.object(experiment, "kaggle_download_token", return_value="token"),
                  patch.object(experiment.subprocess, "run", side_effect=commands)):
                with self.assertRaises(experiment.BundleUnavailable) as ctx:
                    experiment.download_historical_output(Path(tmp), "owner/notebook/7")
            self.assertIn("HTTP 403", str(ctx.exception))
            self.assertNotIn("signed-url", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
