#!/usr/bin/env python3
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "generate_service_manifest.py"


class ManifestGeneratorTests(unittest.TestCase):
    def invoke(self, output: Path, blob: Path, pvm: Path) -> dict:
        subprocess.run([
            "python3", str(GENERATOR), "--output", str(output), "--blob", str(blob),
            "--pvm", str(pvm), "--genesis-hash", "0xgenesis", "--source-ref", "source-ref",
            "--source-tree", "tree-ref", "--source-dirty", "false", "--minijam-ref", "mini-ref",
            "--jambda-ref", "jambda-ref", "--converter-ref", "converter-ref",
            "--jambda-adapter-ref", "jambda-adapter-ref",
            "--rust-version", "rust", "--target-hash", "target",
        ], check=True)
        return json.loads(output.read_text())

    def test_stale_fields_do_not_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, blob, pvm = root / "manifest.json", root / "blob", root / "service.pvm"
            output.write_text(json.dumps({"mini_cells_git_ref": "old", "jamscript_git_ref": "old", "garbage": "old"}))
            blob.write_bytes(b"blob")
            pvm.write_bytes(b"pvm")
            manifest = self.invoke(output, blob, pvm)
            self.assertNotIn("garbage", manifest)
            self.assertNotIn("mini_cells_git_ref", manifest)
            self.assertNotIn("jamscript_git_ref", manifest)

    def test_source_dependency_and_javascript_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "blob").write_bytes(b"blob")
            (root / "pvm").write_bytes(b"pvm")
            manifest = self.invoke(root / "manifest.json", root / "blob", root / "pvm")
            provenance = manifest["build_provenance"]
            self.assertEqual(provenance["mini_cells_source_ref"], "source-ref")
            self.assertEqual(provenance["mini_cells_source_tree"], "tree-ref")
            self.assertFalse(provenance["mini_cells_source_dirty"])
            self.assertEqual(provenance["minijam_build_ref"], "mini-ref")
            self.assertEqual(provenance["jambda_build_ref"], "jambda-ref")
            self.assertEqual(provenance["jambda_standalone_adapter_ref"], "jambda-adapter-ref")
            self.assertFalse(provenance["jamscript_used_for_build"])
            self.assertIsNone(provenance["jamscript_build_ref"])


if __name__ == "__main__":
    unittest.main()
