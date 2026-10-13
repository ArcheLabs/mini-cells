"""Exercise the Notebook publication cells with real local Git and no GPU/network."""
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/research"))
import publish_core_validation_007 as publisher

NOTEBOOK = REPO / "research/projects/cellular-language-models/experiments/004-cell-graph/notebook.ipynb"
CELLS = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
PUBLICATION = "".join(CELLS[7]["source"])
RECOVERY = "".join(CELLS[8]["source"]).split("```python\n", 1)[1].split("```", 1)[0]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ManualPublicationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "checkout"
        self.root.mkdir()
        self.git("init")
        self.git("config", "user.name", "Publication test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("remote", "add", "origin", "https://github.com/ArcheLabs/mini-cells.git")
        definition = self.root / "definition.json"
        protocol = self.root / "protocol.json"
        config = self.root / "config.json"
        for path in (definition, protocol, config):
            path.write_text("{}\n", encoding="utf-8")
        registry = self.root / "research/governance/registry.yaml"
        registry.parent.mkdir()
        registry.write_text("{}\n", encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-m", "source")
        source = self.git("rev-parse", "HEAD").stdout.strip()
        branch = "results/cell-graph-" + source[:12]
        self.git("checkout", "-b", branch)
        archive = self.root / "research/projects/cellular-language-models/experiments/004-cell-graph/artifacts" / source / "seed-42/reference"
        archive.mkdir(parents=True)
        required = ("metrics.json", "logs.jsonl", "routing_stats.json", "decision.json", "config.json", "experiment.json")
        for name in required:
            (archive / name).write_text("{}\n", encoding="utf-8")
        manifest = {
            "source_commit": source, "seed": 42, "session": "reference",
            "definition_sha256": digest(definition), "config_sha256": digest(config),
            "protocol_sha256": digest(protocol), "scientific_decision": False,
            "files": {name: digest(archive / name) for name in required},
        }
        (archive / "evidence-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        matrix = archive.parent.parent / "matrix-analysis"
        matrix.mkdir()
        (matrix / "decision.json").write_text("{}\n", encoding="utf-8")
        registry.write_text('{"collected": true}\n', encoding="utf-8")

        def expired_setup(*args, **kwargs):
            raise TimeoutError("Notebook setup exceeded 300 seconds")

        self.scope = {
            "ROOT": self.root, "ARCHIVE_TARGET": archive, "ALREADY_ARCHIVED": False,
            "SOURCE_SHA": source, "RESULT_BRANCH": branch, "SEED": 42, "SESSION": "reference",
            "DEFINITION": definition, "PROTOCOL": protocol, "descriptor": {"config": "config.json"},
            "digest": digest, "json": json, "run_git": expired_setup,
            "SETUP_DEADLINE": 0,
        }

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def publish(self):
        exec(compile(PUBLICATION, str(NOTEBOOK), "exec"), self.scope)

    def test_expired_setup_and_stale_archive_flag_commit_once(self):
        before = self.git("rev-parse", "HEAD").stdout.strip()
        self.publish()
        after = self.git("rev-parse", "HEAD").stdout.strip()
        self.assertNotEqual(before, after)
        self.publish()
        self.assertEqual(after, self.git("rev-parse", "HEAD").stdout.strip())
        self.assertFalse(self.git("status", "--porcelain").stdout.strip())

    def test_compatibility_repair_uses_existing_git_and_detects_archive(self):
        exec(RECOVERY, self.scope)
        self.assertTrue(self.scope["ALREADY_ARCHIVED"])
        self.assertEqual(0, self.scope["run_git"]("status", "--porcelain").returncode)

    def test_corrupt_archive_is_not_committed(self):
        (self.scope["ARCHIVE_TARGET"] / "metrics.json").write_text("corrupted", encoding="utf-8")
        before = self.git("rev-parse", "HEAD").stdout
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            self.publish()
        self.assertEqual(before, self.git("rev-parse", "HEAD").stdout)

    def test_wrong_session_identity_is_not_committed(self):
        self.scope["SESSION"] = "graph"
        with self.assertRaisesRegex(RuntimeError, "identity mismatch: session"):
            self.publish()

    def test_unrelated_staged_file_is_not_committed(self):
        (self.root / "unrelated.txt").write_text("preserve", encoding="utf-8")
        self.git("add", "unrelated.txt")
        with self.assertRaisesRegex(RuntimeError, "Unrelated files are staged"):
            self.publish()

    def test_existing_publisher_pushes_commit_to_local_remote(self):
        self.publish()
        bare = Path(self.directory.name) / "remote.git"
        subprocess.run(["git", "init", "--bare", str(bare)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.git("remote", "set-url", "origin", str(bare))

        @contextmanager
        def local_auth(_secret_name):
            yield os.environ.copy()

        with patch.object(publisher, "ROOT", self.root), \
             patch.object(publisher, "EXPECTED_ORIGIN", str(bare)[:-4]), \
             patch.object(publisher, "validate_origin"), \
             patch.object(publisher, "_authenticated_git_env", local_auth):
            self.scope["push_branch"] = publisher.push_branch
            exec("".join(CELLS[9]["source"]), self.scope)
        remote = subprocess.check_output(
            ["git", "--git-dir", str(bare), "rev-parse", self.scope["RESULT_BRANCH"]], text=True)
        self.assertEqual(self.git("rev-parse", "HEAD").stdout.strip(), remote.strip())

    def test_failed_push_keeps_the_local_archive_commit(self):
        self.publish()
        before = self.git("rev-parse", "HEAD").stdout
        missing = Path(self.directory.name) / "missing.git"
        self.git("remote", "set-url", "origin", str(missing))

        @contextmanager
        def local_auth(_secret_name):
            yield os.environ.copy()

        with patch.object(publisher, "ROOT", self.root), \
             patch.object(publisher, "EXPECTED_ORIGIN", str(missing)[:-4]), \
             patch.object(publisher, "validate_origin"), \
             patch.object(publisher, "_authenticated_git_env", local_auth):
            with self.assertRaisesRegex(RuntimeError, "results remain committed locally"):
                publisher.push_branch(self.scope["RESULT_BRANCH"])
        self.assertEqual(before, self.git("rev-parse", "HEAD").stdout)
        self.assertFalse(self.git("status", "--porcelain").stdout.strip())

    def test_notebook_code_cells_parse(self):
        for cell in CELLS:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))


if __name__ == "__main__":
    unittest.main()
