"""Actual D1 operators with synthetic CPU states; no historical model or real-data updates."""

import copy
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = (
    ROOT
    / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3m-f-d1"
)
bootstrap = runpy.run_path(str(BASE / "run.py"), run_name="d1_cpu_tests")
bootstrap["_bootstrap"]()
sys.path.insert(0, str(BASE / "implementation"))

from minicells_r3mf_d1 import execution as exe  # noqa: E402
from minicells_r3mf_d1.data import build_update, split_independent  # noqa: E402
from minicells_r3mf_d1.identity import canonical, sha256_bytes  # noqa: E402
from minicells_r3mf_d1.routing import make_model_class, source_random_inputs  # noqa: E402
from minicells_r3mf_d1.source_identity import code_sha, verify_source_tree  # noqa: E402
from minicells_r3mf_d1.training import torch_batches, update  # noqa: E402
from minicells_r3c_moe_bridge.common import synthetic_configuration  # noqa: E402
from minicells_r3e.common import fingerprint  # noqa: E402
from minicells_r3h.trainer import Trainer  # noqa: E402


class Tokenizer:
    def decode(self, tokens):
        return " ".join(map(str, tokens))


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        cls.runtime = json.loads((BASE.parent / "r3h/config/frozen.json").read_text())
        cls.runtime["engine"] = synthetic_configuration()
        cls.runtime["engine"]["routing"]["cells"] = 6
        cls.runtime["engine"]["model"].update(vocab_size=32, max_sequence_length=128)
        cls.runtime["engine"]["training"].update(
            effective_batch=8, microbatch=1, sequence_length=64
        )
        cls.runtime["training"]["tokens_per_step"] = 512
        cls.config = json.loads((BASE / "config/frozen.json").read_text())
        # This is a test-only copy. Production run_bcd validates the immutable 200-step config.
        cls.config.update(steps=2, continuation_steps=2, evaluation_steps=[0, 1, 2])
        cls.records = [
            dict(
                prompt_ids=[2, 3],
                answer_tokens=[i + 4],
                ids=[2, 3, i + 4],
                group=str(i),
                answer=str(i),
            )
            for i in range(8)
        ]
        model = make_model_class()(copy.deepcopy(cls.runtime), cls.config["seed"], "B-H")
        model.route_switch_margin.fill_(1e-6)
        trainer = Trainer(model, cls.runtime, "B-H")
        trainer.parent_identity = {"synthetic": True}
        trainer.step = 299
        trainer.data_cursor = 299 * 512
        trainer.curve = [{"step": 299, "historical_fixture": True}]
        rows = build_update(trainer.data_rng, cls.records)
        update(
            trainer,
            torch_batches(rows, "PACKED", "cpu"),
            objective="UNIFORM",
            context="PACKED",
            d1_config=cls.config,
        )
        cls.parent = copy.deepcopy(trainer.state())
        assert cls.parent["cell_optimizer"]["state"]  # Nonempty AdamW moments are retained.
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(cls.parent["data_rng"])
        cls.plan = {"rng_start": copy.deepcopy(cls.parent["data_rng"]), "relative_updates": []}
        for i in range(2):
            rows = build_update(rng, cls.records)
            cls.plan["relative_updates"].append(
                dict(
                    relative_step=i + 1,
                    rows=[
                        dict(
                            source_microbatch=r.microbatch,
                            sampled_record_indices=list(r.sampled_indices),
                            source_token_sha256=sha256_bytes(canonical(r.tokens).encode()),
                        )
                        for r in rows
                    ],
                    rng_state_after=copy.deepcopy(rng.bit_generator.state),
                )
            )
        cls.identity = dict(
            experiment_id="SYNTHETIC_CPU_D1_CONTROL_FLOW",
            source_sha="fixture",
            source_code_sha256="fixture-code",
            anchor_step=300,
            max_steps=200,
        )

    def prepare(self, root):
        root.mkdir(parents=True, exist_ok=True)
        exe._atomic_json(root / "A_REUSED_PREDICTIONS.json", {"reused": True})
        manifest = dict(
            run_id="synthetic",
            experiment_id=self.identity["experiment_id"],
            arms={a: dict(status="NOT_STARTED", updates=0) for a in "BCD"},
            artifacts={},
            logical_optimizer_updates=0,
            gpu_used=False,
        )
        return manifest

    def execute(self, root, manifest):
        proof = {"arm_budgets": {a: {"synthetic": True} for a in "BCD"}}
        with patch.object(
            exe,
            "_load_parent_state",
            side_effect=lambda *a, **k: (copy.deepcopy(self.parent), {"synthetic": True}),
        ):
            return exe._execute_run(
                copy.deepcopy(self.config),
                self.plan,
                proof,
                self.records,
                Tokenizer(),
                copy.deepcopy(self.runtime),
                root,
                manifest,
                self.identity,
                "cpu",
            )

    def test_actual_model_routes_use_source_offsets_and_qa_state_is_isolated(self):
        trainer = exe._restore_d1_trainer(
            copy.deepcopy(self.runtime), self.parent, self.config["seed"], "cpu"
        )
        m = trainer.model
        rows = build_update(np.random.default_rng(9), self.records)
        row = rows[2]
        m.active_step = 300
        m.microbatch_index = 2
        with torch.no_grad():
            packed = m(torch.tensor([row.inputs]), epsilon=1.0)
            for fragment in split_independent(row):
                x = torch.tensor([fragment.input_tokens])
                ctx = dict(
                    absolute_step=300, source_microbatch=2, source_start=fragment.source_start
                )
                with patch(
                    "minicells_r3mf_d1.routing.source_random_inputs", wraps=source_random_inputs
                ) as draws:
                    actual = m.forward_fragment(x, source_context=ctx, epsilon=1.0, keep_trace=True)
                self.assertEqual(draws.call_count, 8)
                self.assertTrue(
                    all(
                        call.args[2] == 2 and call.args[4] == fragment.source_start
                        for call in draws.call_args_list
                    )
                )
                expected = packed.actions[
                    fragment.source_start : fragment.source_start + len(fragment.input_tokens)
                ]
                self.assertTrue(torch.equal(actual.actions, expected))
                self.assertTrue(
                    torch.equal(
                        actual.trace[0]["previous"], torch.full_like(actual.trace[0]["previous"], 6)
                    )
                )
            fragment = split_independent(row)[1]
            ctx = dict(absolute_step=300, source_microbatch=2, source_start=fragment.source_start)
            observed_positions = []
            handle = m.position.register_forward_pre_hook(
                lambda module, args: observed_positions.append(args[0].clone())
            )
            first = m.forward_fragment(
                torch.tensor([fragment.input_tokens]), source_context=ctx, epsilon=0.2
            )
            m.forward_fragment(
                torch.tensor([[20, 21, 22]]), source_context=dict(ctx, source_start=0), epsilon=0.2
            )
            second = m.forward_fragment(
                torch.tensor([fragment.input_tokens]), source_context=ctx, epsilon=0.2
            )
            handle.remove()
            self.assertTrue(torch.equal(first.logits, second.logits))
            self.assertTrue(torch.equal(first.actions, second.actions))
            self.assertTrue(
                torch.equal(observed_positions[0], torch.arange(len(fragment.input_tokens)))
            )
            self.assertIsNone(m.d1_source_context)
            self.assertIsNone(m.training_context)

    def test_actual_checkpoint_roundtrip_and_evaluation_restore_full_state(self):
        trainer = exe._restore_d1_trainer(
            copy.deepcopy(self.runtime), self.parent, self.config["seed"], "cpu"
        )
        before = fingerprint(trainer.state())
        self.assertEqual(fingerprint(exe._cpu_tree(trainer.state())), before)
        evaluation = exe._evaluate(trainer, self.records, Tokenizer(), "cpu", 0)
        self.assertEqual(len(evaluation["predictions"]), 8)
        self.assertEqual(fingerprint(trainer.state()), before)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = dict(self.identity, arm="B")
            result = dict(
                identity=identity,
                arm="B",
                curve=[],
                evaluations=[],
                schedule_state={"anchor_lr": 0.01, "anchor_epsilon": 0.1},
            )
            path, _ = exe._checkpoint(
                trainer, identity, 0, self.config, root / "arm-B", {"artifacts": {}}, root, result
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            exe._validate_checkpoint(payload, identity, self.plan, self.records, 300)
            restored = exe._restore_d1_trainer(
                copy.deepcopy(self.runtime), payload["trainer_state"], self.config["seed"], "cpu"
            )
            self.assertEqual(fingerprint(restored.state()), before)
            changed = dict(identity, source_code_sha256="changed-code")
            with self.assertRaisesRegex(ValueError, "IDENTITY"):
                exe._validate_checkpoint(payload, changed, self.plan, self.records, 300)

    def test_checkpoint_before_result_failure_recovers_stale_and_missing_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            self.execute(baseline, self.prepare(baseline))
            original_evaluate = exe._evaluate
            for mode in ("stale", "missing", "evaluated", "after_result"):
                with self.subTest(interruption=mode):
                    target = root / mode
                    manifest = self.prepare(target)

                    def fail_after_checkpoint(trainer, records, tokenizer, device, step):
                        if step == 1 and mode in ("stale", "missing"):
                            raise RuntimeError("INJECTED_AFTER_CHECKPOINT_BEFORE_RESULT")
                        return original_evaluate(trainer, records, tokenizer, device, step)

                    original_save = exe._save_arm_result

                    def save_result(arm_dir, report, manifest, run_root):
                        due = (
                            mode in ("evaluated", "after_result")
                            and arm_dir.name == "arm-B"
                            and len(report["curve"]) == 1
                        )
                        if due and mode == "evaluated":
                            raise RuntimeError("INJECTED_AFTER_EVALUATION_BEFORE_RESULT")
                        original_save(arm_dir, report, manifest, run_root)
                        if due:
                            raise RuntimeError("INJECTED_AFTER_RESULT_BEFORE_MANIFEST")

                    with patch.object(exe, "_evaluate", side_effect=fail_after_checkpoint), patch.object(exe, "_save_arm_result", side_effect=save_result):
                        with self.assertRaisesRegex(RuntimeError, "INJECTED"):
                            self.execute(target, manifest)
                    self.assertTrue((target / "arm-B/checkpoint-step-000001.pt").exists())
                    if mode == "missing":
                        (target / "arm-B/ARM_RESULT.json").unlink()
                    # A fresh process sees the last manifest committed to disk, not mutated memory.
                    manifest = exe._read(target / "RUN_MANIFEST.json")
                    result = self.execute(target, manifest)
                    self.assertEqual(result["manifest"]["logical_optimizer_updates"], 6)
                    for arm in "BCD":
                        report = exe._read(target / ("arm-" + arm) / "ARM_RESULT.json")
                        self.assertEqual([x["step"] for x in report["curve"]], [301, 302])
                        self.assertEqual(
                            [x["relative_step"] for x in report["evaluations"]], [0, 1, 2]
                        )
                        actual = torch.load(
                            target / ("arm-" + arm) / "checkpoint-step-000002.pt",
                            map_location="cpu",
                            weights_only=False,
                        )["trainer_state"]
                        expected = torch.load(
                            baseline / ("arm-" + arm) / "checkpoint-step-000002.pt",
                            map_location="cpu",
                            weights_only=False,
                        )["trainer_state"]
                        for field in (
                            "model",
                            "cell_optimizer",
                            "data_rng",
                            "route_rng",
                            "cf_rng",
                            "data_cursor",
                            "step",
                            "global_cpu_rng",
                            "numpy_rng",
                            "python_rng",
                        ):
                            self.assertEqual(
                                fingerprint(actual[field]),
                                fingerprint(expected[field]),
                                (arm, field),
                            )
                        for a, b in zip(
                            report["curve"],
                            exe._read(baseline / ("arm-" + arm) / "ARM_RESULT.json")["curve"],
                        ):
                            self.assertEqual(a["loss"], b["loss"])
                    # Re-running a completed diagnostic performs zero further updates.
                    with patch.object(
                        exe, "advance", side_effect=AssertionError("REPEATED_UPDATE")
                    ):
                        self.execute(target, manifest)

    def test_parent_loader_adapter_binds_original_source_slot_budget(self):
        from types import SimpleNamespace

        config = copy.deepcopy(self.config)
        config["sampling_start"]["data_rng"] = copy.deepcopy(self.parent["data_rng"])
        config["anchor_checkpoint"]["state_fingerprint"] = fingerprint(self.parent)
        receipt = {
            "receipt": {
                "sha256": config["anchor_checkpoint"]["checkpoint_sha256"],
                "state_fingerprint": fingerprint(self.parent),
            }
        }
        backend = SimpleNamespace(info=lambda revision: SimpleNamespace(sha=revision))
        anchor = SimpleNamespace(state=lambda: copy.deepcopy(self.parent))

        def load_original(received, *args, **kwargs):
            self.assertEqual(received["tokens_per_step"], 512)
            self.assertEqual(received["baseline"], config["baseline"])
            self.assertEqual(kwargs["step"], 300)
            return anchor, receipt

        # Simulate only admission/API boundaries; no CUDA or remote I/O is performed.
        with tempfile.TemporaryDirectory() as tmp, patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2), patch.object(exe, "_runtime_secret"), patch("minicells_artifacts.hf_storage.HubBackend", return_value=backend), patch("minicells_r3m.anchor.load", side_effect=load_original):
            state, actual_receipt = exe._load_parent_state(config, tmp, "cuda:0")
        self.assertEqual(fingerprint(state), fingerprint(self.parent))
        self.assertEqual(actual_receipt, receipt)

    def test_rejected_gpu_request_is_not_reported_as_actual_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "runs/fixture/RUN_MANIFEST.json"
            exe._atomic_json(path, dict(source_sha="fixture", run_status="RUNNING",
                                      gpu_requested=True, gpu_used=False, arms={}))
            with patch("torch.cuda.is_available", return_value=False), patch.object(exe, "_runtime_secret", side_effect=AssertionError("SECRET_MUST_NOT_BE_READ")):
                with self.assertRaisesRegex(RuntimeError, "TWO_VISIBLE_CUDA"):
                    exe._load_parent_state(self.config, root, "cuda:0")
            with patch("torch.cuda.is_initialized", return_value=False), patch("torch.cuda.max_memory_allocated", side_effect=AssertionError("MUST_NOT_INITIALIZE_CUDA")):
                bootstrap["_record_run_failure"](root, "fixture")
            retained = exe._read(path)
            self.assertTrue(retained["gpu_requested"])
            self.assertFalse(retained["gpu_used"])
            self.assertEqual(retained["run_status"], "FAILED_RESUMABLE")

    def test_gpu_allocator_evidence_is_observed_and_unknown_is_explicit(self):
        from types import SimpleNamespace

        # These are simulated allocator API responses, not CUDA execution.
        module = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True,
            device_count=lambda: 2, max_memory_allocated=lambda device: 128 if device == 0 else 0))
        manifest = dict(gpu_requested=True, gpu_used=False)
        self.assertTrue(exe.observe_gpu_usage(manifest, torch_module=module))
        self.assertEqual(manifest["gpu_usage_evidence"][0]["bytes"], 128)
        def unavailable(device):
            raise RuntimeError("SIMULATED_QUERY_FAILURE")
        module.cuda.max_memory_allocated = unavailable
        unknown = dict(gpu_requested=True, gpu_used=False)
        self.assertIsNone(exe.observe_gpu_usage(unknown, torch_module=module))
        self.assertEqual(unknown["gpu_usage_status"], "UNKNOWN_OBSERVATION_FAILED")
        # An earlier retained observation remains true even when a later query fails.
        self.assertTrue(exe.observe_gpu_usage(manifest, torch_module=module))

    def test_source_dirty_or_untracked_code_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "implementation/minicells_r3mf_d1"
            package.mkdir(parents=True)
            (root / "run.py").write_text("original = True\n")
            (package / "__init__.py").write_text("")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=CPU Test",
                    "-c",
                    "user.email=cpu@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                check=True,
            )
            verify_source_tree(root)
            initial = code_sha(root, root)
            (root / "run.py").write_text("original = False\n")
            self.assertNotEqual(initial, code_sha(root, root))
            with self.assertRaisesRegex(RuntimeError, "UNCOMMITTED"):
                verify_source_tree(root)
            (root / "run.py").write_text("original = True\n")
            (package / "injected.py").write_text("changed = True\n")
            with self.assertRaisesRegex(RuntimeError, "UNTRACKED"):
                verify_source_tree(root)

    def test_run_bcd_checks_code_content_before_model_or_parent_loading(self):
        config = json.loads((BASE / "config/frozen.json").read_text())
        config_sha = sha256_bytes(canonical(config).encode())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("SAMPLE_PLAN.json", "BUDGET_PROOF.json"):
                exe._atomic_json(root / name, {})
            exe._atomic_json(
                root / "RUN_MANIFEST.json",
                dict(
                    experiment_id=config["experiment_id"],
                    source_sha="fixture",
                    config_sha256=config_sha,
                    data_sha256=config["input_records"]["sha256"],
                    source_code_sha256="different-content",
                ),
            )
            with patch( "minicells_r3mf_d1.source_identity.verify_source_tree", return_value="fixture" ), patch("minicells_r3mf_d1.source_identity.code_sha", return_value="current-content"), patch.object( exe, "_load_parent_state", side_effect=AssertionError("PARENT_MUST_NOT_LOAD") ):
                with self.assertRaisesRegex(RuntimeError, "PREFLIGHT_MANIFEST_IDENTITY"):
                    exe.run_bcd(config, root, root / "output", "fixture", "cpu")


if __name__ == "__main__":
    unittest.main()
