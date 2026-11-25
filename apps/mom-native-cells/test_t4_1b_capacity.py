import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from minicells_native.generation import generate
from minicells_native.query import run_query
from capacity_meter import QueryProbe, nvml_device
from kaggle_t4_1b_capacity import t4_uuid
from preflight_t4_1b import FORMAT, GATES, approved, check_logits
from test_sdk import Fixture, Reader, fixture_patch
from minicells_native.sdk import MiniCells
from run_t4_1b_capacity import (
    BASELINE_REVISION, CELLS, HERE, PROMPTS, build_prompts, main, matrix, read, summarize, validate_target, write,
)


class CapacityTest(unittest.TestCase):
    def test_dual_t4_launcher_and_meter_use_same_full_uuid(self):
        first = "GPU-01234567-89ab-cdef-0123-456789abcdef"
        second = "GPU-98765432-10ab-cdef-0123-456789abcdef"
        output = f"{first}, Tesla T4, 7.5\n{second}, Tesla T4, 7.5\n"
        nvml = Mock()
        nvml.nvmlDeviceGetCount.return_value = 2
        with patch("kaggle_t4_1b_capacity.subprocess.check_output", return_value=output):
            visible = t4_uuid()
        self.assertEqual(visible, first)
        self.assertIs(nvml_device(nvml, None, visible), nvml.nvmlDeviceGetHandleByUUID.return_value)
        nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with(first)
        nvml.nvmlDeviceGetHandleByIndex.assert_not_called()

        with self.assertRaisesRegex(ValueError, "differs"):
            nvml_device(nvml, second, visible)
        with self.assertRaisesRegex(ValueError, "full GPU UUID"):
            nvml_device(nvml, None, "0")

    def test_model_gate_blocks_before_hardware_worker(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                "run_t4_1b_capacity.environment", return_value={"hardware_status": "NON_TARGET"}), patch(
                "run_t4_1b_capacity.subprocess.run") as worker:
            target = Path(tmp) / "target.json"
            target.write_text(json.dumps(dict(approved_target=None, reason="Fixture blocked")))
            with patch("sys.argv", ["capacity", "--output", str(Path(tmp) / "run"), "--target", str(target)]):
                main()
            worker.assert_not_called()
            out = Path(tmp) / "run"
            summary = read(out, "summary.json")
            self.assertEqual(summary["status"], "BLOCKED")
            self.assertEqual(set(summary["gates"].values()), {"NOT_RUN"})
            self.assertIsNone(summary["peak_device_bytes"])
            self.assertFalse((out / "work").exists())

    def test_rejects_historical_and_unregistered_checkpoints(self):
        manifest = json.loads((HERE / "capacity_target.json").read_text(encoding="utf-8"))
        target = dict(id="any", revision=BASELINE_REVISION, repository_designation="test",
                      weights_sha256="a"*64, tokenizer_hash="a"*64, tokenizer_revision="test")
        manifest["approved_target"] = target
        self.assertIn("historical", validate_target(manifest))
        target["revision"] = "a"*40
        self.assertIn("not registered", validate_target(manifest))

    def test_preflight_failure_stops_before_capacity(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                "sys.argv", ["capacity", "--output", str(Path(tmp) / "run")]), patch(
                "run_t4_1b_capacity.environment", return_value={"hardware_status": "NON_TARGET"}), patch(
                "run_t4_1b_capacity.subprocess.run", return_value=SimpleNamespace(returncode=1)) as worker:
            main()
            self.assertEqual(worker.call_count, 1)
            self.assertIn("preflight_t4_1b.py", worker.call_args.args[0][2])
            self.assertEqual(set(read(Path(tmp) / "run", "summary.json")["gates"].values()), {"NOT_RUN"})

    def test_passing_standalone_preflight_still_leaves_capacity_unrun(self):
        target = read(HERE, "capacity_target.json")["approved_target"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"

            def completed(*args, **kwargs):
                write(out, "preflight_result.json", dict(format=FORMAT, status="PASS", target=target,
                      source_commit="fixture", gates=dict.fromkeys(GATES, "PASS")))
                return SimpleNamespace(returncode=0)

            with patch("sys.argv", ["capacity", "--output", str(out), "--stage", "preflight"]), patch(
                    "run_t4_1b_capacity.environment", return_value={}), patch(
                    "run_t4_1b_capacity.subprocess.run", side_effect=completed) as worker, patch(
                    "preflight_t4_1b.source_commit", return_value="fixture"):
                main()
            self.assertEqual(worker.call_count, 1)
            self.assertEqual(read(out, "decision.json")["status"], "PASS")
            self.assertEqual(read(out, "summary.json")["status"], "NOT_RUN")

    def test_preflight_receipt_is_bound_to_complete_gates_target_and_commit(self):
        target = json.loads((HERE / "capacity_target.json").read_text())["approved_target"]
        self.assertIsNone(validate_target({"approved_target": target}))
        receipt = dict(format=FORMAT, status="PASS", target=target, source_commit="example",
                       gates=dict.fromkeys(GATES, "PASS"))
        self.assertTrue(approved(receipt, target, "example"))
        self.assertFalse(approved(receipt, target, "other"))
        receipt["gates"]["FLA_NATIVE_KERNEL"] = "NOT_RUN"
        self.assertFalse(approved(receipt, target, "example"))

    def test_logits_gate_rejects_constant_and_nonfinite(self):
        self.assertEqual(check_logits(torch, torch.tensor([[[1., 0.]]]), 1, 2)["shape"], [1, 1, 2])
        for values in ([1., 1.], [1., float("nan")]):
            with self.assertRaisesRegex(ValueError, "finite, nonconstant"):
                check_logits(torch, torch.tensor([[values]]), 1, 2)

    def test_matrix_exact_lengths_and_repeats(self):
        cases = matrix()
        self.assertEqual(len(cases), 47)
        self.assertTrue(all((n, p, 12, mode) in cases for n in CELLS
                            for p in PROMPTS for mode in ("manual", "auto")))
        self.assertEqual(cases[8, 1024, 128, "auto"], 3)
        tokenizer = SimpleNamespace(encode=lambda *a, **kw: list(range(2048)))
        self.assertTrue(all(len(v) == int(k) for k, v in build_prompts(tokenizer).items()))

    def test_normal_eos_and_capacity_decode_share_generator(self):
        model = SimpleNamespace(model=SimpleNamespace(lm_head=lambda x: x),
                                tokenizer=SimpleNamespace(eos_token_id=0,
                                                          decode=lambda tokens, **kw: str(tokens)))
        step = lambda _: torch.tensor([[[5., 0.]]])
        seen = []
        self.assertEqual(generate(model, [1], step, 3, 0)[1], [])
        self.assertEqual(generate(model, [1], step, 3, 0, stop_on_eos=False,
                                  observe=lambda _, token: seen.append(token))[1], [0, 0, 0])
        self.assertEqual(seen, [0, 0, 0])


    def test_capacity_path_reuses_sdk_scoring_and_fuses_two(self):
        with tempfile.TemporaryDirectory() as tmp, fixture_patch(), patch(
                "minicells_native.query.FusionReader", Reader), patch("torch.cuda.synchronize"):
            model = Fixture()
            with MiniCells._from_backbone(model, tmp) as mc:
                for name in ("a", "bb", "ccc", "dddd"):
                    mc.cells.write(mc.cells.create(name).id, name)
                model.calls.clear()
                probe = QueryProbe(torch, eos=0)
                encoded = [7] * 128
                with mc._operation():
                    result = run_query(mc, "fixture", "auto", 12, 0, token_ids=encoded, probe=probe)
                self.assertEqual(len(result.cell_scores), 4)
                self.assertEqual(len(result.selected_cell_ids), 2)
                self.assertEqual(len(probe.tokens), 12)
                self.assertEqual(probe.cache_bytes, 32)
                self.assertEqual([len(ids) for mode, ids, _ in model.calls if mode == "score"],
                                 [64] * 8)

    def test_summary_requires_all_repetitions_and_recovery(self):
        def row(case, repetitions):
            n, prompt, generated, mode = case
            return dict(registered_cells=n, prompt_tokens=prompt, requested_new_tokens=generated,
                        mode=mode, repetitions_expected=repetitions,
                        trials=[dict(kind="measured", status="PASS") for _ in range(repetitions)])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for name in ("model_load.json", "no_memory_smoke.json", "cell_storage.json",
                         "recovery_profile.json"):
                write(out, name, dict(status="PASS"))
            cases = [row(case, reps) for case, reps in matrix().items()]
            write(out, "generation_profile.json", cases)
            write(out, "read_all_profile.json", [row((n, p, 0, "auto"), 1)
                                                for n in CELLS for p in PROMPTS])
            smoke = dict(status="PASS", peak=dict(device_used=1), other_process_peak=0,
                         device_sampler_errors=[])
            write(out, "no_memory_smoke.json", smoke)
            write(out, "environment.json", dict(runtime_gpu=dict(threshold_bytes=100)))
            self.assertEqual(summarize(out), "FULL PASS")
            self.assertEqual(read(out, "summary.json")["maximum_measured_configurations"]["auto"],
                             [[8, 1024, 128]])
            cases[-1]["trials"].pop()
            write(out, "generation_profile.json", cases)
            self.assertEqual(summarize(out), "PARTIAL PASS")
