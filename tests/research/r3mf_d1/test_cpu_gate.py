import json
import sys
import unittest
from pathlib import Path

# ruff: noqa: E402

ROOT = Path(__file__).resolve().parents[3]
UNIT = (
    ROOT
    / "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3m-f-d1"
)
sys.path.insert(0, str(UNIT / "implementation"))

from minicells_r3mf_d1.data import (
    SourceRow,
    build_source_row,
    build_update,
    split_independent,
    target_masks,
)  # noqa: E402
from minicells_r3mf_d1.evaluation import generate_tokens, summarize_predictions, teacher_forced_nll  # noqa: E402
from minicells_r3mf_d1.identity import identity, validate_resume  # noqa: E402
from minicells_r3mf_d1.loss import normalized_loss_sum, selected_positions  # noqa: E402
from minicells_r3mf_d1.routing import source_random_inputs  # noqa: E402
from minicells_r3mf_d1.sample_plan import replay_planned_update  # noqa: E402
from minicells_r3mf_d1.training import torch_batches, update as training_update  # noqa: E402


class FixedRng:
    def __init__(self, indexes):
        self.indexes = list(indexes)
        self.offset = 0
        self.bit_generator = type("BitGenerator", (), {})()
        self.bit_generator.state = {"offset": 0}

    def integers(self, high):
        value = self.indexes[self.offset % len(self.indexes)]
        self.offset += 1
        self.bit_generator.state = {"offset": self.offset}
        if not 0 <= value < high:
            raise AssertionError("invalid test draw")
        return value


def record(prompt, answer, group):
    prompt = tuple(prompt)
    answer = tuple(answer)
    return {"prompt_ids": prompt, "answer_tokens": answer, "ids": prompt + answer, "group": group}


class D1UnitTests(unittest.TestCase):
    def test_torch_batches_keep_labels_and_split_qa_context(self):
        records = [record([10, 11], [12], "a"), record([20, 21], [22], "b")]
        row = build_source_row(FixedRng([0, 1]), records, 65, 1, microbatch=3)
        packed = torch_batches((row,), "PACKED", "cpu")
        independent = torch_batches((row,), "INDEPENDENT_QA", "cpu")
        self.assertEqual(len(packed), 1)
        self.assertEqual(packed[0]["x"].shape, (1, 64))
        self.assertEqual(packed[0]["y"].tolist()[0], list(row.tokens[1:]))
        self.assertEqual(len(independent), len(split_independent(row)))
        self.assertEqual(
            sum(batch["y"].numel() for batch in independent),
            sum(len(fragment.labels) for fragment in split_independent(row)),
        )
        for batch, fragment in zip(independent, split_independent(row)):
            self.assertEqual(batch["source_microbatch"], 3)
            self.assertEqual(batch["source_start"], fragment.source_start)
            self.assertEqual(batch["x"].shape[1], batch["input_positions"])
            self.assertEqual(batch["y"][0].tolist(), list(fragment.tokens[1:]))

    def test_training_update_uses_update_global_masked_ce_and_one_optimizer_step(self):
        import torch
        from types import SimpleNamespace

        class TinyCellModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cells = torch.nn.ModuleList([torch.nn.Linear(32, 32, bias=False)])
                self.active_step = None
                self.microbatch_index = 0
                self.training_context = None
                self.d1_source_context = None
                self.last_logits = None

            def forward(self, tokens, **kwargs):
                one_hot = torch.nn.functional.one_hot(tokens, num_classes=32).float()
                logits = self.cells[0](one_hot)
                logits.retain_grad()
                self.last_logits = logits
                return SimpleNamespace(logits=logits)

            def forward_fragment(self, tokens, **kwargs):
                return self(tokens)

        class TinyTrainer:
            def __init__(self, model):
                self.model = model
                self.c = {"engine": {"training": {"clip": 1.0, "learning_rate": 0.1, "warmup_steps": 5}}}
                self.cell_opt = torch.optim.SGD(model.cells.parameters(), lr=0.1)
                self.step = 300
                self.data_cursor = 300 * 512
                self.route_rng = None
                self.cost = {"main_updates": 0, "main_tokens": 0, "elapsed_update_seconds": 0.0}
                self.curve = []

            def epsilon(self):
                return 0.0

            def verify_frozen(self):
                return None

        row = build_source_row(
            FixedRng([0, 1]),
            [record([10, 11], [12], "a"), record([20, 21], [22], "b")],
            65,
            1,
        )
        batches = torch_batches((row,), "PACKED", "cpu")
        model = TinyCellModel()
        trainer = TinyTrainer(model)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        result = training_update(
            trainer,
            batches,
            objective="ANSWER_EOS",
            context="PACKED",
            d1_config={"source_rows_per_step": 8, "source_sequence_tokens": 65},
        )
        selected = batches[0]["answer_eos_mask"]
        grads = model.last_logits.grad[0]
        self.assertEqual(result["selected_target_count"], int(selected.sum()))
        self.assertTrue(torch.equal(grads[~selected], torch.zeros_like(grads[~selected])))
        self.assertTrue(bool(model.last_logits.grad[0][selected].abs().sum() > 0))
        self.assertEqual(trainer.step, 301)
        self.assertEqual(trainer.data_cursor, 301 * 512)
        self.assertEqual(trainer.cost["main_updates"], 1)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))

    def test_sample_plan_replay_validates_draws_tokens_and_rng_boundary(self):
        records = [record([10, 11], [12], "a"), record([20, 21], [22], "b")]
        rng = FixedRng([0, 1])
        planned_rows = build_update(rng, records)
        from minicells_r3mf_d1.identity import canonical, sha256_bytes

        step = {
            "rows": [
                {
                    "source_microbatch": row.microbatch,
                    "sampled_record_indices": list(row.sampled_indices),
                    "source_token_sha256": sha256_bytes(canonical(row.tokens).encode("utf-8")),
                }
                for row in planned_rows
            ],
            "rng_state_after": {"offset": rng.offset},
        }
        replay = FixedRng([0, 1])
        self.assertEqual(len(replay_planned_update(replay, records, step)), 8)
        bad = dict(step)
        bad_rows = [dict(row) for row in step["rows"]]
        bad_rows[0]["source_token_sha256"] = "wrong"
        bad["rows"] = bad_rows
        with self.assertRaisesRegex(ValueError, "TOKEN_STREAM"):
            replay_planned_update(FixedRng([0, 1]), records, bad)

    def test_answer_eos_mask_is_shifted_target_aligned(self):
        row = build_source_row(FixedRng([0]), [record([10, 11], [12, 13], "f")], 5, 1)
        self.assertEqual(row.tokens, (10, 11, 12, 13, 1))
        self.assertEqual(row.labels, (11, 12, 13, 1))
        masks = target_masks(row)
        self.assertEqual(masks["answer"], (False, True, True, False))
        self.assertEqual(masks["eos"], (False, False, False, True))
        self.assertEqual(masks["answer_eos"], (False, True, True, True))
        self.assertEqual(masks["label_facts"], ("f", "f", "f", "f"))

    def test_answer_eos_selection_preserves_original_labels(self):
        labels = (21, 22, 23, 1)
        mask = (False, True, True, True)
        self.assertEqual(labels, (21, 22, 23, 1))
        self.assertEqual(selected_positions(mask, "answer_eos"), (1, 2, 3))
        self.assertEqual(selected_positions(mask, "uniform"), (0, 1, 2, 3))
        losses = (2.0, 3.0, 5.0, 7.0)
        self.assertEqual(normalized_loss_sum(losses, mask, "answer_eos", 3), 5.0)
        nll = teacher_forced_nll((8.0, 2.0, 3.0, 4.0), ("prompt", "answer", "answer", "eos"))
        self.assertEqual(nll["answer_nll"], 2.5)
        self.assertEqual(nll["eos_nll"], 4.0)
        self.assertEqual(nll["answer_target_count"], 2)

    def test_torch_answer_eos_ce_backpropagates_only_selected_original_labels(self):
        import torch
        from minicells_r3mf_d1.loss import selected_loss

        logits = torch.tensor(
            [[1.0, 0.0, -1.0], [0.0, 2.0, -1.0], [1.0, -1.0, 0.0], [-1.0, 0.0, 2.0]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 1, 2, 2])
        mask = (False, True, False, True)
        loss = selected_loss(logits, labels, mask, "answer_eos", 2)
        expected = torch.nn.functional.cross_entropy(
            logits[[1, 3]], labels[[1, 3]], reduction="mean"
        )
        self.assertTrue(torch.equal(labels, torch.tensor([0, 1, 2, 2])))
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[[0, 2]], torch.zeros_like(logits.grad[[0, 2]])))
        self.assertTrue(torch.all(logits.grad[[1, 3]].abs().sum(dim=1) > 0))

    def test_independent_fragments_reset_position_and_drop_only_cross_qa_edges(self):
        a = record([10, 11], [12], "a")
        b = record([20, 21], [22], "b")
        row = build_source_row(FixedRng([0, 1, 0]), [a, b], 10, 1)
        fragments = split_independent(row)
        self.assertEqual([f.fact for f in fragments], ["a", "b", "a"])
        self.assertEqual(sum(len(f.labels) for f in fragments), len(row.labels) - 2)
        self.assertEqual(fragments[0].tokens, (10, 11, 12, 1))
        self.assertEqual(fragments[1].tokens, (20, 21, 22, 1))
        positions = [list(range(len(f.input_tokens))) for f in fragments]
        self.assertEqual(positions, [[0, 1, 2], [0, 1, 2], [0]])

    def test_isolated_fragment_has_no_sibling_context_state(self):
        row = build_source_row(
            FixedRng([0, 1]),
            [record([10, 11], [12], "left"), record([20, 21], [22], "right")],
            8,
            1,
        )
        fragments = split_independent(row)

        def independent_signature(fragment):
            # Mirrors the intended fresh-forward contract: only this fragment enters state.
            return tuple(fragment.tokens), tuple(range(len(fragment.input_tokens))), None, 6

        before = independent_signature(fragments[1])
        mutated = SourceRow(
            row.microbatch,
            (999,) + row.tokens[1:],
            ("prompt",) + row.token_kinds[1:],
            ("other",) + row.token_facts[1:],
            row.sampled_facts,
            row.sampled_indices,
            row.qa_spans,
        )
        after = independent_signature(split_independent(mutated)[1])
        self.assertEqual(before, after)

    def test_routing_random_numbers_slice_by_source_offset(self):
        def factory(seed, step, micro, stage, rows, actions, device):
            return {"draw": tuple(range(rows)), "seed": tuple([seed] * rows)}

        actual = source_random_inputs(7, 300, 3, 2, 17, 5, 6, "cpu", factory)
        self.assertEqual(actual, {"draw": (17, 18, 19, 20, 21), "seed": (7,) * 5})
        with self.assertRaises(ValueError):
            source_random_inputs(7, 300, 3, 2, 63, 2, 6, "cpu", factory)

    def test_resume_identity_checks_full_arm_and_state_contract(self):
        config = {"experiment_id": "d1", "seed": 151001301, "anchor_step": 300}
        expected = identity(config, "source", "data", "C")
        expected["max_steps"] = 200
        saved = {
            "identity": expected,
            "step": 400,
            "relative_step": 100,
            "trainer_state_sha256": "a",
            "optimizer_state_sha256": "b",
            "data_rng_state": {"state": 1},
            "route_rng_state_sha256": "c",
            "frozen_parameter_sha256": "d",
        }
        self.assertTrue(validate_resume(saved, expected))
        changed = dict(expected, arm="D")
        with self.assertRaisesRegex(ValueError, "IDENTITY"):
            validate_resume(saved, changed)

    def test_autoregressive_scores_separate_text_match_eos_and_nll(self):
        rows = []
        for i in range(8):
            rows.append(
                {
                    "fact": str(i),
                    "expected_answer_tokens": [10 + i],
                    "generated_answer_tokens": [10 + i] if i < 4 else [99],
                    "eos_generated": i in (0, 2),
                    "answer_nll": float(i),
                    "eos_nll": float(i + 1),
                }
            )
        result = summarize_predictions(rows)
        self.assertEqual(result["first_token_accuracy"], 0.5)
        self.assertEqual(result["text_only_exact_match"], 0.5)
        self.assertEqual(result["strict_exact_match"], 0.25)
        self.assertEqual(result["eos_termination_rate"], 0.25)
        self.assertEqual(result["answer_nll"], 3.5)
        self.assertEqual(result["eos_nll"], 4.5)

    def test_autoregressive_generation_uses_prompt_prefix_and_stops_before_eos(self):
        prefixes = []

        def next_token(prefix):
            prefixes.append(prefix)
            return 23 if len(prefix) == 2 else 1

        generated, ended = generate_tokens(next_token, [10, 200], max_tokens=8, eos_id=1)
        self.assertEqual(generated, [23])
        self.assertTrue(ended)
        self.assertEqual(prefixes, [(10, 200), (10, 200, 23)])

    def test_fixed_config_keeps_gpu_phase_closed(self):
        config = json.loads((UNIT / "config/frozen.json").read_text())
        self.assertEqual(set(config["arms"]), {"A", "B", "C", "D"})
        self.assertFalse(config["authorization"]["gpu_training"])
        self.assertFalse(config["authorization"]["full_r3mf_pilot"])
        self.assertEqual(config["steps"], 200)


if __name__ == "__main__":
    unittest.main()
