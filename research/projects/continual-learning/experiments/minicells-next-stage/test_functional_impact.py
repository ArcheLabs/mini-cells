"""CPU regression tests for strict functional decisions and the same-process seam."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch

from common import write
import m1_functional_impact as exp


def row(episode, arm, variant, wording, gold, wrong):
    other = "b" if variant == "a" else "a"
    return dict(episode=episode, arm=arm, variant=variant, wording=wording,
                candidate_order=["a", "b"], gold_logp=gold, logp={variant: gold, other: wrong},
                margin=gold - wrong, gold_nll=-gold, correct=gold > wrong)


def lane():
    return {(0, arm, label, w): row(0, arm, label, w, -1.0, -2.0)
            for arm in exp.ARMS for label in ("a", "b") for w in (0, 1)}


class FunctionalImpactTest(unittest.TestCase):
    def test_episode_requires_four_correct_and_deltas_are_right_minus_left(self):
        left = lane()
        right = dict(left)
        key = (0, "combined", "b", 1)
        right[key] = row(*key, -2.0, -2.0)
        a, b = exp.summarize(left), exp.summarize(right)
        self.assertEqual(a["combined"]["successes"], 1)
        self.assertEqual(b["combined"]["successes"], 0)
        result = exp.compare(left, right, a, b)
        self.assertEqual(result["arms"]["combined"]["episode_losses"], ["0"])
        self.assertEqual(result["arms"]["combined"]["query_losses"], 1)
        self.assertEqual(result["arms"]["combined"]["gold_order_changes"], 1)
        comparison = next(q for q in result["cases"]
                          if (q["episode"], q["arm"], q["variant"], q["wording"]) == key)
        self.assertEqual(comparison["margin_delta"], -1.0)
        self.assertEqual(comparison["gold_nll_delta"], 1.0)
        gains = exp.compare(right, left, b, a)
        self.assertEqual(gains["arms"]["combined"]["episode_gains"], ["0"])

    def test_functional_and_history_decisions_remain_independent(self):
        from itertools import combinations

        def compare_groups(groups):
            return exp.functional_decision({
                f"{a}_vs_{b}": {"gold_order_changes": int(groups[a] != groups[b])}
                for a, b in combinations(exp.LANES, 2)
            })

        common = dict.fromkeys(exp.LANES, 0)
        identical = compare_groups(common)
        self.assertEqual(identical["functional_status"], "NO_FUNCTIONAL_EFFECT_OBSERVED")
        stable = compare_groups(dict(common, **{"B1-E": 1, "B2-E": 1, "P0-A": 2}))
        self.assertEqual(stable["functional_status"], "FUNCTIONAL_EFFECT_PRESENT")
        self.assertEqual(stable["process_history_status"], "STABLE_BETWEEN_ARM_DIFFERENCE")
        self.assertEqual(stable["live_vs_restored_status"], "DIFFERENT")
        unstable = compare_groups(dict(common, **{"A1-E": 1}))
        self.assertEqual(unstable["functional_status"], "FUNCTIONAL_INSTABILITY_PRESENT")
        self.assertEqual(unstable["unstable_arms"], ["A"])

    def test_reuses_worker_context_and_keeps_probe_and_bridge_launch_receipts(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "work").mkdir()
            mc = object()
            events = []

            def probe_then_save(_mc, root, label):
                self.assertIs(_mc, mc)
                events.append("probe")
                write(root / "work" / f"{label}.json",
                      dict(start_failures=[], bundle_unchanged=True))

            def mock_history(root, label):
                calls = Counter()
                with exp.history.fixed_forward(None, calls, lambda: "query"):
                    exp.history.trial(mc, root, label)
                    self.assertEqual(calls, {})
                write(root / f"policy-{label}.json", dict(pid=os.getpid()))

            def bridge(instance, item, paths, segments):
                self.assertIs(instance, mc)
                self.assertEqual(paths, ("A", "E"))
                self.assertEqual(item["index"], 0)
                events.append("bridge")
                for segment in ("bridge/sdk_write", "bridge/live_prefill", "bridge/query/A", "bridge/query/E"):
                    with exp.segment_label(segments, segment):
                        active()
                return [dict(path=path) for path in paths]

            from contextlib import contextmanager

            @contextmanager
            def fixed(_module, calls, current_segment):
                nonlocal active
                active = lambda: calls.update([(current_segment(), 256, 8, 1, 256, 32, 4, 3, 1)])
                yield exp.FIXED

            active = None
            with (patch.object(exp.history, "fixed_forward", fixed),
                  patch.object(exp.history, "trial", probe_then_save),
                  patch.object(exp.history, "worker", mock_history),
                  patch.object(exp, "dataset", return_value=[dict(index=0)]),
                  patch.object(exp, "bridge_case", side_effect=bridge)):
                exp.worker(out, "P0")
            self.assertEqual(events, ["probe", "bridge"])
            receipt = exp.load_json(out / "worker-P0.json")
            self.assertEqual(receipt["pid"], os.getpid())
            self.assertEqual(receipt["expected_query_count"], 576)
            self.assertEqual(len(receipt["bridge_launches"]), 4)

    def test_invalid_frozen_asset_stops_before_gpu_work(self):
        with TemporaryDirectory() as tmp:
            with patch.object(exp, "urlopen", side_effect=OSError("unavailable")):
                with self.assertRaisesRegex(exp.InvalidProtocol, "Frozen Bridge dataset"):
                    exp.fixture(Path(tmp))


if __name__ == "__main__":
    unittest.main()
