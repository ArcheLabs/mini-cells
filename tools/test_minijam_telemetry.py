#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_minijam_telemetry import summarize


class MiniJamTelemetryTests(unittest.TestCase):
    def test_percentiles_and_headroom_use_mini_jam_limits(self):
        result = summarize({"samples": [
            {"measured_refine_gas": 100, "measured_accumulate_gas": 40,
             "wall_time_ms": 5, "peak_memory_bytes": 1000,
             "batch_count": 1, "sample_count": 8},
            {"measured_refine_gas": 200, "measured_accumulate_gas": 60,
             "wall_time_ms": 7, "peak_memory_bytes": 1200,
             "batch_count": 2, "sample_count": 16},
            {"measured_refine_gas": 300, "measured_accumulate_gas": 80,
             "wall_time_ms": 9, "peak_memory_bytes": 1400,
             "batch_count": 3, "sample_count": 24},
        ]})
        self.assertEqual(result["sample_count"], 48)
        self.assertEqual(result["batch_count"], 6)
        self.assertEqual(result["metrics"]["refine_gas"]["p95"], 300)
        self.assertEqual(result["headroom"]["refine"]["limit"], 1_000_000_000)
        self.assertEqual(result["headroom"]["refine"]["p95_margin"], 999_999_700)


if __name__ == "__main__":
    unittest.main()
