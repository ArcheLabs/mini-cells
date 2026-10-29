"""Small CPU tests for the isolated auditor, not a training or Kaggle test suite."""

import hashlib
import json
from pathlib import Path
import unittest
from audit_text_boundaries import hashes, locate, spans

BASE = Path(__file__).resolve().parent
AUDIT = BASE / "registration/amendment-002"


class AuditChecks(unittest.TestCase):
    def test_all_shards_and_cross_boundary_mapping(self):
        layout = json.loads((AUDIT / "SOURCE_LAYOUT.json").read_text())
        self.assertEqual(layout["total_rows"], 2119719)
        self.assertEqual([r["rows"] for r in layout["shards"]], [529930, 529930, 529930, 529929])
        self.assertEqual(locate(layout, 529929)["local_row"], 529929)
        self.assertEqual(locate(layout, 529930)["local_row"], 0)
        self.assertEqual(len(spans(layout, 529929, 529931)), 2)
        self.assertEqual(locate(layout, 2119718)["local_row"], 529928)
        with self.assertRaises(ValueError):
            locate(layout, 2119719)
        self.assertEqual(
            sum(r["global_range"][1] - r["global_range"][0] for r in spans(layout, 500000, 501024)),
            1024,
        )

    def test_raw_and_format_variants_are_distinct_checks(self):
        a = hashes("A full-width Ａ story.\n")
        b = hashes("a full-width A story.  ")
        self.assertNotEqual(a["raw_utf8"], b["raw_utf8"])
        self.assertEqual(a["nfkc_casefold_whitespace"], b["nfkc_casefold_whitespace"])
        self.assertEqual(
            hashes("A story!")["nfkc_word_sequence"], hashes("A story.\n")["nfkc_word_sequence"]
        )

    def test_failed_content_gate_cannot_be_called_registered_or_window_verified(self):
        proof = json.loads((AUDIT / "DATA_BOUNDARY_PROOF.json").read_text())
        self.assertEqual(proof["status"], "REGISTRATION_BLOCKED_TEXT_LEAKAGE")
        self.assertEqual(proof["historical_affected_candidate_count"], 150)
        self.assertEqual(proof["summary"]["main_report"]["exact_historical_duplicates"], 150)
        self.assertEqual(proof["summary"]["trigger"]["exact_historical_duplicates"], 0)
        self.assertEqual(proof["summary"]["pilot_report"]["exact_historical_duplicates"], 0)
        self.assertEqual(
            proof["historical_complete_scan"]["nonempty_records"],
            {"train": 476847, "validation": 4396},
        )
        self.assertFalse(proof["registration_changed"])
        self.assertFalse(proof["gpu_training_started"])
        self.assertFalse(proof["evaluation_windows"]["exact_r3m_token_window_ids_verified"])
        self.assertEqual(proof["configuration_before"], proof["configuration_after"])
        self.assertEqual(len(proof["registry_audit"]["uncovered_declared_intervals"]), 0)

    def test_proof_and_original_registration_remain_hash_bound(self):
        proof = json.loads((AUDIT / "DATA_BOUNDARY_PROOF.json").read_text())
        amendment = json.loads((AUDIT / "STORY_RANGE_AMENDMENT_002.json").read_text())
        self.assertEqual(
            amendment["proof_sha256"],
            hashlib.sha256((AUDIT / "DATA_BOUNDARY_PROOF.json").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            proof["configuration_after"]["config_sha256"],
            hashlib.sha256((BASE / "config/pilot.json").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            proof["auditor"]["code_sha256"],
            hashlib.sha256((BASE / "audit_text_boundaries.py").read_bytes()).hexdigest(),
        )
        self.assertTrue((BASE / "registration/DATA_BOUNDARY_PROOF.json").exists())


if __name__ == "__main__":
    unittest.main()
