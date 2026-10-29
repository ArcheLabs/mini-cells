"""Small CPU integrity checks for fragment counts and hash-collision rejection."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / 'shared/python'))
from minicells_data_registry.fragments import compile_scanner, matches, scan, write_patterns  # noqa: E402


class FragmentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.binary = compile_scanner(cls.root / 'scan')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_scan(self, sequences, source, budget):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            path = Path(directory)
            patterns = [(len(x), struct.pack('<' + 'I' * len(x), *x)) for x in sequences]
            write_patterns(path / 'patterns', patterns)
            (path / 'source').write_bytes(struct.pack('<' + 'I' * len(source), *source))
            scan(self.binary, path / 'patterns', path / 'source', path / 'matches', budget)
            return list(matches(path / 'matches'))

    def test_stride_one_cross_boundary_and_budget(self):
        result = self.run_scan([[2, 0, 3]], [1, 2, 0, 3, 4, 1, 2, 0, 3], 7)
        self.assertEqual(result[0]['positions'], [1, 6])
        self.assertEqual(result[0]['occurrences'], 2)
        self.assertEqual(result[0]['budget_occurrences'], 1)

    def test_hash_collision_requires_full_sequence(self):
        a = [0] * 14 + [1, 0]
        b = [0] * 14 + [0, 1000003]
        def polynomial(values):
            return sum((v + 1) * pow(1000003, len(values) - 1 - i, 2**64)
                       for i, v in enumerate(values)) % 2**64
        self.assertEqual(polynomial(a), polynomial(b))
        self.assertEqual(self.run_scan([a], b, len(b)), [])

    def test_last_position_included_and_short_source(self):
        result = self.run_scan([[9] * 16, [9] * 65], [9] * 17, 17)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['positions'], [0, 1])
        self.assertEqual(result[0]['occurrences'], 2)


if __name__ == '__main__':
    unittest.main()
