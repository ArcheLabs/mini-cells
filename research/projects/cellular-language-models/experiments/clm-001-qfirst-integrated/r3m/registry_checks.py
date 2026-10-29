"""Small synthetic CPU tests; no dataset/model access."""
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[6] / 'shared/python'))
from minicells_data_registry import ExclusionIndex, content_hashes, release_allowed, select


class RegistryTests(unittest.TestCase):
    def test_format_equivalence(self):
        self.assertEqual(content_hashes('Ａ  Story')[1], content_hashes('a story')[1])

    def test_deterministic_order_and_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            index = ExclusionIndex(Path(d) / 'index.sqlite')
            index.add('old story', 'history', 'confirmation', 12)
            records = [(1, 'OLD   STORY'), (2, 'new story'), (3, 'other story'), (4, 'NEW story')]
            a = select(records, index, 'frozen', 2)
            self.assertEqual(a, select(list(reversed(records)), index, 'frozen', 2))
            self.assertNotIn(1, [x['raw_index'] for x in a])
            self.assertTrue(index.matches('old story'))
            index.close()

    def test_unknown_never_releases(self):
        self.assertFalse(release_allowed(None, True, True, True))
        self.assertFalse(release_allowed(True, True, True, False))
        self.assertTrue(release_allowed(True, True, True, True))

    def test_insufficient_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            index = ExclusionIndex(Path(d) / 'index.sqlite')
            with self.assertRaises(ValueError):
                select([(0, 'same'), (1, 'SAME')], index, 'frozen', 2)
            index.close()


if __name__ == '__main__':
    unittest.main()
