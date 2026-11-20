import unittest

from corpus import select


class CorpusTest(unittest.TestCase):
    def test_preserves_order_and_excludes_repeated_prefixes(self):
        def rows(names):
            return [{"text": f" = {name} = "} for name in names]

        def encode(text):
            return list(text.split()[1].encode())

        args = dict(tokens=1, train_size=2, holdout_size=2, compat_size=1)
        result, manifest = select(rows("ABACDE"), rows("AF"), encode, **args)
        self.assertEqual(result["train"], [[65], [66]])
        self.assertEqual(result["p1_compat_validation"], [[70]])
        self.assertEqual(len(result["primary_holdout"]), 2)
        self.assertEqual((result, manifest), select(rows("ABACDE"), rows("AF"), encode, **args))
