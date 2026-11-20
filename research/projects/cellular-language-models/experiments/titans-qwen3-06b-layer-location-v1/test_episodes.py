import unittest

from episodes import digest, encode, generate


class WhitespaceTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [len(word) for word in text.split()]


class EpisodesTest(unittest.TestCase):
    def test_paired_and_split_isolated(self):
        train, validation, test = [generate(split, 5)
                                   for split in ("train", "validation", "test")]
        self.assertEqual(digest(train), digest(generate("train", 5)))
        entities = lambda split: {entity for episode in split
                                  for entity, _ in episode.facts}
        self.assertFalse(entities(train) & entities(test))
        self.assertFalse(entities(validation) & entities(test))

    def test_only_answer_labels_and_paired_donor_queries(self):
        episode = generate("test", 1, 16)[0]
        correct = encode(episode, WhitespaceTokenizer())
        donor = encode(episode, WhitespaceTokenizer(), donor=True)
        self.assertEqual(correct[1:], donor[1:])
        self.assertTrue(all(label == -100 for _, labels in correct[:2] for label in labels))
        self.assertEqual(sum(label != -100 for label in correct[2][1]), 4)


if __name__ == "__main__":
    unittest.main()
