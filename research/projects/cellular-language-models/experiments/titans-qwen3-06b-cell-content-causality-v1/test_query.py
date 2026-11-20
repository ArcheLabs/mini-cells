import unittest
from types import SimpleNamespace

import torch

from query import query, snapshot


class Wrapper:
    def reset_episode(self):
        self.cell = None
        self.position = 0


class Model:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.calls = []

    def get_input_embeddings(self):
        return SimpleNamespace(weight=torch.empty(0))

    def __call__(self, input_ids, use_cache):
        assert not use_cache and not self.wrapper.write_enabled
        value = self.wrapper.cell["fact"].item() if self.wrapper.cell else 0
        self.calls.append((input_ids.tolist()[0], value))
        logits = torch.zeros(1, input_ids.shape[1], 8)
        logits[..., value] = 10
        return SimpleNamespace(logits=logits)


class QueryTests(unittest.TestCase):
    def test_counterfactual_and_empty_with_independent_generation(self):
        wrapper = Wrapper()
        model = Model(wrapper)
        wrapper.cell = {"fact": torch.tensor(3)}
        wrapper.position = 8
        correct = snapshot(wrapper)
        wrapper.cell["fact"].fill_(4)
        donor = snapshot(wrapper)
        first = query(model, wrapper, correct, [1], [3, 5], max_new_tokens=2)
        self.assertEqual(model.calls[:3], [([1, 3], 3), ([1], 3), ([1, 3], 3)])
        self.assertEqual(first["generated_ids"], [3, 3])
        self.assertEqual(query(model, wrapper, donor, [1], [4], max_new_tokens=1)
                         ["generated_ids"], [4])
        self.assertEqual(query(model, wrapper, (None, 0), [1], [3], max_new_tokens=1)
                         ["generated_ids"], [0])
        self.assertEqual(query(model, wrapper, correct, [1], [3, 5], max_new_tokens=2),
                         first)
        self.assertEqual(correct[0]["fact"].item(), 3)
        self.assertIsNone(wrapper.cell)
        self.assertEqual(wrapper.position, 0)
