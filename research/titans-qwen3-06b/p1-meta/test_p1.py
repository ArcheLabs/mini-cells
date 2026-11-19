import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from p1 import episode_loss


class EpisodeTest(unittest.TestCase):
    def test_later_loss_reaches_earlier_write_without_qwen_cache(self):
        slow = torch.nn.Parameter(torch.tensor(0.2))
        wrapper = SimpleNamespace(state=None, pending=None)
        reads = []

        def commit():
            wrapper.state = wrapper.pending if wrapper.state is None else wrapper.state + wrapper.pending

        def model(input_ids, position_ids, use_cache):
            self.assertFalse(use_cache)
            reads.append(wrapper.state)
            wrapper.pending = slow * input_ids.sum()
            value = slow * 0 if wrapper.state is None else wrapper.state
            return SimpleNamespace(logits=torch.stack((value, -value)).expand(1, 2, 2))

        wrapper.commit = commit
        with patch("p1.CHUNK", 2):
            losses = episode_loss(model, wrapper, torch.tensor([[1, 1] * 4]))
        gradient, = torch.autograd.grad(losses[1], reads[1])
        self.assertNotEqual(gradient.item(), 0)
        self.assertIsNone(reads[0])
        self.assertIsNone(wrapper.state)
        self.assertIsNone(wrapper.pending)
