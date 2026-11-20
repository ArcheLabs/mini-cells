import io
import os
from pathlib import Path
import unittest

import torch
from fp32_adamw import FP32AdamW
from run import profiles, lr_scale


class TrainingProtocolTest(unittest.TestCase):
    def test_upstream_schedule_and_step_budget(self):
        cfgs = profiles(Path(os.environ['TITANS_SOURCE']))
        for stage, expected in [('pretrain', 953), ('gold_sft', 200), ('quality_sft', 59)]:
            cfg = cfgs[stage]
            steps = cfg['total_tokens'] // (cfg['batch_size'] * cfg['sequence_length'])
            self.assertEqual(steps, expected)
            self.assertEqual(cfg['sequence_length'], 16384)
            for step, lr in [(0, cfg['neural_mem_warmup_start_learning_rate']),
                             (cfg['warmup_steps'], cfg['neural_mem_learning_rate']),
                             (steps, cfg['min_neural_mem_learning_rate'])]:
                self.assertAlmostEqual(lr_scale(step, cfg, steps)*cfg['neural_mem_learning_rate'], lr)

    def test_fp32_optimizer_and_scheduler_roundtrip(self):
        cfg = profiles(Path(os.environ['TITANS_SOURCE']))['pretrain']

        def session(weight):
            param = torch.nn.Parameter(weight.clone())
            opt = FP32AdamW([param], lr=5e-4, betas=(.9, .95), weight_decay=.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: lr_scale(step, cfg, 953))
            return param, opt, scheduler

        def update(param, opt, scheduler, target):
            opt.zero_grad()
            (param-target).square().sum().backward()
            opt.step()
            scheduler.step()

        a, opt_a, lr_a = session(torch.tensor([.3, -.1]))
        update(a, opt_a, lr_a, torch.tensor([1., .4]))
        checkpoint = io.BytesIO()
        torch.save((a.detach(), opt_a.state_dict(), lr_a.state_dict()), checkpoint)
        checkpoint.seek(0)
        weight, state, sched = torch.load(checkpoint, weights_only=True)
        b, opt_b, lr_b = session(weight)
        opt_b.load_state_dict(state)
        lr_b.load_state_dict(sched)
        for param, opt, lr in [(a, opt_a, lr_a), (b, opt_b, lr_b)]:
            update(param, opt, lr, torch.tensor([.2, -.5]))
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        for key in ('exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(opt_a.state[a][key], opt_b.state[b][key], atol=0, rtol=0)
        self.assertEqual(lr_a.state_dict(), lr_b.state_dict())

    def test_objective_aligns_final_tile_at_16k(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from run import objective

        model = torch.nn.Module()
        model.lm_head = torch.nn.Linear(4, 8)
        length = 16384
        ids = torch.arange(length).reshape(1, -1) % 8
        hidden = torch.randn(1, length, 4, requires_grad=True)
        batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
        with patch.dict(sys.modules, {'model': SimpleNamespace(hidden_states=lambda *args: hidden)}):
            loss, count, _ = objective(model, batch, use_checkpoint=True)
        reference = torch.nn.functional.cross_entropy(
            model.lm_head(hidden[:, :-1]).reshape(-1, 8), ids[:, 1:].flatten(), reduction='sum')
        self.assertEqual(count, 16383)
        torch.testing.assert_close(loss, reference)
        loss.backward()
        self.assertEqual(hidden.grad[:, -1].count_nonzero().item(), 0)
