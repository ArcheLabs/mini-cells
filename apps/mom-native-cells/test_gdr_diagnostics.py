import unittest

import torch

from gdr_diagnostics import FIELDS, ResidualTrace, parity_report, reference, statistics


class GatedDeltaTest(unittest.TestCase):
    def test_prediction_precedes_update_and_output_follows_it(self):
        q = k = torch.tensor([[[[1., 0.]], [[1., 0.]]]])
        v = torch.tensor([[[[3., 4.]], [[2., 1.]]]])
        state = torch.tensor([[[[2., 0.], [0., 0.]]]])
        decay = torch.log(torch.full((1, 2, 1), 0.5))
        beta = torch.full_like(decay, 0.5)
        output, final, predicted, error = reference(q, k, v, decay, beta, state)
        torch.testing.assert_close(predicted, torch.tensor([[[[1., 0.]], [[1., 1.]]]]))
        torch.testing.assert_close(error, torch.tensor([[[[2., 4.]], [[1., 0.]]]]))
        torch.testing.assert_close(output * 2**0.5, torch.tensor([[[[2., 2.]], [[1.5, 1.]]]]))
        torch.testing.assert_close(final, torch.tensor([[[[1.5, 1.], [0., 0.]]]]))
        torch.testing.assert_close(state, torch.tensor([[[[2., 0.], [0., 0.]]]]))
        stats = statistics(output, predicted, error, v, beta, decay)
        self.assertAlmostEqual(float(stats[0, 1, 0, FIELDS.index("delta_norm")]), 0.5)
        self.assertAlmostEqual(float(stats[0, 0, 0, FIELDS.index("relative_residual_norm")]),
                               20**0.5 / (5 + 1e-6))

    def test_missing_state_is_zero_and_padding_is_not_recorded(self):
        q = k = v = torch.ones((1, 2, 1, 1))
        params = torch.zeros((1, 2, 1))
        trace = ResidualTrace(reference, verify=True)
        p = torch.full((1, 1, 4), .25)
        trace.begin_layer(0, p, torch.tensor([[[0, 1]]]), torch.full((1, 1, 2), .5),
                          torch.tensor([[-1, 0]]))
        observer = trace.for_cell(0)
        probe = observer.before(q, k, v, params, params, None)
        observer.after(0, probe, *probe[:2], v, params, params)
        self.assertEqual(trace.banks[0][3].tolist(), [0])
        self.assertEqual(trace.routers[0]["topk_mass"], [.5])
        self.assertTrue(trace.parity["state"]["passed"])
        self.assertFalse(parity_report(q, q + 1, atol=1e-6, rtol=0)["passed"])
