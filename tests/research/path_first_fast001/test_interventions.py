import torch
import pytest
from fast001.models import make_model
from fast001.interventions import pair


@pytest.mark.parametrize("arm", ["CLM-S8", "CLM-S16"])
def test_symmetric_interventions_are_real_and_legal(tiny, arm):
    model = make_model(tiny, arm).eval()
    x = torch.tensor([[1, 2, 3, 4]])
    y = x + 1
    before = {n: p.clone() for n, p in model.state_dict().items()}
    for kind in (
        "edge_zero",
        "edge_shuffle",
        "order_permutation",
        "context_zero_replay",
        "truncate",
        "self_loop_replacement",
    ):
        row = pair(model, x, y, kind, step=4, seed=17)
        assert row.get("scientific_decision") is False
        assert row["applicable_tokens"] <= x.numel()
        if row["applicable_tokens"]:
            assert row["nll_after_minus_before"] is not None
    assert all(torch.equal(p, before[n]) for n, p in model.state_dict().items())


def test_empty_self_loop_subset_is_not_zero_effect(tiny, monkeypatch):
    model = make_model(tiny, "CLM-S8").eval()
    original = model.forward

    # Greedy reference is a valid alternating path with no repeated Cell.
    def replay(tokens, **kwargs):
        kwargs.setdefault(
            "replay", torch.tensor([0, 1] * 4)[None, None, :].expand(*tokens.shape, 8)
        )
        return original(tokens, **kwargs)

    monkeypatch.setattr(model, "forward", replay)
    row = pair(model, torch.tensor([[1, 2]]), torch.tensor([[2, 3]]), "self_loop_replacement")
    assert row["applicable_tokens"] == 0 and row["nll_after_minus_before"] is None
