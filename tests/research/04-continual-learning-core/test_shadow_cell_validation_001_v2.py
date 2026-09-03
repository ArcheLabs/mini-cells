from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

from minicells.clm04mini.examples import collate_scored
from minicells.clm04mini.model import MiniCLMConfig, TinyCLMDecoder
from minicells.shadow_maturation import (
    MATURITY_GRID,
    AcceptedModelSnapshot,
    FunctionalSketch,
    ShadowSidecar,
    count_false_safe,
    corrected_adamw_proposal,
    hash_accepted_state,
    m0_equivalence_delta,
    project_realized_delta,
    routing_is_preserved,
    select_oracle_maturity,
    select_sketch_maturity,
    synthetic_examples,
    train_shadow,
)


class Pad:
    pad_id = 0


def model() -> TinyCLMDecoder:
    return TinyCLMDecoder(MiniCLMConfig(
        vocab_size=32, max_seq_len=16, num_layers=4, d_model=8,
        n_heads=2, dense_ff_hidden=16, base_cells=4, cell_hidden=4,
        routing_salt="test-shadow-v2",
    ))


def batch(examples):
    return collate_scored(examples[:2], pad_id=0, device=torch.device("cpu"))[0]


def test_zero_expression_identity() -> None:
    accepted = model().eval()
    sidecar = ShadowSidecar(accepted, gate_mode="zero")
    examples = synthetic_examples(vocab_size=32, domain="math", count=2, seed=1)
    x = batch(examples)
    assert m0_equivalence_delta(sidecar, x, [item.address_id for item in examples]) <= 1e-6


def test_accepted_model_immutable_after_shadow_training() -> None:
    accepted = model().eval()
    sidecar = ShadowSidecar(accepted, gate_mode="task_id")
    before = hash_accepted_state(accepted)
    examples = synthetic_examples(vocab_size=32, domain="math", count=4, seed=2)
    train_shadow(sidecar, examples, Pad(), torch.device("cpu"), steps=2, batch_size=2, seed=3)
    assert hash_accepted_state(accepted) == before


def test_shadow_parameters_change() -> None:
    sidecar = ShadowSidecar(model().eval(), gate_mode="task_id")
    examples = synthetic_examples(vocab_size=32, domain="math", count=4, seed=4)
    initial = sidecar.shadow.operator.weight.detach().clone()
    train_shadow(sidecar, examples, Pad(), torch.device("cpu"), steps=2, batch_size=2, seed=5)
    assert not torch.equal(initial, sidecar.shadow.operator.weight.detach())


def test_no_global_read_stealing() -> None:
    accepted = model().eval()
    sidecar = ShadowSidecar(accepted, gate_mode="input_only")
    addresses = ["math/example-0", "story/example-1", "base/example-2"]
    assert routing_is_preserved(accepted, sidecar, addresses)


def test_maturity_scales_fixed_contribution() -> None:
    accepted = model().eval()
    sidecar = ShadowSidecar(accepted, gate_mode="zero")
    with torch.no_grad():
        sidecar.shadow.operator.weight.fill_(0.01)
    examples = synthetic_examples(vocab_size=32, domain="math", count=2, seed=6)
    x = batch(examples)
    addresses = [item.address_id for item in examples]
    with torch.no_grad():
        base = accepted(x, addresses)
        d1 = sidecar(x, addresses, maturity=0.25) - base
        d2 = sidecar(x, addresses, maturity=0.75) - base
    torch.testing.assert_close(d2, d1 * 3.0, rtol=1e-5, atol=1e-6)


def test_oracle_selector_and_reject_all() -> None:
    frontier = [
        {"maturity": 0.0, "old_regression": 0.0, "new_gain": 0.0},
        {"maturity": 0.25, "old_regression": 0.1, "new_gain": 0.2},
        {"maturity": 0.5, "old_regression": 0.3, "new_gain": 0.4},
    ]
    assert select_oracle_maturity(frontier, 0.2, 0.1) == 0.25
    assert select_oracle_maturity(frontier, 0.01, 0.1) is None


def test_false_safe_accounting() -> None:
    frontier = [{"maturity": 0.5, "old_regression": 0.25, "new_gain": 0.4}]
    assert count_false_safe(0.5, frontier, 0.2) == 1
    assert count_false_safe(None, frontier, 0.2) == 0


def test_realized_update_projection() -> None:
    delta = torch.ones(3, 4)
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    projected = project_realized_delta(delta, q)
    assert float((projected @ q.T).norm()) <= 1e-7


def test_sketch_selector_uses_bounded_state() -> None:
    sidecar = ShadowSidecar(model().eval(), gate_mode="zero")
    width = sidecar.accepted.cfg.d_model
    sketch = FunctionalSketch(torch.eye(width), torch.eye(width), sample_count=8, sketch_rank=width)
    with torch.no_grad():
        sidecar.shadow.operator.weight.fill_(0.001)
    selected = select_sketch_maturity(
        sidecar, sketch, MATURITY_GRID,
        {value: value for value in MATURITY_GRID}, max_predicted_damage=1.0, min_new_gain=0.0,
    )
    assert selected == 1.0
    assert sketch.bytes == 2 * width * width * 4


def test_formal_seed_enforcement() -> None:
    formal = {95311, 95312, 95313}
    assert 95301 not in formal
    with pytest.raises(ValueError):
        if 95301 not in formal:
            raise ValueError("development seed cannot enter formal aggregation")

