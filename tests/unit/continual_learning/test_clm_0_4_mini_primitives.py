import copy

import pytest
import torch

from minicells.clm04mini import DependencyIndex, MiniCLMConfig, StableAddressRouter, TinyCLMDecoder
from minicells.clm04mini.m0 import make_examples, max_logit_delta


@pytest.mark.unit
def test_stable_address_router_is_deterministic_and_out_of_band():
    router = StableAddressRouter(num_cells=8, salt="test-salt")
    assert router.route(3, "math/mul") == router.route(3, "math/mul")
    assert router.route(4, "math/mul") == router.route(4, "math/mul")
    first, second = router.route(3, "math/mul")
    assert first != second
    assert 0 <= first < 8
    assert 0 <= second < 8


@pytest.mark.unit
def test_zero_output_growth_bundle_is_functionally_identical_before_training():
    torch.manual_seed(1)
    cfg = MiniCLMConfig()
    model = TinyCLMDecoder(cfg).eval()
    candidate = copy.deepcopy(model)
    candidate.spawn_growth_bundle("math/mul")
    examples = make_examples(
        address_id="math/mul",
        family="zero-growth-test",
        target_token=31,
        count=2,
        vocab_size=cfg.vocab_size,
    )
    assert max_logit_delta(model, candidate, examples, torch.device("cpu")) <= 1e-7


@pytest.mark.unit
def test_dependency_index_is_exact_union():
    index = DependencyIndex()
    index.register("p0", ["base:L3:C00", "base:L4:C01"])
    index.register("p1", ["base:L3:C02"])
    index.register("p2", ["base:L3:C00"])
    assert index.scope(["base:L3:C00"]) == {"p0", "p2"}
    assert index.scope(["base:L3:C00", "base:L3:C02"]) == {"p0", "p1", "p2"}
    assert index.coverage(["base:L3:C02"], 3) == pytest.approx(1 / 3)
