import torch

from minicells.growth_router import (
    BinaryLineageRouter,
    HierarchicalGrowthRouter,
    RouteLeaf,
    RouteSplit,
    iter_leaves,
)
from minicells.upcycled_cellular_textnca import UpcyclingConfig, convert_textnca_to_upcycled
from minicells.language_models import TextNCALM


def _router() -> HierarchicalGrowthRouter:
    source = TextNCALM(vocab_size=17, max_context=8, dim=8, heads=2, ffn_dim=12,
                       windows=(2, 3, 4), iterations=(1, 1, 1), carry_bias=2.0)
    upcycled = convert_textnca_to_upcycled(source, config=UpcyclingConfig(num_experts=4, top_k=1))
    return HierarchicalGrowthRouter(0, upcycled.stages[0].program_bank.router, 8)


def test_recursive_route_tree_is_explicit_and_reconstructable() -> None:
    router = _router()
    prototypes = torch.tensor([[1.0] + [0.0] * 7, [-1.0] + [0.0] * 7])
    router.add_split("s0-e2", "s0-e4", "s0-split0", prototypes)
    router.add_split("s0-e4", "s0-e5", "s0-split1", prototypes)
    structure = router.structure()
    restored = _router()
    restored.restore_structure(structure)
    assert structure == restored.structure()
    assert list(iter_leaves(router.roots[2])) == ["s0-e2", "s0-e4", "s0-e5"]


def test_binary_router_is_pointwise() -> None:
    router = BinaryLineageRouter(4)
    perception = torch.randn(2, 5, 4)
    baseline = router(perception)
    changed = perception.clone()
    changed[0, 0] += 3
    actual = router(changed)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[0, 0] = False
    torch.testing.assert_close(actual[mask], baseline[mask])
