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


def test_dynamic_split_inherits_root_router_device_and_dtype() -> None:
    # float64 catches placement drift even on CPU; CUDA additionally exercises
    # the exact failure mode seen in the formal Kaggle birth.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router = _router().to(device=device, dtype=torch.float64)
    prototypes = torch.tensor(
        [[1.0] + [0.0] * 7, [-1.0] + [0.0] * 7],
        device=device,
        dtype=torch.float64,
    )
    split = router.add_split("s0-e2", "s0-e4", "s0-split0", prototypes)
    assert split.prototypes.device == device
    assert split.prototypes.dtype == torch.float64
    perception = torch.randn(2, 5, 8, device=device, dtype=torch.float64)
    router.route_with_details(perception)

    structure = router.structure()
    restored = _router().to(device=device, dtype=torch.float64)
    restored.restore_structure(structure)
    restored_split = restored.split_routers["s0-split0"]
    assert restored_split.prototypes.device == device
    assert restored_split.prototypes.dtype == torch.float64
    restored.route_with_details(perception)
