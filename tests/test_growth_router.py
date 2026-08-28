import math

import torch

from minicells.clm_growth import ProgressiveGrowthCLM
from minicells.growth_router import (
    BinaryLineageRouter,
    HierarchicalGrowthRouter,
    RouteLeaf,
    RouteSplit,
    iter_leaves,
)
from minicells.growth_validation import newborn_causal_diagnostics
from minicells.upcycled_cellular_textnca import UpcyclingConfig, convert_textnca_to_upcycled
from minicells.language_models import TextNCALM


def _router() -> HierarchicalGrowthRouter:
    source = TextNCALM(vocab_size=17, max_context=8, dim=8, heads=2, ffn_dim=12,
                       windows=(2, 3, 4), iterations=(1, 1, 1), carry_bias=2.0)
    upcycled = convert_textnca_to_upcycled(source, config=UpcyclingConfig(num_experts=4, top_k=1))
    return HierarchicalGrowthRouter(0, upcycled.stages[0].program_bank.router, 8)


def _growth_model() -> ProgressiveGrowthCLM:
    source = TextNCALM(vocab_size=17, max_context=8, dim=8, heads=2, ffn_dim=12,
                       windows=(2, 3, 4), iterations=(1, 1, 1), carry_bias=2.0)
    upcycled = convert_textnca_to_upcycled(source, config=UpcyclingConfig(num_experts=4, top_k=1))
    return ProgressiveGrowthCLM(upcycled)


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
    requested_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router = _router().to(device=requested_device, dtype=torch.float64)
    root_parameter = next(router.root_router.parameters())
    actual_device = root_parameter.device
    prototypes = torch.tensor(
        [[1.0] + [0.0] * 7, [-1.0] + [0.0] * 7],
        device=actual_device,
        dtype=torch.float64,
    )
    split = router.add_split("s0-e2", "s0-e4", "s0-split0", prototypes)
    assert split.prototypes.device == actual_device
    assert split.prototypes.dtype == root_parameter.dtype == torch.float64
    perception = torch.randn(2, 5, 8, device=actual_device, dtype=torch.float64)
    router.route_with_details(perception)

    structure = router.structure()
    restored = _router().to(device=requested_device, dtype=torch.float64)
    restored_root_parameter = next(restored.root_router.parameters())
    restored.restore_structure(structure)
    restored_split = restored.split_routers["s0-split0"]
    assert restored_split.prototypes.device == restored_root_parameter.device
    assert restored_split.prototypes.dtype == restored_root_parameter.dtype == torch.float64
    restored.route_with_details(perception.to(restored_root_parameter.device))


def test_newborn_diagnostics_rehydrates_cpu_perceptions_to_router_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _growth_model().to(device)
    event = model.birth(
        stage=0,
        parent_id="s0-e0",
        routed_perceptions=torch.randn(512, 8, device=device),
        token=500_000,
    )
    inputs = torch.randint(0, 17, (2, 6), device=device)
    targets = torch.randint(0, 17, (2, 6), device=device)
    diagnostics = newborn_causal_diagnostics(
        model,
        [(inputs, targets)],
        stage=0,
        parent_id="s0-e0",
        child_id=str(event["child"]),
    )
    assert math.isfinite(diagnostics["router_logit_variance"])
    # Collection remains CPU-backed by design; diagnostics must rehydrate only
    # the transient concatenated batch rather than retaining GPU perceptions.
    bank = model.stages[0].program_bank
    collected = [
        item
        for expert_id in ("s0-e0", str(event["child"]))
        for item in bank.last_perceptions.get(expert_id, [])
    ]
    assert collected
    assert all(item.device.type == "cpu" for item in collected)
