import math

import torch
from torch.nn import functional as F

from minicells.clm_growth import ProgressiveGrowthCLM
from minicells.growth_router import (
    BinaryLineageRouter,
    HierarchicalGrowthRouter,
    RouteLeaf,
    RouteSplit,
    iter_leaves,
    straight_through_top1,
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


def test_straight_through_top1_is_forward_exact_and_keeps_soft_gradient() -> None:
    logits = torch.tensor(
        [[0.125, -0.75, 0.333], [1.1, 1.2, -0.4]],
        dtype=torch.float32,
        requires_grad=True,
    )
    gates, probabilities, indices = straight_through_top1(logits)
    expected = F.one_hot(indices, num_classes=logits.shape[-1]).to(logits.dtype)
    assert torch.equal(gates.detach(), expected)
    assert torch.equal(gates.detach().unique().sort().values, torch.tensor([0.0, 1.0]))

    weights = torch.tensor([[1.0, 2.0, -1.0], [-2.0, 0.5, 3.0]])
    (gates * weights).sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0
    assert torch.isfinite(probabilities).all()


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


def test_birth_rollback_restores_tree_without_moduledict_pop_error() -> None:
    model = _growth_model()
    bank = model.stages[0].program_bank
    before = bank.router.structure()
    child_id, _parent, _router_module, _centroids = bank.add_birth(
        "s0-e0",
        torch.randn(512, 8),
    )
    split_id = bank.split_by_child[child_id]
    assert child_id in bank.experts
    assert split_id in bank.router.split_routers

    bank.remove_birth(child_id, split_id, before)
    assert child_id not in bank.experts
    assert split_id not in bank.router.split_routers
    assert bank.router.structure() == before
    # Rollback is intentionally idempotent for abort/error paths.
    bank.remove_birth(child_id, split_id, before)


def test_function_preserving_birth_passes_parity_on_runtime_device() -> None:
    requested_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _growth_model().to(requested_device)
    runtime_device = next(model.parameters()).device
    inputs = torch.randint(0, 17, (2, 6), device=runtime_device)
    targets = torch.randint(0, 17, (2, 6), device=runtime_device)
    event = model.birth(
        stage=0,
        parent_id="s0-e0",
        routed_perceptions=torch.randn(512, 8, device=runtime_device),
        token=500_000,
        validation_inputs=inputs,
        validation_targets=targets,
    )
    parity = event["parity"]
    assert parity["status"] == "CLM_GROWTH_EQUIVALENCE"
    assert parity["non_parent_root_routes_unchanged"] is True
    assert parity["child_parameters_equal_parent"] is True


def test_newborn_diagnostics_rehydrates_cpu_perceptions_to_router_device() -> None:
    requested_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _growth_model().to(requested_device)
    # Resolve the concrete runtime device (e.g. cuda:0) from the model itself.
    # torch.device("cuda") has index=None and does not necessarily compare equal
    # to tensors/modules resident on torch.device("cuda:0").
    runtime_device = next(model.parameters()).device

    # Pick a parent that this exact diagnostic batch actually visits.  The test
    # must verify device rehydration, not depend on a randomly chosen root leaf
    # receiving traffic.
    inputs = torch.randint(0, 17, (2, 6), device=runtime_device)
    targets = torch.randint(0, 17, (2, 6), device=runtime_device)
    with torch.no_grad():
        _output, pre_stats = model(inputs, return_stats=True)
    root_indices = pre_stats.root_routes[0].reshape(-1)
    counts = torch.bincount(root_indices, minlength=4)
    parent_index = int(counts.argmax().item())
    assert int(counts[parent_index].item()) > 0
    parent_id = f"s0-e{parent_index}"

    event = model.birth(
        stage=0,
        parent_id=parent_id,
        routed_perceptions=torch.randn(512, 8, device=runtime_device),
        token=500_000,
    )
    child_id = str(event["child"])
    diagnostics = newborn_causal_diagnostics(
        model,
        [(inputs, targets)],
        stage=0,
        parent_id=parent_id,
        child_id=child_id,
    )
    assert math.isfinite(diagnostics["router_logit_variance"])

    # Collection remains CPU-backed by design; diagnostics must rehydrate only
    # the transient concatenated batch to the split router's concrete runtime
    # device/dtype and must not retain the GPU copy in the pressure cache.
    bank = model.stages[0].program_bank
    split_id = bank.split_by_child[child_id]
    split_router = bank.router.split_routers[split_id]
    assert split_router.prototypes.device == runtime_device
    collected = [
        item
        for expert_id in (parent_id, child_id)
        for item in bank.last_perceptions.get(expert_id, [])
    ]
    assert collected
    assert all(item.device.type == "cpu" for item in collected)
