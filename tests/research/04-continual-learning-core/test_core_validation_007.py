from __future__ import annotations

from pathlib import Path

import torch

from minicells.real_representation_007_config import CoreValidation007Config, smoke_config
from minicells.real_representation_007_core import (
    FunctionalModeCatalog,
    cut_interference_fraction,
    partition_modes,
    update_low_rank_basis,
)
from minicells.real_representation_007_experiment import FunctionalSystem, summarize_discovery

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "research" / "validations" / "core-007-functional-boundary-discovery" / "protocol.json"


def _catalog(dim: int = 4) -> FunctionalModeCatalog:
    return FunctionalModeCatalog(
        dim=dim,
        maximum_modes_per_address=4,
        maximum_write_rank=3,
        creation_cosine_threshold=0.8,
    )


def _add_mode(catalog: FunctionalModeCatalog, address: int, axis: int) -> int:
    z = torch.zeros(catalog.dim, dtype=torch.float64)
    z[axis] = 1.0
    w = torch.zeros(catalog.dim, catalog.dim, dtype=torch.float64)
    w[axis, axis] = 1.0
    mode, _, _ = catalog.locate_or_create(address=address, pooled_z=z, write_matrix=w)
    catalog.modes[mode].register_dependency(z[None, :], sequences=1)
    return mode


def test_write_basis_is_bounded_and_orthonormal() -> None:
    q = torch.zeros(9, 0, dtype=torch.float64)
    for i in range(9):
        v = torch.zeros(9, dtype=torch.float64)
        v[i] = 1.0
        q = update_low_rank_basis(q, v, maximum_rank=3)
    assert q.shape == (9, 3)
    assert torch.allclose(q.T @ q, torch.eye(3, dtype=torch.float64), atol=1e-10)


def test_micro_modes_split_opposed_write_demands() -> None:
    catalog = _catalog()
    z = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    plus = torch.eye(4, dtype=torch.float64)
    minus = -torch.eye(4, dtype=torch.float64)
    a, created_a, _ = catalog.locate_or_create(address=0, pooled_z=z, write_matrix=plus)
    b, created_b, similarity = catalog.locate_or_create(address=0, pooled_z=z, write_matrix=minus)
    assert created_a and created_b
    assert a != b
    assert similarity == 1.0  # newly created mode reports self identity


def test_interference_partition_places_damage_across_cut() -> None:
    catalog = _catalog()
    modes = [_add_mode(catalog, address=i, axis=i // 2) for i in range(4)]
    left, right = partition_modes(
        "interference_cut", modes, catalog, trigger_mode=modes[0]
    )
    assert left and right
    assert set(left).isdisjoint(right)
    assert set(left) | set(right) == set(modes)
    assert cut_interference_fraction(modes, left, right, catalog) >= 0.5


def test_functional_split_is_parameter_preserving_and_covariance_conserving() -> None:
    system = FunctionalSystem.initialize(
        dim=4,
        base_address_owner={0: 0, 1: 0, 2: 0, 3: 0},
        base_cells=1,
        certificate_energy=0.995,
        maximum_modes_per_address=4,
        maximum_write_rank=3,
        mode_creation_cosine_threshold=0.8,
    )
    # Use the system's catalog so ownership and dependency state are shared.
    modes = [_add_mode(system.catalog, address=i, axis=i // 2) for i in range(4)]
    for mode in modes:
        system.ensure_mode_owner(mode)
    system.cells[0].a = torch.arange(16, dtype=torch.float64).reshape(4, 4) / 10
    before_a = system.cells[0].a.clone()
    before_cov = system.cell_covariance(0).clone()
    record = system.split(
        parent=0,
        candidate="interference_cut",
        trigger_mode=modes[0],
        transaction=3,
    )
    child = int(record["child_id"])
    assert torch.equal(system.cells[0].a, before_a)
    assert torch.equal(system.cells[child].a, before_a)
    after_cov = system.cell_covariance(0) + system.cell_covariance(child)
    assert torch.allclose(before_cov, after_cov, atol=1e-12)
    assert set(system.modes_for_cell(0)).isdisjoint(system.modes_for_cell(child))


def test_protocol_keeps_discovery_and_confirmation_disjoint() -> None:
    cfg = CoreValidation007Config.from_protocol(PROTOCOL)
    assert cfg.discovery_seeds == (80701, 80702)
    assert cfg.confirmation_seeds == (80711, 80712, 80713)
    assert not (set(cfg.discovery_seeds) & set(cfg.confirmation_seeds))
    smoke = smoke_config(cfg)
    assert smoke.base.cell_dim <= 8
    assert smoke.maximum_modes_per_address <= 2


def test_discovery_selection_is_deterministic() -> None:
    cfg = CoreValidation007Config.from_protocol(PROTOCOL)
    rows = []
    for seed in cfg.discovery_seeds:
        candidates = []
        for i, name in enumerate(cfg.boundary_candidates):
            candidates.append(
                {
                    "candidate": name,
                    "median_interference_cut_fraction": 0.2 + 0.1 * i,
                    "median_balance": 0.5,
                    "routing_agreement": 0.8,
                    "soft_top2_coverage": 0.9,
                    "selection_score": 0.55 * (0.2 + 0.1 * i) + 0.3 * 0.8 + 0.15 * 0.5,
                }
            )
        rows.append({"seed": seed, "candidate_rows": candidates})
    decision = summarize_discovery(rows, cfg)
    assert decision["status"] == "FUNCTIONAL_BOUNDARY_DISCOVERY_COMPLETED"
    assert decision["provisional_winner"] == "interference_cut"
    assert decision["scientific_decision"] is False
