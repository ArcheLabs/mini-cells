from __future__ import annotations

import json
from pathlib import Path

import torch

from minicells.real_representation_008_experiment import (
    CertifiedAtom,
    FunctionalTarget,
    VariantSpec,
    _certificate_from_covariance,
    _constraint_violation,
    _safe_delta,
    reconstruct,
    sparse_coefficients,
    train_variant,
)

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "research" / "validations" / "core-008-certified-functional-atoms" / "protocol.json"


def _atom(matrix: torch.Tensor, atom_id: int) -> CertifiedAtom:
    dim = matrix.shape[0]
    return CertifiedAtom(
        atom_id=atom_id,
        matrix=matrix.clone().to(torch.float64),
        rank_units=1,
        max_rank=4,
        covariance=torch.zeros(dim, dim, dtype=torch.float64),
        q=torch.zeros(dim, 0, dtype=torch.float64),
        key_sum=torch.zeros(dim, dtype=torch.float64),
    )


def test_safe_low_rank_delta_annihilates_certificate() -> None:
    torch.manual_seed(8)
    dim = 8
    q, _ = torch.linalg.qr(torch.randn(dim, 3, dtype=torch.float64))
    target = torch.randn(dim, dim, dtype=torch.float64)
    delta = _safe_delta(target, q, rank=2)
    assert torch.linalg.matrix_rank(delta, tol=1e-9) <= 2
    assert _constraint_violation(delta, q) <= 1e-10


def test_sparse_composition_recovers_known_signed_basis() -> None:
    dim = 4
    a = torch.zeros(dim, dim, dtype=torch.float64); a[0, 0] = 1.0
    b = torch.zeros(dim, dim, dtype=torch.float64); b[1, 2] = 1.0
    atoms = [_atom(a, 0), _atom(b, 1)]
    target = 2.0 * a - 0.5 * b
    coeff = sparse_coefficients(target, atoms, top_k=2)
    got = reconstruct(coeff, atoms, dim)
    assert torch.allclose(got, target, atol=1e-10)


def test_certificate_energy_is_bounded_subspace_state() -> None:
    cov = torch.diag(torch.tensor([10.0, 3.0, 0.1, 0.01], dtype=torch.float64))
    q = _certificate_from_covariance(cov, 0.95)
    assert q.shape[0] == 4
    assert 1 <= q.shape[1] < 4
    assert torch.allclose(q.T @ q, torch.eye(q.shape[1], dtype=torch.float64), atol=1e-10)


def test_online_atom_training_respects_rank_and_factor_budget() -> None:
    dim = 4
    targets = []
    for i in range(6):
        g = torch.zeros(dim, dim, dtype=torch.float64)
        g[i % dim, (i + 1) % dim] = 1.0
        z = torch.eye(dim, dtype=torch.float64)
        targets.append(
            FunctionalTarget(
                token_sha256=str(i),
                partition="train",
                pooled_z=z.mean(dim=0),
                z=z,
                target=g,
            )
        )
    spec = VariantSpec(
        name="rank2_atoms",
        maximum_atoms=8,
        maximum_rank_per_atom=2,
        adaptive_append_rank=False,
    )
    state = train_variant(
        targets,
        spec,
        certificate_energy=0.95,
        factor_budget=2 * dim * 4,
        top_k=2,
        target_residual=0.2,
        minimum_update_improvement=0.01,
        max_growth_actions=2,
    )
    assert state.used_factor_scalars <= state.factor_budget
    assert sum(a.rank_units for a in state.atoms) <= 4
    assert all(a.rank_units <= 2 for a in state.atoms)


def test_protocol_compares_rank_variants_under_one_budget() -> None:
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert p["budget"]["conceptual_factor_scalars"] == 4096
    assert p["composition"]["maximum_active_atoms"] == 4
    assert p["replication"]["formal_seeds"] == [80821, 80822, 80823]
    for name in ("rank1_atoms", "rank2_atoms", "rank4_atoms", "adaptive_atoms"):
        assert p["variants"][name]["maximum_atoms"] == 32
    assert p["secondary_metrics"][-1].startswith("whole-model NLL")
