from __future__ import annotations

import torch

from .clm_routing import CLMRoutingConfig
from .language_models import TextNCALM
from .sparse_cellular_textnca import SparseCellularTextNCA


def convert_textnca_to_sparse_cellular(
    model: TextNCALM,
    *,
    num_programs: int = 8,
    receptor_dim: int | None = None,
    phenotype_dim: int = 0,
) -> SparseCellularTextNCA:
    dim = model.token_embedding.embedding_dim
    config = CLMRoutingConfig(
        num_programs=num_programs,
        receptor_dim=receptor_dim or min(32, max(1, dim // 4)),
        phenotype_dim=phenotype_dim,
    )
    converted = SparseCellularTextNCA(model, config)
    converted.train(model.training)
    return converted


@torch.no_grad()
def verify_dense_equivalence(
    teacher: TextNCALM,
    student: SparseCellularTextNCA,
    input_ids: torch.Tensor,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    student.set_routing_mode("dense")
    expected = teacher(input_ids)
    actual = student(input_ids)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=rtol, atol=atol)
    if len(actual.stage_logits) != len(expected.stage_logits):
        raise AssertionError("stage supervision output count changed during conversion")
    for got, want in zip(actual.stage_logits, expected.stage_logits):
        torch.testing.assert_close(got, want, rtol=rtol, atol=atol)
    student.conversion_metadata["dense_equivalence_verified"] = True
