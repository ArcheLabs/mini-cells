"""M3W-0: checkpoint-only write-drift counterfactual restoration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .native_clm_m3l2 import OnlineAddressNativeCLM
from .native_clm_v0 import NativeCLM


@dataclass(frozen=True)
class M3W0Thresholds:
    maximum_all_lineage_vs_m1_loss_delta_each_domain: float = 1e-4
    minimum_all_lineage_A_excess_recovery_fraction: float = 0.95
    root_write_dominant_fraction: float = 0.60
    descendant_write_dominant_fraction: float = 0.60
    minimum_new_domain_gain_retention_each_for_children_carry_plasticity: float = 0.70


def root_ancestor(model: OnlineAddressNativeCLM, cell_id: int) -> int:
    """Return the original M1 root ancestor of any lineage Cell."""
    current = int(cell_id)
    seen: set[int] = set()
    while current >= model.lineage_root_count:
        if current in seen:
            raise RuntimeError("cycle in lineage ancestry")
        seen.add(current)
        parent = int(model.cellular.cells[current].parent_id.item())
        if not (0 <= parent < current):
            raise RuntimeError(f"invalid parent {parent} for cell {current}")
        current = parent
    return current


@torch.no_grad()
def restore_operator_groups(
    model: OnlineAddressNativeCLM,
    m1: NativeCLM,
    *,
    restore_roots: bool,
    restore_descendants: bool,
) -> dict[str, Any]:
    """Restore selected operator groups to exact M1 root-ancestor weights."""
    if m1.cell_count != model.lineage_root_count:
        raise ValueError("M1 root count does not match lineage root count")
    restored: list[int] = []
    for cell_id, cell in enumerate(model.cellular.cells):
        is_root = cell_id < model.lineage_root_count
        if (is_root and not restore_roots) or ((not is_root) and not restore_descendants):
            continue
        root_id = root_ancestor(model, cell_id)
        source = m1.cellular.cells[root_id].weight
        cell.weight.copy_(source.to(device=cell.weight.device, dtype=cell.weight.dtype))
        restored.append(cell_id)
    return {
        "restore_roots": bool(restore_roots),
        "restore_descendants": bool(restore_descendants),
        "restored_cell_ids": restored,
    }


def _loss(matrix: dict[str, dict[str, Any]], domain: str) -> float:
    return float(matrix[domain]["loss"])


def _gain_ratio(
    m1: dict[str, dict[str, Any]],
    final: dict[str, dict[str, Any]],
    counterfactual: dict[str, dict[str, Any]],
    domain: str,
) -> float:
    base = _loss(m1, domain) - _loss(final, domain)
    if base <= 1e-12:
        raise ValueError(f"final checkpoint has no positive adaptation gain on {domain}")
    return (_loss(m1, domain) - _loss(counterfactual, domain)) / base


def analyze_factorial(
    *,
    seed: int,
    m1_matrix: dict[str, dict[str, Any]],
    final_matrix: dict[str, dict[str, Any]],
    root_restore_matrix: dict[str, dict[str, Any]],
    descendant_root_restore_matrix: dict[str, dict[str, Any]],
    all_lineage_restore_matrix: dict[str, dict[str, Any]],
    thresholds: M3W0Thresholds,
) -> dict[str, Any]:
    """Compute exact two-factor Shapley attribution for root/descendant writes."""
    domains = ("A", "B", "C", "D")
    identity_deltas = {
        domain: abs(_loss(all_lineage_restore_matrix, domain) - _loss(m1_matrix, domain))
        for domain in domains
    }

    l00 = _loss(all_lineage_restore_matrix, "A")
    l10 = _loss(descendant_root_restore_matrix, "A")
    l01 = _loss(root_restore_matrix, "A")
    l11 = _loss(final_matrix, "A")
    lm1 = _loss(m1_matrix, "A")

    root_shapley = 0.5 * ((l10 - l00) + (l11 - l01))
    descendant_shapley = 0.5 * ((l01 - l00) + (l11 - l10))
    total = root_shapley + descendant_shapley
    if total <= 1e-12:
        root_fraction = float("nan")
        descendant_fraction = float("nan")
    else:
        root_fraction = root_shapley / total
        descendant_fraction = descendant_shapley / total

    final_excess = l11 - lm1
    if final_excess <= 1e-12:
        all_recovery = float("nan")
    else:
        all_recovery = (l11 - l00) / final_excess

    root_restore_gain_retention = {
        domain: _gain_ratio(m1_matrix, final_matrix, root_restore_matrix, domain)
        for domain in ("B", "C", "D")
    }
    descendant_restore_gain_retention = {
        domain: _gain_ratio(m1_matrix, final_matrix, descendant_root_restore_matrix, domain)
        for domain in ("B", "C", "D")
    }

    identity_ok = (
        max(identity_deltas.values())
        <= thresholds.maximum_all_lineage_vs_m1_loss_delta_each_domain
        and all_recovery >= thresholds.minimum_all_lineage_A_excess_recovery_fraction
    )

    return {
        "seed": int(seed),
        "identity_ok": bool(identity_ok),
        "identity_loss_deltas": identity_deltas,
        "A_losses": {
            "M1": lm1,
            "00_all_lineage_restore": l00,
            "10_descendant_root_restore": l10,
            "01_root_restore": l01,
            "11_final": l11,
        },
        "A_all_lineage_excess_recovery_fraction": all_recovery,
        "A_root_shapley": root_shapley,
        "A_descendant_shapley": descendant_shapley,
        "A_root_fraction": root_fraction,
        "A_descendant_fraction": descendant_fraction,
        "root_restore_new_domain_gain_retention": root_restore_gain_retention,
        "descendant_restore_new_domain_gain_retention": descendant_restore_gain_retention,
        "matrices": {
            "m1": m1_matrix,
            "final": final_matrix,
            "root_restore": root_restore_matrix,
            "descendant_root_restore": descendant_root_restore_matrix,
            "all_lineage_restore": all_lineage_restore_matrix,
        },
    }


def classify_results(results: list[dict[str, Any]], thresholds: M3W0Thresholds) -> str:
    if not results or any(not result["identity_ok"] for result in results):
        return "INCONCLUSIVE_IDENTITY"

    if all(float(result["A_root_fraction"]) >= thresholds.root_write_dominant_fraction for result in results):
        children_carry = all(
            all(
                float(value)
                >= thresholds.minimum_new_domain_gain_retention_each_for_children_carry_plasticity
                for value in result["root_restore_new_domain_gain_retention"].values()
            )
            for result in results
        )
        return (
            "ROOT_WRITE_DOMINANT_CHILDREN_CARRY_PLASTICITY"
            if children_carry
            else "ROOT_WRITE_DOMINANT_TRANSFER_GAP"
        )

    if all(
        float(result["A_descendant_fraction"]) >= thresholds.descendant_write_dominant_fraction
        for result in results
    ):
        return "DESCENDANT_WRITE_DOMINANT"
    return "DISTRIBUTED_WRITE_DRIFT"
