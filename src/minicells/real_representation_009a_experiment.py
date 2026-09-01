"""Core Validation 009A — Factorized Functional Coordinates.

Geometry-only test of shared left/output-effect and right/input-condition
subspaces for normalized frozen-Pythia write demands. No routing, certificate,
growth, or continual-learning mechanism is present in this module.
"""
from __future__ import annotations

import math
import statistics
from typing import Any

import torch

_EPS = 1e-12


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs)) if xs else 0.0


def normalize_write(write: torch.Tensor) -> torch.Tensor:
    x = write.detach().cpu().to(torch.float64)
    return x / max(float(torch.linalg.norm(x).item()), _EPS)


def make_rows(projected: list[Any], signatures: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seq in projected:
        rows.append(
            {
                "token_sha256": seq.token_sha256,
                "partition": str(seq.partition),
                "z": seq.z.detach().cpu().to(torch.float64),
                "g": normalize_write(signatures[seq.token_sha256].write_matrix),
            }
        )
    return rows


def _covariance_bases(train: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dim = int(train[0]["g"].shape[0])
    left_cov = torch.zeros(dim, dim, dtype=torch.float64)
    right_cov = torch.zeros(dim, dim, dtype=torch.float64)
    for row in train:
        g = row["g"]
        left_cov += g @ g.T
        right_cov += g.T @ g
    left_cov /= max(len(train), 1)
    right_cov /= max(len(train), 1)

    lvals, lvecs = torch.linalg.eigh(left_cov)
    rvals, rvecs = torch.linalg.eigh(right_cov)
    li = torch.argsort(lvals, descending=True)
    ri = torch.argsort(rvals, descending=True)
    lvals, lvecs = lvals[li].clamp_min(0.0), lvecs[:, li]
    rvals, rvecs = rvals[ri].clamp_min(0.0), rvecs[:, ri]
    return lvecs.contiguous(), rvecs.contiguous(), lvals, rvals


def _cumulative_energy(values: torch.Tensor, dims: list[int]) -> list[dict[str, Any]]:
    total = max(float(values.sum().item()), _EPS)
    csum = torch.cumsum(values, dim=0) / total
    return [
        {
            "dimension": int(d),
            "cumulative_energy": float(csum[min(int(d), len(csum)) - 1].item()),
        }
        for d in dims
    ]


def _fro_residual(g: torch.Tensor, approx: torch.Tensor) -> float:
    return float(torch.linalg.norm(g - approx).item() / max(float(torch.linalg.norm(g).item()), _EPS))


def _action_residual(z: torch.Tensor, g: torch.Tensor, approx: torch.Tensor) -> float:
    target = z @ g.T
    pred = z @ approx.T
    return float(torch.linalg.norm(target - pred).item() / max(float(torch.linalg.norm(target).item()), _EPS))


def _project_left(g: torch.Tensor, left: torch.Tensor, m: int) -> torch.Tensor:
    l = left[:, :m]
    return l @ (l.T @ g)


def _project_right(g: torch.Tensor, right: torch.Tensor, n: int) -> torch.Tensor:
    r = right[:, :n]
    return (g @ r) @ r.T


def _project_two_sided(g: torch.Tensor, left: torch.Tensor, right: torch.Tensor, m: int, n: int) -> torch.Tensor:
    l, r = left[:, :m], right[:, :n]
    core = l.T @ g @ r
    return l @ core @ r.T


def _rank1(g: torch.Tensor) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(g, full_matrices=False)
    return (u[:, :1] * s[:1]) @ vh[:1]


def _summary(subset: list[dict[str, Any]], approx_fn: Any) -> dict[str, float | int]:
    fro, action = [], []
    for row in subset:
        approx = approx_fn(row["g"])
        fro.append(_fro_residual(row["g"], approx))
        action.append(_action_residual(row["z"], row["g"], approx))
    return {
        "count": len(subset),
        "median_frobenius_residual": _median(fro),
        "mean_frobenius_residual": _mean(fro),
        "median_local_action_residual": _median(action),
        "mean_local_action_residual": _mean(action),
    }


def run_geometry(rows: list[dict[str, Any]], protocol: dict[str, Any], *, seed: int) -> dict[str, Any]:
    train = [r for r in rows if r["partition"] == "train"]
    eval_rows = [r for r in rows if r["partition"] == "eval"]
    geom = protocol["write_geometry"]
    dims = [int(x) for x in geom["dimension_grid"]]
    splits = [(int(a), int(b)) for a, b in geom["budget_matched_splits"]]
    left, right, lvals, rvals = _covariance_bases(train)

    left_rows: list[dict[str, Any]] = []
    right_rows: list[dict[str, Any]] = []
    for d in dims:
        for partition, subset in (("train", train), ("eval", eval_rows)):
            left_rows.append(
                {
                    "seed": seed,
                    "partition": partition,
                    "dimension": d,
                    **_summary(subset, lambda g, d=d: _project_left(g, left, d)),
                }
            )
            right_rows.append(
                {
                    "seed": seed,
                    "partition": partition,
                    "dimension": d,
                    **_summary(subset, lambda g, d=d: _project_right(g, right, d)),
                }
            )

    # Full landscape is heldout-only: it is diagnostic and never used for the
    # frozen budget winner except where (m,n) is also an explicit budget split.
    landscape: list[dict[str, Any]] = []
    for m in dims:
        for n in dims:
            landscape.append(
                {
                    "seed": seed,
                    "partition": "eval",
                    "left_dim": m,
                    "right_dim": n,
                    "basis_parameter_count": 64 * (m + n),
                    **_summary(eval_rows, lambda g, m=m, n=n: _project_two_sided(g, left, right, m, n)),
                }
            )

    budget_rows: list[dict[str, Any]] = []
    for m, n in splits:
        for partition, subset in (("train", train), ("eval", eval_rows)):
            budget_rows.append(
                {
                    "seed": seed,
                    "partition": partition,
                    "left_dim": m,
                    "right_dim": n,
                    "basis_parameter_count": 64 * (m + n),
                    **_summary(subset, lambda g, m=m, n=n: _project_two_sided(g, left, right, m, n)),
                }
            )

    rank1_rows = []
    for partition, subset in (("train", train), ("eval", eval_rows)):
        rank1_rows.append(
            {
                "seed": seed,
                "partition": partition,
                **_summary(subset, _rank1),
            }
        )

    return {
        "format": "minicells.core-validation.factorized-functional-coordinates-seed.v1",
        "experiment_id": "core-validation-009a",
        "seed": seed,
        "scientific_decision": False,
        "left_spectrum": _cumulative_energy(lvals, dims),
        "right_spectrum": _cumulative_energy(rvals, dims),
        "left_only": left_rows,
        "right_only": right_rows,
        "two_sided_landscape": landscape,
        "budget_splits": budget_rows,
        "per_write_rank1": rank1_rows,
    }


def _budget_eval(run: dict[str, Any], m: int, n: int) -> dict[str, Any]:
    return next(
        row
        for row in run["budget_splits"]
        if row["partition"] == "eval" and int(row["left_dim"]) == m and int(row["right_dim"]) == n
    )


def _budget_train(run: dict[str, Any], m: int, n: int) -> dict[str, Any]:
    return next(
        row
        for row in run["budget_splits"]
        if row["partition"] == "train" and int(row["left_dim"]) == m and int(row["right_dim"]) == n
    )


def summarize_discovery(runs: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    expected = [int(x) for x in protocol["discovery"]["seeds"]]
    completed = sorted(int(r["seed"]) for r in runs)
    missing = [s for s in expected if s not in completed]
    splits = [(int(a), int(b)) for a, b in protocol["write_geometry"]["budget_matched_splits"]]
    candidates: list[dict[str, Any]] = []
    for m, n in splits:
        vals = [float(_budget_eval(run, m, n)["median_local_action_residual"]) for run in runs]
        if not vals:
            continue
        candidates.append(
            {
                "left_dim": m,
                "right_dim": n,
                "mean_eval_local_action_residual": _mean(vals),
                "worst_eval_local_action_residual": max(vals),
                "per_seed_eval_local_action_residual": vals,
                "basis_parameter_count": 64 * (m + n),
            }
        )
    winner = None
    viable = False
    if not missing and candidates:
        winner = min(
            candidates,
            key=lambda r: (
                float(r["mean_eval_local_action_residual"]),
                float(r["worst_eval_local_action_residual"]),
                abs(int(r["left_dim"]) - int(r["right_dim"])),
                int(r["left_dim"]),
            ),
        )
        ref = float(protocol["discovery"]["viability_reference"])
        viable = all(float(x) <= ref for x in winner["per_seed_eval_local_action_residual"])
    return {
        "status": "FUNCTIONAL_COORDINATE_DISCOVERY_COMPLETED" if not missing else "DISCOVERY_INCOMPLETE",
        "scientific_decision": False,
        "completed_seeds": completed,
        "missing_seeds": missing,
        "candidate_summary": candidates,
        "provisional_winner": None if winner is None else {"left_dim": int(winner["left_dim"]), "right_dim": int(winner["right_dim"])},
        "winner_metrics": winner,
        "winner_meets_viability": viable,
        "confirmation_allowed": bool(not missing and viable),
    }


def confirmation_gate_row(run: dict[str, Any], protocol: dict[str, Any], *, left_dim: int, right_dim: int) -> dict[str, Any]:
    eval_row = _budget_eval(run, left_dim, right_dim)
    train_row = _budget_train(run, left_dim, right_dim)
    rank1 = next(r for r in run["per_write_rank1"] if r["partition"] == "eval")
    gates = protocol["confirmation"]["gates"]
    gap = max(
        0.0,
        float(eval_row["median_local_action_residual"]) - float(train_row["median_local_action_residual"]),
    )
    checks = {
        "heldout_action": float(eval_row["median_local_action_residual"]) <= float(gates["maximum_heldout_median_local_action_residual"]),
        "heldout_frobenius": float(eval_row["median_frobenius_residual"]) <= float(gates["maximum_heldout_median_frobenius_residual"]),
        "generalization_gap": gap <= float(gates["maximum_train_eval_local_action_gap"]),
        "rank1_identity_guard": float(rank1["median_local_action_residual"]) <= float(gates["maximum_per_write_rank1_oracle_action_residual"]),
        "budget": int(eval_row["basis_parameter_count"]) == int(gates["required_basis_parameter_count"]),
    }
    return {
        "seed": int(run["seed"]),
        "left_dim": left_dim,
        "right_dim": right_dim,
        "train_median_local_action_residual": float(train_row["median_local_action_residual"]),
        "eval_median_local_action_residual": float(eval_row["median_local_action_residual"]),
        "eval_median_frobenius_residual": float(eval_row["median_frobenius_residual"]),
        "train_eval_local_action_gap": gap,
        "rank1_eval_local_action_residual": float(rank1["median_local_action_residual"]),
        "basis_parameter_count": int(eval_row["basis_parameter_count"]),
        **checks,
        "pass": all(checks.values()),
    }


def summarize_confirmation(runs: list[dict[str, Any]], protocol: dict[str, Any], *, left_dim: int, right_dim: int) -> dict[str, Any]:
    expected = [int(x) for x in protocol["confirmation"]["seeds"]]
    completed = sorted(int(r["seed"]) for r in runs)
    missing = [s for s in expected if s not in completed]
    rows = [confirmation_gate_row(r, protocol, left_dim=left_dim, right_dim=right_dim) for r in runs]
    if missing:
        return {
            "status": "CONFIRMATION_INCOMPLETE",
            "scientific_decision": False,
            "supported": None,
            "completed_seeds": completed,
            "missing_seeds": missing,
            "locked_split": {"left_dim": left_dim, "right_dim": right_dim},
            "gate_rows": rows,
        }
    supported = all(bool(r["pass"]) for r in rows)
    conf = protocol["confirmation"]
    return {
        "status": conf["positive_status"] if supported else conf["negative_status"],
        "scientific_decision": True,
        "supported": supported,
        "completed_seeds": completed,
        "missing_seeds": [],
        "locked_split": {"left_dim": left_dim, "right_dim": right_dim},
        "passed_seeds": sum(bool(r["pass"]) for r in rows),
        "total_formal_seeds": len(expected),
        "gate_rows": rows,
    }
