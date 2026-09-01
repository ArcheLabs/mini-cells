"""Core Validation 009B-1 — Carrier Causal Sufficiency.

Causal test of the decomposition G ~= G_parallel + G_perp discovered after
Core 009A. The key invariant is causal magnitude: carrier and residual are
projected from the full normalized write and are NEVER renormalized before
using the same eta as the full intervention.
"""
from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .real_representation_006_experiment import ProjectedSequence

_EPS = 1e-12


@dataclass(frozen=True)
class CausalSequence:
    partition: str
    source: str
    token_sha256: str
    hidden: torch.Tensor
    labels: torch.Tensor
    z: torch.Tensor
    ghat: torch.Tensor
    raw_write_norm: float

    @property
    def tokens(self) -> int:
        return int(self.labels.numel())


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _mean(xs: list[float]) -> float:
    return float(statistics.fmean(xs)) if xs else 0.0


def _p90(xs: list[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(float(x) for x in xs)
    idx = max(0, min(len(ys) - 1, math.ceil(0.90 * len(ys)) - 1))
    return float(ys[idx])


def extract_causal_sequences(
    sequences: list[ProjectedSequence],
    u: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 16,
) -> list[CausalSequence]:
    """Extract the exact frozen-path write signature plus causal eval tensors."""
    out: list[CausalSequence] = []
    weight = lm_head_weight.to(device=device, dtype=torch.float32)
    u_dev = u.to(device=device, dtype=torch.float32)
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        hidden = torch.stack([s.hidden for s in batch]).to(device=device, dtype=torch.float32)
        labels = torch.stack([s.labels for s in batch]).to(device=device, dtype=torch.long)
        hidden_req = hidden.detach().requires_grad_(True)
        logits = F.linear(hidden_req, weight)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="sum",
        )
        grad_h = torch.autograd.grad(loss, hidden_req)[0]
        q = torch.einsum("bth,hd->btd", grad_h, u_dev)
        for i, seq in enumerate(batch):
            z32 = seq.z.to(device=device, dtype=torch.float32)
            raw = torch.einsum("to,ti->oi", q[i], z32) / max(seq.tokens, 1)
            norm = float(torch.linalg.norm(raw).detach().cpu().item())
            ghat = raw / max(norm, _EPS)
            out.append(
                CausalSequence(
                    partition=str(seq.partition),
                    source=str(seq.source),
                    token_sha256=seq.token_sha256,
                    hidden=seq.hidden.detach().cpu().to(dtype=torch.float32),
                    labels=seq.labels.detach().cpu().to(dtype=torch.long),
                    z=seq.z.detach().cpu().to(dtype=torch.float64),
                    ghat=ghat.detach().cpu().to(dtype=torch.float64),
                    raw_write_norm=norm,
                )
            )
    return out


def analysis_sequences(sequences: list[CausalSequence]) -> list[CausalSequence]:
    """Exclude router-only examples after extraction so batching remains frozen."""
    return [s for s in sequences if s.partition in {"train", "eval"}]


def fit_train_carrier(sequences: list[CausalSequence]) -> torch.Tensor:
    train = [s.z for s in sequences if s.partition == "train"]
    if not train:
        raise ValueError("carrier fit requires train sequences")
    mean = torch.cat(train, dim=0).mean(dim=0)
    norm = float(torch.linalg.norm(mean).item())
    if norm <= _EPS:
        raise RuntimeError("train activation mean has zero norm")
    return (mean / norm).to(dtype=torch.float64)


def decompose_direction(
    ghat: torch.Tensor, carrier: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rr = torch.outer(carrier, carrier)
    parallel = ghat @ rr
    residual = ghat - parallel
    return parallel, residual


def _unit_delta_norm(seq: CausalSequence, direction: torch.Tensor) -> float:
    coeff = seq.z @ direction.T
    return float(torch.linalg.norm(coeff).item())


def eta_for_target_ratio(seq: CausalSequence, full_direction: torch.Tensor, rho: float) -> float:
    hidden_norm = float(torch.linalg.norm(seq.hidden.to(dtype=torch.float64)).item())
    delta_norm = _unit_delta_norm(seq, full_direction)
    if delta_norm <= _EPS:
        raise RuntimeError("full write has zero target hidden action")
    return float(rho) * hidden_norm / delta_norm


def _nlls(
    sequences: list[CausalSequence],
    direction: torch.Tensor | None,
    eta: float,
    u: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 8,
) -> list[float]:
    if not sequences:
        return []
    out: list[float] = []
    u_dev = u.to(device=device, dtype=torch.float32)
    weight = lm_head_weight.to(device=device, dtype=torch.float32)
    a = None if direction is None else (-float(eta) * direction).to(device=device, dtype=torch.float32)

    for start in range(0, len(sequences), batch_size):
        rows = sequences[start : start + batch_size]
        lengths = {int(s.labels.numel()) for s in rows}
        if len(lengths) != 1:
            for row in rows:
                out.extend(
                    _nlls([row], direction, eta, u, lm_head_weight, device=device, batch_size=1)
                )
            continue
        hidden = torch.stack([s.hidden for s in rows]).to(device=device, dtype=torch.float32)
        labels = torch.stack([s.labels for s in rows]).to(device=device, dtype=torch.long)
        if a is not None:
            z = torch.stack([s.z for s in rows]).to(device=device, dtype=torch.float32)
            coeff = torch.einsum("bti,oi->bto", z, a)
            delta_h = torch.einsum("btd,hd->bth", coeff, u_dev)
            hidden = hidden + delta_h
        with torch.no_grad():
            logits = F.linear(hidden, weight)
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape).mean(dim=1)
        out.extend(float(x) for x in losses.detach().cpu().tolist())
    return out


def baseline_nlls(
    sequences: list[CausalSequence],
    u: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, float]:
    vals = _nlls(sequences, None, 0.0, u, lm_head_weight, device=device)
    return {s.token_sha256: v for s, v in zip(sequences, vals)}


def run_discovery(
    sequences: list[CausalSequence],
    protocol: dict[str, Any],
    *,
    seed: int,
    u: torch.Tensor,
    lm_head_weight: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    eval_rows = [s for s in sequences if s.partition == "eval"]
    carrier = fit_train_carrier(sequences)
    base = baseline_nlls(eval_rows, u, lm_head_weight, device=device)
    scales = [float(x) for x in protocol["discovery"]["perturbation_ratio_grid"]]
    scale_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []

    for rho in scales:
        gains: list[float] = []
        norm_gains: list[float] = []
        half_errors: list[float] = []
        descent = 0
        for seq in eval_rows:
            eta = eta_for_target_ratio(seq, seq.ghat, rho)
            full_nll = _nlls([seq], seq.ghat, eta, u, lm_head_weight, device=device)[0]
            half_nll = _nlls([seq], seq.ghat, 0.5 * eta, u, lm_head_weight, device=device)[0]
            baseline = base[seq.token_sha256]
            gain = baseline - full_nll
            half_gain = baseline - half_nll
            linearity = abs(gain - 2.0 * half_gain) / max(abs(gain), _EPS)
            if gain > 0:
                descent += 1
            gains.append(gain)
            norm_gains.append(gain / max(abs(baseline), _EPS))
            half_errors.append(linearity)
            target_rows.append(
                {
                    "seed": int(seed),
                    "rho": rho,
                    "token_sha256": seq.token_sha256,
                    "source": seq.source,
                    "baseline_nll": baseline,
                    "full_nll": full_nll,
                    "half_nll": half_nll,
                    "full_gain": gain,
                    "half_gain": half_gain,
                    "normalized_full_gain": norm_gains[-1],
                    "half_step_linearity_error": linearity,
                    "eta": eta,
                }
            )
        scale_rows.append(
            {
                "seed": int(seed),
                "rho": rho,
                "count": len(eval_rows),
                "full_descent_fraction": descent / max(len(eval_rows), 1),
                "median_full_nll_gain": _median(gains),
                "median_full_normalized_nll_gain": _median(norm_gains),
                "median_half_step_linearity_error": _median(half_errors),
                "p90_half_step_linearity_error": _p90(half_errors),
            }
        )

    return {
        "format": "minicells.core-validation.carrier-causal-sufficiency-discovery-seed.v1",
        "experiment_id": protocol["experiment_id"],
        "seed": int(seed),
        "scientific_decision": False,
        "carrier_fit": {
            "train_sequence_count": sum(s.partition == "train" for s in sequences),
            "eval_sequence_count": len(eval_rows),
            "carrier_norm": float(torch.linalg.norm(carrier).item()),
        },
        "scale_summary": scale_rows,
        "target_rows": target_rows,
    }


def _peer_hash(target: CausalSequence, peer: CausalSequence, seed: int) -> str:
    raw = f"{seed}:{target.token_sha256}:{peer.token_sha256}".encode()
    return hashlib.sha256(raw).hexdigest()


def select_peers(
    target: CausalSequence,
    eval_rows: list[CausalSequence],
    *,
    seed: int,
    matched_count: int,
    unrelated_count: int,
) -> tuple[list[CausalSequence], list[CausalSequence]]:
    same = [
        s for s in eval_rows
        if s.token_sha256 != target.token_sha256 and s.source == target.source
    ]
    same.sort(key=lambda s: _peer_hash(target, s, seed))
    matched = same[:matched_count]

    by_source: dict[str, list[CausalSequence]] = {}
    for s in eval_rows:
        if s.token_sha256 == target.token_sha256 or s.source == target.source:
            continue
        by_source.setdefault(s.source, []).append(s)
    unrelated: list[CausalSequence] = []
    for source in sorted(by_source):
        rows = sorted(by_source[source], key=lambda s: _peer_hash(target, s, seed))
        if rows:
            unrelated.append(rows[0])
    unrelated.sort(key=lambda s: _peer_hash(target, s, seed))
    return matched, unrelated[:unrelated_count]


def _gain_ratio(part: float, full: float) -> float | None:
    if full <= _EPS:
        return None
    return float(part / full)


def run_confirmation(
    sequences: list[CausalSequence],
    protocol: dict[str, Any],
    *,
    seed: int,
    rho: float,
    u: torch.Tensor,
    lm_head_weight: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    eval_rows = [s for s in sequences if s.partition == "eval"]
    carrier = fit_train_carrier(sequences)
    base = baseline_nlls(eval_rows, u, lm_head_weight, device=device)
    conf = protocol["confirmation"]
    matched_count = int(conf["matched_same_source_peers_per_target"])
    unrelated_count = int(conf["unrelated_different_source_peers_per_target"])

    target_rows: list[dict[str, Any]] = []
    full_positive = 0
    carrier_positive = 0
    carrier_ratios: list[float] = []
    residual_ratios: list[float] = []
    excess_harm_ratios: list[float] = []

    for target in eval_rows:
        full = target.ghat
        parallel, residual = decompose_direction(full, carrier)
        eta = eta_for_target_ratio(target, full, rho)
        matched, unrelated = select_peers(
            target, eval_rows, seed=seed,
            matched_count=matched_count, unrelated_count=unrelated_count
        )
        eval_set = [target, *matched, *unrelated]
        variants = {"full": full, "carrier": parallel, "residual": residual}
        variant_nll: dict[str, list[float]] = {
            name: _nlls(eval_set, direction, eta, u, lm_head_weight, device=device)
            for name, direction in variants.items()
        }

        baseline_target = base[target.token_sha256]
        gains = {
            name: baseline_target - vals[0]
            for name, vals in variant_nll.items()
        }
        if gains["full"] > 0:
            full_positive += 1
        if gains["carrier"] > 0:
            carrier_positive += 1

        offset = 1
        matched_gains: dict[str, float] = {}
        for name, vals in variant_nll.items():
            xs = []
            for peer, after in zip(matched, vals[offset : offset + len(matched)]):
                xs.append(base[peer.token_sha256] - after)
            matched_gains[name] = _mean(xs)
        offset += len(matched)

        unrelated_harm: dict[str, float] = {}
        unrelated_abs: dict[str, float] = {}
        for name, vals in variant_nll.items():
            harms, changes = [], []
            for peer, after in zip(unrelated, vals[offset : offset + len(unrelated)]):
                delta = after - base[peer.token_sha256]
                harms.append(max(delta, 0.0))
                changes.append(abs(delta))
            unrelated_harm[name] = _mean(harms)
            unrelated_abs[name] = _mean(changes)

        c_ratio = _gain_ratio(gains["carrier"], gains["full"])
        r_ratio = _gain_ratio(gains["residual"], gains["full"])
        excess = None
        if gains["full"] > _EPS:
            excess = (
                unrelated_harm["carrier"] - unrelated_harm["full"]
            ) / gains["full"]
        if c_ratio is not None:
            carrier_ratios.append(c_ratio)
        if r_ratio is not None:
            residual_ratios.append(r_ratio)
        if excess is not None:
            excess_harm_ratios.append(excess)

        target_rows.append(
            {
                "seed": int(seed),
                "token_sha256": target.token_sha256,
                "source": target.source,
                "rho": float(rho),
                "eta": eta,
                "baseline_nll": baseline_target,
                "full_target_gain": gains["full"],
                "carrier_target_gain": gains["carrier"],
                "residual_target_gain": gains["residual"],
                "carrier_over_full_target_gain": c_ratio,
                "residual_over_full_target_gain": r_ratio,
                "full_matched_transfer_gain": matched_gains["full"],
                "carrier_matched_transfer_gain": matched_gains["carrier"],
                "residual_matched_transfer_gain": matched_gains["residual"],
                "full_unrelated_positive_harm": unrelated_harm["full"],
                "carrier_unrelated_positive_harm": unrelated_harm["carrier"],
                "residual_unrelated_positive_harm": unrelated_harm["residual"],
                "full_unrelated_absolute_change": unrelated_abs["full"],
                "carrier_unrelated_absolute_change": unrelated_abs["carrier"],
                "residual_unrelated_absolute_change": unrelated_abs["residual"],
                "carrier_excess_unrelated_harm_over_full_target_gain": excess,
                "carrier_frobenius_fraction": float(torch.linalg.norm(parallel).item()),
                "residual_frobenius_fraction": float(torch.linalg.norm(residual).item()),
                "matched_peer_count": len(matched),
                "unrelated_peer_count": len(unrelated),
            }
        )

    summary = {
        "seed": int(seed),
        "rho": float(rho),
        "count": len(eval_rows),
        "full_descent_fraction": full_positive / max(len(eval_rows), 1),
        "carrier_descent_fraction": carrier_positive / max(len(eval_rows), 1),
        "median_carrier_over_full_target_gain": _median(carrier_ratios),
        "median_residual_over_full_target_gain": _median(residual_ratios),
        "median_carrier_excess_unrelated_harm_over_full_target_gain": _median(excess_harm_ratios),
        "median_full_target_gain": _median([r["full_target_gain"] for r in target_rows]),
        "median_carrier_target_gain": _median([r["carrier_target_gain"] for r in target_rows]),
        "median_residual_target_gain": _median([r["residual_target_gain"] for r in target_rows]),
        "median_full_unrelated_positive_harm": _median([r["full_unrelated_positive_harm"] for r in target_rows]),
        "median_carrier_unrelated_positive_harm": _median([r["carrier_unrelated_positive_harm"] for r in target_rows]),
        "median_carrier_frobenius_fraction": _median([r["carrier_frobenius_fraction"] for r in target_rows]),
        "median_residual_frobenius_fraction": _median([r["residual_frobenius_fraction"] for r in target_rows]),
    }
    return {
        "format": "minicells.core-validation.carrier-causal-sufficiency-confirmation-seed.v1",
        "experiment_id": protocol["experiment_id"],
        "seed": int(seed),
        "scientific_decision": False,
        "rho": float(rho),
        "summary": summary,
        "target_rows": target_rows,
    }


def discovery_scale_is_viable(row: dict[str, Any], protocol: dict[str, Any]) -> bool:
    gates = protocol["discovery"]["gates"]
    return (
        float(row["full_descent_fraction"]) >= float(gates["minimum_full_descent_fraction"])
        and float(row["median_full_normalized_nll_gain"]) >= float(gates["minimum_median_full_normalized_nll_gain"])
        and float(row["median_half_step_linearity_error"]) <= float(gates["maximum_median_half_step_linearity_error"])
        and float(row["p90_half_step_linearity_error"]) <= float(gates["maximum_p90_half_step_linearity_error"])
    )


def summarize_discovery(runs: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    expected = [int(x) for x in protocol["discovery"]["seeds"]]
    completed = sorted(int(r["seed"]) for r in runs)
    missing = [s for s in expected if s not in completed]
    candidates = []
    for rho in [float(x) for x in protocol["discovery"]["perturbation_ratio_grid"]]:
        per_seed = []
        for run in runs:
            row = next(r for r in run["scale_summary"] if abs(float(r["rho"]) - rho) < 1e-15)
            per_seed.append({"seed": int(run["seed"]), **row, "viable": discovery_scale_is_viable(row, protocol)})
        candidates.append(
            {
                "rho": rho,
                "all_completed_seed_rows_viable": bool(per_seed) and all(r["viable"] for r in per_seed),
                "per_seed": per_seed,
            }
        )
    locked = None
    if not missing:
        viable = [c for c in candidates if c["all_completed_seed_rows_viable"]]
        if viable:
            locked = max(viable, key=lambda c: float(c["rho"]))
    return {
        "status": "CAUSAL_SCALE_DISCOVERY_COMPLETE" if not missing else "DISCOVERY_INCOMPLETE",
        "scientific_decision": False,
        "completed_seeds": completed,
        "missing_seeds": missing,
        "candidate_summary": candidates,
        "locked_rho": None if locked is None else float(locked["rho"]),
        "confirmation_allowed": bool(not missing and locked is not None),
    }


def confirmation_gate_row(run: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    s = run["summary"]
    g = protocol["confirmation"]["gates"]
    checks = {
        "full_descent": float(s["full_descent_fraction"]) >= float(g["minimum_full_descent_fraction"]),
        "carrier_descent": float(s["carrier_descent_fraction"]) >= float(g["minimum_carrier_descent_fraction"]),
        "carrier_gain": float(s["median_carrier_over_full_target_gain"]) >= float(g["minimum_median_carrier_over_full_target_gain"]),
        "residual_gain": float(s["median_residual_over_full_target_gain"]) <= float(g["maximum_median_residual_over_full_target_gain"]),
        "unrelated_harm": float(s["median_carrier_excess_unrelated_harm_over_full_target_gain"]) <= float(g["maximum_median_carrier_excess_unrelated_harm_over_full_target_gain"]),
    }
    return {**s, **checks, "pass": all(checks.values())}


def summarize_confirmation(runs: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    expected = [int(x) for x in protocol["confirmation"]["seeds"]]
    completed = sorted(int(r["seed"]) for r in runs)
    missing = [s for s in expected if s not in completed]
    rows = [confirmation_gate_row(r, protocol) for r in runs]
    if missing:
        return {
            "status": "CONFIRMATION_INCOMPLETE",
            "scientific_decision": False,
            "supported": None,
            "completed_seeds": completed,
            "missing_seeds": missing,
            "gate_rows": rows,
        }
    supported = all(bool(r["pass"]) for r in rows)
    return {
        "status": protocol["confirmation"]["positive_status"] if supported else protocol["confirmation"]["negative_status"],
        "scientific_decision": True,
        "supported": supported,
        "completed_seeds": completed,
        "missing_seeds": [],
        "passed_seeds": sum(bool(r["pass"]) for r in rows),
        "total_formal_seeds": len(expected),
        "gate_rows": rows,
    }
