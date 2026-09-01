#!/usr/bin/env python3
"""Aggregate Core Validation 009D seed checkpoints into frozen decisions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from minicells.real_representation_009d_experiment import select_discovery_lock

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "core-validation-009d-compositional-operator-geometry"
VALIDATION = ROOT / "research" / "validations" / "core-009d-compositional-operator-geometry"
PROTOCOL = VALIDATION / "protocol.json"
LOCK = VALIDATION / "representation-lock.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_seed(path: Path, phase: str, seed: int) -> dict[str, Any] | None:
    if not path.is_file(): return None
    try: payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception: return None
    expected_format = f"minicells.core-validation.compositional-operator-geometry-{phase}-seed.v1"
    if payload.get("format") != expected_format or payload.get("phase") != phase or int(payload.get("seed", -1)) != seed: return None
    if payload.get("protocol_sha256") != _sha256(PROTOCOL): return None
    return payload


def _flatten(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        seed = int(payload["seed"]); guard = payload["rank1_core_guard"]; dense = payload["dense_56x8"]["eval"]; compact = payload["rank1_core_56x8"]["eval"]
        rows.append({"seed": seed, "family": "rank1_core_guard", "configuration": "56x8->rank1", "eval_median_action": compact["median_local_action_residual"], "eval_p90_action": compact["p90_local_action_residual"], "eval_median_frobenius": compact["median_frobenius_residual"], "baseline_median_action": dense["median_local_action_residual"], "relative_or_excess": guard["eval_median_local_action_excess_over_dense_56x8"], "viable": bool(guard["pass"])})
        for config in payload.get("sparse_tensor_configs", []):
            rows.append({"seed": seed, "family": "sparse_tensor", "configuration": str(config["active_coordinates"]), "eval_median_action": config["eval"]["median_local_action_residual"], "eval_p90_action": config["eval"]["p90_local_action_residual"], "eval_median_frobenius": config["eval"]["median_frobenius_residual"], "baseline_median_action": config["rotated_null_eval"]["median_local_action_residual"], "relative_or_excess": config["relative_median_action_improvement_over_rotated_null"], "viable": bool(config["viable"])})
        cond = payload.get("right_conditioned")
        if cond is not None:
            rows.append({"seed": seed, "family": "right_conditioned", "configuration": f"ridge:{cond['selected_ridge_lambda']}", "eval_median_action": cond["eval"]["median_local_action_residual"], "eval_p90_action": cond["eval"]["p90_local_action_residual"], "eval_median_frobenius": cond["eval"]["median_frobenius_residual"], "baseline_median_action": cond["mean_left_eval"]["median_local_action_residual"], "relative_or_excess": cond["relative_median_action_improvement_over_mean_left"], "viable": bool(cond["viable"])})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)


def discovery_decision(protocol: dict[str, Any], payloads: list[dict[str, Any]], missing: list[int]) -> dict[str, Any]:
    lock, summary = select_discovery_lock(payloads, protocol)
    if missing: status, allowed = "DISCOVERY_INCOMPLETE", False
    elif lock is not None and lock["family"] == "sparse_tensor": status, allowed = protocol["discovery"]["sparse_positive_status"], True
    elif lock is not None and lock["family"] == "right_conditioned": status, allowed = protocol["discovery"]["conditional_positive_status"], True
    elif summary["rank1_core_guard_all_completed_seeds"]: status, allowed = protocol["discovery"]["factor_compression_only_status"], False
    else: status, allowed = protocol["discovery"]["failure_status"], False
    return {"format": "minicells.core-validation.compositional-operator-geometry-discovery-decision.v1", "experiment_id": protocol["experiment_id"], "protocol_version": protocol["protocol_version"], "protocol_sha256": _sha256(PROTOCOL), "scientific_decision": False, "status": status, "completed_seeds": sorted(int(p["seed"]) for p in payloads), "missing_seeds": missing, "confirmation_allowed": allowed, "locked_configuration": lock, "candidate_summary": summary, "data_manifest_sha256": payloads[0].get("data_manifest_sha256") if payloads else None}


def confirmation_decision(protocol: dict[str, Any], payloads: list[dict[str, Any]], missing: list[int]) -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.is_file() else None
    if lock is None: raise RuntimeError("009D confirmation decision requires committed representation lock")
    family = lock["configuration"]["family"]; rows: list[dict[str, Any]] = []
    for payload in payloads:
        guard = bool(payload["rank1_core_guard"]["pass"])
        if family == "sparse_tensor":
            configs = payload.get("sparse_tensor_configs", [])
            if len(configs) != 1: raise RuntimeError("locked sparse 009D confirmation seed must contain one configuration")
            candidate = configs[0]
        elif family == "right_conditioned":
            candidate = payload.get("right_conditioned")
            if candidate is None: raise RuntimeError("locked right-conditioned 009D confirmation seed missing candidate")
        else: raise RuntimeError(f"unknown 009D locked family {family}")
        rows.append({"seed": int(payload["seed"]), "family": family, "rank1_core_guard": guard, "candidate_viable": bool(candidate["viable"]), "pass": guard and bool(candidate["viable"]), "candidate": candidate})
    if missing: status, scientific, supported = "CONFIRMATION_INCOMPLETE", False, False
    else:
        supported = bool(rows) and all(bool(r["pass"]) for r in rows); status = protocol["confirmation"]["positive_status"] if supported else protocol["confirmation"]["negative_status"]; scientific = True
    return {"format": "minicells.core-validation.compositional-operator-geometry-confirmation-decision.v1", "experiment_id": protocol["experiment_id"], "protocol_version": protocol["protocol_version"], "protocol_sha256": _sha256(PROTOCOL), "scientific_decision": scientific, "status": status, "supported": supported, "completed_seeds": sorted(int(p["seed"]) for p in payloads), "missing_seeds": missing, "representation_lock": lock, "gate_rows": rows, "data_manifest_sha256": payloads[0].get("data_manifest_sha256") if payloads else None}


def write_results_md(path: Path, phase: str, decision: dict[str, Any]) -> None:
    lines = ["# Core Validation 009D — Compositional Operator Geometry", "", f"- Phase: `{phase}`", f"- Status: `{decision['status']}`", f"- Completed seeds: `{decision['completed_seeds']}`", f"- Missing seeds: `{decision['missing_seeds']}`"]
    if phase == "discovery":
        lines.extend([f"- Confirmation allowed: `{decision['confirmation_allowed']}`", f"- Locked configuration: `{decision['locked_configuration']}`", f"- Rank-1-in-56x8 compression guard: `{decision['candidate_summary']['rank1_core_guard_all_completed_seeds']}`", "", "A positive reusable result requires one fixed sparse tensor coordinate count on both seeds, or—only if sparse tensor reuse fails—the frozen train-CV right-conditioned predictor on both seeds. Rank-1 factor compression alone does not permit confirmation."])
    else:
        lines.extend([f"- Scientific decision: `{decision['scientific_decision']}`", f"- Supported: `{decision['supported']}`", "", "Confirmation evaluates only the committed operator representation lock on untouched seeds."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--phase", choices=("discovery", "confirmation"), required=True); args = p.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8")); seeds = [int(x) for x in protocol[args.phase]["seeds"]]
    phase_dir = RESULTS / args.phase; phase_dir.mkdir(parents=True, exist_ok=True); payloads: list[dict[str, Any]] = []; missing: list[int] = []
    for seed in seeds:
        payload = _load_seed(phase_dir / "seeds" / f"seed-{seed}.json", args.phase, seed)
        if payload is None: missing.append(seed)
        else: payloads.append(payload)
    if args.phase == "discovery":
        decision = discovery_decision(protocol, payloads, missing)
        if decision["confirmation_allowed"]:
            lock = {"format": "minicells.core-validation.compositional-operator-geometry-lock.v1", "protocol_sha256": _sha256(PROTOCOL), "confirmation_allowed": True, "configuration": decision["locked_configuration"], "discovery_seeds": decision["completed_seeds"]}
            (phase_dir / "representation-lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else: decision = confirmation_decision(protocol, payloads, missing)
    (phase_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"); _write_csv(phase_dir / "candidate-summary.csv", _flatten(payloads)); write_results_md(phase_dir / "RESULTS.md", args.phase, decision); print(json.dumps(decision, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
