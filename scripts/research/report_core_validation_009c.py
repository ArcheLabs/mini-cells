#!/usr/bin/env python3
"""Aggregate Core Validation 009C seed checkpoints into a frozen decision."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from minicells.real_representation_009c_experiment import select_discovery_lock

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "core-validation-009c-sparse-local-effect-geometry"
VALIDATION = ROOT / "research" / "validations" / "core-009c-sparse-local-effect-geometry"
PROTOCOL = VALIDATION / "protocol.json"
LOCK = VALIDATION / "representation-lock.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_seed(path: Path, phase: str, seed: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("phase") != phase or int(payload.get("seed", -1)) != seed:
        return None
    if payload.get("protocol_sha256") != _sha256(PROTOCOL):
        return None
    return payload


def _flatten_configs(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for config in payload.get("sparse_configs", []) + payload.get("local_configs", []):
            row = {
                "seed": int(payload["seed"]),
                "family": config["family"],
                "eval_median_residual": float(config["eval"]["median_normalized_residual"]),
                "eval_p90_residual": float(config["eval"]["p90_normalized_residual"]),
                "null_eval_median_residual": float(config["null_eval"]["median_normalized_residual"]),
                "relative_improvement_over_global32": float(config["relative_median_improvement_over_global32"]),
                "relative_improvement_over_null": float(config["relative_median_improvement_over_matched_null"]),
                "viable": bool(config["viable"]),
            }
            if config["family"] == "sparse":
                row.update({"atom_count": int(config["atom_count"]), "sparsity": int(config["sparsity"]), "chart_count": "", "local_dimension": ""})
            else:
                row.update({"atom_count": "", "sparsity": "", "chart_count": int(config["chart_count"]), "local_dimension": int(config["local_dimension"])})
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def discovery_decision(protocol: dict[str, Any], payloads: list[dict[str, Any]], missing: list[int]) -> dict[str, Any]:
    lock, summary = select_discovery_lock(payloads, protocol)
    complete = not missing
    if not complete:
        status = "DISCOVERY_INCOMPLETE"
        allowed = False
    elif lock is None:
        status = protocol["discovery"]["failure_status"]
        allowed = False
    elif lock["family"] == "sparse":
        status = protocol["discovery"]["sparse_positive_status"]
        allowed = True
    else:
        status = protocol["discovery"]["local_positive_status"]
        allowed = True
    return {
        "format": "minicells.core-validation.sparse-local-effect-geometry-discovery-decision.v1",
        "experiment_id": protocol["experiment_id"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": _sha256(PROTOCOL),
        "scientific_decision": False,
        "status": status,
        "completed_seeds": [int(p["seed"]) for p in payloads],
        "missing_seeds": missing,
        "confirmation_allowed": allowed,
        "locked_configuration": lock,
        "candidate_summary": summary,
        "data_manifest_sha256": payloads[0].get("data_manifest_sha256") if payloads else None,
    }


def confirmation_decision(protocol: dict[str, Any], payloads: list[dict[str, Any]], missing: list[int]) -> dict[str, Any]:
    gates = protocol["confirmation"]["gates"]
    gate_rows: list[dict[str, Any]] = []
    for payload in payloads:
        configs = payload.get("sparse_configs", []) + payload.get("local_configs", [])
        if len(configs) != 1:
            raise RuntimeError("009C confirmation seed must contain exactly one locked configuration")
        row = configs[0]
        checks = {
            "eval_median": float(row["eval"]["median_normalized_residual"]) <= float(gates["maximum_eval_median_residual"]),
            "eval_p90": float(row["eval"]["p90_normalized_residual"]) <= float(gates["maximum_eval_p90_residual"]),
            "beats_global32": float(row["relative_median_improvement_over_global32"]) >= float(gates["minimum_relative_median_improvement_over_global32"]),
            "beats_null": float(row["relative_median_improvement_over_matched_null"]) >= float(gates["minimum_relative_median_improvement_over_matched_null"]),
        }
        gate_rows.append({
            "seed": int(payload["seed"]),
            "family": row["family"],
            "eval_median_residual": float(row["eval"]["median_normalized_residual"]),
            "eval_p90_residual": float(row["eval"]["p90_normalized_residual"]),
            "relative_improvement_over_global32": float(row["relative_median_improvement_over_global32"]),
            "relative_improvement_over_null": float(row["relative_median_improvement_over_matched_null"]),
            "checks": checks,
            "pass": all(checks.values()),
        })
    if missing:
        status = "CONFIRMATION_INCOMPLETE"; scientific = False; supported = False
    else:
        supported = bool(gate_rows) and all(r["pass"] for r in gate_rows)
        status = protocol["confirmation"]["positive_status"] if supported else protocol["confirmation"]["negative_status"]
        scientific = True
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.is_file() else None
    return {
        "format": "minicells.core-validation.sparse-local-effect-geometry-confirmation-decision.v1",
        "experiment_id": protocol["experiment_id"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": _sha256(PROTOCOL),
        "scientific_decision": scientific,
        "status": status,
        "supported": supported,
        "completed_seeds": [int(p["seed"]) for p in payloads],
        "missing_seeds": missing,
        "representation_lock": lock,
        "gate_rows": gate_rows,
        "data_manifest_sha256": payloads[0].get("data_manifest_sha256") if payloads else None,
    }


def write_results_md(path: Path, phase: str, decision: dict[str, Any]) -> None:
    lines = [
        "# Core Validation 009C — Sparse / Local Effect Geometry",
        "",
        f"- Phase: `{phase}`",
        f"- Status: `{decision['status']}`",
        f"- Completed seeds: `{decision['completed_seeds']}`",
        f"- Missing seeds: `{decision['missing_seeds']}`",
    ]
    if phase == "discovery":
        lines.extend([
            f"- Confirmation allowed: `{decision['confirmation_allowed']}`",
            f"- Locked configuration: `{decision['locked_configuration']}`",
            "",
            "Discovery requires one fixed sparse or local configuration to pass heldout residual, global-32D improvement, matched-null improvement and complexity gates on both seeds. Semantic labels and eval-assisted fitting are forbidden.",
        ])
    else:
        lines.extend([
            f"- Scientific decision: `{decision['scientific_decision']}`",
            f"- Supported: `{decision['supported']}`",
            "",
            "Confirmation evaluates only the committed discovery representation lock on untouched seeds.",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--phase", choices=("discovery", "confirmation"), required=True); args = p.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    seeds = [int(x) for x in protocol[args.phase]["seeds"]]
    phase_dir = RESULTS / args.phase; phase_dir.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, Any]] = []; missing: list[int] = []
    for seed in seeds:
        payload = _load_seed(phase_dir / "seeds" / f"seed-{seed}.json", args.phase, seed)
        if payload is None: missing.append(seed)
        else: payloads.append(payload)
    if args.phase == "discovery":
        decision = discovery_decision(protocol, payloads, missing)
        if decision["confirmation_allowed"]:
            lock = {
                "format": "minicells.core-validation.sparse-local-effect-geometry-lock.v1",
                "protocol_sha256": _sha256(PROTOCOL),
                "confirmation_allowed": True,
                "configuration": decision["locked_configuration"],
                "discovery_seeds": decision["completed_seeds"],
            }
            (phase_dir / "representation-lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        decision = confirmation_decision(protocol, payloads, missing)
    (phase_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(phase_dir / "config-summary.csv", _flatten_configs(payloads))
    write_results_md(phase_dir / "RESULTS.md", args.phase, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
