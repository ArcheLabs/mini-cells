#!/usr/bin/env python3
"""Report Core Validation 009B-2 discovery or confirmation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from minicells.real_representation_009b2_experiment import confirmation_gate_row, select_discovery_dimension

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-009b2-persistent-effect-geometry"
PROTOCOL = VALIDATION / "protocol.json"
RESULTS = ROOT / "results" / "core-validation-009b2-persistent-effect-geometry"


def _protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def _sha() -> str:
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def _seed_format(phase: str) -> str:
    return f"minicells.core-validation.persistent-effect-geometry-{phase}-seed.v1"


def _load_seed(phase: str, seed: int) -> dict[str, Any] | None:
    path = RESULTS / phase / "seeds" / f"seed-{seed}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != _seed_format(phase) or payload.get("phase") != phase or int(payload.get("seed", -1)) != seed or payload.get("protocol_sha256") != _sha() or payload.get("scientific_decision") is not False:
        raise RuntimeError(f"invalid/stale 009B-2 seed artifact: {path}")
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def _discovery(protocol: dict[str, Any]) -> dict[str, Any]:
    out = RESULTS / "discovery"; out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in protocol["discovery"]["seeds"]]
    payloads = [p for s in seeds if (p := _load_seed("discovery", s)) is not None]
    completed = [int(p["seed"]) for p in payloads]; missing = [s for s in seeds if s not in completed]
    locked, candidates = select_discovery_dimension(payloads, protocol)

    dimension_csv = []
    spectrum_csv = []
    for p in payloads:
        for row in p["dimension_rows"]:
            dimension_csv.append({"seed": p["seed"], "dimension": row["dimension"], "train_median_residual": row["train"]["median_normalized_residual"], "train_p90_residual": row["train"]["p90_normalized_residual"], "eval_median_residual": row["eval"]["median_normalized_residual"], "eval_p90_residual": row["eval"]["p90_normalized_residual"], "eval_median_projected_cosine": row["eval"]["median_projected_cosine"], "train_to_eval_median_residual_gap": row["train_to_eval_median_residual_gap"]})
        for row in p["spectrum"]:
            spectrum_csv.append({"seed": p["seed"], **row})
    candidate_csv = []
    for row in candidates:
        for ps in row["per_seed"]:
            candidate_csv.append({"dimension": row["dimension"], "all_completed_seed_rows_viable": row["all_completed_seed_rows_viable"], **ps})
    _write_csv(out / "offline-dimensions.csv", dimension_csv)
    _write_csv(out / "effect-spectrum.csv", spectrum_csv)
    _write_csv(out / "candidate-summary.csv", candidate_csv)

    complete = not missing
    confirmation_allowed = bool(complete and locked is not None)
    status = "DISCOVERY_INCOMPLETE" if not complete else ("EFFECT_GEOMETRY_DISCOVERY_COMPLETE" if confirmation_allowed else str(protocol["discovery"]["failure_status"]))
    decision = {"format": "minicells.core-validation.persistent-effect-geometry-discovery-decision.v1", "experiment_id": protocol["experiment_id"], "protocol_version": protocol["protocol_version"], "protocol_sha256": _sha(), "data_manifest_sha256": payloads[0].get("data_manifest_sha256") if payloads else None, "completed_seeds": completed, "missing_seeds": missing, "candidate_summary": candidates, "locked_dimension": locked, "confirmation_allowed": confirmation_allowed, "scientific_decision": False, "status": status}
    (out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lock_path = out / "basis-lock.json"
    if confirmation_allowed:
        lock = {"format": "minicells.core-validation.persistent-effect-geometry-basis-lock.v1", "experiment_id": protocol["experiment_id"], "protocol_version": protocol["protocol_version"], "protocol_sha256": _sha(), "discovery_seeds": seeds, "locked_dimension": int(locked), "residual_threshold_tau": float(protocol["online_growth"]["residual_threshold_tau"]), "confirmation_allowed": True, "selection_uses_only_offline_effect_geometry": True, "online_metrics_used_for_selection": False, "scientific_decision": False}
        lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif lock_path.exists():
        lock_path.unlink()

    (out / "RESULTS.md").write_text("\n".join(["# Core Validation 009B-2 — Persistent Effect Geometry", "", "- Phase: `discovery`", f"- Status: `{status}`", f"- Completed seeds: `{completed}`", f"- Missing seeds: `{missing}`", f"- Locked dimension: `{locked}`", f"- Confirmation allowed: `{confirmation_allowed}`", "", "Discovery selects only the smallest <=32-dimensional offline effect subspace satisfying the frozen heldout residual/generalization gates on both seeds. Online growth cannot influence the lock."]) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True)); return decision


def _confirmation(protocol: dict[str, Any]) -> dict[str, Any]:
    out = RESULTS / "confirmation"; out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in protocol["confirmation"]["seeds"]]
    payloads = [p for s in seeds if (p := _load_seed("confirmation", s)) is not None]
    completed = [int(p["seed"]) for p in payloads]; missing = [s for s in seeds if s not in completed]
    gates = [confirmation_gate_row(p, protocol) for p in payloads]

    offline_csv = []; online_csv = []; growth_csv = []
    for p in payloads:
        offline_csv.append({"seed": p["seed"], "locked_dimension": p["locked_dimension"], "train_median_residual": p["offline"]["train"]["median_normalized_residual"], "eval_median_residual": p["offline"]["eval"]["median_normalized_residual"], "eval_p90_residual": p["offline"]["eval"]["p90_normalized_residual"], "train_to_eval_median_residual_gap": p["offline"]["train_to_eval_median_residual_gap"]})
        for row in p["online"]:
            online_csv.append({"seed": p["seed"], "ordering": row["ordering"], "train_writes": row["train_writes"], "final_dimension": row["final_dimension"], "new_coordinates_per_100_writes": row["new_coordinates_per_100_writes"], "late_growth_per_100_writes": row["late_growth_per_100_writes"], "independent_memory_compression_ratio": row["independent_memory_compression_ratio"], "eval_median_normalized_residual": row["eval_median_normalized_residual"], "eval_p90_normalized_residual": row["eval_p90_normalized_residual"]})
            for point in row["growth_curve"]:
                growth_csv.append({"seed": p["seed"], "ordering": row["ordering"], **point})
    _write_csv(out / "offline-summary.csv", offline_csv)
    _write_csv(out / "online-summary.csv", online_csv)
    _write_csv(out / "online-growth.csv", growth_csv)
    _write_csv(out / "gate-summary.csv", gates)

    complete = not missing
    if complete:
        supported = all(g["pass"] for g in gates) and len(gates) == len(seeds)
        scientific = True
        status = protocol["confirmation"]["positive_status"] if supported else protocol["confirmation"]["negative_status"]
    else:
        supported = None; scientific = False; status = "CONFIRMATION_INCOMPLETE"
    locked_values = sorted({int(p["locked_dimension"]) for p in payloads})
    decision = {"format": "minicells.core-validation.persistent-effect-geometry-confirmation-decision.v1", "experiment_id": protocol["experiment_id"], "protocol_version": protocol["protocol_version"], "protocol_sha256": _sha(), "data_manifest_sha256": payloads[0].get("data_manifest_sha256") if payloads else None, "completed_seeds": completed, "missing_seeds": missing, "locked_dimension": locked_values[0] if len(locked_values) == 1 else None, "gate_rows": gates, "passed_seeds": sum(bool(g["pass"]) for g in gates), "total_formal_seeds": len(seeds), "scientific_decision": scientific, "supported": supported, "status": status}
    (out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "RESULTS.md").write_text("\n".join(["# Core Validation 009B-2 — Persistent Effect Geometry", "", "- Phase: `confirmation`", f"- Status: `{status}`", f"- Scientific decision: `{scientific}`", f"- Supported: `{supported}`", f"- Completed seeds: `{completed}`", f"- Missing seeds: `{missing}`", f"- Locked dimension: `{decision['locked_dimension']}`", "", "A positive result supports compact reusable effect geometry only; it does not establish sparse coefficients, deployable addressability, certificates, continual mutation, or a confirmed CLM architecture."]) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True)); return decision


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--phase", choices=("discovery", "confirmation"), required=True); args = p.parse_args()
    protocol = _protocol(); _discovery(protocol) if args.phase == "discovery" else _confirmation(protocol); return 0


if __name__ == "__main__":
    raise SystemExit(main())
