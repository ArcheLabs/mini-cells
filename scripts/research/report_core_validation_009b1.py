#!/usr/bin/env python3
"""Generate Core Validation 009B-1 discovery/confirmation reports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from minicells.real_representation_009b1_experiment import (
    summarize_confirmation,
    summarize_discovery,
)

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-009b1-carrier-causal-sufficiency"
PROTOCOL = VALIDATION / "protocol.json"
SCALE_LOCK = VALIDATION / "scale-lock.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-009b1-carrier-causal-sufficiency"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_format(phase: str) -> str:
    return f"minicells.core-validation.carrier-causal-sufficiency-{phase}-seed.v1"


def _load_runs(phase: str, out: Path, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    runs = []
    for seed in [int(x) for x in protocol[phase]["seeds"]]:
        path = out / phase / "seeds" / f"seed-{seed}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != _expected_format(phase):
            raise RuntimeError(f"invalid 009B-1 seed format: {path}")
        if payload.get("phase") != phase or int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"invalid 009B-1 seed identity: {path}")
        if payload.get("protocol_sha256") != _sha256(PROTOCOL):
            raise RuntimeError(f"009B-1 seed protocol mismatch: {path}")
        if payload.get("data_manifest_sha256") != protocol["data"]["expected_manifest_sha256"]:
            raise RuntimeError(f"009B-1 seed manifest mismatch: {path}")
        runs.append(payload)
    return runs


def _discovery(runs: list[dict[str, Any]], protocol: dict[str, Any], out: Path) -> dict[str, Any]:
    phase_out = out / "discovery"
    phase_out.mkdir(parents=True, exist_ok=True)
    decision = summarize_discovery(runs, protocol)
    decision.update({
        "format": "minicells.core-validation.carrier-causal-sufficiency-discovery-decision.v1",
        "experiment_id": protocol["experiment_id"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": _sha256(PROTOCOL),
        "data_manifest_sha256": protocol["data"]["expected_manifest_sha256"],
    })
    if not decision["missing_seeds"] and not decision["confirmation_allowed"]:
        decision["status"] = "CAUSAL_SCALE_DISCOVERY_NO_VIABLE_SCALE"
    (phase_out / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    scale_rows, target_rows = [], []
    for run in runs:
        scale_rows.extend(run["scale_summary"])
        target_rows.extend(run["target_rows"])
    pd.DataFrame(scale_rows).to_csv(phase_out / "scale-summary.csv", index=False)
    pd.DataFrame(target_rows).to_csv(phase_out / "target-discovery.csv", index=False)

    if decision.get("confirmation_allowed"):
        lock = {
            "format": "minicells.core-validation.carrier-causal-sufficiency-scale-lock.v1",
            "experiment_id": protocol["experiment_id"],
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": _sha256(PROTOCOL),
            "discovery_seeds": protocol["discovery"]["seeds"],
            "locked_rho": float(decision["locked_rho"]),
            "confirmation_allowed": True,
            "selection_uses_only_full_write": True,
            "scientific_decision": False,
        }
        (phase_out / "scale-lock.json").write_text(
            json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if scale_rows:
        frame = pd.DataFrame(scale_rows)
        fig, ax = plt.subplots(figsize=(7, 4))
        for seed, group in frame.groupby("seed"):
            group = group.sort_values("rho")
            ax.plot(group["rho"], group["median_full_normalized_nll_gain"], marker="o", label=str(seed))
        ax.set_xscale("log")
        ax.set_xlabel("target hidden perturbation ratio rho")
        ax.set_ylabel("median normalized full-write NLL gain")
        ax.set_title("009B-1 discovery: full-write causal scale")
        ax.legend()
        fig.tight_layout()
        fig.savefig(phase_out / "causal-scale-discovery.png", dpi=180)
        plt.close(fig)

    lines = [
        "# Core Validation 009B-1 Discovery",
        "",
        f"- Status: `{decision['status']}`",
        f"- Completed seeds: `{decision['completed_seeds']}`",
        f"- Missing seeds: `{decision['missing_seeds']}`",
        f"- Locked rho: `{decision.get('locked_rho')}`",
        f"- Confirmation allowed: `{decision.get('confirmation_allowed')}`",
        "",
        "Scale selection uses full-write measurability/linearity only. Carrier and residual outcomes are not computed in discovery.",
        "",
    ]
    (phase_out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return decision


def _confirmation(runs: list[dict[str, Any]], protocol: dict[str, Any], out: Path) -> dict[str, Any]:
    if not SCALE_LOCK.is_file():
        raise RuntimeError("009B-1 confirmation report requires committed scale-lock.json")
    lock = json.loads(SCALE_LOCK.read_text(encoding="utf-8"))
    if lock.get("protocol_sha256") != _sha256(PROTOCOL):
        raise RuntimeError("009B-1 scale lock protocol mismatch")
    phase_out = out / "confirmation"
    phase_out.mkdir(parents=True, exist_ok=True)
    decision = summarize_confirmation(runs, protocol)
    decision.update({
        "format": "minicells.core-validation.carrier-causal-sufficiency-confirmation-decision.v1",
        "experiment_id": protocol["experiment_id"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": _sha256(PROTOCOL),
        "data_manifest_sha256": protocol["data"]["expected_manifest_sha256"],
        "locked_rho": float(lock["locked_rho"]),
    })
    (phase_out / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gates = pd.DataFrame(decision.get("gate_rows", []))
    gates.to_csv(phase_out / "gate-summary.csv", index=False)
    targets = []
    for run in runs:
        targets.extend(run["target_rows"])
    pd.DataFrame(targets).to_csv(phase_out / "target-causal-effects.csv", index=False)

    if not gates.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = range(len(gates))
        ax.plot(x, gates["median_carrier_over_full_target_gain"], marker="o", label="carrier/full")
        ax.plot(x, gates["median_residual_over_full_target_gain"], marker="o", label="residual/full")
        ax.axhline(float(protocol["confirmation"]["gates"]["minimum_median_carrier_over_full_target_gain"]), linestyle="--")
        ax.set_xticks(list(x), [str(int(s)) for s in gates["seed"]])
        ax.set_ylabel("median target-gain ratio")
        ax.set_xlabel("confirmation seed")
        ax.set_title("009B-1 causal decomposition")
        ax.legend()
        fig.tight_layout()
        fig.savefig(phase_out / "causal-decomposition.png", dpi=180)
        plt.close(fig)

    lines = [
        "# Core Validation 009B-1 Confirmation",
        "",
        f"- Status: `{decision['status']}`",
        f"- Scientific decision: `{decision['scientific_decision']}`",
        f"- Supported: `{decision.get('supported')}`",
        f"- Locked rho: `{decision['locked_rho']}`",
        f"- Completed seeds: `{decision['completed_seeds']}`",
        f"- Missing seeds: `{decision['missing_seeds']}`",
        "",
        "## Gate summary",
        "",
        gates.to_markdown(index=False) if not gates.empty else "No completed confirmation seeds.",
        "",
        "A positive result establishes carrier causal sufficiency only. It does not establish effect reuse, addressability, certificates, growth, or an end-to-end CLM.",
        "",
    ]
    (phase_out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return decision


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("discovery", "confirmation"), required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    runs = _load_runs(args.phase, args.out, protocol)
    decision = _discovery(runs, protocol, args.out) if args.phase == "discovery" else _confirmation(runs, protocol, args.out)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
