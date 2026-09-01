#!/usr/bin/env python3
"""Generate Core Validation 009A discovery/confirmation reports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from minicells.real_representation_009a_experiment import summarize_confirmation, summarize_discovery

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-009a-factorized-functional-coordinates"
PROTOCOL = VALIDATION / "protocol.json"
WINNER_LOCK = VALIDATION / "winner-lock.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-009a-factorized-functional-coordinates"
SEED_FORMAT = "minicells.core-validation.factorized-functional-coordinates-seed.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runs(phase: str, out: Path, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    runs = []
    for seed in [int(x) for x in protocol[phase]["seeds"]]:
        path = out / phase / "seeds" / f"seed-{seed}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != SEED_FORMAT or payload.get("phase") != phase or int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"invalid seed artifact identity: {path}")
        if payload.get("protocol_sha256") != _sha256(PROTOCOL):
            raise RuntimeError(f"seed artifact protocol mismatch: {path}")
        if payload.get("data_manifest_sha256") != protocol["data"]["expected_manifest_sha256"]:
            raise RuntimeError(f"seed artifact data manifest mismatch: {path}")
        runs.append(payload)
    return runs


def _flatten(runs: list[dict[str, Any]], key: str) -> pd.DataFrame:
    rows = []
    for run in runs:
        seed = int(run["seed"])
        rows.extend({"seed": seed, **row} for row in run[key])
    return pd.DataFrame(rows)


def _write_common_tables(runs: list[dict[str, Any]], phase_out: Path) -> dict[str, pd.DataFrame]:
    frames = {
        "left-only.csv": _flatten(runs, "left_only"),
        "right-only.csv": _flatten(runs, "right_only"),
        "two-sided-landscape.csv": _flatten(runs, "two_sided_landscape"),
        "budget-splits.csv": _flatten(runs, "budget_splits"),
        "per-write-rank1.csv": _flatten(runs, "per_write_rank1"),
    }
    spectra = []
    for run in runs:
        seed = int(run["seed"])
        spectra.extend({"seed": seed, "side": "left", **r} for r in run["left_spectrum"])
        spectra.extend({"seed": seed, "side": "right", **r} for r in run["right_spectrum"])
    frames["factor-spectrum.csv"] = pd.DataFrame(spectra)
    for name, frame in frames.items():
        frame.to_csv(phase_out / name, index=False)
    return frames


def _plot_discovery(frames: dict[str, pd.DataFrame], phase_out: Path) -> None:
    budget = frames["budget-splits.csv"]
    if not budget.empty:
        ev = budget[budget["partition"] == "eval"].copy()
        grouped = ev.groupby(["left_dim", "right_dim"], as_index=False)["median_local_action_residual"].mean()
        grouped = grouped.sort_values("left_dim")
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [f"{int(a)}/{int(b)}" for a, b in zip(grouped["left_dim"], grouped["right_dim"])]
        ax.plot(range(len(grouped)), grouped["median_local_action_residual"], marker="o")
        ax.axhline(0.45, linestyle="--")
        ax.set_xticks(range(len(grouped)), labels, rotation=45, ha="right")
        ax.set_ylabel("mean heldout local-action residual")
        ax.set_xlabel("left/right dimensions (m+n=64)")
        ax.set_title("Core 009A discovery: budget-matched factor splits")
        fig.tight_layout()
        fig.savefig(phase_out / "budget-split-discovery.png", dpi=180)
        plt.close(fig)

    landscape = frames["two-sided-landscape.csv"]
    if not landscape.empty:
        grouped = landscape.groupby(["left_dim", "right_dim"], as_index=False)["median_local_action_residual"].mean()
        pivot = grouped.pivot(index="left_dim", columns="right_dim", values="median_local_action_residual").sort_index().sort_index(axis=1)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(pivot.values, origin="lower", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), [str(int(x)) for x in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)), [str(int(x)) for x in pivot.index])
        ax.set_xlabel("right/input-condition dimension n")
        ax.set_ylabel("left/output-effect dimension m")
        ax.set_title("Core 009A heldout two-sided residual landscape")
        fig.colorbar(im, ax=ax, label="local-action residual")
        fig.tight_layout()
        fig.savefig(phase_out / "two-sided-landscape.png", dpi=180)
        plt.close(fig)


def _discovery(runs: list[dict[str, Any]], protocol: dict[str, Any], out: Path) -> dict[str, Any]:
    phase_out = out / "discovery"
    phase_out.mkdir(parents=True, exist_ok=True)
    frames = _write_common_tables(runs, phase_out)
    decision = summarize_discovery(runs, protocol)
    if not decision["missing_seeds"] and not decision["winner_meets_viability"]:
        decision["status"] = "GEOMETRY_DISCOVERY_NO_VIABLE_BUDGET_SPLIT"
        decision["confirmation_allowed"] = False
    decision.update(
        {
            "format": "minicells.core-validation.factorized-functional-coordinates-discovery-decision.v1",
            "experiment_id": protocol["experiment_id"],
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": _sha256(PROTOCOL),
            "data_manifest_sha256": protocol["data"]["expected_manifest_sha256"],
        }
    )
    (phase_out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(decision.get("candidate_summary", [])).to_csv(phase_out / "candidate-summary.csv", index=False)

    if decision.get("confirmation_allowed"):
        lock = {
            "format": "minicells.core-validation.factorized-functional-coordinates-winner-lock.v1",
            "experiment_id": protocol["experiment_id"],
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": _sha256(PROTOCOL),
            "discovery_seeds": protocol["discovery"]["seeds"],
            "locked_split": decision["provisional_winner"],
            "winner_metrics": decision["winner_metrics"],
            "winner_meets_viability": True,
            "scientific_decision": False,
        }
        (phase_out / "winner-lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _plot_discovery(frames, phase_out)
    lines = [
        "# Core Validation 009A Discovery",
        "",
        f"- Status: `{decision['status']}`",
        f"- Completed seeds: `{decision['completed_seeds']}`",
        f"- Missing seeds: `{decision['missing_seeds']}`",
        f"- Provisional winner: `{decision.get('provisional_winner')}`",
        f"- Winner meets viability: `{decision.get('winner_meets_viability')}`",
        f"- Confirmation allowed: `{decision.get('confirmation_allowed')}`",
        "",
        "Discovery selects factor dimensions only. It is not a scientific supported/not-supported decision.",
        "",
    ]
    (phase_out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return decision


def _confirmation(runs: list[dict[str, Any]], protocol: dict[str, Any], out: Path) -> dict[str, Any]:
    if not WINNER_LOCK.is_file():
        raise RuntimeError("confirmation report requires committed winner-lock.json")
    lock = json.loads(WINNER_LOCK.read_text(encoding="utf-8"))
    if lock.get("protocol_sha256") != _sha256(PROTOCOL):
        raise RuntimeError("winner lock protocol hash mismatch")
    split = lock["locked_split"]
    m, n = int(split["left_dim"]), int(split["right_dim"])
    phase_out = out / "confirmation"
    phase_out.mkdir(parents=True, exist_ok=True)
    _write_common_tables(runs, phase_out)
    decision = summarize_confirmation(runs, protocol, left_dim=m, right_dim=n)
    decision.update(
        {
            "format": "minicells.core-validation.factorized-functional-coordinates-confirmation-decision.v1",
            "experiment_id": protocol["experiment_id"],
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": _sha256(PROTOCOL),
            "data_manifest_sha256": protocol["data"]["expected_manifest_sha256"],
        }
    )
    (phase_out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gates = pd.DataFrame(decision.get("gate_rows", []))
    gates.to_csv(phase_out / "gate-summary.csv", index=False)
    lines = [
        "# Core Validation 009A Confirmation",
        "",
        f"- Status: `{decision['status']}`",
        f"- Scientific decision: `{decision['scientific_decision']}`",
        f"- Locked split: `{decision['locked_split']}`",
        f"- Completed seeds: `{decision['completed_seeds']}`",
        f"- Missing seeds: `{decision['missing_seeds']}`",
        "",
        "## Gate summary",
        "",
        gates.to_markdown(index=False) if not gates.empty else "No completed confirmation seeds.",
        "",
        "009A confirmation concerns factor geometry only; routing, sparsity, certificates, growth and continual learning remain outside scope.",
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
