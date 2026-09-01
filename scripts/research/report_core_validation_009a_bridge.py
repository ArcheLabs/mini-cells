#!/usr/bin/env python3
"""Report the Core Validation 009A right-collapse diagnostic bridge."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from minicells.real_representation_009a_bridge import spectrum_energy

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "research" / "validations" / "core-009a-right-collapse-bridge"
PROTOCOL = VALIDATION / "protocol.json"
DEFAULT_OUT = ROOT / "results" / "core-validation-009a-right-collapse-bridge"
SEED_FORMAT = "minicells.core-validation.009a-right-collapse-bridge-seed.v1"
CONDITIONS = ("raw", "centered", "whitened", "mean_direction_removed")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _load_runs(out: Path, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    runs = []
    for seed in [int(x) for x in protocol["replication"]["diagnostic_seeds"]]:
        path = out / "seeds" / f"seed-{seed}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != SEED_FORMAT or int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"invalid bridge seed artifact identity: {path}")
        if payload.get("protocol_sha256") != _sha256(PROTOCOL):
            raise RuntimeError(f"bridge seed protocol mismatch: {path}")
        if payload.get("data_manifest_sha256") != protocol["data"]["expected_manifest_sha256"]:
            raise RuntimeError(f"bridge seed data manifest mismatch: {path}")
        if payload.get("scientific_decision") is not False or payload.get("source_009a_status_changed") is not False:
            raise RuntimeError(f"bridge seed illegally changes scientific/source status: {path}")
        if payload.get("source_009a_reproduction", {}).get("pass") is not True:
            raise RuntimeError(f"bridge seed lacks required 009A reproduction: {path}")
        runs.append(payload)
    return runs


def _projection_value(condition: dict[str, Any], key: str, *, partition: str, right_dim: int) -> float:
    row = next(
        r for r in condition[key]
        if r["partition"] == partition and int(r["right_dim"] if "right_dim" in r else r["dimension"]) == right_dim
    )
    return float(row["median_local_action_residual"])


def _condition_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for name in CONDITIONS:
            c = run["conditions"][name]
            row: dict[str, Any] = {
                "seed": int(run["seed"]),
                "condition": name,
                "sequence_right_top1_energy": spectrum_energy(c, "sequence_right_spectrum", 1),
                "sequence_right_top8_energy": spectrum_energy(c, "sequence_right_spectrum", 8),
                "sequence_right_participation_rank": float(c["sequence_right_spectrum"]["participation_rank"]),
                "sequence_right_dim95": int(c["sequence_right_spectrum"]["dimension_at_energy"]["0.95"]),
                "token_normalized_right_top1_energy": spectrum_energy(c, "token_normalized_right_spectrum", 1),
                "token_normalized_right_top8_energy": spectrum_energy(c, "token_normalized_right_spectrum", 8),
                "token_energy_weighted_right_top1_energy": spectrum_energy(c, "token_energy_weighted_right_spectrum", 1),
                "token_energy_weighted_right_top8_energy": spectrum_energy(c, "token_energy_weighted_right_spectrum", 8),
                "sequence_left_top56_energy": spectrum_energy(c, "sequence_left_spectrum", 56),
                "token_count": int(c["token_count"]),
            }
            for d in (1, 2, 4, 8):
                row[f"eval_right_only_n{d}_action_residual"] = _projection_value(
                    c, "right_only", partition="eval", right_dim=d
                )
                row[f"eval_two_sided_56x{d}_action_residual"] = _projection_value(
                    c, "two_sided_56", partition="eval", right_dim=d
                )
            rows.append(row)
    return rows


def _spectra_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for name in CONDITIONS:
            c = run["conditions"][name]
            for key in (
                "sequence_left_spectrum",
                "sequence_right_spectrum",
                "token_normalized_right_spectrum",
                "token_energy_weighted_right_spectrum",
            ):
                for point in c[key]["curve"]:
                    rows.append(
                        {
                            "seed": int(run["seed"]),
                            "condition": name,
                            "spectrum": key,
                            "dimension": int(point["dimension"]),
                            "cumulative_energy": float(point["cumulative_energy"]),
                        }
                    )
    return rows


def _projection_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for name in CONDITIONS:
            c = run["conditions"][name]
            for key in ("right_only", "two_sided_56"):
                for r in c[key]:
                    rows.append({"seed": int(run["seed"]), "condition": name, "kind": key, **r})
    return rows


def _alignment_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"seed": int(run["seed"]), **run["alignment"]} for run in runs]


def _ablation_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        ab = run["top1_ablation"]
        ev = ab["partition_summary"]["eval"]
        spectrum = ab["residual_right_spectrum"]
        rows.append(
            {
                "seed": int(run["seed"]),
                **ev,
                "residual_right_top1_energy": spectrum_energy(
                    {"x": spectrum}, "x", 1
                ),
                "residual_right_top8_energy": spectrum_energy(
                    {"x": spectrum}, "x", 8
                ),
                "residual_right_participation_rank": float(spectrum["participation_rank"]),
                "residual_right_dim95": int(spectrum["dimension_at_energy"]["0.95"]),
                "source_reproduction_absolute_delta": float(
                    run["source_009a_reproduction"]["absolute_delta"]
                ),
            }
        )
    return rows


def _aggregate_metrics(condition_df: pd.DataFrame, ablation_df: pd.DataFrame) -> dict[str, Any]:
    def cm(condition: str, column: str) -> float:
        values = condition_df.loc[condition_df["condition"] == condition, column].astype(float).tolist()
        return _median(values)

    return {
        "raw_sequence_right_top1_energy": cm("raw", "sequence_right_top1_energy"),
        "centered_sequence_right_top1_energy": cm("centered", "sequence_right_top1_energy"),
        "whitened_sequence_right_top1_energy": cm("whitened", "sequence_right_top1_energy"),
        "mean_direction_removed_sequence_right_top1_energy": cm("mean_direction_removed", "sequence_right_top1_energy"),
        "raw_token_normalized_right_top1_energy": cm("raw", "token_normalized_right_top1_energy"),
        "raw_token_energy_weighted_right_top1_energy": cm("raw", "token_energy_weighted_right_top1_energy"),
        "raw_eval_two_sided_56x1_action_residual": cm("raw", "eval_two_sided_56x1_action_residual"),
        "raw_eval_two_sided_56x8_action_residual": cm("raw", "eval_two_sided_56x8_action_residual"),
        "top1_ablation_eval_residual_action_fraction": _median(
            ablation_df["median_residual_local_action_fraction"].astype(float).tolist()
        ),
        "top1_ablation_residual_right_top1_energy": _median(
            ablation_df["residual_right_top1_energy"].astype(float).tolist()
        ),
        "maximum_source_reproduction_absolute_delta": float(
            ablation_df["source_reproduction_absolute_delta"].astype(float).max()
        ),
    }


def _flags(metrics: dict[str, Any], protocol: dict[str, Any]) -> dict[str, bool]:
    t = protocol["descriptive_flags"]
    collapse = float(t["right_collapse_top1_energy"])
    drop = float(t["substantial_top1_energy_drop"])
    amp = float(t["sequence_over_token_amplification"])
    small = float(t["small_top1_ablation_residual_action_fraction"])
    residual_low = float(t["residual_low_dimensional_top1_energy"])
    raw = float(metrics["raw_sequence_right_top1_energy"])
    centered = float(metrics["centered_sequence_right_top1_energy"])
    whitened = float(metrics["whitened_sequence_right_top1_energy"])
    mean_removed = float(metrics["mean_direction_removed_sequence_right_top1_energy"])
    token = float(metrics["raw_token_normalized_right_top1_energy"])
    return {
        "raw_right_collapse_reproduced": raw >= collapse,
        "centering_sensitive": raw - centered >= drop,
        "whitening_sensitive": raw - whitened >= drop,
        "mean_direction_sensitive": raw - mean_removed >= drop,
        "sequence_aggregation_sensitive": raw - token >= amp,
        "robust_common_right_direction_across_controls": min(centered, whitened, mean_removed) >= collapse,
        "top1_functionally_dominant": float(metrics["top1_ablation_eval_residual_action_fraction"]) <= small,
        "post_top1_residual_still_low_dimensional": float(metrics["top1_ablation_residual_right_top1_energy"]) >= residual_low,
    }


def _plot(condition_df: pd.DataFrame, out: Path) -> None:
    if condition_df.empty:
        return
    grouped = condition_df.groupby("condition", as_index=False)[
        [
            "sequence_right_top1_energy",
            "token_normalized_right_top1_energy",
            "token_energy_weighted_right_top1_energy",
        ]
    ].mean()
    order = list(CONDITIONS)
    grouped["order"] = grouped["condition"].map({x: i for i, x in enumerate(order)})
    grouped = grouped.sort_values("order")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(grouped))
    ax.plot(x, grouped["sequence_right_top1_energy"], marker="o", label="sequence")
    ax.plot(x, grouped["token_normalized_right_top1_energy"], marker="o", label="token normalized")
    ax.plot(x, grouped["token_energy_weighted_right_top1_energy"], marker="o", label="token energy weighted")
    ax.set_xticks(list(x), grouped["condition"], rotation=25, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("right top-1 cumulative energy")
    ax.set_title("Core 009A bridge: right-side collapse controls")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "right-collapse-controls.png", dpi=180)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    runs = _load_runs(args.out, protocol)
    args.out.mkdir(parents=True, exist_ok=True)

    expected = [int(x) for x in protocol["replication"]["diagnostic_seeds"]]
    completed = sorted(int(r["seed"]) for r in runs)
    missing = [s for s in expected if s not in completed]

    condition_df = pd.DataFrame(_condition_rows(runs))
    spectra_df = pd.DataFrame(_spectra_rows(runs))
    projection_df = pd.DataFrame(_projection_rows(runs))
    alignment_df = pd.DataFrame(_alignment_rows(runs))
    ablation_df = pd.DataFrame(_ablation_rows(runs))
    condition_df.to_csv(args.out / "condition-summary.csv", index=False)
    spectra_df.to_csv(args.out / "right-collapse-spectra.csv", index=False)
    projection_df.to_csv(args.out / "projection-residuals.csv", index=False)
    alignment_df.to_csv(args.out / "alignment.csv", index=False)
    ablation_df.to_csv(args.out / "top1-ablation.csv", index=False)

    aggregate = None
    flags = None
    if not missing and not condition_df.empty and not ablation_df.empty:
        aggregate = _aggregate_metrics(condition_df, ablation_df)
        flags = _flags(aggregate, protocol)

    decision = {
        "format": "minicells.core-validation.009a-right-collapse-bridge-decision.v1",
        "experiment_id": protocol["experiment_id"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": _sha256(PROTOCOL),
        "data_manifest_sha256": protocol["data"]["expected_manifest_sha256"],
        "status": "RIGHT_COLLAPSE_DIAGNOSTIC_COMPLETE" if not missing else "RIGHT_COLLAPSE_DIAGNOSTIC_INCOMPLETE",
        "scientific_decision": False,
        "supported": None,
        "source_009a_status": protocol["source_009a"]["status"],
        "source_009a_scientific_decision": True,
        "source_009a_status_changed": False,
        "completed_seeds": completed,
        "missing_seeds": missing,
        "aggregate_metrics": aggregate,
        "interpretation_flags": flags,
        "interpretation_boundary": "Post-confirmation diagnostic only; no Core 009A gate, winner, or scientific result is changed.",
    }
    (args.out / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot(condition_df, args.out)

    lines = [
        "# Core Validation 009A Bridge — Right-Side Collapse Robustness",
        "",
        f"- Status: `{decision['status']}`",
        "- Scientific decision: `False` (diagnostic bridge by construction)",
        f"- Source 009A remains: `{decision['source_009a_status']}`",
        f"- Completed seeds: `{completed}`",
        f"- Missing seeds: `{missing}`",
        "",
    ]
    if aggregate is not None:
        lines.extend(
            [
                "## Aggregate diagnostics",
                "",
                "```json",
                json.dumps(aggregate, indent=2, sort_keys=True),
                "```",
                "",
                "## Descriptive flags",
                "",
                "```json",
                json.dumps(flags, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Per-condition summary",
            "",
            condition_df.to_markdown(index=False) if not condition_df.empty else "No completed seeds.",
            "",
            "This bridge explains the 009A asymmetry only. It cannot revoke or strengthen the formal 009A support decision and does not test routing, sparsity, certificates, growth, or continual learning.",
            "",
        ]
    )
    (args.out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
