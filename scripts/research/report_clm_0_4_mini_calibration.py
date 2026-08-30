#!/usr/bin/env python3
"""Render CLM-0.4-mini M1 calibration report without scientific claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results" / "clm-0.4-mini-m1-calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    decision = json.loads((args.results / "decision.json").read_text(encoding="utf-8"))
    status = str(decision["status"])
    if decision.get("scientific_decision") is not False:
        raise RuntimeError("calibration report refuses scientific decisions")
    lines = [
        "# CLM-0.4-mini M1 Calibration",
        "",
        f"> Status: **{status}** — development calibration only; no M1 scientific decision.",
        "",
    ]
    if status == "CALIBRATION_PLAN_ONLY":
        lines.extend([
            f"- Candidate count: `{decision['candidate_count']}`",
            f"- Plan SHA-256: `{decision['plan_sha256']}`",
            "- Development seed observed: `false`",
            "- Formal seeds observed: `false`",
        ])
    else:
        summary = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
        base = summary["base"]
        lines.extend([
            "## Base prerequisites",
            "",
            f"- Math exact match: `{base['math_exact_match']:.4f}`",
            f"- Story exact match: `{base['story_exact_match']:.4f}`",
            f"- Minimum base Cell activation: `{base['minimum_base_cell_activation']}`",
            f"- Base prerequisite pass: `{base['prerequisites']['pass']}`",
            "",
        ])
        if status == "CALIBRATION_CONFIGURATION_SELECTED":
            selected = summary["selected"]
            candidate = selected["candidate"]
            lines.extend([
                "## Selected configuration",
                "",
                f"- Candidate: `{candidate['candidate_id']}`",
                f"- First passing ordinal: `{selected['first_passing_ordinal']}`",
                f"- Candidates evaluated: `{selected['candidates_evaluated']}`",
                f"- Direct: `lr={candidate['direct']['learning_rate']}, steps={candidate['direct']['steps']}`",
                f"- Growth/private: `lr={candidate['growth_private']['learning_rate']}, steps={candidate['growth_private']['steps']}`",
                "- Protocol-lock candidate: `protocol-lock.candidate.json`",
                "",
                "Formal execution is still **not authorized**. The generated lock must be reviewed and committed as the canonical `protocol-lock.json` before seeds `90411/90412/90413` may be opened.",
            ])
        elif status == "CALIBRATION_NO_CONFIGURATION_PASSED":
            lines.extend([
                "## Outcome",
                "",
                "No registered configuration passed in the frozen order. Do not open formal seeds and do not expand the grid post hoc. A protocol revision is required before further scientific execution.",
            ])
        elif status == "CALIBRATION_BASE_PREREQUISITES_FAILED":
            lines.extend([
                "## Outcome",
                "",
                "The base model failed one or more pre-registered prerequisites. Candidate search did not proceed. Formal seeds remain unopened.",
            ])
    lines.extend([
        "",
        "## Scientific boundary",
        "",
        "This calibration uses development seed `90401` only. It does not establish CLM-0.4-mini support or non-support; that decision is reserved for the three unseen formal seeds after protocol lock.",
        "",
    ])
    (args.results / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.results / "RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
