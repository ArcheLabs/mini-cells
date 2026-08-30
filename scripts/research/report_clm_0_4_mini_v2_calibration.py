#!/usr/bin/env python3
"""Render CLM-0.4-mini M1-v2 calibration report without formal claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results" / "clm-0.4-mini-m1-v2-calibration"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    decision = json.loads(
        (args.results / "decision.json").read_text(encoding="utf-8")
    )
    status = str(decision["status"])
    if decision.get("scientific_decision") is not False:
        raise RuntimeError("v2 calibration report refuses scientific decisions")
    lines = [
        "# CLM-0.4-mini M1-v2 Calibration",
        "",
        f"> Status: **{status}** — development calibration only.",
        "",
        "- v1 development seed `90401` is historical diagnosis only.",
        f"- v2 development seed observed: `{decision.get('development_seed_observed', False)}`",
        "- Formal seeds `90411/90412/90413` observed: `false`",
        "",
    ]
    summary_path = args.results / "summary.json"
    if summary_path.is_file() and status != "V2_CALIBRATION_PLAN_ONLY":
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        comparison = summary.get("base_comparison", {})
        clm = comparison.get("clm", {})
        capability = clm.get("capability", {})
        diagnostics = capability.get("diagnostics", {})
        lines.extend(
            [
                "## Aligned base admission",
                "",
                f"- Math teacher-forced answer exact: `{capability.get('math_teacher_forced_answer_exact')}`",
                f"- Story teacher-forced answer exact: `{capability.get('story_teacher_forced_answer_exact')}`",
                f"- Math greedy exact (diagnostic): `{diagnostics.get('math_greedy_exact_match')}`",
                f"- Story greedy exact (diagnostic): `{diagnostics.get('story_greedy_exact_match')}`",
                f"- Base prerequisite pass: `{clm.get('prerequisites', {}).get('pass')}`",
                "",
                "## Static dense controls",
                "",
            ]
        )
        for kind, payload in comparison.get("dense", {}).items():
            cap = payload.get("capability", {})
            lines.append(
                f"- `{kind}` params `{payload.get('parameter_count')}`: "
                f"math TF answer exact `{cap.get('math_teacher_forced_answer_exact')}`, "
                f"story `{cap.get('story_teacher_forced_answer_exact')}`"
            )
        if "dense_continual" in summary:
            lines.extend(["", "## Dense continual diagnostics", ""])
            for variant, payload in summary["dense_continual"].get(
                "variants", {}
            ).items():
                lines.append(
                    f"- `{variant}`: acceptance `{payload.get('effective_acceptance_rate')}`, "
                    f"retention `{payload.get('final_protected_retention_ratio')}`"
                )
    if status == "V2_CALIBRATION_CONFIGURATION_SELECTED":
        selected = json.loads(
            (args.results / "selected.json").read_text(encoding="utf-8")
        )
        candidate = selected["candidate"]
        lines.extend(
            [
                "",
                "## Selected CLM configuration",
                "",
                f"- Candidate: `{candidate['candidate_id']}`",
                f"- Direct: `{candidate['direct']}`",
                f"- Growth/private: `{candidate['growth_private']}`",
                "",
                "Formal execution is not authorized until the generated "
                "`protocol-lock.candidate.json` is reviewed and committed.",
            ]
        )
    elif status == "V2_CALIBRATION_BASE_PREREQUISITES_FAILED":
        lines.extend(
            [
                "",
                "Candidate search did not start because the aligned CLM base "
                "admission gate failed.",
            ]
        )
    elif status == "V2_CALIBRATION_NO_CONFIGURATION_PASSED":
        lines.extend(
            [
                "",
                "No registered configuration passed. Do not expand the grid "
                "post hoc; revise the protocol before formal execution.",
            ]
        )
    lines.extend(
        [
            "",
            "## Scientific boundary",
            "",
            "Dense baselines are diagnostic-only and never control CLM commit, "
            "growth, candidate selection, or the formal scientific decision.",
            "",
        ]
    )
    path = args.results / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
