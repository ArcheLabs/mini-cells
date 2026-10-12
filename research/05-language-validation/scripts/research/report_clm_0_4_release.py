#!/usr/bin/env python3
"""Render a compact CLM-0.4 Release report from completed artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    decision = json.loads((root / "decision.json").read_text(encoding="utf-8"))
    comparison = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    if decision.get("release_track") != "release":
        raise RuntimeError("not a CLM-0.4 Release result directory")

    clm = comparison["clm"]
    dense = comparison["dense"]
    smoke = decision["profile"] == "smoke-1m"
    lines = [
        "# CLM-0.4 Release Results",
        "",
        f"- Profile: **{decision['profile']}**",
        f"- Status: **{decision['status']}**",
        f"- Base tokens: **{int(decision['base_tokens']):,}**",
        f"- Transactions: **{int(decision['transactions'])}**",
        f"- Pipeline SHA-256: `{decision['pipeline_sha256']}`",
        "",
        "> The 1M profile is an end-to-end engineering smoke and is not a release-quality capability benchmark."
        if smoke else
        "> The 30M profile is the CLM-0.4 release benchmark/output.",
        "",
        "## Equal-parameter base comparison",
        "",
        f"- CLM parameters: **{int(clm['parameter_count_base']):,}**",
        f"- Dense parameters: **{int(dense['parameter_count_base']):,}**",
        f"- Difference: **{int(comparison['equal_parameter_difference'])} parameters**",
        f"- CLM base Math: **{pct(clm['base_capability']['math_teacher_forced_answer_exact'])}**",
        f"- Dense base Math: **{pct(dense['base_capability']['math_teacher_forced_answer_exact'])}**",
        f"- CLM base Story: **{pct(clm['base_capability']['story_teacher_forced_answer_exact'])}**",
        f"- Dense base Story: **{pct(dense['base_capability']['story_teacher_forced_answer_exact'])}**",
        "",
        "## After continual learning",
        "",
        f"- CLM final Math: **{pct(clm['final_capability']['math_teacher_forced_answer_exact'])}**",
        f"- Dense final Math: **{pct(dense['final_capability']['math_teacher_forced_answer_exact'])}**",
        f"- CLM final Story: **{pct(clm['final_capability']['story_teacher_forced_answer_exact'])}**",
        f"- Dense final Story: **{pct(dense['final_capability']['story_teacher_forced_answer_exact'])}**",
        f"- CLM protected retention: **{pct(clm['protected_retention_ratio'])}**",
        f"- Dense protected retention: **{pct(dense['protected_retention_ratio'])}**",
        f"- CLM effective commits: **{int(clm['effective_commits'])}**",
        f"- Dense commits (always-finetune baseline): **{int(dense['effective_commits'])}**",
        f"- CLM parameter growth: **{pct(clm['growth_parameter_overhead_ratio'])}**",
        f"- Dense parameter growth: **0.00%**",
        "",
        "## Comparison visualizations",
        "",
        *[f"- `{path}`" for path in comparison.get("visualizations", [])],
    ]
    if smoke:
        readiness = json.loads((root / "release-readiness.json").read_text(encoding="utf-8"))
        lines.extend([
            "",
            "## 30M readiness",
            "",
            f"- Status: **{readiness['status']}**",
            *[f"- {key}: **{'PASS' if value else 'FAIL'}**" for key, value in readiness["checks"].items()],
        ])
    text = "\n".join(lines) + "\n"
    (root / "RESULTS.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
