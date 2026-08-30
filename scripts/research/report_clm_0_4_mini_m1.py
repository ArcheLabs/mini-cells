#!/usr/bin/env python3
"""Render the execution-only CLM-0.4-mini M1 infrastructure smoke report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results" / "clm-0.4-mini-m1-infrastructure"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
    decision = json.loads((args.results / "decision.json").read_text(encoding="utf-8"))
    if decision["status"] != "SMOKE_ONLY" or decision["scientific_decision"] is not False:
        raise RuntimeError("M1 infrastructure report refuses non-smoke decisions")
    diagnostic = summary["diagnostic_gate_snapshot"]
    variants = diagnostic["variant_summaries"]
    lines = [
        "# CLM-0.4-mini M1 Infrastructure Smoke",
        "",
        "> Status: **SMOKE_ONLY** — no scientific decision.",
        "",
        "This run validates the formal-scale interfaces before development-seed calibration. ",
        "It does not use seed `90401` and does not open `90411/90412/90413`.",
        "",
        "## Infrastructure",
        "",
        f"- Formal model parameter count: `{summary['formal_model_parameter_count']:,}`",
        f"- Smoke seed: `{summary['seed']}`",
        f"- Curriculum manifest: `{summary['curriculum_manifest_sha256']}`",
        f"- Base smoke tokens: `{summary['base_corpus_manifest']['actual_tokens']:,}`",
        f"- Smoke transaction projection: `{summary['transaction_projection_ids']}`",
        "",
        "## Variant projection",
        "",
        "| Variant | Commits | Acceptance | Growth bundles | FSR | Mean dependency coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("local_always", "local_tx", "local_tx_growth"):
        item = variants[name]
        lines.append(
            f"| `{name}` | {item['effective_commits']} | {item['effective_acceptance_rate']:.3f} | "
            f"{item['spawned_bundles']} | {item['false_safe_rate']:.3f} | "
            f"{item['mean_direct_dependency_coverage']:.3f} |"
        )
    lines.extend(
        [
            "",
            "These numbers are diagnostics from a reduced smoke projection. They must not be interpreted as M1 gate results.",
            "",
            "## Checkpoint replay",
            "",
        ]
    )
    for name, item in summary["checkpoint_replay"].items():
        lines.append(f"- `{name}`: `{'PASS' if item['match'] else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## Next boundary",
            "",
            "After this infrastructure is merged and independently smoke-tested, the next allowed scientific action is development-seed `90401` calibration over the already registered finite candidate grid. Formal seeds remain unopened until a committed `protocol-lock.json` exists.",
            "",
        ]
    )
    (args.results / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.results / "RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
