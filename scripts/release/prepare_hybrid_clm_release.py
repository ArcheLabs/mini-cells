"""Validate and stage a HybridCLM mutation for Hugging Face publication.

This command copies only the safe artifact files. It never uploads, executes a
formal protocol, or includes a foundation-model checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from minicells.hybrid.mutation import CellMutation


REQUIRED = ("mutation.safetensors", "cell_config.json", "manifest.json", "provenance.json", "evaluation.json")


def _read_config(path: Path | None) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path else None


def _readme(config: dict | None) -> str:
    if config is None:
        return (
            "# MiniCells HybridCLM Cell mutation\n\n"
            "Artifact validated by `CellMutation.from_pretrained`; foundation weights are not included.\n"
        )
    package = config["package"]
    foundation = config["foundation"]
    mutation = config["mutation"]
    return (
        f"# {package['release_title']}\n\n"
        "This repository contains a Cell mutation and metadata only; it does not\n"
        "redistribute the foundation model.\n\n"
        f"- Base model: `{foundation['repo_id']}`\n"
        f"- Immutable base revision: `{foundation['revision']}`\n"
        f"- Cell placement: layer `{mutation['layer']}`, budget `{mutation['cell_budget']}`\n\n"
        "Scientific status: **Engineering Evidence · Formal Validation Pending**.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mutation", type=Path, help="validated local mutation directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-config", type=Path)
    parser.add_argument("--validation-report", type=Path)
    args = parser.parse_args()
    config = _read_config(args.release_config)
    if args.validation_report:
        report = json.loads(args.validation_report.read_text(encoding="utf-8"))
        if report.get("status") != "READY_FOR_PUBLICATION":
            raise SystemExit("publication validation report is not READY_FOR_PUBLICATION")
    mutation = CellMutation.from_pretrained(args.mutation)
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        source = args.mutation / name
        if not source.is_file():
            raise SystemExit(f"missing required artifact file: {source}")
        shutil.copy2(source, destination / name)
    (destination / "README.md").write_text(_readme(config), encoding="utf-8")
    print({"status": "READY_FOR_HF_REVIEW", "digest": mutation.digest(), "output": str(destination)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
