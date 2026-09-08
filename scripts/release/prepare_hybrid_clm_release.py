"""Validate and stage a HybridCLM mutation for Hugging Face publication.

This command copies only the safe artifact files. It never uploads, executes a
formal protocol, or includes a foundation-model checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from minicells.hybrid.mutation import CellMutation


REQUIRED = ("mutation.safetensors", "cell_config.json", "manifest.json", "provenance.json", "evaluation.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mutation", type=Path, help="validated local mutation directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mutation = CellMutation.from_pretrained(args.mutation)
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        source = args.mutation / name
        if not source.is_file():
            raise SystemExit(f"missing required artifact file: {source}")
        shutil.copy2(source, destination / name)
    (destination / "README.md").write_text(
        "# MiniCells HybridCLM Cell mutation\n\n"
        "Artifact validated by `CellMutation.from_pretrained`; foundation weights are not included.\n",
        encoding="utf-8",
    )
    print({"status": "READY_FOR_HF_REVIEW", "digest": mutation.digest(), "output": str(destination)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

