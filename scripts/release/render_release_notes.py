#!/usr/bin/env python3
"""Render conservative GitHub release notes from a checked-in config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(config: dict) -> str:
    package = config["package"]
    foundation = config["foundation"]
    mutation = config["mutation"]
    expected = config["expected_metrics"]
    publication = config.get("publication", {})
    return f"""# {package['release_title']}

Scientific status: **Engineering Evidence · Formal Validation Pending**.

## HybridCLM

This preview packages the verified Granite MoE backend, explicit `CellPlacement`,
reversible `CellMutation` attach/detach, alpha scaling, rollback/restoration,
and the safetensors artifact schema with provenance validation.

## Granite MoE backend

- Base model: `{foundation['repo_id']}`
- Immutable base revision: `{foundation['revision']}`
- Cell placement: layer `{mutation['layer']}`, budget `{mutation['cell_budget']}`
- Engineering seed: `{mutation['engineering_seed']}`

## Current scientific evidence

The frozen engineering evidence reports ranking OFF `{expected['ranking_off']:.4f}`
and ranking ON `{expected['ranking_on']:.4f}` (gain `{expected['ranking_gain']:.4f}`).
The same-graph zero-state and restoration checks pass. Locality at alpha=1
remains unresolved under the frozen protocol.

## Known limitations

This release does not claim standalone CLM conversion, superiority to LoRA,
continual learning, or resolved locality. Formal validation has not started;
formal seeds remain reserved and untouched.

## Hugging Face

The separately published Cell artifact, when available:

- Model: `https://huggingface.co/{publication.get('hf_namespace', 'archelabs-org')}/{publication.get('model_repo', 'granite-3.1-1b-hybrid-cell-l7-k64')}`
- Space: `https://huggingface.co/spaces/{publication.get('hf_namespace', 'archelabs-org')}/{publication.get('space_repo', 'MiniCells-HybridCLM')}`
- Collection: `{publication.get('collection_title', 'MiniCells — Hybrid CLM')}`

The software package does not contain foundation-model weights.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("RELEASE_NOTES.md"))
    args = parser.parse_args()
    config = json.loads(args.release_config.read_text(encoding="utf-8"))
    args.output.write_text(render(config), encoding="utf-8")
    print(json.dumps({"status": "RELEASE_NOTES_RENDERED", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
