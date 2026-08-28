#!/usr/bin/env python3
"""Formal publisher entrypoint for the CLM-0.3 release benchmark.

The release-only source-corpus equivalence proof is required, validated, copied,
and therefore included in the publication metadata hash manifest.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CORE = HERE / "publish_clm_0_3_release_benchmark.py"
spec = importlib.util.spec_from_file_location("clm_0_3_release_publisher_core", CORE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load release publisher: {CORE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.TOP_LEVEL = (*module.TOP_LEVEL, "source-corpus-equivalence.json")
_original_validate = module._validate


def _validated(source: Path):
    result = _original_validate(source)
    proof = json.loads((source / "source-corpus-equivalence.json").read_text(encoding="utf-8"))
    if proof.get("status") != "CLM_RELEASE_SOURCE_006_CORPUS_EQUIVALENCE":
        raise RuntimeError("release source-corpus equivalence was not established")
    decision = json.loads((source / "decision.json").read_text(encoding="utf-8"))
    bridge = json.loads((source / "bridge-summary.json").read_text(encoding="utf-8"))
    if decision.get("source_corpus_equivalence") != proof:
        raise RuntimeError("release decision is not bound to the source-corpus proof")
    if bridge.get("source_corpus_equivalence") != proof:
        raise RuntimeError("release bridge summary is not bound to the source-corpus proof")
    return result


module._validate = _validated

if __name__ == "__main__":
    raise SystemExit(module.main())
