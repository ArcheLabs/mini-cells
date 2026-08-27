from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "scripts"))

import minicells.language_proposal_utility as proposal_utility  # noqa: E402
from minicells.language_recruitment_numerics import stable_gated_replicator_activity  # noqa: E402

# The utility forward imported the gated replicator into module scope. Replace
# only that numerical primitive; all scientific definitions, models, seeds,
# corpora, losses, epsilon probes and feature extraction remain unchanged.
proposal_utility._gated_replicator_activity = stable_gated_replicator_activity

import run_proposal_utility_discovery_worker as worker  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(worker.main())
