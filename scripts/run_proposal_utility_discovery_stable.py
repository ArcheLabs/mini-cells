from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "research").exists():
    ROOT = Path("/kaggle/working/mini-cells")
if not (ROOT / "research").exists():
    raise RuntimeError("Run from a mini-cells checkout or /kaggle/working/mini-cells")
sys.path.insert(0, str(ROOT / "scripts"))

import run_proposal_utility_discovery as base  # noqa: E402
import run_proposal_utility_discovery_resumable as resumable  # noqa: E402


OUT = ROOT / "results" / "proposal-utility-discovery-stable-v1"
WORKER = ROOT / "scripts" / "run_proposal_utility_discovery_worker_stable.py"


def main() -> int:
    # Keep the original failed 019 worker artifacts untouched. The stable rerun
    # has its own result directory so gradient-oracle comparisons remain auditable.
    base.OUT = OUT
    base.WORKER = WORKER
    cache, _ = base.prepare_corpus()
    gpu_count = base.run_workers(cache)
    resumable._postprocess(gpu_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
