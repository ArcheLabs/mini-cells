"""Read-only reuse of existing accounting/statistics/checkpoint mechanisms."""
from __future__ import annotations
import sys
from pathlib import Path
OLD=Path(__file__).resolve().parents[3]/'005-compositional-cell-graph/implementation'
sys.path.insert(0,str(OLD))
from exp005.checkpoint import save_checkpoint,restore_checkpoint,file_sha256,canonical_sha256
from exp005.accounting import ObservedGemmFlops
from exp005.report import paired_bootstrap_delta
from experiments.common.train import make_batch,write_json
