"""CLM-0.4-mini token-level continual-learning primitives."""

from .model import MiniCLMConfig, StableAddressRouter, TinyCLMDecoder
from .protocol import CandidateOptimizerConfig, ProtocolError, candidate_grid, load_protocol

# The model and protocol are usable for lightweight infrastructure checks
# without the optional byte-BPE dependency.  Keep the historical convenience
# exports when the full research dependency set is installed.
from .curriculum import TransactionSpec, build_curriculum

try:
    from .m0 import M0Config, M0_SEED, replay_journal, run_m0
    from .m1 import prepare_formal_data_assets, run_m1_infrastructure_smoke, run_m1_stream
    from .state import CellRegistry, DependencyIndex, TokenExample, model_state_hash
except ModuleNotFoundError as exc:
    if exc.name != "tokenizers":
        raise

__all__ = [
    "MiniCLMConfig",
    "StableAddressRouter",
    "TinyCLMDecoder",
    "M0Config",
    "M0_SEED",
    "run_m0",
    "replay_journal",
    "CellRegistry",
    "DependencyIndex",
    "TokenExample",
    "model_state_hash",
    "TransactionSpec",
    "build_curriculum",
    "CandidateOptimizerConfig",
    "ProtocolError",
    "candidate_grid",
    "load_protocol",
    "prepare_formal_data_assets",
    "run_m1_infrastructure_smoke",
    "run_m1_stream",
]
