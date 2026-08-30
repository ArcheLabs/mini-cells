"""CLM-0.4-mini token-level continual-learning primitives."""

from .curriculum import TransactionSpec, build_curriculum
from .m0 import M0Config, M0_SEED, replay_journal, run_m0
from .m1 import prepare_formal_data_assets, run_m1_infrastructure_smoke, run_m1_stream
from .model import MiniCLMConfig, StableAddressRouter, TinyCLMDecoder
from .protocol import CandidateOptimizerConfig, ProtocolError, candidate_grid, load_protocol
from .state import CellRegistry, DependencyIndex, TokenExample, model_state_hash

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
