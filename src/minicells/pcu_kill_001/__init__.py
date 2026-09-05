"""PCU-KILL-001: a fail-closed, registry-first MoE cellularization harness.

The package deliberately keeps the pretrained router outside the Cell runtime.
It is useful both with the Hugging Face Granite-MoE implementation and with
small deterministic test doubles used by the engineering test suite.
"""

from .cellular import (
    CellPartition,
    CellProjection,
    CellularExpert,
    CellularExperts,
    GraniteArchitectureInspector,
    UnsupportedFoundationArchitecture,
    extract_expert_projections,
    patch_moe_block,
)
from .governance import (
    DEVELOPMENT_SEED,
    FORMAL_SEEDS,
    ProtocolMismatch,
    assert_engineering_seed,
    assert_formal_preflight,
    git_provenance,
    sha256_file,
)
from .registry import CellRecord, CellRegistry, merge_registries, rollback_registry
from .synthetic import DatasetAudit, SyntheticWorld, audit_dataset, generate_world
from .cache import CacheEquivalence, CacheSemanticsInvalid, CachedTailRunner, TailCache
from .training import ForkedCell, ForkedCellularExpert, ForkedCellularExperts, allocate_topk

__all__ = [
    "CellPartition",
    "CellProjection",
    "CellularExpert",
    "CellularExperts",
    "GraniteArchitectureInspector",
    "UnsupportedFoundationArchitecture",
    "extract_expert_projections",
    "patch_moe_block",
    "DEVELOPMENT_SEED",
    "FORMAL_SEEDS",
    "ProtocolMismatch",
    "assert_engineering_seed",
    "assert_formal_preflight",
    "git_provenance",
    "sha256_file",
    "CellRecord",
    "CellRegistry",
    "merge_registries",
    "rollback_registry",
    "DatasetAudit",
    "SyntheticWorld",
    "audit_dataset",
    "generate_world",
    "CacheEquivalence",
    "CacheSemanticsInvalid",
    "CachedTailRunner",
    "TailCache",
    "ForkedCell",
    "ForkedCellularExpert",
    "ForkedCellularExperts",
    "allocate_topk",
]
