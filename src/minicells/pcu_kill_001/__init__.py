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
    mark_formal_seed_running,
    git_provenance,
    sha256_file,
)
from .registry import CellRecord, CellRegistry, merge_registries, rollback_registry
from .synthetic import (
    DatasetAudit,
    SyntheticWorld,
    POSITIVE_CONTROL_VERSION,
    audit_dataset,
    context_oracle,
    generate_world,
)
from .cache import CacheEquivalence, CacheSemanticsInvalid, CachedTailRunner, TailCache, validate_cache_identity
from .training import ForkedCell, ForkedCellularExpert, ForkedCellularExperts, allocate_topk
from .composition import ComposedCell, ComposedCellularExpert, ComposedCellularExperts, FunctionalCellDelta, compose_cellular_experts
from .lora import ComposedLoRACell, LoRAConfig, LoRACell, MatchedLoRAExpert, MatchedLoRAExperts, merge_lora_factors, merged_effective_deltas
from .task import IGNORE_INDEX, TailTrainingCache, TaskSequences, answer_token_cross_entropy, build_task_sequences, cache_task_sequences, load_task_cache, save_task_cache, validate_answer_only_labels
from .task_training import TaskBranchResult, slice_task_cache, task_conditioned_allocation, train_cached_branch, train_cached_lora_branch
from .evaluation import EvaluationSummary, evaluate_matrix, evaluate_samples, greedy_generate
from .overlay import ExpertsOverlayModel, model_with_experts_overlay

# The scientific pipeline historically deep-copied the entire foundation for
# every A/B/AB/LoRA evaluation state. All states differ only in the final MoE
# expert runtime, so make the resource-bounded overlay the package default.
# The swap is restored in a finally block on every forward and does not mutate
# foundation weights or routing state.
from . import experiment as _experiment

_experiment._model_with_experts = model_with_experts_overlay

# Persist G0/cache/model/dataset identity before any context-oracle or capacity
# early exit. This changes only audit timing, never the scientific worker.
from .pipeline_guard import install_pipeline_guard, persist_pre_science_evidence

install_pipeline_guard(_experiment)

# Granite E0/formal execution must also avoid materializing a second complete
# 1.3B FP32 foundation. Install the single-foundation runtime after experiment
# is fully imported, then patch execution's fail-fast G0 entry point. The
# scientific protocol is unchanged; only resident model layout and inference
# graph retention differ.
from .resource_runtime import (
    cellularize_in_place,
    full_moe_overlay_equivalence,
    g0_preflight as _resource_g0_preflight,
    inference_logits,
    run_formal_execution as _resource_formal_execution,
    run_granite_engineering as _resource_granite_engineering,
)

_experiment._logits = inference_logits
_experiment._run_granite_engineering = _resource_granite_engineering
_experiment.run_formal_execution = _resource_formal_execution

from . import execution as _execution

_execution._g0_preflight = _resource_g0_preflight


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
    "mark_formal_seed_running",
    "git_provenance",
    "sha256_file",
    "CellRecord",
    "CellRegistry",
    "merge_registries",
    "rollback_registry",
    "DatasetAudit",
    "SyntheticWorld",
    "POSITIVE_CONTROL_VERSION",
    "audit_dataset",
    "generate_world",
    "context_oracle",
    "CacheEquivalence",
    "CacheSemanticsInvalid",
    "CachedTailRunner",
    "TailCache",
    "validate_cache_identity",
    "ForkedCell",
    "ForkedCellularExpert",
    "ForkedCellularExperts",
    "allocate_topk",
    "FunctionalCellDelta",
    "ComposedCell",
    "ComposedCellularExpert",
    "ComposedCellularExperts",
    "compose_cellular_experts",
    "LoRAConfig",
    "LoRACell",
    "MatchedLoRAExpert",
    "MatchedLoRAExperts",
    "ComposedLoRACell",
    "merge_lora_factors",
    "merged_effective_deltas",
    "IGNORE_INDEX",
    "TaskSequences",
    "TailTrainingCache",
    "answer_token_cross_entropy",
    "build_task_sequences",
    "cache_task_sequences",
    "save_task_cache",
    "load_task_cache",
    "validate_answer_only_labels",
    "TaskBranchResult",
    "slice_task_cache",
    "task_conditioned_allocation",
    "train_cached_branch",
    "train_cached_lora_branch",
    "EvaluationSummary",
    "evaluate_matrix",
    "evaluate_samples",
    "greedy_generate",
    "ExpertsOverlayModel",
    "model_with_experts_overlay",
    "cellularize_in_place",
    "full_moe_overlay_equivalence",
    "inference_logits",
    "install_pipeline_guard",
    "persist_pre_science_evidence",
]
