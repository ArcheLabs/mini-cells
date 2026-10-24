"""CPU inspection of a CUDA parent; never a production-resume adapter."""

import copy
from minicells_r3m import anchor
from minicells_r3l.continuation import restore as parent_restore
from minicells_r3k.reuse import bind
from .common import fingerprint


def _restore_for_cpu_inspection(runtime, state, seed, arm, device):
    if str(device) != "cpu":
        raise RuntimeError("R3ME_INSPECTION_ADAPTER_CPU_ONLY")
    original_sha = fingerprint(state)
    inspection = copy.deepcopy(state)
    cuda_state = inspection["global_cuda_rng"]
    evidence = {
        "scope": "CPU_MODEL_OPTIMIZER_IDENTITY_INSPECTION_ONLY",
        "original_cuda_rng_count": len(cuda_state),
        "original_cuda_rng_sha256": fingerprint(cuda_state),
        "cuda_rng_applied": False,
        "gpu_resume_verified": False,
    }
    # The original artifact remains intact. CUDA RNG is not applicable to CPU inspection.
    inspection["global_cuda_rng"] = []
    trainer = parent_restore(runtime, inspection, seed, arm, "cpu")
    if fingerprint(state) != original_sha:
        raise RuntimeError("R3ME_INSPECTION_MUTATED_PARENT_STATE")
    trainer.cpu_inspection_evidence = evidence
    return trainer


def load_for_cpu_inspection(c, backend, destination, seed):
    loader = bind(anchor.load, restore=_restore_for_cpu_inspection)
    trainer, proof = loader(c, backend, destination, seed, "cpu")
    return trainer, dict(proof, cpu_inspection=trainer.cpu_inspection_evidence)
