# Experiment 002 local training architecture

The research path has three deliberately separate layers:

1. `minicells-sim::trainer` is persistent native training. It calls the same
   `minicells-runtime` refine and accumulate functions used by the guest,
   writes deterministic checkpoints and JSONL metrics, and validates model
   hashes on resume.
2. `minicells-dataset` is the canonical JSONL compiler. Its output contains a
   deterministic dataset root, sorted/deduplicated samples, train/validation
   split, and deterministic batch identity. It also emits Merkle membership
   proofs for the eventual guest verifier.
3. `minicells-pvm` owns the chain-free host boundary and the real artifact
   identity. It refuses to claim execution until the pinned Jambda
   `RefineCtx` can be supplied with a chain-free state adapter.

The synthetic V1 path is not removed: its golden generation vector remains a
regression test. Dataset-backed native training uses the same model, fixed
point math, SIGN-SPSA optimizer, and evaluation code, with `BatchSelection`
providing the explicit dataset/batch identity.

The current direct PVM blocker is external to the native implementation. The
pinned MiniJAM/Jambda API exposes `VmEngine<InterpBackend>` through a
chain-oriented `RefineCtx` and `StateView`; implementing a local adapter needs
the corresponding state backend and hostcall ABI in the pinned supporting
repository. The CLI reports this typed blocker instead of substituting native
execution under a PVM label.
