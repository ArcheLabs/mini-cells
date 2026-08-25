# MiniJAM V0 implementation

MINI Cells V0 implements deterministic Echo inference and one-generation-at-a-time guarded SIGN-SPSA v2 training as a Rust `no_std` MiniJAM service. Refine evaluates BASE and the candidate on the same canonical batch. Accumulate only validates and pairs PLUS/MINUS results, accepts only a unique candidate strictly better than BASE, hashes the retained model, and writes bounded state.

Compatibility is pinned to JAM semantics 0.7.2, MiniJAM `1dceda20d501b19207fc33252f180e658dc064d7`, Jambda `0fb3591ed6f3a7479e3fa0c672711aa80827d486`, and the Rust-to-PVM converter at the MiniJAM ref. The build reuses the current MiniJAM SDK host ABI and converter; it does not define a parallel ABI.

The crates separate the pure fixed-point kernel, explicit wire protocol, host-independent runtime, in-memory simulator, service guest, and scheduler CLI. Capability IDs 1–5 and modality IDs 1–4 are reserved; only ECHO/TEXT is implemented and unknown values are rejected.

The canonical status and exact validation evidence are maintained in `IMPLEMENTATION_STATUS.md` and `artifacts/implementation-status.json`.
