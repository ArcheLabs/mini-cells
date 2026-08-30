# State provider boundary

MINI Cells V0.1 uses inline `META` and `MODEL` witnesses in the WorkPackage.
The chain crate exposes `ModelWitnessProvider` so a later state-plane provider
can replace the inline model without changing submission or verification. The
JamScript M0 facade in `../JamScript/crates/service-state-plane` provides the
versioned commitment/request/witness/provider seam over the existing
`FullState` and `ProofState` implementations.
