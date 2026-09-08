# Public deterministic fixtures

This directory contains small, deterministic test vectors used by the Rust
and Python parity tests:

* `v1/` — protocol-v1 fixed-model and generation vectors.
* `training-fidelity-v1/` — generated FP32 training traces and expected state.

The files are synthetic/reproducible engineering fixtures, not private or
copyrighted training data. Larger or restricted datasets belong in external
storage and must be represented here by a generator, content hash, and
provenance manifest rather than by committing the dataset itself.

The former root `fixtures/` path was consolidated here so fixtures are clearly
test assets rather than a product/runtime surface.
