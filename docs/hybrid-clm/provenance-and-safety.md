# Provenance and safety

Mutation weights use safetensors only. Loading validates the schema, tensor
keys/shapes, file size and SHA256, pinned base revision, architecture and
module signatures, placement identity, and target parameter shapes. Public
artifacts never execute arbitrary pickle deserialization.

Attach fails closed on a mismatch. An advanced override may be added by a
future backend, but v0.1 does not silently coerce a mutation. Attach/detach
tracks only declared targets, and `verify_restoration()` reports whether the
cellularized baseline was restored exactly.

Every published artifact should include source commit, protocol, dataset
identity/hash, seed, optimizer configuration, training steps, trainable count,
and a clear engineering/formal status.
