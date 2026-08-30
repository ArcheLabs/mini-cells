# Protocol V1

All MINI Cells payloads use explicit little-endian codecs with fixed magic/version fields. They do not use Rust layout, SCALE derive, JSON, or floating point inside the service.

Work starts with `MCW1`, version 1, operation, flags, and a 64-bit request ID. Operations are inference, training PLUS, training MINUS, and status probe. Inference carries an expected generation and at most 32 normalized text bytes. Training carries the generation and parent model hash.

Results start with `MCR1`, version 1, operation, status, request ID, generation, and model hash. Training results add side, integer loss, correct/total token counts, and evaluation digest. Inference results add bounded input/output and matching-token count.

Canonical storage uses `mc:v1:*` keys. Metadata identifies protocol/model/optimizer/capability/modality, generation, current hash, seeds, and ring cursors. Model bytes are exactly 8,952 bytes. Pending PLUS/MINUS entries are paired only when generation, parent hash, batch identity, and evaluation identity agree. History has 64 slots and inference results 16 slots. A stale result is a successful no-op and cannot rewrite canonical state.

MiniJAM Work carries finalized `MetaV1` and model bytes as external-data items 0 and 1 after initialization. The canonical work package commits to these sidecars. Refine verifies the model bytes against the metadata hash before evaluation. This is required because JAM Refine host call 3 is historical preimage lookup, while authoritative service-key storage is available only during Accumulate.
