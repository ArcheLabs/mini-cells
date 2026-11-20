# Architecture boundaries

## TNT and MiniCells

TNT couples query projection onto a key subspace with hierarchical
memory and two training stages. The first stage uses hierarchical memory
to make chunkwise training efficient. The second adapts the model to its
target inference form. Q–K projection addresses the mismatch between
keys used to train memory and queries used to retrieve from it. Evaluate
those choices together when claiming a TNT comparison. A frozen
projection-only intervention is a separate ablation.

MiniCells uses layer-local and KV-head-local shards. Its persistent
addressing state belongs to each shard's versioned Cell. MiniCells
adaptations require their own training, budget and causal read/write
contract. Record the mapping to TNT mechanisms and any deviations before
reporting results. [TNT paper](https://openreview.net/pdf/424813e8424a5a7c3af990a764d389ec5b389490.pdf)

## Ownership and compatibility

A Cell stores per-layer/head fast weights, momentum, addressing tensors,
update counters and lineage. Store tensors in an allowlisted tensor
artifact. Bind the Cell to fingerprints of the backbone, tokenizer,
layer map, memory architecture and slow parameters. Slow parameters
live in a separate versioned model artifact.

Session owns attention KV, DeltaNet recurrent and convolution state,
token position, pending writes and any live session buffers. Restore
a Cell into a fresh Session and validate its contents before attaching.
Reject extra state fields and incompatible fingerprints.

Initialize every new Cell from the new architecture's independently
trained meta-parameters. Reject Stage 6C fast weights and momentum in
the new codec. Shape padding cannot establish a function-preserving
migration. Any future conversion requires an explicit state mapping
and independent retrieval evidence.
