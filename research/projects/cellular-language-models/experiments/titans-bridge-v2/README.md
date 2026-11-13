# Titans Bridge v2

## Track A reproduction

From this directory, run the frozen P4 entry points in the same CPU PyTorch
environment. Use fresh output paths.

```bash
OMP_NUM_THREADS=1 python ../p4-peer-titans-collaboration/run_m1.py --output /tmp/bridge-m1
OMP_NUM_THREADS=1 python ../p4-peer-titans-collaboration/m1.1-capacity/smoke.py /tmp/bridge-m11
```

The historical manifests are engineering evidence. Inspect the per-episode
retrieval metrics and the forgetting series before treating the positive
control or the retention failure as reproduced. A runner PASS means the
measurements completed.

## Open implementation decisions

The comparison requires a verified MiniCells 0.8 adapter checkpoint and its
training split identities. The previous pilot trains with LM loss and its
weights are retained in Kaggle Output. Bridge evaluation freezes the
checkpoint and reads the Titans scores directly. Compare the historical
initialization and learned adapter separately, recording the projection
identities and holding values fixed within each matched comparison.

At eight associations, Chunk 8 and Batch have identical update widths.
Record both labels as aliases there and distinguish them at 32 and 64.
Choose a single geometry across loads for the new matrix. Preserve the
historical geometry exactly in Track A. Freeze the fixed-control constants,
negative construction and validation-only rejection threshold rule before
the test matrix. Use at least 100 paired episodes per primary condition
and bootstrap at episode level.

Tracks B and later remain pending. Test data have not been materialized.
