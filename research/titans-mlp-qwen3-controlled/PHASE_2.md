# Phase 2 — Frozen representation pilot

The pretrained audit passed before this extraction. CPU FP32 inference
captured the output of block index 22 and selected the last non-padding
token. Eight facts in each split have disjoint entities, values and
template pairs. There are three vectors per fact: write, value and query.
The manifest records shape, mean, standard deviation, norm, token and
padding counts and artifact hashes.

The projection check uses three independent, fixed seeded Gaussian
1024 by 512 matrices to inspect shape and scale. It provides no evidence of
semantic address alignment. The pilot used no optimizer or GPU. Phase 3
requires a separate training decision and remains pending.
