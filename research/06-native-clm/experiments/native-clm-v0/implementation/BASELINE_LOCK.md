# Native CLM baseline lock criteria

`NCLM-001` must not be promoted until every item below is complete. The code/protocol is now substantially frozen; data hashes and accelerator evidence are produced by the first real runs.

## Required lock

- [x] Dataset identity/revision frozen: `roneneldan/TinyStories@f54c09fd23315a6f9c86f9e14f9f4f06615b22e3`.
- [ ] Tokenizer artifact/hash frozen and shared by Transformer and CLM. The deterministic 2,048 BPE recipe is frozen; the concrete artifact/hash is created by the first pinned corpus build.
- [x] Context length frozen at 128; training sequence length 125.
- [x] Baseline train-token budget frozen at 10M/model (`baseline` profile); `smoke` and `dev` are explicitly non-decision profiles.
- [x] Optimizer, LR schedule, warmup, weight decay, clipping, precision and batch-token policy frozen in `native_clm_runtime.py`.
- [x] Development seeds (`91001–91003`) separated from untouched confirmation seeds (`91101–91103`).
- [x] `T1` modern Transformer architecture frozen for the initial baseline.
- [x] `C0` provenance reconstructed from historical Experiment 007 and scaled to the ~2M regime.
- [x] Parameter mismatch registered below 1%: C0 1,899,576 vs T1 1,894,080 (~0.29%).
- [ ] Training-FLOP estimator independently checked against a profiler/reference counter on both model families. The analytical estimator is implemented and labelled as an estimate.
- [x] Inference analytical FLOP/token estimator accounts for all 12 recurrent C0 steps and hierarchical local-attention windows.
- [x] PPL path uses identical tokenizer, batch schedule family, causal targets and fixed validation examples within each paired seed.
- [ ] Accelerator smoke/baseline execution passes with the frozen corpus and produces resumable checkpoints + paired summaries.
- [ ] All three development baseline seeds complete without protocol change.
- [ ] Baseline interpretation/candidate-selection rule frozen before confirmation.
- [ ] All three untouched confirmation seeds complete without architecture/protocol change.

## Promotion rule

A development result can motivate a candidate but cannot establish a validated optimization. Freeze the candidate first, then run untouched confirmation seeds.

Every claimed optimization must state which comparison it supports:

- quality / parameter;
- quality / training compute;
- quality / inference compute.

Do not infer one category from another. In particular, the equal-parameter baseline is expected to use materially more recurrent compute for C0; a lower C0 PPL would initially establish parameter efficiency only.
