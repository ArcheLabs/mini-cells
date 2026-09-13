# Native CLM baseline lock criteria

The initial notebook implementation is an engineering scaffold until all items below are satisfied. Do not promote `NCLM-001` before this lock is complete.

## Required lock

- [ ] Dataset revision and train/validation split identity frozen.
- [ ] Tokenizer artifact/hash frozen and shared by Transformer and CLM.
- [ ] Context length frozen.
- [ ] Train-token budget frozen.
- [ ] Optimizer, LR schedule, warmup, weight decay, clipping, precision and batch-token policy frozen.
- [ ] Development seeds separated from untouched confirmation seeds.
- [ ] `T1` modern Transformer architecture frozen.
- [ ] `C0` historical/native CLM baseline provenance documented.
- [ ] Parameter mismatch is within the registered tolerance or explicitly controlled by a parameter sweep.
- [ ] Training-FLOP estimator checked on both model families.
- [ ] Inference-FLOP/token estimator accounts for recurrent CLM steps.
- [ ] PPL uses identical tokenization, masking and validation examples.
- [ ] Smoke tests pass on CPU and accelerator execution path.

## Promotion rule

A development result can motivate a candidate but cannot establish a validated optimization. Freeze the candidate first, then run untouched confirmation seeds.

Every claimed optimization must state which comparison it supports:

- quality / parameter;
- quality / training compute;
- quality / inference compute.

Do not infer one category from another.
