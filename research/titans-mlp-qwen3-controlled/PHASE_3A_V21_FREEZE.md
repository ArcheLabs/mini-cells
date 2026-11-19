# Freeze review

**NOT_FROZEN**. Formal Test is sealed and the formal experiment has not started.

| Item | Status |
|---|---|
| Template semantics and grammar | Codex review complete, owner review pending |
| Template families and pinned tokenizer offsets | CPU checks complete |
| Dataset balance and disjoint splits | CPU unit checks complete |
| Kaggle T4 and remote publication | PASS; independent artifact readback at cf6b359 |
| Formal configuration | Draft, formal runner integration pending |
| Owner sign-off | Pending |

Formal configuration SHA256:
`cb7948ab25cdea760483ee4cb95ed8045b250ef7766ecef79e6d8579fdff7e28`.

Template registry SHA256:
`e7fccd1fc0c1f4932e437cc19d882c5f908893cebe5693b86c4e3669eb252ba1`.

The draft specifies full-scale training, baselines, conditions and thresholds.
The current executable is the separate eight-update Smoke. A formal freeze
requires the formal runner consuming this draft, audited T4 Smoke and
publication evidence, and explicit owner approval.

The T4 run used CUDA:0, FP32 and PyTorch 2.11.0+cu130. It completed eight
adapter updates with a frozen Backbone. The remote commit was fetched
independently and all four publisher-listed artifacts matched their
byte counts and SHA256 hashes. The dataset contains train and validation
only. Validation Recall@1 matched CPU Smoke in all four conditions.

## Config-driven CPU preflight

The validation-only runner loads the draft, checks the verified snapshot
identity and template policy, consumes optimizer and loss settings, and
selects a checkpoint by held-both validation MRR. A reduced one-seed
32/32-fact run completed eight epochs and restored the selected checkpoint.
A config unit test rejects any attempt to open Test in this runner.
See smoke/cpu/config-preflight.json for the exact config digest, losses,
ranks and checkpoint digest.

The limited preflight does not exercise formal-scale data, all baselines,
rejection-threshold calibration or final Test reporting. These paths must
be implemented and verified before freeze approval.

The follow-up comparison preflight exercised all seven configured method
names and validation Value retrieval at 32 candidates. V1 Ridge uses
full-text last-token features on the corrected dataset with fixed ridge
regularization of 1. Each learned adapter selected a validation checkpoint
and calibrated a rejection cutoff on unknown entities and unseen relations.
The observed calibration FPR is in smoke/cpu/comparison-preflight.json.
This is an in-sample validation calibration statistic.

Use the same preflight with --full-validation to generate the configured
Train/Validation sizes for the first configured seed. That path remains
unexecuted at full scale. The 512-way evaluation, independent final
rejection assessment and five-seed acceptance are still pending.
