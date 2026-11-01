# GC-001 Baseline Repair

Status: IMPLEMENTATION_READY; real P1/P2 training and P3 GPU diagnostics NOT_RUN.
Source branch: `codex/epclm-gc001-baseline-repair`. The supplied [specification](SPEC.md)
is retained as reference; [protocol.json](protocol.json) is the executable frozen design.

## Execute in order

1. [Batch A — P1](batch-a.ipynb): import into the existing private
   `bingjiangli/minicells` Kaggle Notebook. Enable GPU, Internet, KAGGLE_API_TOKEN and
   GITHUB_TOKEN, then Save & Run All. Train C1-BROADCAST, IC-ALL-SELF and F0-PM-BROADCAST
   from fresh initialization, each for 250K tokens. Original C1/F0 and all original
   GC samples/metrics are frozen references, not retrained or regenerated.
2. [Batch B — P2](batch-b.ipynb): only after A is published COMPLETE. Reuse all A
   results; train the existing Native T1 adapter for 250K tokens. No repeated C1 pilot
   or completed generation. Context128, KV disabled, actual bounded context replay.
3. [Batch C — P3](batch-c.ipynb): reuse compatible published A/B and original GC.
   No training, sampling reruns or model extension. Aggregate all retained samples,
   training/state diagnostics, EOS observations and prepare a blind review package.
   Missing/incomplete T1 is explicit and excluded from comparisons.

Each batch is one manual launch with automatic preflight, recovery, source/data
verification, execution and metadata-only publication. No Add Input/Output mounting,
manual SHA fields, CPU fallback training, automatic tuning or hidden 1M/10M expansion.
Published COMPLETE batches retry report publication without model inference/training.
Matching local progress takes priority over older remote progress. If a published
partial run has committed work but its private checkpoint cannot be recovered, stop
instead of restarting training. Transient recovery failures are archived and a later
manual launch may retry acquisition. Numerical/integrity failures remain terminal.

## Repository/provenance audit

Original result `49c6fef619d51cac9ce5079d698e2876ff2d144b` was independently fetched and
verified before editing. Original source `fc999f7677200ea875b8aacf2fe56ee896bd43ec`,
original train source 398a191, data manifest 1eac9053… and historical NO-COMM checkpoint
hash are pinned. Original report bytes are SHA-256 verified, not guessed from a branch
name. Original GC implementation, protocol, notebooks and result branch stay unchanged.
Only new repair modules, shared publisher registration and research indexes are added.

The old center-only NO-COMM stays historical: 15 blocks are zero and most pool inputs
are structurally inactive. Its recorded allocated/trainable budgets remain preserved;
new post-hoc structural audits are separate annotations. Old F0 also contains unused
cell projections/update: 1493120 allocated versus 1312896 structurally forward-used
parameters. Do not treat its allocated count as an active-parameter matching result.

## Matched architecture controls

[Independent/control architecture](architecture/independent_cellular.md) and
[Flat matching](architecture/flat_parameter_matched.md) describe the exact transitions.

C1-BROADCAST and IC-ALL-SELF have identical 966784 trainable parameters and initialization.
Every cell receives `roll(e,8*i)`; a constant int64 gather buffer implements this in
one indexed operation, with no new trainable parameters. The buffer is reported
separately from latent-state and weight memory. IC uses left/center/right projections
of self, with all three active; broadcast uses radius1 ring neighbors. Shared update,
readout, alpha .1, K4, input and training protocol are otherwise equal. Global readout
is allowed; independent evolution never reads another cell. Initial full parameter
hashes of the companion controls and embedding hashes across all new arms are checked.

Flat has a standalone 2048-vector RMSNorm/down/SiLU/up residual transition. No inherited
unused cellular projections are allocated. Structural integer sizing chooses rank43,
964736 active/allocated parameters, relative error about .212% from 966784. Rank44
has the same absolute error; ties choose the smaller rank. Parameter matching never
uses validation/holdout results. Stored coordinate activity and gradient participation
are not claims of 2048 independent information degrees of freedom: the low-rank update
bottleneck is explicit. This is parameter/state-storage matching, not compute matching.

All three start fresh model/optimizer states, use seed173018021, batch4/TBPTT64,
AdamW3e-4/.9/.95/decay.1, clip1, FP32, original warmup/cosine and exact original document
order/alignment/reset. Tokenizer and all asset hashes are verified. The audit must
match original GC exactly: 1827 training documents/416027 prediction tokens, 16 validation
documents/3322 tokens, 205 holdout documents/47403 tokens. Independent Test is unavailable;
tokenizer development exposure remains possible, confirmation is never accessed.

## Observations and costs

Prespecified updates250/500/750/977 record learning curves and first4 validation/first4
holdout diagnostic probes; these are labeled subset trends, not full validation curves
or selection criteria. Final quality evaluates all 16/205 documents. Milestone weights
are kept privately so interruption does not silently replace earlier states. If an
interrupted earlier diagnostic was not observed, it is marked missing, not evaluated
using a later checkpoint. Training completion requires all intended parameter tensors
have observed finite nonzero gradients; Flat output-row participation is CPU-tested.

Every model uses all original 16 prompts, greedy and .8/top-k40/top-p.95 sampling,
max128 tokens, seed713+prompt index. All outputs including poor samples are retained.
First-prompt greedy/sampling continuation checks use the same environment and private
sampling RNG. Numerical/token reproducibility is not promised across different hardware.
IC must show 16 nonzero blocks on real prefill. Prefill norms, nonzero count, full pairwise
similarity, dominant block share, norm/change progression and boundary proximity are
recorded. Pairwise means exclude zero blocks, so inactive zeros are not mistaken for
diversity. Flat storage blocks have no cellular semantic interpretation.

To avoid contaminating decode timings, immutable diagnostic state references are kept
only for the current sample, never fed back into the model. Norm/similarity calculations
run after generation timing. This temporary diagnostic memory is not latent model memory;
process peak VRAM includes it and other resident models. Standard history-length prefill/
decode tests reuse existing GC evaluation. Training and inference peaks are recorded
separately, along with analytical linear FLOPs, checkpoint size and measured times.

The communication comparison is C1-BROADCAST minus IC-ALL-SELF. Paired document bootstrap
intervals reuse the existing helper; one training seed and development-only data still
cannot establish architectural superiority. Original C1 versus center-only NO-COMM
is not a full-capacity independent-cell comparison. T1 is context-based reference only,
not an equivalent state-first model or a KV-optimized implementation.

## P3 and human review

C reuses all saved new-model numeric trajectories and original generated text. Historical
free-running state trajectories were not recorded and cannot be faithfully reconstructed
from decoded text; they remain NOT_AVAILABLE, with no rerun of completed generation.
C performs only new frozen teacher-forced EOS/calibration/state observations on the fixed
validation documents, using original C1/F0 and optional recovered old NO-COMM weights.
Original checkpoint curves supply genuine first50/last50 means and gradient norms;
historical validation trajectories remain NOT_AVAILABLE. Historical private-model
unavailability is reported, never repaired by retraining. Flags for repetition,
similarity, confidence or numerical range are descriptive, not coherence or causality.

The blind CSV includes every eligible model/prompt/mode with shuffled model-independent
sample IDs. Reviewers receive only this package, not named public outputs; the mapping
stays in private Kaggle Output. Six dimensions use 1–5, higher better (including less
degeneration). No reviewers are contacted or scores invented. Until supplied reviews
exist, status is PENDING and inter-reviewer agreement unavailable. A supplied
`human-review-scored.csv` is validated; source comments and ratings are preserved privately,
with aggregate ratings reported. Templates are not overwritten by score ingestion.
Changing the sample set requires a new review round rather than silently remapping IDs.

A conditional extension proposal checks recent training-loss improvement and fixed
validation-probe deterioration tolerance across the same selected arms. It authorizes
zero new training, creates no optimizer continuation and requires a separately frozen
protocol/identity and equal extension budget. Insufficient evidence yields no proposal.

## Budgets and publication

Every batch: execution5400s, private recovery cap180s **inside execution**, publication600s,
safety1200s, Notebook hard cap7200s. Recovery is not added a second time to the 7200-second total. Nested timers preserve the outer
deadline, and per-event checks bound inference. No CPU performance benchmark is run. Full synthetic gradient preflight is executed once for Batch A; B/C and publication retries reuse the matching A proof and only perform cheap configuration checks.

New namespaces: `gc001-baseline-repair/batch-a`, `batch-b`, `batch-c` under the existing
Core artifact tree. Results descend from main through the existing authenticated
metadata-only publisher; remote SHA is verified. Weights, optimizers, runtime snapshots,
raw stories/tokens and blind keys stay private. Named generated samples and numerical
summaries are public. Blind CSV is separate; reviewers must not browse named outputs.

## Validation and pending execution

Focused synthetic CPU contracts cover event permutation/access, isolation, synchronous
ring boundary, all-parameter gradients, Flat rank/activity, all-cell snapshots and source
checkpoint rejection, plus blind-review preservation and nested timers. All original GC
contracts/integration tests remain applicable. No actual GPU experiment, full local
language training or human review has been run for this implementation task. P1/P2/P3
execution and real findings remain pending the three Kaggle batches; H1–H5 NOT_EVALUATED.


## Recovery optimization after the 1a3bd87 attempt

The original failed result `0a404b94df2afc42779f3c7d5f47a4ce8c2ce59a` is unchanged:
RECOVERY exhausted ~601.5 seconds, with no new models/training entries. This operational
amendment changes recovery limits, not architectures, token budgets, data or evaluation.
New code/protocol identities do not reuse incompatible optimizer checkpoints.

- Local verified progress first; current-source history search is capped at two recent
  versions / 30 seconds, not 25 versions / ten minutes. Backend setup has a 30-second
  cap and is cached once per process across recovery helpers. Backend failure is not
  mistaken for proof that no checkpoint exists.
- A/B directly retrieve version142 input manifest, tokenizer, train and development
  files only. They neither download nor load original C1/F0 weights. Original metrics
  and generated text are reused from pinned GC publication. C retrieves historical
  weights only when it actually needs frozen diagnostics.
- Known input lookup has a 120-second cap and does not fall back to a wide history scan.
  All files are verified against the pinned data manifest and nested file hashes;
  complete local inputs bypass the network entirely. No new tokenizer/data generation.
- A short search with no known committed training may end and proceed to first training.
  Once matching progress or published committed training is known, failed recovery
  blocks instead of silently resetting a model. Completed models are never retrained.
- Resume transfers report metadata and the unfinished model checkpoint only, not complete
  model weights or old milestone copies. Deferred files keep explicit private-version /
  namespace / SHA locators in the report for later retrieval; saved outputs are not deleted.
- Initial recovery totals are capped at 180 seconds, within unchanged execution5400s /
  publication600s / safety1200s / hard7200s. Phase timings, searched versions, requested
  and verified inputs, backend initialization and failed substage are retained even
  on timeout. Recovery timeouts report BLOCKED_MODEL_ASSET_UNAVAILABLE, not a training
  interruption. No budget is enlarged and no automatic retraining is introduced.

Validation for this amendment is one mocked recovery contract (known-version data only,
no historical weights, local hash-checked reuse, shared backend, selective unfinished
checkpoint restore), plus static checks. No full CPU suite, model run or GPU pilot.

### Checkpoint metadata hotfix

Fix `st_size` attribute access after completed training, and return TypeError failures through the existing report publisher. No training, model, data, or budget changes. The exact `6ac7cc1e465eb7d985a23ed35caab6c81f4cef89` source may resume only while the Git diff is confined to the audited recovery/metadata files, this note and its regression test; protocol/data/report identity and checkpoint hashes remain checked. Original per-model training identities are retained, and the new execution records its predecessor. No general cross-source optimizer reuse is enabled. Recent-version search remains bounded; missing progress must not be interpreted as evidence that training never happened.
