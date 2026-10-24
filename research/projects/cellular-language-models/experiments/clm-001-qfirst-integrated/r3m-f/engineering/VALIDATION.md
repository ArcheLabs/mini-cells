# R3m-F engineering validation

Scope: synthetic CPU execution plus the exactly recovered public Tokenizer bytes. **No GPU training, private historical checkpoint download or model-quality evaluation was performed by this implementation task.**

## Checks

- Frozen matrix: three arms × three seeds × 900 updates = 8,100 updates / 4,147,200 tokens.
- Tiny Overfit: seed 151001301, eight fixed Train facts, 200 updates / 102,400 tokens, final 6/8 EM reference. No automatic Pilot release or budget extension.
- Fact generation with original Tokenizer: 44/11 disjoint A1 facts; 15/4 disjoint Copy numbers. Prompt-prefix exactness, digit-only answer offsets, EOS separation and multi-token numerical outputs checked.
- All facts/templates/orders remain in their assigned roles. Train-only probes; reused Story Development explicitly disclosed; sealed roles not opened.
- Finite effective Cell gradients, actual Cell updates, frozen Connection and unchanged embedding/head. Synthetic nine-job workflow verifies same-seed starting full state and all barriers.
- Full-state checkpoint round trip, RNG/cursor/optimizer identity, exact next update; simulated second-session immutable HF restore. This is **not** evidence of an actual Kaggle cross-session run for F.
- Free generation never receives target answer tokens. Complete EM requires the whole decoded answer and EOS; first-token result uses the actual generated first token. Fact aggregation does not inflate sample size with templates.
- Every Cell/each timepoint gets three lawful replacements. Exact full signed gradient cosine includes unused parameters as zero vectors, without a sketch. Diagnostics preserve model/RNG/work state.
- Local bare-Git result return verifies current-main ancestry, result-only files, remote commit SHA and idempotence. No public checkpoints or workflow edits in results branches.
- Original R3m-E regression suite runs alongside new tests. CPU protocol Gate and static checks pass. Final local run: **37 passed in 146.38s**, Python 3.8 / Torch CPU, using the repository's existing dependency cache. GitHub Actions separately runs Python 3.12. Ruff and compile checks pass.

## Actual unchanged model size

| Metric | Parameters |
|---|---:|
| Total | 1,006,694 |
| Trainable / Cells | 380,928 |
| Frozen Connection | 864 |
| Added | 0 |

## Pending real-run evidence and limitations

1. Tiny Overfit has **not run on GPU**. Whether the existing frozen embedding/head and Cell optimizer can memorize the eight facts is unknown. A failure cannot identify a unique architectural cause.
2. Actual Kaggle/HF persistence of this experiment remains unverified until a real saved run/recovery is observed. The existing transport is reused, not reimplemented. Uploaded checkpoint readback alone is not cross-session equivalence.
3. Full Pilot remains unreleased until independent Tiny-result review and authorization; A2/A3 and Connection reorganization remain prohibited. Its complete worker/analysis matrix is exercised only with synthetic CPU fixtures.
4. No real GPU performance estimate is claimed. Fixed finite workloads, serialized incremental uploads and resumable isolated analysis replace runtime termination limits. Tiny uses one training lane while two visible devices preserve the historical RNG layout.
5. Copy's disjoint numbers create TV distance 0.25 from A1 answer frequencies. One-digit mass is closely matched, but a positive F2−F1 contrast still needs this limitation considered.
6. Eleven unseen A1 facts and three seeds support exploratory evidence only. Wilson intervals refer to the stricter all-variants-correct fact success; variant-averaged EM is reported separately.
7. Historical Development is reused; no complete global historical content-isolation claim is made. Original R3m registration stays BLOCKED.
8. Cell 1 was selected from R3m-E. A five-timepoint correlation cannot establish causation or justify Connection reorganization.
