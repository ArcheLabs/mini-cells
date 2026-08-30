# CLM-0.3d — Probationary Mitosis

## Question

Can a trained CLM decide whether a prospective lineage deserves persistent existence by letting a **shadow lineage develop through a multi-horizon probation**, rather than making a one-shot short-horizon birth decision?

CLM-0.3d is the direct successor to CLM-0.3c. It keeps the validated CLM-0.3 birth operator frozen and changes only the developmental decision protocol.

The central hypothesis is:

> A useful lineage may require a non-trivial maturation horizon. Shadow creation is therefore a proposal, not a birth. A lineage is promoted only after sustained future advantage over the matched no-growth continuation; otherwise it undergoes apoptosis.

## Prior evidence

CLM-0.3 established function-preserving hierarchical lineage birth. CLM-0.3b confirmed that newborns can acquire statistically detectable causal specialization while whole-model marginal utility remains small. CLM-0.3c then evaluated all 12 possible single births from a common checkpoint. All 39 executed births were function preserving, but the 100K one-shot policy was not calibrated to 500K value in any replicate. One replicate was inconclusive at 100K yet developed a statistically positive and practically meaningful 500K advantage.

Earlier MiniCells Experiment 023b independently established a narrower developmental principle in a phenotype substrate: temporary shadows should be evaluated over multiple prequential windows and committed only when their advantage is sustained. Experiment 024b further showed that a new trait should correspond to functional non-substitutability rather than mere ability to specialize.

CLM-0.3d combines these lines in the formal CLM lineage substrate.

## Frozen substrate

- Base release: CLM-0.1
- Base SHA-256: `87d36c408ae3873ffd567ebf17050661b42ddae2c8d5d1bab84b2c27c3c7e7a0`
- Recurrent substrate: unchanged GRUCell core
- Root topology: 3 stages × 4 CLM-0.1 experts
- Birth: exact `child = deepcopy(parent)`
- Child competes only inside the parent lineage
- Geometry initialization: deterministic cosine k-means, k=2
- Optimizer: AdamW; child inherits parent moments; split router starts fresh
- Scheduler: continuous global continuation schedule
- Root balance auxiliary loss: `0.0`
- Replicate seeds: `55031`, `55032`, `55033`

No CLM-0.3d result may be used to modify the birth operator.

## Developmental vocabulary

CLM-0.3d distinguishes three states:

1. **proposal** — a prospective parent lineage is selected for shadow evaluation;
2. **shadow lineage** — the duplicated child develops counterfactually but has no persistent structural authority;
3. **promoted lineage** — the shadow has survived probation and is allowed to become persistent structure.

Creating a shadow is therefore not counted as a scientific claim of birth. `PROMOTE` is the persistent-birth event. `REJECT` is apoptosis.

## Shared decision checkpoint

Each replicate starts from the released CLM-0.1 model and receives exactly **1.5M TinyStories continuation tokens**. The resulting serialized model, optimizer, scheduler, RNG state, and data schedule form one immutable decision checkpoint.

Both formal environmental conditions start from this exact checkpoint.

## Formal environmental conditions

Each replicate runs both conditions.

### 1. `stationary_story`

The future stream remains TinyStories. This is the negative/control environment. A robust growth controller should not be forced to invent persistent structure merely because a copied lineage can specialize.

Expected developmental outcome: `REJECT`.

### 2. `story_arithmetic_shift`

The future stream is a deterministic 50/50 mixture of TinyStories and the existing tokenizer-compatible synthetic Arithmetic corpus used by the earlier cellular differentiation studies.

This is a controlled capability-shift environment, not a semantic routing rule. Domain identity is used only to construct the experimental data/objective and to perform posthoc retention diagnostics. It is never provided to the CLM router, geometry initializer, shortlist rule, or promotion rule.

Training objective:

- Story batches: `CE + 0.5 * KL(student || frozen CLM-0.1)`
- Arithmetic batches: `CE`

The frozen-teacher KL protects inherited-language behavior; it is deliberately not imposed on the novel arithmetic capability. Control and all shadows consume the identical chronological future batches and objective.

A shift is not assumed to require mitosis. The matched no-growth arm may economically absorb the capability. This is measured explicitly.

## Shadow candidates

At the decision checkpoint, all 12 CLM-0.1 root lineages are eligible proposals.

CLM-0.3d does **not** use CLM-0.3c Split Regret to decide WHERE. The 0.3c formal result showed negative Spearman correlation in all three replicates. The existing calibration machinery is reused only to collect enough routed perceptions for deterministic geometry initialization.

For every condition:

1. one no-growth control is forked from the decision checkpoint;
2. one shadow is created for each of the 12 root lineages from the same checkpoint;
3. every shadow must pass `CLM_GROWTH_EQUIVALENCE` before its first optimizer step.

Across 3 replicates × 2 conditions, the formal matrix therefore contains **72 executed shadow births**.

## Multi-horizon probation

The no-growth control develops continuously to 500K future tokens.

Every shadow first develops to 100K. All branches are evaluated at:

- 50K
- 100K

The evaluation uses holdout **A** and a deterministic paired 2,000-sample bootstrap.

For lineage `i` at age `h`:

`Q_i(h) = (NLL_control(h) - NLL_shadow_i(h)) / NLL_control(h)`

The first 100K is **not** a birth/death gate.

### Shortlisting

After all 12 shadows reach 100K, retain the four highest point-estimate `Q_i(100K)` values. Ties are broken deterministically by expert ID.

The shortlist exists only to bound experimental compute. Analytic gradient scores do not enter it.

The four shortlisted shadows continue from their actual 100K states and are evaluated at:

- 200K
- 300K
- 500K

Thus each shortlisted lineage has a developmental value trajectory rather than a single endpoint.

## Holdout-A probation gate

For a shortlisted lineage to become eligible for independent confirmation, all must hold:

1. `LCB95[Q_i(300K)] > 0`;
2. `LCB95[Q_i(500K)] > 0`;
3. mean point `Q_i` across 200K / 300K / 500K is positive;
4. 500K shadow/control PPL ratio is `<= 0.995`.

The last rule retains the preregistered practical effect threshold from CLM-0.3c: at least 0.5% lower PPL. It is kept separate from NLL utility rather than subtracting an arbitrary structural cost in mismatched units.

If several shadows pass, choose the one with the largest mean late-horizon relative improvement, followed by larger final lower confidence bound and deterministic expert-ID tie breaking.

If none pass, the condition returns `REJECT` without examining holdout B for selection.

## Independent holdout-B confirmation

Holdout **B** is completely disjoint from holdout A and is never used for shortlisting or probation selection.

A holdout-A winner is promoted only if, at 500K on holdout B:

1. paired `LCB95[Q_i] > 0`;
2. shadow/control PPL ratio is `<= 0.995`;
3. under `story_arithmetic_shift`, Story NLL relative to the matched no-growth control is `<= 1.01`.

Failure on holdout B produces `REJECT_OVERFIT`, not promotion.

## Functional absorption diagnostic

The shift condition separately asks whether the no-growth model can absorb the new capability.

Let the decision checkpoint have Story/Arithmetic NLLs `L0_s`, `L0_a`, and the 500K no-growth shift continuation have `Lc_s`, `Lc_a`.

`G_arithmetic = (L0_a - Lc_a) / |L0_a|`

`D_story = max((Lc_s - L0_s) / |L0_s|, 0)`

`U_absorb = G_arithmetic - D_story`

The shift is diagnostically `absorbable_without_mitosis` when:

- arithmetic gain `>= 0.02`;
- Story damage `<= 0.01`.

This diagnostic does not force rejection or promotion. It exists to distinguish "controller failed to grow" from "existing structure learned the shift economically".

## Maturation rescue

CLM-0.3c suggested that useful structure can be inconclusive at 100K but valuable at 500K.

A promoted lineage is counted as a **maturation rescue** when:

- its 100K holdout-A lower confidence bound was `<= 0`; and
- its 500K independent holdout-B promotion is confirmed.

At least 2/3 shift replicates meeting this pattern gives `CLM_LINEAGE_MATURATION_SIGNAL`.

## Formal decisions

### Growth equivalence

All 72 shadow births must pass exact parity:

`CLM_PROBATIONARY_GROWTH_EQUIVALENCE`

### Stationary specificity

At least 2/3 `stationary_story` replicates must end without confirmed promotion:

`CLM_STATIONARY_REJECTION_SIGNAL`

If at least 2/3 stationary replicates promote, report:

`CLM_STATIONARY_OVERGROWTH`

### Shift sensitivity

At least 2/3 `story_arithmetic_shift` replicates must independently confirm promotion:

`CLM_SHIFT_PROMOTION_SIGNAL`

### Overall signal

Strong positive requires:

- 72/72 growth equivalence;
- stationary rejection in at least 2/3 replicates;
- shift promotion in at least 2/3 replicates.

Status:

`CLM_PROBATIONARY_MITOSIS_SIGNAL`

If shift promotion fails but the no-growth model economically absorbs the shift in at least 2/3 replicates, report:

`CLM_SHIFT_ABSORBED_WITHOUT_MITOSIS`

Otherwise report:

`CLM_PROBATIONARY_MITOSIS_NOT_CONFIRMED`

## Provenance and resume

Every formal worker records:

- immutable Git commit and tree SHA;
- clean tracked-tree requirement;
- CLM-0.1 SHA;
- Story corpus/tokenizer hashes;
- Arithmetic corpus hashes;
- replicate seed;
- trunk schedule hash;
- both condition schedule hashes;
- disjoint holdout-A and holdout-B schedule hashes;
- formal horizons, shortlist size, practical/retention thresholds;
- all 72 parity records;
- per-horizon candidate/control batch NLLs and paired bootstrap intervals;
- shortlist decisions;
- probation decisions;
- holdout-B confirmations;
- absorption diagnostics.

Evidence from different commits/trees must never be mixed. A code change requires `--restart-existing` for any existing 0.3d output directory.

Training checkpoints are resume artifacts only and are excluded from publication.

## Out of scope

CLM-0.3d does not establish:

- repeated autonomous births after promotion;
- production-efficient persistent shadows;
- learned or adaptive probation horizons;
- expert death/merge after persistent birth;
- hardware/energy pricing in the utility objective;
- local-only sensing without backpropagation;
- 2D cellular topology;
- multimodal growth;
- real-world continual-learning capability boundaries.

A positive result supports the narrower claim that a full CLM lineage can use multi-horizon shadow development to discriminate stationary continuation from a controlled capability shift and promote persistent structure only after sustained counterfactual advantage.
