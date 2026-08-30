# Experiment 024b — Independent Third-Trait Challenge

## Motivation

Experiment 024 produced `FIRST_BIRTH_WITHOUT_SECOND_TRAIT_GENESIS` rather than the preregistered sequential `1 -> 2 -> 3` signal. The first Story/Arithmetic bifurcation was supported, but the resulting computational trait also improved the planned Transform benchmark strongly. The scientific diagnosis is therefore not simply that the second birth rule failed. The planned third benchmark was substantially absorbable by the already-existing phenotype basis.

024b tests a stricter idea:

> A new expert boundary should correspond to functional non-substitutability, not a human benchmark label.

The experiment first identifies a third capability that the current two-trait organism cannot economically absorb, and only then asks whether the unchanged probationary birth rule produces a third persistent trait.

## Research question

Can a Story/Arithmetic two-trait TextNCA:

1. screen several synthetic candidate capabilities for functional non-substitutability;
2. select one candidate by a deterministic preregistered cross-replicate rule;
3. reject a weak exposure to that selected capability;
4. later commit a `2 -> 3` birth under strong exposure;
5. form three-domain functional identity without task labels entering proposal, geometry routing, or commit?

## Fixed substrate

024b inherits the validated 022–024 substrate:

- Story-pretrained TextNCA;
- frozen shared genome after pretraining;
- persistent phenotype rows;
- fixed pretrained-parent phenotype gradient as the developmental sensor;
- geometry proposal/routing without task labels;
- newborn Adam-row inheritance;
- matched Capacity shadow;
- prequential probation;
- structural cost `0.005`;
- geometry-over-capacity advantage threshold `0.005`;
- routing purity threshold `0.75`.

These thresholds are not tuned for 024b.

## Phase 1 — first trait birth

Each of three replicates independently pretrains the same architecture on TinyStories, then runs the established 50/50 Story + Arithmetic probationary birth stage.

The committed `K=2` state is the screening foundation. Full first-birth scientific support requires:

- the geometry shadow commits;
- two-domain functional identity passes;
- routing purity is at least `0.75`.

A replicate that does not commit `K=2` is retained as a negative first-birth outcome and is not silently forced into screening.

## Phase 2 — candidate independence screening

### Fixed candidate pool

The pool is frozen before GPU execution:

1. `TRANSFORM` — reverse a six-digit sequence; retained as the known 024 reference capability;
2. `PARITY` — parity of a ten-bit sequence;
3. `MODSUM` — sum six digits modulo seven;
4. `SORT` — sort six digits;
5. `ROTATE` — rotate a six-digit sequence left by two positions;
6. `DELAY_COPY` — recall a four-digit prefix after an eight-digit distractor;
7. `STATE_MACHINE` — compute the terminal state of a deterministic three-state process.

All corpora are deterministic, tokenizer-compatible, and generated from frozen seeds.

### Why screening must include retention

A capability is not independent merely because it is difficult. The existing computational trait may learn it by overwriting Arithmetic. Therefore screening evaluates both candidate gain and Arithmetic damage.

For each candidate and replicate, starting from the exact same committed two-trait checkpoint:

### Existing-trait absorption arm

Only the posthoc Arithmetic-best computational branch is adapted to the candidate for 128 steps.

Let:

- `L0_c` be its candidate NLL before adaptation;
- `Le_c` candidate NLL after adaptation;
- `L0_a` Arithmetic NLL before adaptation;
- `Le_a` Arithmetic NLL after adaptation.

Define normalized candidate gain:

`G_existing = (L0_c - Le_c) / |L0_c|`

and Arithmetic damage:

`D_arith = max((Le_a - L0_a) / |L0_a|, 0)`.

Existing-trait value is:

`U_existing = G_existing - D_arith`.

### Newborn arm

A third phenotype is initialized from the same computational parent plus the same small fork-scale perturbation along the mean candidate gradient direction. Its optimizer moments inherit the parent row. Existing Story/Arithmetic phenotype rows are not updated. Only the newborn learns the candidate for the same 128 steps and sees the same schedule.

Let `Ln_c` be its final candidate NLL.

`G_newborn = (L0_c - Ln_c) / |L0_c|`

`U_newborn = G_newborn - 0.005`.

### Independence score

`I = U_newborn - U_existing`.

Absorption ratio:

`A = max(U_existing, 0) / U_newborn` when `U_newborn > 0`.

A candidate qualifies in a replicate iff all hold:

- `G_newborn >= 0.02`;
- `I >= 0.01`;
- `A <= 0.50`.

A candidate enters the formal selection pool iff it qualifies in at least 2/3 replicates.

### Deterministic cross-replicate selection

Among formally qualifying candidates, choose the candidate with the largest median `I` across available screening replicates. Exact ties are broken lexicographically.

If no candidate qualifies, the script still chooses the largest-median candidate for an explicitly exploratory challenge so the run remains diagnostically useful, but the experiment cannot receive the strong positive status.

The selected candidate is written to `selection.json` before any challenge worker starts. All replicates then use that same frozen candidate.

## Phase 3 — weak selected-capability control

Starting again from the pre-screening committed `K=2` checkpoint, not from any screening-adapted state, each replicate sees 256 probation steps:

- 115 Story;
- 115 Arithmetic;
- 26 selected capability (~10%).

A `K=3` geometry shadow is proposed and evaluated exactly as in 024.

Preregistered outcome: reject, retain `K=2` Story/Arithmetic identity.

This stage tests whether screening accidentally selected a capability that causes hypersensitive overgrowth under weak exposure.

## Phase 4 — strong selected-capability challenge

If the weak stage rejects, the continuously adapted `K=2` incumbent proceeds to a 256-step approximately balanced stream:

- 86 Story;
- 85 Arithmetic;
- 85 selected capability.

Again compare:

- incumbent `K=2`;
- matched local Capacity `K=3` shadow;
- geometry-routed `K=3` shadow.

Birth commit uses the unchanged 023b/024 rule:

- final two geometry windows have positive net utility;
- final-three mean geometry net utility is positive;
- final-three mean geometry advantage over Capacity is at least `0.005`.

No task/domain label enters proposal fitting, geometry routing, or commit.

## Functional validation

After each challenge stage, evaluate every active phenotype on held-out:

- Story;
- Arithmetic;
- selected capability.

For a committed `K=3` system, three-domain identity is permutation-invariant. Each domain must have a distinct best branch with normalized identity margins satisfying the inherited identity threshold. Geometry routing purity over the three posthoc families must be at least `0.75`.

The weak stage additionally requires retention of the existing Story/Arithmetic two-trait identity.

## Strong positive status

`INDEPENDENT_THIRD_TRAIT_GENESIS_SIGNAL` requires all of:

1. at least one candidate satisfies the preregistered screening qualification rule;
2. Story/Arithmetic first birth with identity and routing purity passes in at least 2/3 replicates;
3. weak selected-capability stage rejects and retains two-trait identity in 3/3 replicates;
4. strong selected-capability `2 -> 3` birth commits, forms three-domain functional identity, and has routing purity >=0.75 in at least 2/3 replicates;
5. final committed `K=3` in at least 2/3 replicates.

## Diagnostic statuses

- `INDEPENDENT_THIRD_TRAIT_GENESIS_SIGNAL`
- `NO_FUNCTIONALLY_INDEPENDENT_THIRD_CAPABILITY`
- `INDEPENDENT_CAPABILITY_CAUSES_EARLY_BIRTH`
- `NO_STABLE_FIRST_TRAIT_BIRTH`
- `INDEPENDENT_CAPABILITY_WITHOUT_THIRD_TRAIT_GENESIS`

## Causal protections

- Screening clones all candidate arms from the same pre-screening `K=2` state.
- Screening training is never carried into the final challenge.
- Candidate selection is frozen before challenge execution.
- All three challenge arms consume the same chronological future batches.
- Capacity and Geometry newborns start from the same parent phenotype and inherited optimizer row.
- Proposal, geometry routing, and commit do not use semantic labels.
- Screening is explicitly benchmark-controlled and may use task identity to construct the diagnostic absorption test and identify the current Arithmetic-best branch.

## Scope

024b does not establish that real-world capability boundaries are discrete or that this seven-task synthetic pool spans meaningful intelligence. It still uses a global backprop phenotype-gradient oracle, a frozen shared genome, preallocated phenotype slots, synthetic corpora, and experiment-controlled screening. It does not establish local sensing, rewiring, pruning, merging, death, unbounded physical allocation, or inference-time endogenous recruitment.

A positive result would support the narrower claim:

> An existing cellular expert basis can be tested for functional substitutability, and a new persistent trait can be born only after a candidate demand is shown to be economically non-substitutable.
