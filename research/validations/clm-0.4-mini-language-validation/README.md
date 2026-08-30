[English] | [中文](README.zh-CN.md)

# CLM-0.4-mini — Token-Level Continual-Learning Validation

> Status: Protocol frozen for implementation; formal experiment not yet run

## 1. Decision question

Core Validation 004 supported the closed CLM mechanism in a controlled synthetic function world:

`ROUTE -> LEARN -> VALIDATE -> COMMIT | ROLLBACK -> GROW -> LEARN -> VALIDATE -> COMMIT/ROLLBACK`

CLM-0.4-mini asks the next narrower question:

> **Can the same dependency-scoped transactional growth loop preserve useful stability and plasticity when the model is a real autoregressive token-level language model trained and updated by next-token prediction?**

This is a mechanism-transfer experiment, not a general language-model benchmark. It deliberately keeps the addressing plane explicit and stable so that failure can be attributed to the transition to token-level learning rather than to an unsolved semantic-routing problem.

A positive result supports token-level transfer in the registered controlled language curriculum. It does not establish general natural-language continual learning, autonomous semantic addressing, indefinitely bounded growth, LLM-scale behavior, or JAM-native training.

---

## 2. Parent evidence and frozen interpretation

The experiment inherits, without relabeling, the pre-0.4 evidence chain:

- Core Validation 002: `WRITE_ADDRESSABILITY_NOT_SUPPORTED`.
- Core Validation 002B: `SPARSE_WRITE_ASSEMBLY_NOT_SUPPORTED`.
- Core Validation 002C: `ORACLE_SPARSE_ASSEMBLY_NOT_SUPPORTED`.
- Core Validation 003: `DEPENDENCY_SCOPED_TRANSACTIONAL_LEARNING_NOT_SUPPORTED` overall, while its registered gate-level data supported frozen-route structural locality and showed the stability/plasticity bottleneck.
- Core Validation 004: `GROWTH_RESTORED_PLASTICITY_SUPPORTED`, 3/3 formal seeds, in the registered controlled synthetic setting.

The conceptual transition remains:

`knowledge-addressed writing -> execution dependency -> transactional safety -> growth-restored plasticity`.

CLM-0.4-mini does not reopen precise latent write-addressability or learnable router drift.

---

## 3. Three execution phases

### M0 — execution smoke

Purpose: validate the full software path only.

It must exercise:

- tokenization and base-model forward/backward;
- deterministic stable routing;
- dependency trace recording;
- speculative direct candidate creation;
- dependency-scoped validation;
- rollback;
- zero-output growth-bundle creation;
- growth validation and atomic commit;
- Cell registry updates;
- checkpoint + transaction journal replay;
- reporting and artifact generation.

M0 may use a reduced model/data/transaction count. It emits only `SMOKE_ONLY` and has no scientific meaning.

### M1 — formal CLM-0.4-mini pilot

The primary scientific experiment. It uses the frozen architecture and 192 continual-learning transactions over a controlled language curriculum. All three unseen formal model seeds must independently pass every primary gate.

### M2 — scale rehearsal

Runs only if M1 passes. It uses the same mechanism over a substantially longer stream to characterize state growth, dependency-index growth, validation cost, checkpoint size, replayability, and bounded-validator approximations. M2 is an engineering scale gate for the 30–50M candidate; it does not relabel the M1 scientific result.

The 30–50M CLM-0.4 candidate is authorized only when:

`M1_GO AND M2_GO`.

---

## 4. M1 model architecture

The target base model is approximately 5M parameters before growth.

### Decoder

- decoder-only autoregressive Transformer;
- vocabulary: 8,192 BPE tokens;
- sequence length: 256;
- layers: 4;
- model width: 256;
- attention heads: 8;
- positional encoding: RoPE or another parameter-free deterministic positional scheme;
- embedding / LM-head weight tying: required.

### Blocks 1–2: shared language backbone

- ordinary self-attention;
- dense FFN hidden width 768;
- trained during base pretraining;
- completely frozen during the continual-learning stream.

### Blocks 3–4: sparse Cell-FFN layers

Each of the final two blocks contains:

- 32 base Cells;
- Cell hidden width 32;
- Top-2 base Cell activation per sequence address per layer;
- weighted/residual combination defined once in the implementation lock;
- all shared attention, normalization, embedding, and LM-head parameters frozen during continual learning.

Approximate base parameter budget, excluding small bias/norm terms:

- token embedding: ~2.10M;
- attention projections: ~1.05M;
- dense FFNs in blocks 1–2: ~0.79M;
- 32x32 Cell-FFNs in blocks 3–4: ~1.05M;
- total: ~5.0M.

This is intentionally close to the minimum scale that still gives a real Transformer language model and fine-grained Cell dependency domains.

---

## 5. Stable addressing plane

The formal M1 router is deliberately **not a learned semantic router**.

Every training/evaluation example carries an out-of-band `address_id`. The address is metadata and MUST NOT be inserted into the token sequence or exposed as answer-bearing text.

For each growth-capable layer, the base Top-2 route is:

`R_base(layer, address_id) = stable_hash(protocol_salt, layer, address_id) -> two distinct Cell IDs`.

Required invariants:

1. the mapping is deterministic across machines and runs;
2. it never depends on mutable hidden states;
3. it never changes during the continual stream;
4. it can route previously unseen address IDs without retraining;
5. base routing and growth-route additions are versioned separately.

This design avoids an important token-LM failure mode: even a frozen learned router can change downstream routes if it consumes hidden states that were changed by an earlier mutable layer. CLM-0.4-mini therefore uses immutable input-side metadata for the certification route.

A **shadow semantic router** may be trained/evaluated from ordinary text representations, but it is diagnostic only. It may never influence candidate selection, validation scope, commit, rollback, or growth in M1.

---

## 6. Cell state and growth unit

A base or growth Cell is a versioned mutable state unit with at least:

- `cell_id`;
- `layer_id`;
- parameter tensor state;
- parameter count;
- state hash/version;
- dependency/probe IDs;
- activation count;
- accepted update count;
- rejected update count;
- birth transaction for growth Cells;
- parent/base route metadata;
- owner `address_id` for private growth Cells.

A formal **growth bundle** contains exactly one private additive Cell in each growth-capable layer (blocks 3 and 4).

Each private Cell uses hidden width 32. One growth bundle therefore adds roughly 32.8K weights before small bias terms, about 0.65% of the ~5M base model.

Growth rules:

1. a new growth Cell is initialized to produce exactly zero residual output;
2. adding the untrained Cell and route must therefore preserve the pre-growth function within the structural tolerance;
3. a growth bundle is private to one `address_id`;
4. an address may own at most one growth bundle in M1;
5. the private route is monotonic: it may be added on commit, but existing base routes are never rewritten;
6. Cell parameters and private route commit atomically;
7. failed probationary growth is deleted completely;
8. once a private bundle exists, later updates for that address train only that private bundle; M1 does not spawn a second generation.

Thus growth tests reuse rather than one-Cell-per-update behavior.

---

## 7. Base training corpus

Base pretraining target: **30M tokenized tokens ±1%**.

Token mixture by token count:

- 60% language carrier corpus;
- 20% controlled base mathematics;
- 20% controlled base story-world language.

### Language carrier

Use a deterministic pinned subset of TinyStories or an equivalent already-supported repository corpus. The exact dataset revision, sample-ID manifest, preprocessing version, tokenizer training manifest, and hashes MUST be written into `protocol-lock.json` before any formal seed is run.

The carrier corpus establishes ordinary token-level syntax and generation behavior. It is not itself part of the continual-learning decision set.

### Base mathematics

Include only capabilities that remain protected throughout M1, such as:

- small-integer addition;
- subtraction;
- comparison;
- simple one-step verbal arithmetic.

### Base story language

Use generated natural-language micro-worlds containing stable entities, locations, possessions, occupations, and simple relations. Train both declarative statements and question/answer forms.

Train/validation surface templates must be disjoint enough that held-out evaluation is not exact string replay.

### Base route coverage

Base-corpus examples must carry stable route metadata that distributes examples across all 32 base Cells in both Cell layers. Carrier examples may use deterministic sample-hash address buckets. The implementation must report per-Cell base activation counts and fail the base prerequisite if any Cell is effectively untrained according to the lock threshold.

---

## 8. Continual curriculum

M1 contains exactly **192 transactions**, interleaving mathematics and story updates.

The semantic schedule is fixed before formal execution and is identical across the three formal model seeds. Model initialization, minibatch order, dropout/randomness, and optimizer randomness vary by model seed; the curriculum examples and intended truths do not.

### Mathematics: 96 transactions

Use 12 continual math addresses, each revisited 8 times.

Families should cover deterministic small-integer skills not present in base pretraining, for example:

- multiplication;
- exact integer division;
- modulo;
- precedence / two-step expressions;
- bounded affine transforms;
- several synthetic binary operator families with textual definitions and generated examples.

Each revisit increases range, surface form, or compositional difficulty while preserving earlier accepted capability probes. This repeated structure measures whether a private growth bundle is reused rather than repeatedly spawned.

### Story worlds: 96 transactions

Use 24 continual story-world addresses, each revisited 4 times.

Transactions must include both:

- `append`: add a new fact while all existing facts remain protected;
- `supersede`: intentionally replace the current value of one knowledge key, such as a character moving to a new city.

For a supersede transaction, only historical probes for the exact superseded key are removed from the protected old-behavior set. Unrelated facts about the same world remain protected.

This is the token-level analogue of intentionally excluding the current target context from old-regression scoring in the synthetic validations.

### Per-transaction data

Default M1 sizes:

- training examples: 64;
- local new-validation examples: 128;
- newly admitted protected probes after a successful commit: 32;
- sequences truncated/padded to the frozen maximum sequence length.

The exact generator versions and transaction manifest hash are part of `protocol-lock.json`.

---

## 9. Transaction state machine

For transaction `t` with address `a_t`:

### Case A — no private growth bundle exists

1. Resolve the immutable base route in blocks 3 and 4.
2. Copy the current model state into a speculative candidate.
3. Train **only** the four routed base Cells (Top-2 in each of two Cell layers).
4. Compute new-task gain on the transaction validation set.
5. Build the exact dependency-scoped old-validation set from historical probes whose recorded base routes intersect any touched Cell.
6. Evaluate local old regression.
7. If both new gain and local regression pass, atomically commit the four Cell updates.
8. Otherwise roll back the entire direct candidate.
9. On direct rejection, create one probationary zero-output private growth bundle for `a_t`, train only that bundle, validate it, and atomically commit `bundle + private route` only if it passes.
10. If growth fails, delete the probationary bundle and leave model state unchanged.

### Case B — a private bundle already exists

1. Base routes remain unchanged.
2. Train only the address-owned private bundle.
3. Validate against the private bundle's protected dependency probes plus new-task validation.
4. Commit or roll back.
5. Never spawn a second bundle for that address in M1.

The world truth advances according to the curriculum even when the model rejects an update; rejected learning must remain observable as a capability gap rather than silently changing the target back.

---

## 10. Dependency index and validators

### Exact local dependency index

M1 uses an exact inverted execution index. At minimum it must support:

`cell_id -> protected probe IDs`.

No probabilistic structure that can create false negatives is allowed for the formal commit path. Bloom/Cuckoo-style filters may only be used as optional positive-prefilter diagnostics where false negatives are impossible.

For a direct update touching Cell set `B_t`:

`V_local(B_t) = union(probes[cell] for cell in B_t)`.

For an existing private bundle, its validation scope is its exact protected owner-address probes.

For a newly spawned private bundle, the pre-existing protected dependency domain outside the intentionally changed target key is empty because no old example previously routed through that private bundle. The global oracle still checks all unaffected history.

### Hidden global oracle

The formal commit path MUST NOT see the full historical probe set.

For every candidate, an evaluator-only hidden oracle computes the same regression metric over all protected historical probes except intentionally superseded keys.

The oracle:

- never changes local pass/fail;
- never triggers rollback/growth;
- only measures false-safe behavior and global damage.

This preserves the central experimental question: whether local dependency-scoped safety predicts full-history safety.

---

## 11. Language-level regression metrics

Teacher-forced evaluation is the certification surface.

For probe set `S`, define mean token negative log-likelihood `L(S)` over scored answer/continuation tokens.

Old regression:

`R(S) = (L_after(S) - L_before(S)) / max(L_before(S), eps)`.

New gain:

`G_new = (L_before(new) - L_candidate(new)) / max(L_before(new), eps)`.

Frozen candidate commit thresholds:

- minimum `G_new`: 0.02;
- maximum local old regression `R(V_local)`: 0.005.

The hidden global oracle uses the same 0.005 regression threshold to define global failure.

False-safe rate:

`FSR = P(local PASS and global FAIL | local PASS)`.

### Structural escape

For historical probes outside the declared dependency scope, compare teacher-forced logits before and after the candidate. A probe is a structural escape if maximum absolute scored-logit change exceeds `1e-5`.

Primary structural escape rate must be zero.

### Free generation

Free-generation math/story behavior is evaluated separately for capability quality and retention. It is never substituted for teacher-forced structural certification because an early changed token can alter all later autoregressive inputs.

---

## 12. Baselines

Every formal model seed runs the same curriculum on the same pretrained initialization copied into these primary variants:

### `local_always`

Train the same four routed base Cells (or existing private bundle where applicable in the corresponding no-growth form) and always commit direct learning. It is the primary high-plasticity / interference reference for gain and damage ratios.

### `local_tx`

Train the same direct routed base Cells, apply dependency-scoped transactional rejection, and never grow. It tests whether the language model reproduces the stability/plasticity bottleneck seen in 003.

### `local_tx_growth`

The CLM-0.4-mini primary mechanism: direct transactional update, rollback on unsafe learning, probationary private growth, atomic commit, and later private reuse.

### Secondary diagnostic baselines

A dense continual-finetuning arm and/or bounded replay arm may be included for context if compute permits. They are not allowed to change the primary formal decision unless added to a new protocol version before formal execution.

---

## 13. Calibration and protocol lock

Scientific gates, architecture, datasets/curriculum semantics, formal seeds, and decision rules are frozen by this document.

Candidate-training hyperparameters may be selected using **development seed 90401 only** from a finite predeclared grid.

Direct candidate grid:

- optimizer: AdamW;
- batch size: 32;
- learning rate in `{0.001, 0.003, 0.01}`;
- steps in `{16, 32, 64}`;
- weight decay: 0 unless a protocol revision is committed before formal execution.

Growth/private candidate grid:

- optimizer: AdamW;
- batch size: 32;
- learning rate in `{0.001, 0.003, 0.01}`;
- steps in `{32, 64, 96}`;
- weight decay: 0.

Calibration rule:

1. enumerate configurations in increasing estimated candidate compute;
2. use only development curriculum/model seed 90401;
3. select the first configuration satisfying the M1 primary safety/plasticity sanity gates on the development run;
4. if none passes, stop and revise the protocol before any formal seed is opened;
5. write the exact selected values, tokenizer hash, base-corpus manifest hash, curriculum manifest hash, code commit/tree, deterministic routing salt, environment versions, and structural tolerance into `protocol-lock.json`;
6. commit that lock file before running any formal model seed.

After any result from `90411`, `90412`, or `90413` has been observed, no gate, model architecture, dataset manifest, transaction schedule, optimizer choice, learning rate, step count, or tolerance may change for this experiment ID.

---

## 14. Formal seeds and replication

Development/calibration model seed:

`90401`

Formal model seeds:

`90411, 90412, 90413`

All three formal seeds are required.

A seed fails if base prerequisites fail or if any primary continual-learning gate fails.

The semantic curriculum and formal evaluation examples are fixed across model seeds; formal seeds change model/training stochasticity, not the intended knowledge stream.

Formal execution target: Kaggle Tesla T4 or better CUDA GPU. CPU smoke is supported; CPU formal decisions are not.

---

## 15. M1 base prerequisites

Before a formal continual stream starts, each seed must independently satisfy:

- base math exact-match accuracy `>= 0.85`;
- base story QA exact-match accuracy `>= 0.85`;
- every one of the 32 base Cells in each mutable layer receives at least the minimum activation count frozen in `protocol-lock.json`;
- no NaN/Inf parameters or losses;
- tokenizer/model/config hashes match the protocol lock.

Carrier-corpus perplexity/NLL is reported but is not a primary gate unless a protocol revision freezes such a gate before formal seeds are observed.

---

## 16. M1 primary scientific gates

Every formal seed must independently pass **all** gates for a positive M1 result:

### Safety

- false-safe rate `<= 0.02`;
- maximum structural escape rate `= 0` at `1e-5` scored-logit tolerance;
- cumulative positive hidden-global regression damage `<= 0.35x local_always`.

### Plasticity

- effective acceptance rate, counting direct or growth commits, `>= 0.75`;
- cumulative committed new-learning gain `>= 0.75x local_always`;
- final protected-behavior retention ratio `>= 0.95` relative to each probe's accuracy when it entered the protected set;
- `local_tx_growth` cumulative committed gain must exceed `local_tx`.

### Growth

- growth rescue rate after direct rejection `>= 0.70`;
- private-bundle reuse acceptance `>= 0.60`;
- spawned growth bundles / effective commits `<= 0.50`;
- total committed growth-parameter overhead at stream end `<= 0.25x` base parameter count;
- maximum active private growth Cells per growth-capable layer per input `<= 1`.

### Locality

- mean exact dependency coverage for direct candidates `<= 0.30` of protected history.

These thresholds are intentionally somewhat looser than synthetic 004 because this is the first token-level transfer test, while remaining strong enough to reject a mechanism that merely preserves safety by refusing most useful learning.

---

## 17. M1 formal statuses

Positive:

`CLM_0_4_MINI_TOKEN_LEVEL_LOOP_SUPPORTED`

Negative:

`CLM_0_4_MINI_TOKEN_LEVEL_LOOP_NOT_SUPPORTED`

Smoke:

`SMOKE_ONLY`

A positive status requires all primary gates on all three formal seeds.

---

## 18. Observability contract

Every continual transaction must be auditable. The raw record must include at least:

- transaction ID and curriculum operation (`capability`, `append`, `supersede`);
- address ID and knowledge key where applicable;
- train/validation/probe example manifest IDs;
- model state hash before transaction;
- base routed Cell IDs per mutable layer;
- private bundle ID if present;
- candidate kind (`direct`, `spawn`, `private-reuse`);
- touched parameter count and fraction;
- local dependency probe count and coverage;
- new NLL/accuracy before and candidate;
- local old NLL before/candidate and regression;
- hidden global old NLL before/candidate and regression;
- local pass, oracle pass, false-safe flag;
- structural escape count/rate;
- commit/rollback/growth decision;
- Cell births/deletions;
- state hash after transaction;
- active base/private Cells per layer;
- training tokens, validation tokens, optimizer steps;
- candidate wall time, validation wall time, total transaction wall time;
- peak allocated GPU memory;
- logical mutable parameter count;
- checkpoint/journal references.

The experiment must make it possible to reconstruct why every state transition was accepted or rejected.

---

## 19. Cell registry and lineage

Maintain a persistent registry containing at least:

- `cell_id`;
- `layer_id`;
- base/private type;
- owner address for private Cells;
- parent/base route;
- birth transaction;
- parameter count;
- current state hash;
- total activations;
- dependency count;
- accepted/rejected updates;
- growth rescue event;
- reuse events.

Also emit a lineage graph linking base routes to committed private growth bundles.

Any post-hoc semantic specialization label (for example `math.mul`) is analysis metadata only and must never be used as evidence that the model discovered that semantic category autonomously.

---

## 20. Replayability and checkpoints

The formal stream must be journaled as deterministic state transitions.

Required:

- base checkpoint;
- periodic full checkpoints at least every 16 transactions;
- exact RNG seeds/state or deterministic per-transaction seeds;
- transaction manifest hashes;
- Cell registry snapshot/version;
- route table/version;
- state hash after every committed transaction.

A sampled replay test must restore from a prior checkpoint, replay the journal, and reproduce the registered later state hash within the frozen deterministic policy.

---

## 21. M2 scale rehearsal

M2 runs only after M1 Go.

Use the same architecture and mechanism with a new engineering seed `90421` and a longer 768-transaction stream that repeatedly revisits the same bounded set of continual addresses. The purpose is to test time-horizon scaling and reuse, not unlimited domain expansion.

M2 primary engineering checks:

- no OOM or unrecoverable numerical failure on the target GPU;
- growth-parameter overhead at end `<= 0.30x` base parameter count;
- private-reuse acceptance in the second half `>= 0.75`;
- mean direct dependency coverage `<= 0.30`, p95 `<= 0.45`;
- periodic checkpoint + journal replay reproduces sampled state hashes;
- model-only checkpoint size `<= 1.35x` base model-only checkpoint size;
- active private Cell count per growth-capable layer remains `<= 1` per input by construction and observation.

### Bounded-validator shadow study

M2 keeps the exact dependency validator as the real commit authority but additionally evaluates bounded per-Cell regression reservoirs of sizes 32, 64, and 128 probes.

For each reservoir size report:

- decision agreement with exact local validation;
- shadow false-safe rate against the hidden global oracle;
- validation examples/tokens saved;
- memory footprint.

30–50M scale readiness requires at least one bounded reservoir configuration with:

- decision agreement `>= 0.95`;
- shadow FSR `<= 0.02`;

without ever allowing the bounded shadow validator to control M2 commits.

This is an engineering projection only; adopting a bounded validator in the later 30–50M experiment requires its own frozen protocol decision.

---

## 22. Required artifacts

Implementation/smoke results should use a working result directory. Formal publication should eventually produce a canonical artifact directory:

`artifacts/experiments/clm-0.4-mini-language-validation/`

Required outputs after formal execution:

- `protocol-lock.json`;
- `run-manifest.json`;
- `decision.json`;
- `base-metrics.csv`;
- `seed-summary.csv`;
- `gate-summary.csv`;
- `transaction-records.csv` or Parquet plus a stable CSV summary;
- `cell-registry.jsonl`;
- `cell-lineage.json`;
- `dependency-summary.csv`;
- `cost-summary.csv`;
- `scaling-readiness.json` after M2;
- environment/provenance record;
- figures described below.

Recommended figures:

1. protected retention and new-capability gain over transactions;
2. local_always vs local_tx vs local_tx_growth stability/plasticity frontier;
3. direct acceptance, growth rescue, and private reuse over time;
4. total Cells and growth parameters over time;
5. Cell lineage/genealogy;
6. dependency coverage distribution;
7. false-safe / structural-escape audit;
8. transaction training vs validation cost;
9. active vs total Cell count;
10. M2 bounded-validator agreement/cost curves.

---

## 23. Interpretation limits

Even if M1 and M2 pass, the following remain open:

- autonomous semantic address discovery;
- hidden-state learned routing under mutable representations;
- arbitrary router evolution without dependency-index rebuild;
- unrestricted growth and multi-generation mitosis;
- Cell merge/apoptosis;
- open-domain factual continual learning;
- multilingual or multimodal continual learning;
- 30–50M and larger scaling;
- distributed/JAM-native training and state-transition verification;
- physical sparse-kernel speedup comparable to logical sparse activation.

A pass should be described as **controlled token-level language validation of the CLM core loop**, not as a solution to general continual learning.

---

## 24. Go to the 30–50M candidate

The project may start the 30–50M CLM-0.4 formal candidate only after:

1. M1 status is `CLM_0_4_MINI_TOKEN_LEVEL_LOOP_SUPPORTED` on 3/3 formal seeds;
2. M2 scale rehearsal passes all engineering checks;
3. the full observability/replay contract is demonstrated;
4. scaling projection records parameter growth, dependency-index memory, checkpoint growth, transaction time, validation time, and GPU peak memory;
5. no formal gate was changed after formal results were observed.

The main deliverable of CLM-0.4-mini is therefore not merely a small checkpoint. It is a validated and observable **language-level state-transition system** plus the measurements needed to decide whether a 30–50M run is justified.
