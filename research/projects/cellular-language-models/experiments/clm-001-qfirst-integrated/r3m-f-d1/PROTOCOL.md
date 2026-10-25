# R3m-F-D1: Supervision and Context Causal Diagnosis

## Scope

D1 is a bounded mechanism diagnosis based on the frozen Tiny Overfit audit at `2d545f45`. It preserves the original `TINY_OVERFIT_FAILED` result and does not alter the R3m-F configuration, checkpoint, tokenizer, model size, optimizer state, or original result. Arm A is the published 200-update result and is reused as-is. Only B, C, and D are prospective arms. Each prospective arm is capped at 200 optimizer updates, with evaluation at relative steps 0, 50, 100, and 200. No result-dependent extension, full Pilot, Main data, or Confirmation access is allowed.

This protocol and its CPU gate do not authorize or start GPU training. B/C/D require a separate independent review and later explicit run authorization.

## Fixed identity and initialization

- Facts and order: `0+1`, `1+2`, `2+2`, `2+7`, `3+4`, `4+6`, `6+8`, `7+9`, from the exact Tiny records file pinned in `config/frozen.json`.
- Seed: `151001301`.
- Initial state: R3h B-H, global Step 300, exact checkpoint byte hash and full-state fingerprint pinned in the frozen config. Continue its original Cell AdamW moments and the existing continuous R3m schedule; do not reset optimizer state.
- Tokenizer: original 4096-token Tokenizer, SHA-256 `612c2aaa09e7f35c89c41c1e6270bdca1ef861a283a4de859cd808c8e6bfe6ff`.
- Sampler: original PCG64 state at Step 300, eight rows per update, uniform `integers(8)` draws, append each sampled prompt+answer+EOS, truncate each source row independently to 65 tokens, and do not carry tails to the next row.
- B/C/D consume the same CPU-frozen per-step row-index plan. The future runtime must replay each draw and compare the source-token hash and RNG end state before any forward or optimizer update. A mismatch aborts before that update. The original run's terminal RNG was not independently attested, and the original producer/replay NumPy versions differ; the CPU-generated plan is therefore an explicit prospective common sequence, not an assertion that the historical A production trajectory is endpoint-certified.

## Arms

| Arm | Context | Loss | Execution |
|---|---|---|---|
| A | Packed | Uniform CE | Reuse published initial/final predictions and report; no model/checkpoint load or retraining |
| B | Packed | Answer digits + EOS CE | Same 8×65 sampled source rows as the frozen plan |
| C | Independent QA | Uniform CE | Split each already-truncated source row at QA starts; one separate forward per retained QA fragment |
| D | Independent QA | Answer digits + EOS CE | Same fragments and source plan as C |

Answer/EOS masking is applied only after the original shifted labels are constructed. It selects positions; it never rewrites, shifts, or replaces a label. The target mask marks answer digits and EOS separately. The CE numerator is the sum over selected target positions from every chunk in an optimizer update; the denominator is the total selected positions across that entire update. Uniform CE uses all retained targets and divides by the update's total retained target count. Per-fragment averaging is forbidden.

## Independent QA context

The split happens after the sampler has built and truncated each 65-token source row. Every retained QA fragment is passed to the model in a separate forward call, without a previous fragment, KV cache, carried hidden state, or shared predecessor. Its position indices start at zero; the model creates fresh input memory and initializes the first Cell predecessor exactly as on an ordinary fresh forward. Keyed H-routing random arrays are first generated for the original 64 source input offsets, then sliced by the QA's original source-row offset. Thus C and D use the same exogenous routing draws as each other and are not affected by how many earlier QA forwards occurred.

The EOS-to-next-prompt target is omitted because predicting the next QA's first prompt token from the previous QA's EOS would reintroduce a cross-QA state. All other retained within-QA shifted targets, including answer digits and EOS, retain their original labels. Consequently C/D necessarily process fewer input positions and attention pairs than A/B; this is a declared cost and supervision difference, not a hidden implementation detail.

## Training and evaluation invariants

All arms retain the frozen R3h B-H model and original parameter mask: only the original Cell FFN parameters train. Token and position embeddings, tied LM Head, all normalizations/projections, gate and Connection remain frozen. No output representation or model capacity change is introduced.

At each evaluation point, report every one of the eight free autoregressive predictions. Decode only generated continuation tokens, stop on EOS id 1, and exclude EOS from displayed text. Report first generated token accuracy, text-only exact match, strict exact match requiring EOS, termination rate, Teacher-forced answer-digit NLL, and EOS NLL. These are distinct metrics: gold-conditioned answer NLL and the first free token do not establish complete free-generation success. Arm A's initial and final rows are copied from the published negative result with provenance.

Checkpoints must bind arm, source/config/data/sample-plan identities, seed, anchor checkpoint, global and relative steps, full model and Cell-optimizer state, data RNG, route RNG, CPU/CUDA/Python/NumPy RNG states, source cursor, and frozen-parameter fingerprint. Resume rejects any mismatch and performs no implicit update. The inherited data cursor advances by 512 source slots per update; actual input positions, selected CE labels, answer exposures, Cell calls, FFN MAC, tied-head MAC, attention pairs, and wall time remain separate accounting fields.

## CPU budget comparison and interpretation

The CPU replay uses the verified Step-300 data RNG state and the eight frozen records. It produces 1,600 source rows over 200 updates. There are 6,661 answer-digit targets, 4,805 EOS targets, 4,849 complete-answer appearances, and 4,805 complete-answer-plus-EOS appearances. Each D1 prospective arm sees the same answer and EOS target identities and per-fact exposures.

| Measure over 200 updates | A / B packed | C / D independent QA |
|---|---:|---:|
| Source slots | 102,400 | 102,400 |
| Model input / retained target positions | 102,400 | 97,600 |
| Answer + EOS targets | 11,466 | 11,466 |
| Cross-QA EOS→prompt targets omitted | 0 | 4,800 |
| Hard Cell calls (8 per input position) | 819,200 | 780,800 |
| Cell FFN MAC (63,488 per call) | 52,009,369,600 | 49,571,430,400 |
| Tied LM Head MAC (524,288 per input position) | 53,687,091,200 | 51,170,508,800 |
| Causal input-memory query/key pairs | 3,328,000 | 804,607 |

B and D have an answer/EOS loss denominator of 11,466 across the full 200-update plan; per-update denominators are in `SAMPLE_PLAN.json`. C's uniform denominator is 97,600. Cell/Head MAC and causal-pair counts are analytic operation counters, not observed FLOPs or runtime. They omit routing, normalizations, optimizer/backward cost, launch overhead, and other operations.

The diagnostic is **not a perfectly cost-matched 2×2 factorial**. C/D remove 4,800 boundary targets, reset position numbering, issue more separate forwards, and reduce input/attention work. A is historical; its full production terminal RNG was not independently attested, so exact production sample-by-sample matching with B is not claimed. The strongest prospective contrast is D vs C (same independent context and sample plan, loss objective differs). B vs A is a mechanism comparison with a matching declared sampler and update budget, but not endpoint-certified sample identity. Context contrasts and the difference-of-differences must be interpreted with the explicit target and compute differences above. No result alone establishes arithmetic competence, generalization, or Cell specialization.

## CPU gate artifacts

`run.py --mode auto` always runs the CPU Gate first. It validates the frozen input and Tokenizer hashes, prompt/answer token boundaries, sampler replay, per-step source plan/RNG boundaries, masks, fragment isolation structure, restart identity, and evaluation metric contracts. The CPU Gate writes a unique preflight directory containing `SAMPLE_PLAN.json`, `BUDGET_PROOF.json`, `A_REUSED_PREDICTIONS.json`, and `RUN_MANIFEST.json`. It does not import Torch, load model weights, compute logits, perform an optimizer update, or use a GPU.

The Kaggle Notebook sets `RUN_BCD = False` by default. After the required independent review and separate run authorization, changing that one switch to `True` makes the same Run All execute preflight and then B/C/D. The training path still refuses to proceed unless the CPU Gate artifacts match the frozen identity. It restores only the pinned R3h B-H Step 300 checkpoint from the pinned private Hugging Face revision, requires two visible CUDA devices to preserve the saved RNG layout, runs at most 200 updates per arm, and saves full-state resume checkpoints every 25 updates plus at 0/50/100/200. A resumed arm verifies source/config/data/sample-plan identity, the full trainer and optimizer state, frozen parameter mask, cursor, and PCG64 state before continuing. It does not extend an arm beyond 200 updates.

The run output includes an independent manifest, the complete shared sample plan and budget proof, the reused A predictions, per-arm checkpoint/curve/evaluation records, and a combined `RESULTS.json`. At every evaluation point B/C/D report all eight free generations, first-token accuracy, text-only and strict exact match, answer NLL, EOS NLL, and EOS termination. Results remain in Kaggle Output; this Notebook does not publish checkpoints to Hugging Face or write to GitHub. The Notebook's default remains CPU-only, so importing or running it now cannot start GPU training.
