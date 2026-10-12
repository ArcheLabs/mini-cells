# COW-CLM-001 — Minimal Functional Fork

This workflow tests the first frozen mechanism for **Persistent Copy-on-Write Lineage CLM** on the pinned Granite 3.1 1B-A400M foundation.

## Scientific question

Can a mature frozen Granite MoE acquire both a knowledge mutation and a small held-out-composition capability mutation while only selected expert slices become private, with the root model remaining immutable and exactly recoverable?

This is a **Cell formation** test. It is not a routing, deep-lineage, or sibling-composition test.

## Frozen semantics

- Granite is the immutable root model state.
- One explicit execution ticket activates one complete Cell view for the whole forward pass.
- COW-CLM-001 physically stores FP32 zero-initialized deltas only for selected fused expert tensor rows; execution casts them to the parent parameter dtype.
- Unpatched modules and expert slices remain shared immutable parent state.
- Canonical semantics are inheritance/substitution, not additive sibling merging.
- Learned natural-language Cell routing is out of scope; this protocol uses oracle Cell activation.
- Deeper lineage is intentionally rejected by the v0.1 runtime and deferred to COW-CLM-002.

## Frozen capacity ladder

Each track traces the actual frozen Granite top-k expert routing on **training rows only**, ranks `(layer, expert)` sites by hit count, and independently trains root forks with the top:

`1 -> 2 -> 4 -> 8` expert sites.

The protocol does **not** set a private-parameter-fraction success threshold. It records the minimum successful COW fraction instead of defining the desired economic answer in advance.

## Tracks

### Knowledge

Eight synthetic `CowNode-*` mappings. Each fact has three training paraphrases and two held-out paraphrases. Positive track status requires held-out candidate-choice accuracy `1.0`.

### Capability

An invented operation:

`ZOR(a,b) = Z((2a + 3b) mod 10)`

Exact held-out operand pairs are never used for training. Positive track status requires held-out candidate-choice accuracy at least `0.8`.

A formal experiment PASS requires at least one passing capacity on **both** tracks and fresh-runtime verification of each minimum passing artifact.

## Hosted GPU execution

Required Kaggle Secrets:

- `HF_TOKEN`
- `GITHUB_TOKEN`

Secret values are never printed.

If two GPUs are available, the knowledge and capability root forks may run concurrently on `cuda:0` and `cuda:1`; they are scientifically independent. A one-GPU sequential run remains canonical-valid.

Canonical notebook:

`cow-clm-001-kaggle.ipynb`

The notebook performs:

1. branch checkout and dependency installation;
2. CPU lifecycle/protocol guard;
3. authenticated GitHub write preflight;
4. frozen protocol and implementation-blob validation;
5. frozen GPU execution;
6. fresh Granite reload verification;
7. immediate publication of terminal PASS **or** FAIL artifacts to the experiment branch.

## Durable evidence

Local hosted results are written under:

`results/cow-clm-001/seed-26090511/`

Published evidence is written under:

`artifacts/experiments/cow-clm-001/seed-26090511/`

and the terminal decision is:

`artifacts/experiments/cow-clm-001/decision.json`

The numerical result and decision artifacts are authoritative. The notebook is only orchestration.

## Research boundary

COW-CLM-001 does not claim that existing Granite experts are Cells, that expert-only COW is sufficient for every future capability, that sibling Cells automatically compose, or that natural-language routing is solved. It does not rewrite prior CLM conversion, Granite Hybrid CLM, or prompt-address decisions.
