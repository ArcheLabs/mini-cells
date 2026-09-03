# Native CLM v0 M1 — Closure

Status: **COMPLETE**

M1 established the first trained token-predictive Native CLM v0 under the registered engineering boundary. It is not itself a formal continual-learning decision.

Canonical Kaggle result:

```text
status              NATIVE_CLM_V0_M1_NEXT_TOKEN_TRAINING_PASS
parameters          12,154,368
Cells               8
active Cells/token  2
initial val loss    5.7234292984008786
final val loss      0.7885352313518524
initial perplexity  305.9523278270837
final perplexity    2.200171322843134
active fraction     0.25
initial route H     0.6929467022418976
final route H       0.5747594475746155
```

All registered M1 gates passed:

- target parameter scale;
- completed requested steps;
- finite initial/final evaluation;
- validation-loss improvement;
- sparse Cell execution;
- router receives gradient;
- Cells receive gradient;
- generation executes;
- one Cellular Layer;
- autonomous growth is not claimed in M1.

Generation sample began:

```text
Once upon a time there was a little mouse named Timmy. Timmy loved to play in the park...
```

## Canonical checkpoint

The checkpoint was preserved outside Git after the Kaggle publication process became unresponsive.

```text
Hugging Face repo: archelabsxyz/native-clm-v0
file:              final-model.pt
SHA-256:           91cc66f744c97e50105acbb7cdc328a95cb87a32c49baf5b0d6e462d4d4c4c7f
```

M2 never trusts the floating `main` reference alone. The M2 checkpoint fetcher resolves the current Hub commit, downloads the file, verifies this exact SHA-256, and writes the resolved revision into `provenance.json` before any formal seed is touched.

## Closed question

M1 answers:

> Can a nontrivial real next-token model train end-to-end while its forward path contains learned sparse persistent Cells?

Under the registered M1 boundary: **yes**.

M1 does **not** establish replay-free continual language learning, autonomous growth, semantic Cell ontology, Dense/MoE superiority, or 30M+ scaling. Those remain later milestones.

Next milestone: **M2 — Continual Language Stream**.
