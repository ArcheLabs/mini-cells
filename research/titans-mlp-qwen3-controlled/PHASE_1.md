# Phase 1 — Architecture and Memory Sanity

The initial single-seed artifact remains in synthetic-results.json. The
paired follow-up uses seeds 42–46, three arms and all 16 capacity/budget
conditions per seed. Input geometry, initialization and presentation order
are shared within each seed across arms. The raw observations and sample
standard deviations are in multiseed-results.json.

## Synthetic evaluation

The query for a known association uses its write key. Hard negatives are
unbound unit queries constructed by normalizing 0.8 times a known key plus
0.6 times an independent random unit vector. The false positive rate is the
fraction whose maximum cosine score against all stored values reaches 0.5.
This defines an operating point for this diagnostic. No recall calibration
or language claims follow from zero false positives.

| Arm | 4 pairs | 8 pairs | 16 pairs | 32 pairs |
|---|---:|---:|---:|---:|
| Baseline | 90.0 ± 13.7% | 90.0 ± 10.5% | 100 ± 0% | 54.4 ± 15.1% |
| No momentum | 90.0 ± 13.7% | 100 ± 0% | 68.8 ± 9.9% | 21.9 ± 5.4% |
| No forgetting | 90.0 ± 13.7% | 90.0 ± 10.5% | 100 ± 0% | 54.4 ± 15.1% |

The table reports Recall@1 at 64 total updates, including repeated
presentations. Removing momentum reduced recall at 16 and 32 pairs under
this configuration. Removing the small fixed forgetting term left Recall@1
unchanged at 64 updates. Capacity at 32 pairs remained variable. Full
MRR and hard-negative FPR statistics are in the machine-readable artifact.

## Recurrence audit

The implementation matches the ordering and signs of the sequential
surprise recurrence in [Titans §3.1](https://arxiv.org/html/2501.00663#S3.SS1).
It differentiates the squared association loss at the old fast weights,
sets momentum to eta times the previous momentum minus theta times the
gradient, and adds that momentum to the old weights scaled by (1 - alpha).
The implementation applies global norm clipping at 1 before the momentum
update. The fixed controls are scalar eta=0.5, theta=0.02, alpha=0.0001
broadcast over every dynamic parameter. Original Titans conditions the
controls on input. This diagnostic takes a strictly pre-update read and
performs one write per unpadded event. It does not exercise the paper's
parallel training algorithm or adaptive controls.

## Cell tests

CPU tests cover initial storage independence, deterministic recurrence,
read-before-write, padding, read-only state invariance, exact Safetensors
round trip, fingerprint rejection, independent Fork updates, momentum
carry and forgetting in isolation. The existing PCA codec supplies the
hash and tensor validation.

## Qwen structure and gate

**PHASE_1 = FAIL**. The required pretrained-weight CPU audit is incomplete.
Hugging Face requests for the pinned model weights repeatedly timed out
using the snapshot client and direct HTTPS. The inspected official config
is pinned to `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, SHA256
`504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59`.

A CPU structure check built the concrete Transformers Qwen3 class from
that config with fresh parameters. It verified 28 Qwen3 decoder blocks,
dense MLP, SDPA Qwen3 attention, 1024 hidden features, disabled sliding
window, a frozen parameter set and no returned KV cache. Hooks fired
exactly once in layer order. The post-block hook at index 22 saw input
and output [1, 4, 1024]. The evidence is in qwen-config-structure.json.
The pretrained-weight gate requires rerunning audit_qwen.py successfully.

CPU tests: 2 passed. Phase 2 remains closed. No Outer Training or GPU
work was performed.
