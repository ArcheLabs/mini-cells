# Read-only baseline audit and R2 changes

Baseline: `8b7cc5344d0b35f58681286101e93e39ff1fc41d`.
Remote main inspected before implementation: `0cff2c2894a1e6e5cd01ca66b7ff9796857fba80`.
Main adds historical result artifacts and an unrelated native-sparse Notebook adjustment;
R1 QFIRST engine is unchanged. R2 branch starts from that main.

R1: absolute cost regression; self-normalized inverse inclusion (not unbiased HT);
preselected token/depth probes can miss stopped states; one global gradient clip;
unused QF critic competes for clip budget; candidate failures propagate; admission
precedes the step-400 checkpoint; root return omits successful sibling-arm metrics.

R1 log proves a probation exception at step 400 and J0 completion at 500. Its generic
error conflates missing gradients and nonfinite gradients. The first bad operator,
NaN/Inf type and exact cause are **not established** by the log. No R1 retraining is required.

R2 fixes those boundaries in an isolated namespace. Raw operators, input Memory,
parameter inventory, state witness/checkpoint primitives, data dedup/windowing,
accounting and authenticated publishing are reused. Version-sensitive orchestration,
training and lifecycle adapters are explicit R2 derivatives. No monkeypatches alter R1.
A read-only source ledger records reused historical files.

FAST/006 M0 executes attention in every layer before expert FFN. CLM Cells see only
current h; only CTX_READ accesses static causal input memory. R2 does not add an
attention backbone or free context summary. Thus no external MoE equivalence is claimed.

## Frozen candidate choices and unresolved evidence

Config records seed 151001201, 500×8×64 targets per arm, epsilon 0.40/0.25/0.15,
lambda .01, beta .1, 768 counterfactual calls/step, 4-depth buckets and alternating
shallow/deep allocation. Pair confidence uses an approximate normal 1.96 SE band
with 1e-5 numerical tolerance and ≥4 samples; this is not an exact small-sample test.
Adaptive comparisons use the uncertainty band, whereas ranking additionally requires
four samples. Pairwise loss is not claimed unbiased over all action pairs.

Admission candidates use >=32 stories, >=1000 bootstrap replicates, upper95 utility
change < -.005 and upper95 NLL change <=0. These are proposed exploratory thresholds,
not empirically proven constants. The configuration preserves both field names and
validates their exact semantic mapping.

Real data/asset hashes, exact R1 tokenizer reproducibility, GPU timings, device memory,
cross-CUDA restart and the two-hour cost projection remain unverified until actual
Kaggle Pilot. The lock records actual source/config/assets/protocol identities at runtime.
No GPU was available for local verification. 500-step training authorization is false.

## Current execution override

The implementation-stage authorization note above is historical. The current user
instruction enables one-launch development after successful preflight. See
`EXECUTION_WORKFLOW.md`; independent confirmation is still sealed.
