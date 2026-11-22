# MOM-MULTICELL-RESIDUAL-001

Run `apps/mom-native-cells/kaggle_multicell_residual.py` as a Kaggle Python
Script with a T4, Internet, and the existing `GITHUB_TOKEN`. One launch
runs the CPU checks, synthetic and real-input GPU parity, four Smoke groups,
fresh-process restore, 24 Pilot groups and isolated profiling. A failed
engineering check stops progression and publishes its available evidence.
A positive Pilot can expand to 32 groups per seed when the measured time
budget allows. Only a positive final addressing gate starts QA evaluation.

## Source audit

The base is MiniCells `7274be20a196c39dab4706e684e06568ccb46e7f` with frozen
MoM 340M and MoM source `125038087b71b635c171cec857bd0f21d10955b6`.
The verified [FLA 0.2.2 wheel](FLA-LICENSE.txt) implements the fixed-length recurrence in
`fla/ops/gated_delta_rule/fused_recurrent.py`. It decays FP32 state, reduces
over K, subtracts the prediction, applies beta, and updates the state before
reading Q scaled by the inverse square root of key width. Q/K L2 normalization
runs upstream. A separate Triton kernel receives a clone of each initial
state. It emits the pre-update prediction and residual as temporary FP32
tensors. The native kernel alone supplies the output and state that proceed
through the model. Neither the installed wheel nor Cell format is patched.

The shared input and native Top-2 packing come from the existing pre-output
hierarchical equal path. Trace alignment inverts native packing and discards
padding after the kernel returns. The four routed bank identities and the
separate shared bank keep their original labels. Queries use independent
working caches. The experiment hashes both full ZIP bytes and stored tensors
before and after use. A fresh process imports the saved Smoke Cells using the
ordinary public bundle path. It replays the original projected operands with
the imported state and applies the original token/head tolerances to every
residual field. Full forward passes can regenerate different operands.
Their drift is reported per field and routing position, and the primary
Cell scores still have to pass the original score tolerance.

## Data and decision

Four synthetic fact Cells include two-hop facts and same-domain competitors.
A fifth Cell contains unrelated randomized text and a sixth is empty. The
five query types are exact wording, paraphrase, same-domain distractor,
no answer and two-hop composition. Facts and labels are assigned by fixed
seeds with disjoint identifier ranges. Smoke is disjoint from Pilot.

The primary score is the negative relative residual over actual routed
banks, with bank, head, token and layer aggregation. Reports include pooled
tie-aware AUROC and average precision, within-query pairwise accuracy,
Recall@1, mean and complete Recall@2, MRR, and fact-group bootstrap intervals
stratified by seed (2,000 replicates). The same query is evaluated after
reversing load order and after assigning the stored fact payloads to
different Cell IDs. Seed splits, empty/random controls, original router mass,
routed/shared/read-output controls, and leave-one-layer/head-out scores are
included. No-answer rows have no relevant Cell. Forced Top-K selects a Cell
on every no-answer row because the experiment includes no abstention
threshold.

The pilot gate uses AUROC at least 0.70, paired group-bootstrap AUROC lower
bound above 0.60, pairwise accuracy at least 0.65, consistent direction in
two seeds, minimum leave-one-out AUROC above 0.60, and at least 95% assignment
swap ranking agreement on non-tied pairs. The result also must exceed the
random ranking control. This is a development decision. On passing, answer
NLL, candidate Top-1 and greedy exact match compare all Cells, oracle,
residual, and random selection on the same newly materialized snapshots.

## Artifacts

GitHub receives the dataset, full scores, router diagnostics, state digests,
parity evidence, QA decision, performance report and one complete compressed
raw Smoke trace. A hashed index identifies the remaining full scalar traces
in Kaggle output, which are larger. Raw prediction/residual vectors are
transient. No model weights are returned to GitHub. Profiling warms and
measures native and residual paths separately for 1, 2, 4 and 8 Cells.
