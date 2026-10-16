# R3b locked observation and inference definitions — v1

Status: proposed; real assets, selected beta, production timing and formal release
await Gate C–E. The supplied SPEC.md is retained verbatim. Code/config/protocol
identity is fixed before a real run. No algorithm change or R3c intervention.

## Training identity and model mapping

Scientific identity is **CLM-001-R3b**, with Q-R3b/QC-R3b/QS-R3b checkpoint IDs and
separate run/storage/results namespaces. The frozen R3 engine continues to receive
Q-R3/QC-R3/QS-R3 in its own configuration: its Trainer has hard-coded QC and zero-Sham
branches. This explicit alias preserves those branches without modifying historical
code. `implementation_arm` is recorded in each new checkpoint identity. Engine
configuration is not a claim that a new checkpoint belongs to historical R3.
Model/optimizer/data/Oracle/Shadow/Q/LCG/STOP conditions are byte-compared with the
R3 proposal; beta is selected only by original Train-gradient Pilot rules. All
three arms start with identical tensor weights and the same data RNG seed+10.

Real seed:151001301; each arm500 updates×8×64=256000 live training tokens. Shadow
private updates are additionally counted. Critic and reserve slot remain allocated;
requires_grad counts are not per-token touched parameters. QS is same extra forward
with **zero applied LCG gradient**, not an independent random-gradient replication.

## Time points and commit semantics

Post-event series:0,50,100,150,200,250,300,350,400,450,500. At every lifecycle boundary
(100,200,400), retain both pre-event revision0 and all original committed Event
Revisions. At400 retain pre-event snapshot, pre-admission transaction revision,
each admission revision and post-promotion snapshot. Do not mix their labels.
A requested step without an explicit revision selects maximum committed revision;
revision0 explicitly selects the pre-event checkpoint. No missing snapshot is
reconstructed later and presented as a contemporaneous observation.

Workers own isolated RNG/CUDA visibility. They block after an immutable local
checkpoint/handoff until a single host serial publisher acknowledges a verified
HF commit. Other workers can compute against their private live trees; none can
write the master bundle. The master contains exact individually committed arm
states at potentially different steps, never partially copied or in-memory model
state. All progress is reported per arm. HF failures stop advancement at safe
boundaries; no process kill or hard runtime cutoff. Event processing retains
original RNG semantics; only added observations are RNG/mode isolated.

## Probes and metrics

Use the inherited fixed story/position/depth grid and pairs(0,1),(2,3),(4,5), group
feature initial input token ID mod2. Capture Calibration and independent Development
banks once at initial weights; preserve their bytes forever. Stopped/non-Cell
states are excluded without refilling and coverage denominators are reported.
Every snapshot binds arm identity, exact checkpoint SHA/revision, bank SHAs,
runtime **file** SHA and canonical analysis-plan SHA. Runtime semantic SHA is
separately named. No Confirmation role is materialized or read.

Quality uses the same frozen Development evaluation windows for greedy, sampled,
uniform, constant. Original policy-specific hashes are preserved privately; a
common asset/window/seed/role pairing key permits paired *cross-policy* contrasts.
Mean path length is non-STOP operator calls, normalized cost is non-STOP calls/B,
B=16. CTX fraction excludes STOP. Exposures are actual executed Cell events by
initial token parity; they are not specialization evidence by themselves.

Controller Q is **lower is better**. STOP margin=Q_STOP−min Q_CELL (positive means
STOP is worse); CTX margin is analogous. Rank by ascending Q. Score entropy is the
conditional Cell Boltzmann-score entropy, not epsilon-mixture trajectory entropy.
Legal STOP counts, near ties(tolerance1e-6), depth and ctx_seen strata are explicit.
At most32 frozen Calibration states per light snapshot receive independent common
random numbers for8 sampled suffixes comparing best-scored Cell with legal STOP
and CTX. Identifiability uses original paired MC uncertainty and original thresholds,
not only the sign of a noisy loss gap. Illegal STOP states are not measured as legal.

High-cost functional measurements at0/100/200/300/400/450/500 reuse the original
fixed_suffix, post-state derivative, Snapshot.relative and original cluster
summaries. Each accepted Cell's terminal fixed-suffix loss is computed once; expose
Cell rank correlation, oracle agreement and paired regret. No optional four-way
ordered-two-step enumeration is included: this experiment's primary functional
object is single-Cell advantage. Calibration additionally measures independently
replanned suffix agreement; Development uses the independent frozen bank. Neither
is training supervision. Cell advantages D_ij=E[L(j)-L(i)|group] use predefined
pairs. Crossover bands use inherited Bonferroni correction over pair×group family
at each time; longitudinal repeated looks remain exploratory, not confirmatory.

Gradient norms/cosines and clipping ratios come from the unchanged Trainer journal;
no second training backward is added for monitoring. Worker host/Oracle time and
CUDA stream elapsed time (collected at safe boundaries, not synchronized every
update) are separate. Diagnostic host/CUDA time, checkpoint bytes, file counts,
commit/readback time and newly submitted content bytes are recorded. Network wire
bytes, billed HF quota and kernel-only FLOPs are not inferred from file sizes.

## Statistics and interpretation

Primary at each post-event time: (QC greedy−QC own-uniform)−(Q greedy−Q own-uniform)
for NLL and NLL+0.01×normalized cost. Paired Story Cluster Bootstrap:2000 draws,
95% percentile intervals, equal weight to Story cluster means, same original
bootstrap seed and lambda. Report QC−Q greedy, QC−QC uniform and QC−QS negative
control separately. Require identical groups/window pairing keys before inference.
Report actual unique stories and independent seeds=1. Do not treat Q/QS or repeated
windows/steps as additional seeds. Negative/inconclusive results are retained.

H1 local credit-direction signal is distinct from the global system endpoint.
H2/H3 are exploratory patterns with coverage/CIs. Advantage/route/exposure lead-lag
records are descriptive aligned trajectories, not proof of causal reinforcement.
Always retain CAUSAL_FEEDBACK_NOT_TESTED and MULTISEED_NOT_TESTED. Diagnostic
completion is an engineering/evidence completeness outcome, not a requirement
that QC improves. Missing or corrupt identity evidence fails closed.
