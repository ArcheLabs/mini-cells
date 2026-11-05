# Communication Propagation-001 implementation record

Status: IMPLEMENTATION_READY; CP-001 private recovery/GPU execution NOT_RUN.
This is post-diagnostic exploratory mechanism research, not formal confirmation.

## Run

Import [communication-propagation.ipynb](../../communication-propagation.ipynb) into
the existing private `bingjiangli/minicells` Kaggle Notebook. Enable GPU, Internet,
`KAGGLE_API_TOKEN` and `GITHUB_TOKEN`, then Save & Run All. No Add Input, Output
mounting, manual identity fields or new data preparation is required.

The [user specification](../../COMMUNICATION-PROPAGATION-001.md) is preserved as
source material. The executable, separately hashed
[protocol](../../communication-propagation-protocol.json) fixes all choices before
CP-001 execution. Importing the dedicated Notebook selects `--communication-propagation`
and exits before the training launcher. No optimizer, GradScaler, backward, training,
new weights, new dataset, architecture change or confirmation is invoked.

## Existing evidence and asset provenance

The prior result branch
[`kaggle/epclm-core-001-c1-diagnostic-7ee5cd46e10c`](https://github.com/ArcheLabs/mini-cells/blob/kaggle/epclm-core-001-c1-diagnostic-7ee5cd46e10c/artifacts/experiments/epclm-core-001/mechanism-diagnostic/decision.json)
was read during implementation. It reports COMPLETE, exact D0 NLL
5.814626392302277 on 16 documents / 3322 targets, unchanged before/after C1 weight
hash, and PRIVATE_KAGGLE_OUTPUT version 142. Its findings were state and communication
dependence, not semantic memory or a delayed communication benefit.

CP-001 reuses that pinned training identity and frozen loader; it still verifies the
actual recovered bytes, completed checkpoint identity, model configuration, tokenizer,
development data, weights and baseline at runtime. Previous successful recovery is not
proof that private assets remain available now. The latest saved version is inspected
for exact CP-001 progress only, then model retrieval prioritizes known version 142.
Missing preferred versions may use bounded recent-version fallback. Authentication,
429/transport failure and exhausted recovery budgets stop safely. No replacement
training is allowed. Default recovery behavior for prior diagnostic callers is unchanged.

## Interventions and correctness

The normal `Core` and its parameters are unchanged. `DiagnosticCore.evolve()` adds
optional `intervention` / `on_update` arguments; the no-argument path remains the original
forward. The edge pulse is a runtime left-message zero at exactly one receiver in the
first internal update; all remaining rounds/events use normal communication. The
source must equal `(receiver-1) % 16`. Active edge interventions require inference mode.
The callback observes completed updates without injecting another token.

The original 16 stories are processed once with normal K=4 inference, checking the
original NLL within 1e-4 and 3322 prediction tokens. Only pre-event states at floor
25%, 50%, 75% positions are cached. For each selected probe, a short normal trajectory
and a cloned SHAM trajectory are replayed once. The instrumented normal path is checked
against original `model.step()`. SHAM must pass all internal state, logit, NLL, JS and
TV thresholds before any pulse at that probe.

There are at most 48 probes, 48 global branches and 192 local branches. Global removal
lasts exactly the four internal rounds of h=0. Each local edge is removed exactly once
in the first round of h=0. B spatial and C predictive measurements share each local
trajectory; no duplicate B/C replay is performed. All paths teacher-force identical
future input IDs and targets; a private hash records the exact future input identity.
No path crosses a document reset, samples its own tokens or sees future targets before
prediction.

Every completed update uses ring distance from the receiver. At zero-based event h
and internal step k, the allowable radius is `4*h+k`. Measurable difference outside
that cone is an implementation failure. A measurable first-round change can only be
at the receiver. Reachability is an upper bound, not a promise that every reachable
cell changes. Per-cell norms/relative norms and first measured arrivals are recorded.
Readout pools all cells, so immediate output changes are not evidence of spatial spread.

FP32 state-difference thresholds are `1e-6 + 1e-6 * normal_cell_L2_norm`, with epsilon
1e-12. SHAM also bounds maximum logit difference by `1e-6 + 1e-6 * max_abs_logit`, NLL
by 1e-6, JS by 1e-10 and TV by 1e-8. JS/TV probability reductions use FP64 to avoid
cancellation in tiny differences; model parameters, event/state calculations, logits
and NLL remain FP32, with TF32/autocast disabled. No tolerance is relaxed from results.
NaN/Inf, weight change, no-op/SHAM inconsistency and cone violations stop the run.

## Statistics and exploratory decisions

All h=0..16 events are actually consumed and measured. Public temporal summaries report
all fixed horizons 0/1/2/4/8/16, including zero-availability counts. Both signed ΔNLL and
absolute ΔNLL are retained. Inclusive cumulative windows average every h=0..H, for
H=4/8/16; separate future-only windows average h=1..H. Only complete H windows enter
fixed-window estimates, with explicit observation/story counts; truncated probes remain
in the horizons they can reach.

Each statistic first averages probes/edges within a story, then averages stories equally.
The 1000-draw bootstrap resamples whole stories, preserving within-story clustering.
Report global, pooled-local and every fixed receiver separately, without selecting the
best horizon/edge. These intervals describe a single development model, not training-seed
variance or multiple independently confirmed tests.

Spatial transmission requires a first measurable receiver change followed by another
cell's measurable change within the cone. Persistence requires post-event state difference.
Future prediction sensitivity requires JS above max(pre-fixed JS tolerance, observed SHAM
JS floor) at a fixed future horizon in at least two stories. None of these imply utility.

The more demanding USEFUL_DELAYED_COMMUNICATION_SIGNAL label has one fixed primary
comparison: local edges pooled, future-only H=16. It requires at least 8 contributing
stories, mean ΔNLL >=0.001, >=75% positive story means, paired story bootstrap lower
bound >0, and positive future-window means at H=4,8,16. This is an explicitly chosen
exploratory practical threshold, not a validated universal effect-size threshold.
No alternative best-looking horizon substitutes for that rule. H-CP1–4 receive only
exploratory signal/no-clear-signal labels; Core H1–H5 remain NOT_EVALUATED.

## Cost, recovery and publication

Independent recovery cap: 600 seconds; diagnostics cap: 900 seconds. The latter includes
baseline, all pulse/SHAM calculations, metric collection and progress saves. Both have
interruptible timers plus per-update deadline checks. Overall execution stays within
5400 seconds, publication reserves 600 seconds and safety 1200 seconds; Notebook total
hard bound remains 7200 seconds. Recovery/network time is separate from synchronized
forward-plus-metric timing. Progress IO time is reported separately; no kernel profiler,
CPU performance benchmark or repeated GPU pilot is used.

Each completed branch is atomically written with diagnostic identity and SHA-256, then
committed with the existing immutable recovery manifest. Partial resume reconstructs the
small normal/probe caches and skips verified finished pulse branches. COMPLETE retries
only rebuild/publish reports, without model inference. Terminal failures remain failures.
Private model assets are cached outside diagnostic progress, with original checkpoint
bytes unchanged. Per-probe horizon/internal traces remain private; the fixed heatmap
selection publishes only derived state-difference arrays, never tokens, weights or states.

The two figures are a signed/absolute temporal response comparison and four local-edge
heatmaps from the preselected first story's 25% probe. No substitute visual example is
chosen based on results. Public reports include aggregated internal-round/cell profiles,
all horizons, identities, counts, thresholds, numerical controls and limitations.
Publication uses the existing metadata-only publisher, a result commit descending from
main, remote SHA verification and independent `kaggle/epclm-core-001-cp001-<source12>`
branch. Historical training and mechanism-diagnostic results are not overwritten.

## Validation and pending execution

Only three tiny synthetic CPU tests were added: no-op/SHAM equivalence, single-edge
isolation, and causal cone. They also assert no optimizer/backward, no repeated event
injection, unchanged frozen weights and rejection of an intentionally impossible cone.
No GPU experiment, real private checkpoint access, new training or local benchmark was
performed. CP-001 execution, observed costs and findings remain pending Kaggle execution.
