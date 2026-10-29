# JCT-001 — Joint Cell–Connection Training

**MiniCells Research Protocol**

| Field | Value |
|---|---|
| Experiment ID | JCT-001 |
| Full name | Joint Cell–Connection Training |
| Protocol version | 0.1.0 |
| Status | DRAFT — NOT RUN |
| Scientific objective | Establish whether joint Cell–Connection training provides reproducible, attributable learning benefits |
| Primary comparison | Joint training vs. Cell-only training |
| Secondary objective | Identify whether Cell and Connection exhibit a positive interaction effect |
| Execution authorization | BLOCKED pending prerequisite gates |
| Confirmation data | SEALED |

> This document defines a proposed experiment. It does not claim that JCT-001 has been executed or that joint training has been demonstrated.

---

## 1. Executive Summary

MiniCells represents computation as an organization of Cells connected through learned Connections. Its central research hypothesis is that model quality can improve when both the computational units and the organization of computation adapt through training.

The existing research establishes useful foundations for this hypothesis, but it does not yet establish that joint Cell–Connection training reliably outperforms Cell-only training.

JCT-001 will investigate three questions:

1. Can the Connection learning mechanism discover useful computational paths without being given the correct paths as training labels?
2. Does updating Cell and Connection parameters together improve held-out task performance relative to updating Cell parameters alone?
3. Does joint training produce a positive interaction effect beyond the separate contributions of Cell learning and Connection learning?

The experiment is divided into two stages:

- **JCT-001-Pilot:** Verify the implementation and test the hypothesis using a controlled synthetic task, followed by a limited real-model experiment if all prerequisites pass.
- **JCT-001-Confirmation:** Independently confirm the predefined primary result using new training Seeds and sealed evaluation data.

A successful synthetic experiment is necessary evidence that the mechanism can work under controlled conditions. It is not sufficient evidence that the mechanism improves language models.

A successful joint-training experiment is also not, by itself, proof of autonomous reorganization, emergent specialization, or continual learning.

## 2. Existing Evidence and Research Motivation

The protocol must preserve the distinctions between prior findings.

### 2.1 R3j — Long-horizon Connection value

The published R3j metrics record:

- 12 selected cases.
- 2,160 completed new Cell updates.
- 0 new Connection optimizer updates.
- Exploratory long-horizon counterfactual evaluation.

The results demonstrate that short-term candidate quality is not a reliable substitute for long-horizon evaluation. They do not demonstrate joint parameter training.

Reference: [R3j scientific findings](https://github.com/ArcheLabs/mini-cells/blob/results/r3j-1553989692a6159a-edec11d95a0a/research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3j/reports/published/1553989692a6159aa53360abce84be8fd64bec2f86094b997d5316fc6b9e2dae/SCIENTIFIC-FINDINGS.md).

### 2.2 R3k — Future learning value

R3k evaluates whether early signals predict future learning value. Its published manifest records zero Connection updates, and its exploratory predictive-signal analysis does not support the tested predictors.

These findings motivate a direct test of learning outcomes rather than assuming that early metrics or candidate scores establish Connection value.

Reference: [R3k scientific findings](https://github.com/ArcheLabs/mini-cells/blob/results/r3k-8e8b03430f736cfc-edec11d95a0a/research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3k/reports/published/8e8b03430f736cfc2c25768045dc7ccff352a57297a43e6f9b30aeeae828c701/SCIENTIFIC-FINDINGS.md).

### 2.3 R3l — Connection reorganization potential

R3l reports no detectable realizable Connection reorganization opportunity and no observed long-horizon gain under its tested protocol. A target-aware diagnostic identified a path opportunity, but such a diagnostic is not evidence that the ordinary router can discover or exploit that opportunity.

Reference: [R3l scientific findings](https://github.com/ArcheLabs/mini-cells/blob/results/r3l-7c972812699f6d08-edec11d95a0a/research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3l/reports/published/7c972812699f6d08d3f5bdad457dcd50c3d3d56426c26a3b8ff9740c11a50ce1/SCIENTIFIC-FINDINGS.md).

### 2.4 R3m-F-D1 — Cell learning diagnostics

R3m-F-D1 trained Cell parameters and compared context organization and loss supervision. Its best reported arm achieved 3/8 exact answers on the eight-question arithmetic evaluation.

This is evidence about the tested Cell training configuration. It is not evidence of joint Cell–Connection optimization.

Reference: [R3m-F-D1 implementation](https://github.com/ArcheLabs/mini-cells/blob/2e64ee711b204002c5e8d3418dd95bb833543359/research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3m-f-d1/implementation/minicells_r3mf_d1/training.py).

### 2.5 Research conclusion

The current evidence motivates JCT-001 but does not predetermine its outcome.

The first obligation is to establish that the intended Connection optimizer actually executes, changes the intended parameters, and influences the computation being evaluated.

---

## 3. Research Questions and Hypotheses

### H1 — Connection learning

Under a controlled task with known useful computational paths, a trainable Connection mechanism can discover useful paths from task feedback without receiving the correct path as an explicit training label.

**Primary falsification condition:** The ordinary learning mechanism cannot outperform an appropriate frozen or randomized routing control on held-out compositions, despite a validated task and implementation.

### H2 — Joint training benefit

Joint Cell–Connection training improves held-out task quality relative to Cell-only training under matched training data and a matched logical training budget.

**Primary contrast:**

\[
\Delta_{\mathrm{joint}}=L_{11}-L_{10}
\]

Where:

- \(L_{11}\): held-out loss after joint training.
- \(L_{10}\): held-out loss after Cell-only training.

A negative value favors joint training.

### H3 — Positive interaction

Joint training produces a benefit beyond the additive effects of separately updating Cell and Connection parameters.

\[
I=L_{11}-L_{10}-L_{01}+L_{00}
\]

Where:

- \(L_{00}\): both parameter groups frozen.
- \(L_{10}\): Cell-only.
- \(L_{01}\): Connection-only.
- \(L_{11}\): joint training.

A negative value supports a positive interaction on the loss scale.

H2 and H3 are separate hypotheses. Joint training may outperform Cell-only training without exhibiting a positive interaction.

### H4 — Long-horizon retention

Any observed joint-training benefit persists after a predefined continuation period and is not merely a transient improvement immediately after a Connection update.

This is a secondary hypothesis and must not replace the primary endpoint.

---

## 4. Scope and Non-goals

JCT-001 investigates parameter learning and its measurable consequences.

It does not attempt to establish all of the following in one experiment:

- Autonomous detection of when reorganization is necessary.
- Unrestricted addition or deletion of Cells.
- General-purpose language-model capability.
- Emergent specialization across arbitrary tasks.
- Continual learning without interference.
- Superiority to conventional neural architectures in general.

Cell topology, Cell count, model width, tokenizer, and other architectural properties must remain fixed during the primary experiment.

Any dynamic topology experiment must be registered separately.

---

## 5. Stage 0 — Prerequisite Gates

No scientific training may start until the required gates have passed.

### Gate 0.1 — Data registration and isolation

The current R3m preparation is blocked by data-registration conflicts. Previously proposed Story ranges overlap historical data usage, and subsequent candidate ranges have also failed audit.

JCT-001 must not silently select a new range or reinterpret an existing registration.

Required evidence:

- Verified dataset revision and content identity.
- A row-level or content-level allocation ledger covering relevant historical experiments.
- Explicit registration of training, calibration, Pilot Report, and Confirmation roles.
- Duplicate detection and leakage checks.
- A frozen tokenizer identity and source revision.
- Separate Confirmation data that remain inaccessible during protocol development.

If an approved, unused range cannot be established, the real-model experiment remains blocked.

The synthetic pilot may proceed independently only after its own data generator, train/test split, and task identity checks are registered.

### Gate 0.2 — Task learnability

The arithmetic task-learnability gate in the current R3m preparation has not passed.

Before using arithmetic to evaluate joint learning, demonstrate that the baseline task is learnable under the available model capacity and training budget.

The baseline must:

- Beat a no-training or fixed-output reference.
- Beat an appropriate answer-frequency baseline.
- Show nontrivial performance on held-out canonical arithmetic expressions.
- Avoid expression, operand, answer, and template-family leakage.
- Use a frozen evaluation set that is not used to tune the training method.

Failure means the task is not yet suitable for evaluating Connection value. It does not falsify the general Connection hypothesis.

### Gate 0.3 — Checkpoint restoration

Every experimental branch must start from the same verified checkpoint for its Seed.

The restored state must include all relevant components:

- Model and Cell parameters.
- Connection parameters and routing state.
- Cell and Connection optimizer states.
- Learning-rate scheduler and other schedules.
- Random-number generator states.
- Data cursor and sampling state.
- Router temperature, exploration state, and any stateful routing buffers.

The implementation must verify the restored values, not merely copy a checkpoint hash.

The existing research lineage contains checkpoint references with `cross_session_verified=false`. A fresh process within the same session is insufficient to establish independent cross-session recovery.

If the required checkpoint cannot be recovered and verified, the experiment must stop or be registered as a new baseline-generation study. A different checkpoint must not be silently relabeled as the original.

### Gate 0.4 — Connection optimizer audit

Before training, establish that the actual intended Connection learning algorithm is operational.

Required tests:

1. Connection parameters receive the intended update signal.
2. The intended optimizer step is executed.
3. The parameter delta is nonzero when the test requires an update.
4. The update changes the expected routing behavior or Connection state.
5. Cell parameters remain unchanged in the Connection-only arm.
6. Connection parameters remain unchanged in the Cell-only arm.
7. Both parameter groups change in the joint arm when their update rules prescribe changes.
8. Frozen controls remain frozen.
9. Optimizer state restoration produces a reproducible continuation within the declared numerical tolerance.

The Connection optimizer must use the reviewed production learning mechanism. If the implementation uses Q-first collection, a Trust Region, or another specialized optimizer, the exact algorithm and configuration must be frozen and tested.

An optimizer call count alone does not prove that a useful Connection update occurred.

### Gate 0.5 — Runtime and budget audit

The budget system must distinguish:

- Logical training updates.
- Actual optimizer calls.
- Failed and retried updates.
- Total training tokens.
- Measured GPU time.
- Estimated or measured computation.
- Evaluation and checkpointing overhead.

Budget proofs for pre-run checks must not be presented as proofs of actual GPU execution.

The final run manifest must reconcile these records.

### Gate 0.6 — Protocol freeze

Before the Pilot begins, freeze:

- Source revision and configuration.
- Data registration.
- Model and optimizer identity.
- Primary and secondary metrics.
- Training budget.
- Seed allocation.
- Acceptance criteria.
- Failure and stopping conditions.
- Confirmation analysis plan.

No Confirmation data may be materialized to tune the protocol.

---

## 6. Stage 1 — Synthetic Known-Path Pilot

### 6.1 Objective

Determine whether the Connection learning algorithm can discover useful computational organization in a setting where the correct solution is known independently.

This stage isolates the routing mechanism from the difficulty of learning language or arithmetic.

### 6.2 Task design

Construct a deterministic task involving a fixed collection of Cells.

Each Cell implements a known primitive operation. The input contains enough information to determine which composition of operations is required, but the correct path is not directly supplied to the ordinary router.

Different task families require different useful paths.

The generator must create:

- Training compositions.
- Calibration compositions.
- Held-out compositions.
- An independently computed reference output for every example.

The split must be based on canonical task identity, not merely text strings.

The router may use only the state and inputs permitted by the intended architecture. It must not receive the answer, an oracle-selected path, or a hidden task label unavailable at deployment time.

### 6.3 Required controls

Evaluate at least:

1. Trainable Connection.
2. Frozen Connection.
3. Randomized Connection with a registered randomization procedure.
4. Oracle path, used only as a diagnostic upper-bound reference.

The Oracle is not a trainable treatment and must never be used to select the ordinary router's actions during evaluation.

### 6.4 Pilot metrics

Primary metrics:

- Held-out task accuracy.
- Held-out task loss.
- Correct-path selection rate, where a unique correct path is defined by the task.

Secondary metrics:

- Route diversity.
- Per-Cell utilization.
- Route stability across equivalent inputs.
- Execution cost.
- Sensitivity to unseen compositions.

If multiple paths are functionally equivalent, the evaluation must recognize equivalence rather than treating one arbitrary path as uniquely correct.

### 6.5 Pilot decision

Proceed only if:

- The ordinary trainable Connection improves held-out task performance over the frozen and randomized controls.
- The improvement is reproducible across the Pilot Seeds.
- The result is not explained by leakage, oracle information, extra execution budget, or Cell parameter changes in a supposedly Connection-only experiment.
- The optimizer and budget audit passes.

Failure returns the work to implementation or mechanism research. It does not authorize bypassing the gate by using the Oracle.

---

## 7. Stage 2 — Real-Model 2×2 Factorial Experiment

### 7.1 Experimental arms

The primary experiment has four arms.

| Arm | Cell parameters | Connection parameters | Purpose |
|---|---|---|---|
| A — Frozen | Frozen | Frozen | No-learning reference |
| B — Cell-only | Trainable | Frozen | Baseline Cell learning |
| C — Connection-only | Frozen | Trainable | Isolated Connection contribution |
| D — Joint | Trainable | Trainable | Joint learning |

All four arms must begin from the same verified Seed-specific checkpoint.

All arms use the same data examples, sequence lengths, task curriculum, logical update budget, and evaluation schedule within each matched block.

When training stochasticity is present, record the random streams and the rules governing common random numbers. Do not claim exact paired randomness if different routing paths necessarily consume random values differently.

### 7.2 Interpretation of Connection-only training

Freezing Cell parameters must not inadvertently prevent gradients or other intended learning signals from reaching Connection parameters.

The computation must remain differentiable with respect to the relevant routing decisions, or the approved Connection optimizer must provide its intended alternative learning signal.

The audit must verify the actual implementation path.

### 7.3 Training environments

Use two registered environments when the task and data gates permit them.

**Environment N — No New Capability Demand**

Continue the existing Story task under the registered training distribution.

Purpose:

- Measure whether Connection updates introduce unnecessary interference.
- Test whether joint training changes the model when no new capability is required.
- Detect regressions in existing task quality.

**Environment C — Capability Demand**

Introduce a new, learnable task while preserving a fixed portion of the existing Story distribution.

A candidate curriculum from the R3m research design is:

| Relative update range | Story | New task |
|---|---:|---:|
| 0–100 | 90% | 10% |
| 100–200 | 70% | 30% |
| 200–400 | 50% | 50% |
| 400–600 | 50% | 50% |

These are candidate settings, not an authorized run configuration. The final curriculum must be reviewed and frozen after data registration and task learnability are resolved.

The presence of a new task does not automatically prove that the environment creates meaningful organizational pressure. That must be demonstrated through the measured learning and interference results.

### 7.4 Task definitions

If the registered R3m task design is reused, retain its separation between Story and Arithmetic evaluation.

Arithmetic data must be generated and registered as separate collections for:

- Training.
- Trigger calibration.
- Pilot evaluation.
- Confirmation evaluation.

Canonical arithmetic expressions, operand combinations, answers, and template families must be controlled for leakage.

The arithmetic evaluation must report:

- Answer-token NLL.
- Complete-answer exact match.
- Carry and non-carry performance.
- Comparison with an answer-frequency baseline.

Story evaluation must report held-out NLL separately.

A combined score must not conceal a regression on the original task.

### 7.5 Candidate budget

A provisional engineering budget is:

- 600 logical updates per branch.
- 512 training tokens per update.
- Shared initialization work per Seed, rather than duplicated pretraining for each arm.

The resulting branch budget is 307,200 training tokens.

These numbers are planning values aligned with the current R3m candidate design. They are not measured runtime guarantees and do not authorize execution.

The final budget must be frozen only after the task and implementation gates pass.

All four arms must be compared under both:

1. Matched logical training exposure.
2. Measured quality versus actual compute cost.

If joint training requires extra computation, report that cost explicitly. Do not present a quality gain as a free benefit.

---

## 8. Randomization, Seeds, and Statistical Units

### 8.1 Pilot

Use three independently initialized or independently registered Seeds, subject to successful checkpoint verification.

For each Seed, generate one shared initial state and branch all four arms from it.

The Pilot estimates effect direction, variability, implementation reliability, and practical compute cost.

Pilot results are exploratory and cannot be reused as independent Confirmation evidence.

### 8.2 Confirmation

Use new Seeds that were not used to select the final protocol.

Five new Seeds may serve as a minimum planning baseline, but the final sample size must be determined by a prospective power analysis using Pilot variability and a predefined minimum practically important effect.

If five Seeds cannot provide adequate power, increase the sample size or report the result as inconclusive.

The number of test examples does not substitute for the number of independent training Seeds.

### 8.3 Paired comparisons

For each Seed and environment, compute the four arm-level outcomes and derive paired contrasts.

Primary contrast:

\[
D_s=L_{11,s}-L_{10,s}
\]

Interaction contrast:

\[
I_s=L_{11,s}-L_{10,s}-L_{01,s}+L_{00,s}
\]

Aggregate these contrasts across independent Seeds.

Report:

- Mean and median effect.
- Per-Seed effects.
- Confidence intervals.
- Between-Seed variability.
- The prespecified inferential test.
- Adjusted significance for the confirmatory hypotheses.

Use a registered method appropriate for the available number of independent Seeds. Any multiple-testing correction must be specified before Confirmation.

A nonsignificant result is not proof that the effects are exactly zero.

---

## 9. Primary and Secondary Endpoints

### 9.1 Primary endpoint

The primary real-model endpoint is held-out answer-token NLL on the registered capability-demand task.

The primary comparison is Joint versus Cell-only.

The minimum practically important improvement, denoted by \(\delta_{\min}\), must be determined from Pilot results and practical compute considerations before Confirmation data are accessed.

The threshold must not be changed after inspecting Confirmation outcomes.

### 9.2 Confirmatory interaction endpoint

The second confirmatory endpoint is the factorial interaction contrast \(I\).

A positive interaction claim requires evidence that the interaction is favorable under the registered loss scale and inferential procedure.

Joint-versus-Cell-only superiority alone is insufficient.

### 9.3 Secondary endpoints

Report:

- Complete-answer exact match.
- Story held-out NLL.
- Training and evaluation loss trajectories.
- Long-horizon retention.
- Per-Cell utilization and contribution.
- Connection parameter and route changes.
- Gradient and optimizer diagnostics.
- Training tokens and update counts.
- Actual GPU time and compute-cost estimates.
- Peak memory.
- Numerical failures, retries, and rejected updates.

Secondary endpoints must not replace a failed primary endpoint.

---

## 10. Attribution and Mechanism Tests

A lower test loss does not by itself explain why the joint arm improved.

The following analyses are required.

### 10.1 Parameter-update evidence

Record parameter checksums and parameter-delta statistics at registered checkpoints.

For every arm, demonstrate that the intended parameter groups changed and the frozen groups did not.

### 10.2 Routing evidence

Record:

- Route distributions.
- Per-Cell utilization.
- State-conditioned route changes.
- Connection update magnitude.
- Route stability on repeated or equivalent inputs.
- Routing behavior on held-out task compositions.

A more uniform route distribution is not automatically better. Specialization must be evaluated against task performance and useful functional differences.

### 10.3 Post-training interventions

As secondary diagnostic experiments, compare the joint-trained model under:

1. Its learned Connection.
2. Its initial or frozen Connection.
3. A registered route-permutation or randomized Connection control.
4. An Oracle path, if the task supports a valid Oracle definition.

Where practical, retain the learned Cell parameters while changing only the Connection used at evaluation.

Interpret these interventions cautiously: changing routing can create distribution shifts, so a drop in performance alone does not establish that a particular Connection caused the entire training benefit.

### 10.4 Cell specialization

Do not infer specialization solely from utilization imbalance or parameter divergence.

A specialization claim requires evidence that Cells have differentiated functional contributions and that those differences improve held-out task performance.

Cell replacement and capacity-expansion studies are outside the primary JCT-001 scope.

---

## 11. Long-Horizon Evaluation

Short-term improvements may reverse during continued training.

Evaluate each arm at predefined checkpoints, including:

- Initial baseline.
- Early training.
- Mid-training.
- End of training.
- A fixed continuation horizon after the primary training period.

The continuation phase must be registered in advance and use the same rules across arms.

Report the trajectory rather than only the best checkpoint.

A joint-training benefit that appears briefly and then disappears does not satisfy the long-horizon hypothesis.

If an arm is stopped early because of a numerical failure, preserve its failure record. Do not silently replace it with a more favorable run.

---

## 12. Acceptance Criteria

JCT-001 has separate implementation, Pilot, and Confirmation decisions.

### 12.1 Implementation PASS

All of the following are required:

- The correct code revision and configuration are identified.
- Checkpoint restoration is verified.
- The Connection optimizer executes the intended algorithm.
- Parameter-update behavior matches the assigned arm.
- Data and task identities are registered.
- Training and evaluation boundaries are verified.
- Budget accounting reconciles actual execution.
- No Oracle or hidden target information reaches ordinary training or evaluation.

### 12.2 Pilot PASS

All of the following are required:

- The synthetic task is learnable.
- Trainable Connection outperforms appropriate routing controls on held-out compositions.
- The result is not explained by leakage or unequal training exposure.
- The real-model task, if used, passes its independent learnability gate.
- The experiment provides sufficient evidence to justify a powered Confirmation study.

A Pilot PASS does not establish scientific confirmation.

### 12.3 Confirmation PASS — Joint-training benefit

All of the following are required:

- The primary contrast favors Joint over Cell-only.
- The effect meets the predefined practical threshold \(\delta_{\min}\).
- The prespecified inferential criterion is satisfied.
- The result is not driven by a single Seed or a post hoc checkpoint selection.
- The effect is supported on the registered held-out evaluation.
- Budget and execution records pass audit.

### 12.4 Confirmation PASS — Positive interaction

A separate positive interaction claim additionally requires:

- The interaction contrast favors Joint.
- The interaction meets its registered practical threshold.
- The prespecified adjusted inferential criterion is satisfied.

If the primary joint-training contrast passes but the interaction does not, report that Joint training improved the task without claiming demonstrated positive synergy.

### 12.5 Inconclusive

Use `INCONCLUSIVE` when:

- The point estimate favors Joint but uncertainty is too large.
- The experiment lacks sufficient power.
- The task is too easy or too difficult to distinguish the treatments.
- The implementation or budget evidence is incomplete.
- The result depends on a single Seed or unstable evaluation behavior.

### 12.6 FAIL

Use `FAIL` when a required implementation or data-integrity gate is violated, or when the registered scientific criterion is not met and the study has sufficient validity and power to support that conclusion.

A task-learnability failure must not be reported as a falsification of Connection learning in general.

---

## 13. Failure and Stopping Rules

Stop before scientific training if:

- Data registration remains unresolved.
- Checkpoint restoration cannot be verified.
- Connection-only updates cannot be demonstrated.
- Frozen controls mutate unexpectedly.
- The intended Connection learning signal is not available.
- The training task fails its learnability gate.
- Budget accounting cannot reliably capture retries or actual execution.

During training, stop or quarantine a branch if:

- A numerical or state-integrity invariant fails.
- The run accesses sealed evaluation data.
- An unexpected parameter group changes.
- A retry exceeds the registered resource budget.
- The actual training configuration diverges from the frozen manifest.

Do not resume a failed run under a different configuration while retaining the same run identity.

Any change to the experimental treatment requires a new configuration identity and, where scientifically material, a new protocol revision.

---

## 14. Required Research Artifacts

Every run must produce a complete, auditable record.

| Artifact | Required content |
|---|---|
| `JCT001_PROTOCOL.md` | Frozen research protocol and revision history |
| `CONFIG.json` | Architecture, optimizer, schedules, budgets, and treatment definitions |
| `DATA_REGISTRY.json` | Dataset revisions, roles, ranges, hashes, and leakage checks |
| `CHECKPOINT_RECEIPTS.json` | Source checkpoint identities and restoration verification |
| `RUN_MANIFEST.json` | Run identity, Seed, source SHA, status, and execution summary |
| `ARM_RESULT.json` | Metrics, parameter updates, failures, and per-arm outcome |
| `ROUTING_EVOLUTION.json` | Routing changes and per-Cell utilization |
| `CELL_CONNECTION_ATTRIBUTION.json` | Registered attribution analyses |
| `COMPUTE_COST.json` | Actual time, tokens, updates, retries, and cost accounting |
| `BUDGET_PROOF.json` | Reconciled runtime budget evidence |
| `SEED_SUMMARY.json` | Per-Seed paired contrasts and uncertainty |
| `SCIENTIFIC-FINDINGS.md` | Findings, limitations, deviations, and final decision |

Every artifact must be tied to the exact run identity and relevant source revision.

The report must distinguish:

- Planned versus completed work.
- Pre-run checks versus actual execution.
- Exploratory results versus Confirmation results.
- Observed evidence versus interpretation.
- Missing evidence versus negative findings.

No final report may claim that an experiment ran merely because its launcher, notebook, or configuration exists.

---

## 15. Decision Tree

### Outcome A — Synthetic routing fails

Return to Connection learning implementation, optimization, or task design.

Do not start the full real-model matrix.

### Outcome B — Synthetic routing succeeds, real-model task is not learnable

Resolve the task or model-capacity problem first.

Do not conclude that Connection learning is ineffective.

### Outcome C — Joint improves over Cell-only, interaction is unsupported

Report an incremental joint-training benefit if Confirmation criteria pass.

Do not claim that the experiment has demonstrated synergy.

### Outcome D — Joint and interaction both pass Confirmation

Conclude that, under the registered task, architecture, training budget, and evaluation distribution, joint Cell–Connection training produced a reproducible benefit and evidence of a positive interaction.

Then investigate whether the result generalizes to other tasks and training conditions.

### Outcome E — Short-term benefit reverses over the continuation horizon

Do not claim stable joint-learning value.

Investigate update stability, long-horizon effects, and the relationship between Connection changes and subsequent Cell learning.

### Outcome F — Joint does not outperform Cell-only

If the study is valid and sufficiently powered, report that the tested joint-training method did not meet the registered improvement criterion.

This does not establish that all possible Connection learning mechanisms are ineffective.

---

## 16. Relation to R3m

JCT-001 and R3m answer related but distinct questions.

JCT-001 asks:

**Does jointly training Cell and Connection produce a reproducible benefit compared with controlled alternatives?**

R3m-P asks:

**Does the current architecture contain a real, sustained, realizable opportunity for useful reorganization?**

R3m-M asks:

**Can the system learn when reorganization is necessary without being given that decision in advance?**

JCT-001 should not automatically authorize R3m-M.

A successful JCT-001 result is evidence for joint training, not proof of autonomous reorganization. R3m-P must still establish that reorganization has useful, sustained value. R3m-M must independently establish that the model can identify when to reorganize.

The experiments must retain separate identities, manifests, data boundaries, and decision criteria.

---

## 17. Final Research Decision

JCT-001 is designed to turn the current hypothesis into a falsifiable experiment.

The key result is not that both parameter groups change during training. It is that their joint adaptation produces a measurable, reproducible benefit that cannot be adequately explained by Cell-only learning, Connection-only learning, additional compute, data leakage, or a transient routing improvement.

Until the prerequisite gates pass and the Confirmation experiment succeeds, the scientific status remains:

**Joint Cell–Connection training: NOT YET ESTABLISHED.**

No execution or positive scientific result is implied by this protocol.