# CLM Capability Ceiling Audit

Audit date: **2026-09-03**

This document records the strongest claims that the current MiniCells evidence supports without promoting controlled results into stronger Native-CLM or product claims.

## Executive conclusion

The evidence supports a useful set of **model-evolution engineering primitives**. It does **not** currently support autonomous replay-free continual language learning, a natural Cell ontology, or a local rule that can decide whether a model change is globally beneficial.

The near-term ceiling is therefore:

> Localize candidate changes, preserve provenance, bound/verify mutations, add capacity when needed, evaluate candidates globally, and retain commit/rollback control.

The long-term Native-CLM hypothesis remains research, not an engineering dependency.

## Capability matrix

| Capability | Audit status | Current ceiling | Engineering implication |
|---|---|---|---|
| Exact registered-history protection in a controlled writable Cell | **SUPPORTED — controlled** | Core 005: fixed features, linear writable Cells, explicit routing, exact certificate geometry | Safe local mutation can be an engineering primitive when its protected contract is explicit |
| Replay-free subspace state can reduce forgetting on real representations | **SUPPORTED — partial** | Core 006 and Native M2 reduce measured forgetting, but absolute continual-language gates fail | Certificates can be regression/safety aids; they are not a complete CL algorithm |
| Growth can restore writable capacity in controlled settings | **SUPPORTED — controlled** | Core 004/005 and Constructive CLM show capacity rescue; Native M3 shows capacity alone does not produce safe continual behavior | Prefer append/expand over destructive mutation when existing capacity is constrained, but require global validation |
| Learned sparse Cells can compose at model level | **SUPPORTED — controlled** | Constructive CLM-004 passes registered simultaneous/sequential composition gates for linear residual operators | Multi-module computation is a usable structural primitive |
| Learned routing/write/growth control can replace explicit synthetic scaffolds | **SUPPORTED — controlled** | Constructive CLM-005, while fixed certificate safety geometry remains supplied | Learned control is possible in the registered world; do not infer autonomous language-scale CL |
| A small Native CLM can train from next-token loss | **SUPPORTED** | Stage-06 M0/M1; ~12M model trains and executes sparsely | Native architecture trainability is not the blocker |
| Fixed-topology replay-free continual language learning | **NOT SUPPORTED** | Native M2: protection helps but A regression remains far above the registered target | Do not claim replay-free continual language success |
| Global-pool growth restores continual-language retention | **NOT SUPPORTED** | Native M3: growth and child reuse occur, but old-domain retention worsens | New capacity must preserve read ownership; simple global insertion is unsafe |
| Read-preserving lineage growth solves continual retention | **NOT SUPPORTED** | M3R improves the M3 failure mode but remains far outside the absolute retention gate | Root ownership conservation is useful but insufficient |
| Current cosine/centroid address is the correct lineage-local functional boundary | **NOT SUPPORTED** | M3R address diagnostic: current cosine rule near chance while an offline affine query probe is strongly separable | Address representation contains signal; the deployed local rule does not decode it |
| Rank-16 replay-free historical query sketch is sufficient for the local gate | **NOT SUPPORTED** | M3L misses its preregistered median-AUC gate despite being a near miss | Compact address history remains an open capacity/family question |
| Semantic/routing address is a natural Cell split boundary | **NOT SUPPORTED** | Core 006 and later boundary work show routing/semantic identity is not equivalent to writable functional independence | Do not organize engineering Cells by semantic labels and call them natural atoms |
| A stable natural functional Cell boundary has been discovered in pretrained LLMs | **NOT ESTABLISHED** | Core 007/008/009 families provide mixed diagnostics and bridge evidence, not an integrated stable low-growth boundary | Keep natural-boundary discovery on the research track |
| Safe gradient projection implies safe optimizer update | **FALSE for canonical AdamW path** | M2-R0/R0b isolate optimizer mechanics; AdamW preconditioning can leave the protected update subspace | Validate/project the realized optimizer transaction, not only the gradient |
| Final realized-update projection can restore the registered update invariant | **SUPPORTED — mechanics** | M2-R0b reduces committed violation to the numerical reference floor | `optimizer proposal -> final update projection -> commit` is a reusable safety primitive |
| Local safety/improvement implies a globally better model | **NOT ESTABLISHED** | No experiment supplies this implication; historical failures repeatedly show local mechanisms transferring error elsewhere | Candidate generation and global model acceptance must be separate stages |
| Autonomous mitosis/routing can decide what the model should learn | **NOT ESTABLISHED** | Controlled controllers work; trained-model address/growth mechanisms fail registered continual-language goals | Keep autonomous local decision-making off the critical engineering path |
| Parameter consolidation of Cells is globally beneficial | **NOT ESTABLISHED** | No frozen protocol yet demonstrates capability-preserving, plasticity-preserving consolidation with a global acceptance criterion | Consolidation must remain optional and transactional until separately validated |
| Multi-model/MoE merge is improved by CLM factorization | **NOT TESTED** | No comparative merge protocol against strong model/adapter/MoE baselines | Treat as an engineering hypothesis with an early kill test, not an established CLM advantage |
| JAM can determine semantic model quality | **OUT OF SCOPE / FALSE PREMISE** | JAM can provide execution/provenance/state-transition guarantees, not a semantic quality oracle | Use JAM for reproducibility, coordination, provenance and accepted-state transitions; keep quality in the evaluator |

## Strongest supported scientific results

### 1. Core 005: exact controlled safety mechanism

`SUBSPACE_CERTIFIED_MITOSIS_SUPPORTED` establishes that bounded Cell-local certificate state can replace raw replay for exact registered-history protection, saturation detection and growth control in its frozen synthetic linear setting.

**Ceiling:** fixed features, linear writable Cells, explicit routing, known certificate construction. It does not establish language-scale activation certificates or natural routing boundaries.

### 2. Core 006: real representation bridge with a negative system decision

`REAL_REPRESENTATION_CONTINUAL_PLASTICITY_NOT_SUPPORTED` is scientifically important because several subclaims survive the failed overall decision:

- real Pythia representations did not immediately fill the projected Cell space;
- functional reuse increased;
- certificate writes reduced registered forgetting;
- new-learning gain remained substantial relative to replay;
- address-based mitosis failed bounded-growth/conflict-relief gates.

**Ceiling:** useful low-dimensional functional structure exists, but semantic/routing address is not a sufficient functional split unit.

### 3. Constructive CLM 001–005: mechanism stack exists under controlled assumptions

The constructive sequence supports learned coordinate formation, latent discovery under the registered superposition scaffold, finite-horizon structure-tracking growth, protected learned/growing Cells, model-level multi-Cell computation, and a learned routing/write/growth control plane.

**Ceiling:** these are controlled mechanism existence results. The safety certificate remains fixed in CLM-005, operators are not arbitrary Transformer Cells, and the sequence does not establish language-scale continual learning.

### 4. Native M1: trainability is real

The small Native CLM can train from next-token loss and use sparse Cell execution. Therefore the research blocker is not simply that the architecture cannot be optimized at all.

**Ceiling:** successful next-token training is not continual-learning success.

### 5. Native M2/M3/M3R: the trained-model ceiling

The current trained-model sequence progressively isolates the failure boundary:

```text
M2: protection reduces forgetting, but fixed topology fails absolute retention
M3: fresh capacity + global-pool growth does not restore safe retention
M3R: root read ownership can be conserved, but lineage-local functional addressing remains insufficient
```

This sequence is a stronger guide for engineering than the controlled synthetic positives because it establishes where the current mechanisms stop transferring.

### 6. M2-R0/R0b: parameter transactions must be audited after optimization

The numerical reference audit classifies the R0 reference failure as parameter-transaction roundoff and separately identifies AdamW preconditioning as a material update-invariant breaker. Final realized-update projection reaches the numerical floor.

**Ceiling:** this closes an optimizer-mechanics issue only. It does not change the historical M2 decision and does not establish certificate coverage or continual-learning success.

## Current No-Go claims

Until new registered evidence exists, do not state that MiniCells has established any of the following:

- autonomous lifelong/replay-free continual language learning;
- a natural or universal Cell ontology inside pretrained LLMs;
- semantic address as a safe write/growth boundary;
- growth by itself as a forgetting solution;
- local evaluation as a substitute for global model evaluation;
- Cell-level safety as proof that the complete model improved;
- asymptotically bounded Cell growth from finite-horizon constructive experiments;
- safe parameter consolidation of external Cells;
- superiority of CLM for MoE/model merging;
- JAM as a semantic model-quality verifier.

## Engineering primitives safe to carry forward

These mechanisms may be reused in engineering without claiming the stronger research hypothesis:

1. **Explicit modular change boundaries.** A Cell/module can be a deliberately engineered unit of change without claiming it is a natural knowledge atom.
2. **Candidate isolation.** Train a branch or shadow module without immediately mutating the accepted model.
3. **Functional regression contracts.** Use registered probes/certificates to detect local damage.
4. **Realized-update validation.** Judge the actual optimizer transaction before commit.
5. **Append/expand as a default escape from exhausted writable freedom.** Expansion still requires read/routing validation.
6. **Versioning and rollback.** Preserve parent identity, change provenance and reversible candidate state.
7. **Stage-level global evaluation.** Local mechanisms propose; a broader evaluator accepts or rejects the model version.
8. **Optional consolidation.** Only after a separate protocol establishes that consolidation preserves capabilities, locality and future plasticity while providing a measurable resource benefit.

## Audit decision

The current evidence justifies an engineering program around **safe model evolution**. It does not justify making Native-CLM autonomy, natural Cell boundaries or zero-replay continual learning dependencies of that program.

Future theoretical work may raise this ceiling. Until then, engineering claims should remain at or below it.
