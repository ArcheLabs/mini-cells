# Native CLM v0 — JAM Q&A Log

The same deterministic prompts are shown before and after JAM post-training.
Free-generation output is illustrative; benchmark metrics are recorded separately.

## BEFORE JAM TRAINING

### evaluation.jam.services.refine.factual

**Q:** What protocol function or meaning does Refine entry point have in JAM?

**Reference:** Service code entry point 0 is Refine: it executes in-core and is essentially stateless, apart from access to defined historical/preimage context.

**Model:** you can have some cookies and share the beach. It is a big smile on the ground. It was a big smile on it. It was a big s

### evaluation.jam.services.accumulate.factual

**Q:** What protocol function or meaning does Accumulate entry point have in JAM?

**Reference:** Service code entry point 1 is Accumulate: it executes on-chain and may mutate protocol-visible service state.

**Model:** you can have some cookies and started to cry. The end.

### evaluation.jam.work.refinement.factual

**Q:** What protocol function or meaning does Refinement have in JAM?

**Reference:** Refinement is the in-core execution process that validates package inputs and runs each work item's service Refine code to produce work digests and exported segments.

**Model:** you can have some cookies and some cookies and some cookies. You can have a big smile on the ground. You are a good frie

### evaluation.jam.authorization.authorizer.factual

**Q:** What protocol function or meaning does Authorizer have in JAM?

**Reference:** An authorizer is bounded PVM logic that decides whether a work package, including its token, is permitted to execute on a particular core and yields a trace on success.

**Model:** you can have some cookies and share the cookies and the cookies. You can have a big smile on the ground. You can have a

### evaluation.reasoning.001

**Q:** A JAM workload must process a large external input and only afterward update persistent service state. Which phase should do each part, and why?

**Reference:** Use Refine for the high-throughput in-core processing and Accumulate for the state mutation. JAM deliberately separates largely stateless in-core refinement from stateful on-chain accumulation.

**Model:** you can have some coming from the park, and you can have a big star and a big smile. The little girl was so happy to hav

### evaluation.reasoning.002

**Q:** A service preimage exists at the current chain head but was not available at the work package lookup anchor. May Refine assume it exists?

**Reference:** No. Refine must use the historically anchored lookup context. Availability at the current head does not retroactively make the preimage available at the lookup anchor.

**Model:** you can have some coming from the children and share the children with the children and the children were so happy. The

## AFTER JAM TRAINING

### evaluation.jam.services.refine.factual

**Q:** What protocol function or meaning does Refine entry point have in JAM?

**Reference:** Service code entry point 0 is Refine: it executes in-core and is essentially stateless, apart from access to defined historical/preimage context.

**Model:** A work report is the header  execution process that validates package inputs and runs each work item's service Refine co

### evaluation.jam.services.accumulate.factual

**Q:** What protocol function or meaning does Accumulate entry point have in JAM?

**Reference:** Service code entry point 1 is Accumulate: it executes on-chain and may mutate protocol-visible service state.

**Model:** A guarantors are also the availability hash and length preimage become available on-chain accumulation.

### evaluation.jam.work.refinement.factual

**Q:** What protocol function or meaning does Refinement have in JAM?

**Reference:** Refinement is the in-core execution process that validates package inputs and runs each work item's service Refine code to produce work digests and exported segments.

**Model:** The header seal is a signed validator key authorizer as key rotation defined to let independent client implementations a

### evaluation.jam.authorization.authorizer.factual

**Q:** What protocol function or meaning does Authorizer have in JAM?

**Reference:** An authorizer is bounded PVM logic that decides whether a work package, including its token, is permitted to execute on a particular core and yields a trace on success.

**Model:** A guarantee is a signed validator attestation over a work report, submitted as block extrinsic data so the report can en

### evaluation.reasoning.001

**Q:** A JAM workload must process a large external input and only afterward update persistent service state. Which phase should do each part, and why?

**Reference:** Use Refine for the high-throughput in-core processing and Accumulate for the state mutation. JAM deliberately separates largely stateless in-core refinement from stateful on-chain accumulation.

**Model:** No. A null validator has a preimage because that service logic is necelly ets.

### evaluation.reasoning.002

**Q:** A service preimage exists at the current chain head but was not available at the work package lookup anchor. May Refine assume it exists?

**Reference:** No. Refine must use the historically anchored lookup context. Availability at the current head does not retroactively make the preimage available at the lookup anchor.

**Model:** No. A null validator key is an interestinal invalid validator keys to the partior sewly prior state and prior state.

