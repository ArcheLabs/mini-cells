# MiniCells Research Audits

This directory is the stable, cross-experiment audit layer for MiniCells research.

The purpose is deliberately different from a roadmap:

- validation directories preserve experiment-specific protocols and results;
- stage documents describe the active research sequence;
- this directory records what the accumulated evidence allows us to claim, what it does not allow us to claim, and which engineering primitives survive negative scientific results.

## Start here

- [`RESEARCH_LEDGER.md`](RESEARCH_LEDGER.md) — family-level ledger of summary-worthy experiments, including value, limitation, formal status, known ceiling, and engineering primitive obtained.
- [`CLM_CAPABILITY_CEILING.md`](CLM_CAPABILITY_CEILING.md) — conservative capability ceiling as of 2026-09-03.
- [`HISTORICAL_RESEARCH_ASSET_MAP.md`](HISTORICAL_RESEARCH_ASSET_MAP.md) / [`中文`](HISTORICAL_RESEARCH_ASSET_MAP.zh-CN.md) — separates the scientific authority of the historical notebook lineages from the engineering primitives that remain worth reusing, with explicit treatment of Foundations, Self-Organization, Routing/Growth, and CLM-0.4-mini Language Validation.

## Audit rules

1. **Registered results outrank narratives.** A frozen formal decision is never upgraded by later interpretation.
2. **Controlled support stays controlled.** Synthetic/linear support is not described as Transformer-, language-, or scale-level support.
3. **Diagnostics are not milestones.** Checkpoint-only or no-new-seed diagnostics may explain a failure but cannot retroactively change the failed decision.
4. **Negative results are retained.** A failed mechanism remains useful evidence about a boundary and should not disappear from the current narrative.
5. **Every positive claim states what it does not prove.** This is mandatory for future ledger additions.
6. **Engineering utility is separate from scientific support.** A mechanism may be valuable for versioning, rollback, bounded mutation, evaluation, or experimentation even when the stronger continual-learning hypothesis fails.
7. **No local-to-global promotion.** A local safety or improvement signal is not evidence that the globally evaluated model is better.
8. **Historical paths are evidence.** Do not move or rewrite published protocol/result paths merely to make the tree prettier; migrations need reference audits and compatibility shims.
9. **Historical scientific status and engineering reuse are separate axes.** A retired, superseded or negative scientific lineage may still contain a useful primitive, but reuse does not inherit the old scientific claim.

## Required ledger fields

Every summary-worthy family should eventually have:

```text
Claim
Protocol / evidence source
Positive evidence
Negative evidence
Formal status
Known ceiling
What it does NOT prove
Engineering primitive obtained
```

This audit is intentionally conservative. Product positioning should be derived after the capability ceiling is reviewed, not written back into historical scientific decisions.
