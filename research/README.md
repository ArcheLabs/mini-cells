[English] | [中文](README.zh-CN.md)

# MiniCells Research

This tree preserves MiniCells scientific evidence. It is **not** a product roadmap and it does not assume that the strongest Native-CLM hypothesis is true.

## Start here

1. [`audits/CLM_CAPABILITY_CEILING.md`](audits/CLM_CAPABILITY_CEILING.md) — strongest claims currently supported, explicit No-Go claims, and engineering primitives that survive the negative results.
2. [`audits/RESEARCH_LEDGER.md`](audits/RESEARCH_LEDGER.md) — family-level record of summary-worthy experiments with value, limitation, formal status, known ceiling, what each result does **not** prove, and the engineering primitive obtained.
3. [`audits/HISTORICAL_RESEARCH_ASSET_MAP.md`](audits/HISTORICAL_RESEARCH_ASSET_MAP.md) — current scientific authority and surviving engineering value of the historical Foundations, Self-Organization, Routing/Growth and CLM-0.4-mini notebook lineages.
4. [`stages/06-native-clm/`](stages/06-native-clm/) — current trained-model Native-CLM sequence and closures.
5. [`validations/`](validations/) — frozen protocols, formal results and mechanism diagnostics.
6. [`catalog.yaml`](catalog.yaml) — machine-oriented research catalog.

## Repository structure

```text
research/
  README.md                 # navigation only
  audits/                   # cross-experiment claims, historical assets and capability ceiling
  catalog.yaml              # machine-oriented index
  stages/                   # research-stage narratives and closures
  validations/              # frozen protocols/results/diagnostics
  experiments/              # historical experiment organization
  notebooks/                # runnable notebooks / hosted execution entrypoints
  reports/                  # derived reports
  releases/                 # research release records
  previews/                 # preview material
  archive/                  # retired/historical material
```

The directory name does not determine scientific strength. A frozen registered result outranks a roadmap, README, notebook or later interpretation.

## Current evidence boundary

The controlled research stack establishes useful mechanisms: protected local writes, capacity growth, learned sparse coordinates, multi-Cell composition and learned control-plane behavior can all be constructed under registered boundaries. A small Native CLM can also train end-to-end from next-token loss.

The trained-model continual-learning sequence establishes the current ceiling:

```text
M2   fixed-topology replay-free continual language      NOT SUPPORTED
M3   global-pool growth-restored continual language     NOT SUPPORTED
M3R  read-preserving lineage growth                     NOT SUPPORTED
```

Protection has partial causal value, new capacity is usable, and preserving read ownership improves a specific M3 failure mode. None of those facts establishes autonomous replay-free continual language learning.

The optimizer/update audit adds a separate mechanics result: safe-gradient projection is not sufficient for canonical AdamW parameter transactions, while projecting/validating the realized update can restore the registered invariant to the numerical floor. This does not change the M2 scientific decision.

For exact boundaries and source paths, use the audit documents rather than extending this README with another progress table.

## Historical notebooks

The historical notebook tree is retained as a research asset, not as a flat set of equally strong scientific claims.

Current audit classification:

```text
01-foundations          HISTORICAL EXPLORATORY
02-self-organization    HISTORICAL MECHANISTIC EVIDENCE
03-routing-and-growth   ENGINEERING PRECURSOR EVIDENCE
05-language-validation  RETIRED / SUPERSEDED PROTOCOL LINEAGE
```

A historical scientific interpretation may be weakened or superseded while its engineering primitive remains useful. The canonical mapping of those two dimensions is [`audits/HISTORICAL_RESEARCH_ASSET_MAP.md`](audits/HISTORICAL_RESEARCH_ASSET_MAP.md).

Shadow Cell Validation 001 v2 is registered as a separate, unrun architectural
test of copy-on-write candidate development and controlled maturation:
[protocol and implementation](validations/shadow-cell-validation-001-v2-developmental-maturation/README.md).

## Research vs engineering

MiniCells keeps two tracks separate:

- **Long-term research:** natural functional boundaries, future-learning sufficient state, autonomous routing/growth, replay-free continual learning, parameter-level sustained plasticity.
- **Near-term engineering:** explicit modular changes, fork/shadow training, functional regression checks, realized-update validation, append/expand, stage-level global evaluation, versioning and rollback, with consolidation only after a separate acceptance protocol.

Engineering may use a Cell as a deliberately chosen unit of model change without claiming it is a natural knowledge atom.

## Evidence discipline

- Do not reuse observed formal seeds as untouched confirmation seeds.
- Do not upgrade a failed registered decision because a later diagnostic explains the failure.
- Do not promote synthetic/linear support into Transformer-, language-, asymptotic- or product-level support.
- Do not infer global model improvement from a local write/retention signal.
- Do not inherit scientific authority when reusing an old engineering primitive.
- Preserve historical protocol/result paths; restructure entrypoints only with reference audits and compatibility shims.
- Every new summary claim must state what it does **not** prove.

The purpose of the research tree is to make future product decisions depend on the accumulated evidence ceiling rather than on the newest mechanism narrative.
