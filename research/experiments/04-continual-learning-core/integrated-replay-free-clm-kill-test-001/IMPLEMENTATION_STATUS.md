# KT001 Implementation Status

Current state: **UNSEALED — AWAITING DEVELOPMENT SMOKE**

Formal execution is not authorized.

## Implemented

- frozen causal protocol and seed registry;
- frozen formal decision rule before development scientific results;
- five explicit causal arms;
- canonical R0b realized-AdamW transaction binding;
- canonical M3L-2 rank-32 historical address state;
- canonical M3R lineage-local read isolation;
- protocol-forced phase-boundary Shadow expansion;
- B→C→D phase engine with per-step update audit;
- isolated 50/50 matched raw-replay oracle;
- pre/post-Shadow checkpoints and phase evidence;
- per-seed and formal aggregation logic;
- durable Hugging Face/Git publisher, including partial failure evidence;
- pinned deterministic data builder with explicit historical-overlap accounting;
- Kaggle development notebook;
- CPU mechanics tests and CI formal-seed guards.

## Not yet satisfied

- latest CI preflight must be green;
- development seed from `SEEDS.json` must complete all five arms;
- development mechanics/oracle must be valid enough to classify the seed as `PASS` or `VALID_FAIL` rather than `INCONCLUSIVE_MECHANICS` / `INCONCLUSIVE_ORACLE`;
- development evidence must be durably published;
- implementation/data/protocol hashes must then be sealed into `IMPLEMENTATION_LOCK.json`.

## Formal lock rule

`scripts/research/run_integrated_replay_free_clm_kt001.py` refuses a registered formal seed unless `IMPLEMENTATION_LOCK.json` exists with:

```text
status = SEALED_FOR_FORMAL_EXECUTION
```

The formal seed registry is never embedded in CI, runner defaults, or the Kaggle development notebook. Formal seeds remain untouched while this status is UNSEALED.
