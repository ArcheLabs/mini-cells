# PCU-HYBRID-REATTACHMENT-001 — Protocol v3

## 0. Mission

Determine whether a frozen mature Granite MoE can causally consume an already-learned PCU mutation, while separating three questions that protocol v2 accidentally conflated:

1. native Granite ↔ cellularized Granite numerical drift;
2. matched-graph zero-state equivalence and reversibility;
3. useful causal expression versus locality/interference.

This protocol does **not** require autonomous Cell language generation, a new bridge, a new readout, or a newly trained router.

## 1. Historical source and protocol amendment

The source mutation remains exactly:

- experiment: `PCU-OBJECTIVE-ALIGNMENT-001`;
- engineering seed: `26090501`;
- target: Granite layer 7;
- selected Cells: exact published K64 set;
- objective: 16-way candidate-ranking only;
- published A_eval ranking: `0.8203125`;
- published greedy direct accuracy: `0.0`;
- no CE readout regularizer.

The first PCU-HYBRID-REATTACHMENT-001 engineering execution used source commit:

`cadaf6c397000c55deb35db67a3b266003cb3004`

The operator log reported:

- ranking OFF: `0.0625`;
- ranking ON: `0.8203125`;
- ranking gain: `0.7578125`;
- answer-margin gain: `3.1839119061085976`;
- answer-NLL gain: `3.4671419583834133`;
- B-control answer-NLL increase: `0.3189873118037063`;
- base-vs-off A max logit diff: `3.5762786865234375e-05`;
- on-vs-restored A max logit diff: `0.0`;
- v2 status: `ZERO_STATE_EQUIVALENCE_FAILED`.

Protocol v2 used untouched native/fused Granite as one side of the strict zero-state equivalence gate. Cellularization decomposes expert computation into Cell contributions and therefore changes floating-point reduction order even when the mathematical function is unchanged. The observed native↔cellular drift exceeded the predeclared `1e-5` gate.

Protocol v3 is an explicit **post-observation amendment**. It does not retroactively change the v2 result and does not relax any threshold.

The amendment is only:

- native Granite ↔ PARENT_ZERO_DELTA remains recorded as a non-gating G0 numerical diagnostic;
- the strict `1e-5` zero-state gate is applied to the matched cellular path:
  `PARENT_ZERO_DELTA ↔ CELL_OFF`;
- all other thresholds are unchanged.

## 2. Frozen thresholds

The following values are frozen before protocol-v3 GPU execution:

| Gate | Threshold |
| --- | ---: |
| matched cellular zero-state max abs logit diff | `<= 1e-5` |
| ON ↔ RESTORED max abs logit diff | `<= 1e-5` |
| A_eval association ranking | `>= 0.80` |
| ON−OFF causal ranking gain | `>= 0.50` |
| ON−OFF answer-margin gain | `> 0.0` |
| B-control answer-NLL increase | `<= 0.10` |

No threshold may be changed after protocol-v3 execution to obtain a PASS.

## 3. Dual-GPU execution

Protocol v3 reuses the repository's isolated dual-worker execution pattern.

### GPU 0 — primary causal reattachment

Reconstruct the exact published ranking-only K64 mutation and evaluate the following states in one memory-resident frozen Granite instance:

1. `BASE`: untouched native Granite;
2. `PARENT_ZERO_DELTA`: G0 cellularized Granite before mutation replay;
3. `CELL_ON`: exact reconstructed PCU mutation;
4. `CELL_OFF`: same trained runtime with only `delta_*` tensors temporarily zeroed;
5. `CELL_RESTORED`: byte-exact restoration of the same mutation.

Primary causal comparison:

`CELL_ON ↔ CELL_OFF`

Strict zero-state comparison:

`PARENT_ZERO_DELTA ↔ CELL_OFF`

Reversibility comparison:

`CELL_ON ↔ CELL_RESTORED`

Non-gating diagnostic:

`BASE ↔ PARENT_ZERO_DELTA`

### GPU 1 — mutation amplitude sweep

Independently reconstruct the exact same published ranking-only mutation. No new optimizer step is allowed after reconstruction.

Evaluate:

`delta(alpha) = alpha * delta_trained`

for the frozen grid:

`alpha = [0, 0.125, 0.25, 0.5, 0.75, 1.0]`

For every alpha record:

- A_eval 16-way ranking accuracy;
- A answer NLL;
- A target-logit margin;
- B-control answer NLL;
- A ranking gain relative to alpha=0;
- A answer-NLL gain relative to alpha=0;
- B-control answer-NLL increase relative to alpha=0;
- association gate;
- locality gate;
- joint gate.

A locality-compatible point requires simultaneously:

- `A ranking >= 0.80`;
- `B-control NLL increase <= 0.10`;
- `alpha > 0`.

If multiple points satisfy both gates, the reporting selector is frozen as:

1. highest A ranking;
2. then lowest B harm;
3. then lowest alpha.

This selector is descriptive only; every alpha row remains published.

## 4. Scientific interpretation states

Primary arm may emit:

- `ENGINEERING_SIGNAL_HYBRID_REATTACHMENT_SUPPORTED`;
- `CAUSAL_HYBRID_CONSUMPTION_SUPPORTED_LOCALITY_FAILED`;
- `SAME_GRAPH_ZERO_STATE_EQUIVALENCE_FAILED`;
- `REVERSIBILITY_FAILED`;
- `CAUSAL_EXPRESSION_PRESENT_GATES_UNRESOLVED`;
- `NO_CAUSAL_EXPRESSION_ENGINEERING`;
- replay mismatch states.

Amplitude sweep may emit:

- `AMPLITUDE_SWEEP_FINDS_LOCALITY_COMPATIBLE_POINT`;
- `AMPLITUDE_SWEEP_NO_LOCALITY_COMPATIBLE_POINT`;
- replay/reversibility failure states.

Combined engineering decision may emit:

- `HYBRID_REATTACHMENT_SUPPORTED_AT_ALPHA_1`;
- `HYBRID_REATTACHMENT_SUPPORTED_WITH_BOUNDED_AMPLITUDE`;
- `HYBRID_CAUSAL_CONSUMPTION_SUPPORTED_LOCALITY_UNRESOLVED`;
- or a primary protocol-failure state.

None of these strings is a formal-seed scientific PASS.

## 5. Formal boundary

Formal PCU seeds remain:

- `26090511`: `RESERVED_UNTOUCHED`;
- `26090512`: `RESERVED_UNTOUCHED`;
- `26090513`: `RESERVED_UNTOUCHED`.

Protocol-v3 engineering artifacts must contain:

- `scientific_evidence: false`;
- `formal_execution_not_started: true`;
- `formal_decision: RESERVED_UNRUN` where a decision field exists.

The engineering publisher must reject any modified formal seed registry.

## 6. Provenance and publication

Both GPU workers must start from the same clean source commit/tree and write to external worker directories under `/kaggle/working`.

Only after both workers finish may the orchestrator write the repository artifact directory.

The publisher must:

1. validate all JSON schemas and frozen protocol fields;
2. validate worker source commit/tree identity;
3. validate exact Cell and dataset identity between workers;
4. validate the frozen alpha grid;
5. validate that no extra amplitude-sweep training occurred;
6. validate formal seeds are untouched;
7. stage only the current experiment artifact subtree;
8. commit, rebase, and push evidence to `codex/pcu-hybrid-reattachment-001`;
9. refuse remote overwrite of an existing v3 decision artifact.

## 7. Required visualizations

The aggregated artifact must include:

- `equivalence_diffs.png`: native G0 diagnostic versus matched-graph and restoration diffs;
- `causal_ranking_on_off.png`: A_eval ranking OFF versus ON;
- `alpha_sweep_tradeoff.png`: alpha against A ranking and B-control NLL increase;
- `association_locality_pareto.png`: association/locality Pareto view with frozen thresholds.

Raw JSON and CSV remain canonical; plots are explanatory views only.

## 8. Required artifacts

Output root:

`artifacts/research/pcu-hybrid-reattachment-001/engineering/26090501-l7-k64-ranking-causal-reattach-v3/`

Required files:

- `DESIGN.json`
- `RUN_IDENTITY.json`
- `RESULT.json`
- `DECISION.json`
- `PRIMARY_RESULT.json`
- `PRIMARY_DECISION.json`
- `SWEEP_RESULT.json`
- `SWEEP_DECISION.json`
- `AMPLITUDE_SWEEP.csv`
- `REPORT.md`
- four PNG visualizations listed above.

## 9. Canonical execution

On Kaggle T4 x2:

```bash
python scripts/research/run_pcu_hybrid_reattachment_001.py
python scripts/research/publish_pcu_hybrid_reattachment_001.py \
  --branch codex/pcu-hybrid-reattachment-001
```

The provided Kaggle notebook performs the same sequence and pushes validated results back to the branch.
