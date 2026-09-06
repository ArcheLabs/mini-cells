# PCU-HYBRID-REATTACHMENT-001

## Question

**Can a frozen mature Granite model causally consume an already-learned PCU mutation?**

This experiment is intentionally weaker than autonomous parameter takeover and stronger than a local association score. It does not ask a Cell to decode language by itself. It asks whether a mutation that already learned an association changes the final output of the mature frozen model when present at its native MoE location.

## Primary source mutation

The experiment replays the exact published `PCU-OBJECTIVE-ALIGNMENT-001` ranking-only state, not the later Hybrid Objective:

- foundation: `ibm-granite/granite-3.1-1b-a400m-base`
- revision: `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`
- engineering world seed: `26090501`
- layer: L7
- exact selected set: K64 from the published objective-alignment artifact
- optimizer: AdamW
- LR: `1e-3`
- optimizer steps: 128
- effective batch: 8
- training objective: 16-way candidate-ranking only
- published A_eval ranking: `0.8203125`
- published greedy direct accuracy: `0.0`

This is deliberately the historical state that says **association learned, generation unresolved**. No answer-token CE/readout regularizer is added.

The historical artifact contains metrics/provenance but not restorable Cell-delta tensors, so the mutation is reconstructed by deterministic replay of the exact pinned protocol. If the replay does not reproduce the registered ranking result, the run is invalid for causal interpretation.

## Causal states

The same model is evaluated in these states:

1. `BASE`: untouched mature Granite.
2. `PARENT_ZERO_DELTA`: G0 cellularized Granite before any learned mutation.
3. `CELL_ON`: ranking-trained PCU deltas active in the native Granite MoE path.
4. `CELL_OFF`: the same trained model with only Cell delta tensors temporarily zeroed.
5. `CELL_RESTORED`: the exact trained deltas copied back after the OFF intervention.

No new bridge, router, readout head, allocation, CE regularizer, or trainable foundation parameter is permitted.

The decisive comparison is:

\[
M_{\mathrm{Granite+Cell}}(x)
\quad\text{vs}\quad
M_{\mathrm{Granite+zero(Cell)}}(x)
\]

with every other variable held fixed.

## Predeclared engineering gates

`ENGINEERING_SIGNAL_HYBRID_REATTACHMENT_SUPPORTED` requires all of:

- exact replay of the published ranking-only mutation;
- `BASE` vs `PARENT_ZERO_DELTA` max absolute logit difference <= `1e-5`;
- `BASE` vs `CELL_OFF` max absolute logit difference <= `1e-5`;
- `CELL_ON` vs `CELL_RESTORED` max absolute logit difference <= `1e-5`;
- `CELL_ON` A_eval candidate-ranking accuracy >= `0.80`;
- causal ranking gain `ranking(ON) - ranking(OFF) >= 0.50`;
- answer-token target margin improves under ON vs OFF;
- B_eval control answer NLL increase under ON vs OFF <= `0.10` nats/token.

A_eval answer-token NLL, token top-1 accuracy and target-logit margin are read directly from final Granite logits. B_eval is an untrained counter-domain locality control. Greedy generation is reported only as a secondary diagnostic and is **not** a success gate.

## Why this is the decisive missing A/B

The objective-alignment experiment already showed that the K64 PCU state can encode the new A association under a constrained ranking objective. What it did not isolate was the counterfactual:

> with this exact learned delta present versus exactly zeroed, does the frozen mature model's final readout change in the intended direction?

That is the sole causal variable here. Autonomous Cell takeover is outside this protocol.

## Interpretation

A positive engineering result means an association-bearing PCU mutation has a large, reversible and target-specific causal effect on final mature-model readout while Granite remains frozen. It would reject the requirement that PCU must first become an autonomous generator before it can serve as a Hybrid CLM mutation layer.

A positive engineering result is **not** by itself the formal scientific decision. Formal PCU seeds remain untouched.

Possible engineering statuses:

- `ENGINEERING_SIGNAL_HYBRID_REATTACHMENT_SUPPORTED`
- `CAUSAL_EXPRESSION_PRESENT_GATES_UNRESOLVED`
- `NO_CAUSAL_EXPRESSION_ENGINEERING`
- `ZERO_STATE_EQUIVALENCE_FAILED`
- `OFF_STATE_EQUIVALENCE_FAILED`
- `REVERSIBILITY_FAILED`
- `REPLAY_DID_NOT_MATCH_PUBLISHED_MUTATION`

Formal decision remains `RESERVED_UNRUN` in all engineering artifacts.

## Run

```bash
python scripts/research/run_pcu_hybrid_reattachment_001.py --device cuda:0
```

Faster causal pass without the non-gating greedy-generation diagnostic:

```bash
python scripts/research/run_pcu_hybrid_reattachment_001.py \
  --device cuda:0 \
  --skip-direct-generation
```

Artifacts:

`artifacts/research/pcu-hybrid-reattachment-001/engineering/26090501-l7-k64-ranking-causal-reattach/`

Expected files:

- `RUN_IDENTITY.json`
- `DESIGN.json`
- `RESULT.json`
- `DECISION.json`

## Formal boundary

Only after the engineering causal path is valid should the existing reserved PCU formal seeds be consumed. A formal protocol must repeat the same ON/OFF/RESTORED intervention across predeclared seeds and aggregate causal effects; it must not substitute autonomous takeover as the primary endpoint.
