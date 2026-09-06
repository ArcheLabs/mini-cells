# PCU-HYBRID-REATTACHMENT-001

## Question

**Can a frozen mature Granite model causally consume an already-learned PCU mutation?**

This experiment is intentionally weaker than autonomous parameter takeover and stronger than local objective alignment. It does not ask a Cell to decode language by itself. It asks whether a mutation already learned inside a Cell changes the final output of the mature frozen model when reattached at its native MoE location.

## Locked source state

The engineering diagnostic replays the published `PCU-HYBRID-OBJECTIVE-001` state:

- foundation: `ibm-granite/granite-3.1-1b-a400m-base`
- revision: `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`
- engineering world seed: `26090501`
- layer: L7
- exact selected set: K64 from the published objective-alignment artifact
- optimizer: AdamW
- LR: `1e-3`
- optimizer steps: 128
- effective batch: 8
- objective: ranking + `0.25 *` original answer-token CE
- published replay target: A_eval 16-way ranking `0.8359375`
- published free-generation diagnostic: `0.03125`

The historical artifact did not publish restorable Cell-delta tensors, so the mutation is reconstructed by deterministic replay of the exact pinned protocol. A run that does not reproduce the published ranking result is invalid for causal interpretation.

## Causal states

The same model is evaluated in these states:

1. `BASE`: untouched mature Granite.
2. `PARENT_ZERO_DELTA`: G0 cellularized Granite before any Cell mutation.
3. `CELL_ON`: the replayed PCU mutation active in the native Granite MoE path.
4. `CELL_OFF`: the same trained model with only Cell delta tensors temporarily zeroed.
5. `CELL_RESTORED`: the exact trained deltas copied back after the OFF intervention.

No new bridge, router, readout head, allocation, or trainable foundation parameter is permitted.

The decisive comparison is therefore

\[
M_{\mathrm{Granite+Cell}}(x)\;\;\text{vs}\;\;M_{\mathrm{Granite+zero(Cell)}}(x)
\]

with every other variable held fixed.

## Predeclared engineering gates

The engineering signal is called `ENGINEERING_SIGNAL_HYBRID_REATTACHMENT_SUPPORTED` only if all of the following hold:

- published mutation replay matches the registered result;
- `BASE` vs `PARENT_ZERO_DELTA` max absolute logit difference <= `1e-5`;
- `BASE` vs `CELL_OFF` max absolute logit difference <= `1e-5`;
- `CELL_ON` vs `CELL_RESTORED` max absolute logit difference <= `1e-5`;
- `CELL_ON` A_eval candidate-ranking accuracy >= `0.80`;
- causal ranking gain `ranking(ON) - ranking(OFF) >= 0.50`;
- answer-token target margin improves under ON vs OFF;
- B_eval control answer NLL increase under ON vs OFF <= `0.10` nats/token.

A_eval answer-token NLL, token top-1 accuracy and target-logit margin are reported directly from final Granite logits. B_eval is an untrained counter-domain locality control. Greedy generation is secondary and **is not a success gate**.

## Interpretation

A positive engineering result means that an already-learned PCU mutation has a large, reversible, target-specific causal effect on final mature-model readout while the foundation remains frozen. It would reject the claim that PCU must first achieve autonomous generation before it can serve as a Hybrid CLM mutation layer.

A positive engineering result is **not** by itself the formal scientific decision. Formal seeds remain untouched until the engineering causal path is validated.

Possible engineering statuses are:

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

For a faster causal pass that omits the non-gating greedy-generation diagnostic:

```bash
python scripts/research/run_pcu_hybrid_reattachment_001.py \
  --device cuda:0 \
  --skip-direct-generation
```

Artifacts are written to:

`artifacts/research/pcu-hybrid-reattachment-001/engineering/26090501-l7-k64-causal-reattach/`

Expected files:

- `RUN_IDENTITY.json`
- `DESIGN.json`
- `RESULT.json`
- `DECISION.json`

## Formal follow-up boundary

Only after the engineering causal path passes should the existing reserved PCU formal seeds be consumed. A formal protocol must repeat the ON/OFF/RESTORED intervention across the predeclared seeds and aggregate the causal effect; it must not reinterpret autonomous takeover as the primary endpoint.
