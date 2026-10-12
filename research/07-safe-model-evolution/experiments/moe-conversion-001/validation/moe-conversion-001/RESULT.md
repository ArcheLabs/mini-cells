# CLM MoE Conversion 001 — Formal Result

Decision: **PASS — MOE_SUBSTRATE_CONVERSION_SUPPORTED**

Protocol: `research/validations/moe-conversion-001/PROTOCOL.md`

## Formal Stage B target

- Model: `ibm-granite/granite-3.1-1b-a400m-base`
- Resolved Hugging Face revision: `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`
- GitHub Actions run: `33817605547`
- Run URL: <https://github.com/ArcheLabs/mini-cells/actions/runs/33817605547>
- Artifact ID: `9917127794`
- Artifact ZIP SHA-256: `1bce5d9432daa0ffa16d64b6c0542e267b81165786187f0c0ed8982bf4789aae`

## Environment

- Ubuntu 24.04 hosted runner
- Python 3.11.16
- Torch 2.14.0+cpu
- Transformers 5.16.1
- Hugging Face Hub 1.30.0
- safetensors 0.8.0
- Device: CPU, 2 Torch threads
- Execution dtype: BF16
- Parity tolerance: `1e-5`

## Result

```json
{
  "experiment": "CLM_MOE_CONVERSION_001",
  "stage": "real",
  "status": "PASS",
  "source_model_id": "ibm-granite/granite-3.1-1b-a400m-base",
  "source_revision": "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29",
  "manifest_identity_sha256": "dd2b9c750567ff73b1d48e39eb7d1e1213eea9116a68c5164d023420f5a4d670",
  "tensor_count": 218,
  "expert_address_count": 1536,
  "router_outputs_compared": 72,
  "max_abs_logit_error": 0.0,
  "max_abs_router_error": 0.0,
  "bundle_integrity": true,
  "checkpoint_byte_identity": true,
  "logit_argmax_identity": true,
  "router_topk_identity": true,
  "greedy_token_identity": true,
  "expert_is_not_cell": true
}
```

The 1,536 logical expert-slice addresses are exactly consistent with 24 MoE layers × 2 packed expert tensors per layer × 32 experts.

## Stage A prerequisite

The deterministic tiny Granite MoE prerequisite also passed in Actions run `33817333205`:

- unit tests: 2/2 PASS
- checkpoint byte identity: PASS
- max absolute logit error: `0.0`
- max absolute router-logit error: `0.0`
- router outputs compared: 4
- expert addresses: 16

The first Stage A attempt exposed a Transformers 5.16 router-observability API mismatch. The implementation was repaired to observe the actual per-layer Granite router module outputs directly. No frozen scientific threshold or PASS gate was relaxed after seeing the failure.

## Supported conclusion

This result supports the following narrow claim:

> An existing Granite MoE checkpoint can be lifted into a CLM canonical substrate with deterministic tensor and packed-expert mutation addresses, then materialized back into an ordinary Hugging Face MoE checkpoint without changing checkpoint bytes, logits, router logits, router top-k decisions, or greedy token decisions under the frozen protocol.

The experiment therefore finds **no execution-semantics barrier** to placing the current CLM evolution layer above an existing Granite MoE.

## Not established

This result does **not** establish:

- that a MoE expert is a CLM Cell;
- safe or useful CLM mutations;
- branch gain retention;
- mutation composition or mergeability;
- continual learning;
- autonomous Cell discovery or routing.

The next scientific step is `MoE Mutation 001`: create independent sparse mutations over this immutable substrate and test whether they retain branch gains and compose without unacceptable base-model regression.
