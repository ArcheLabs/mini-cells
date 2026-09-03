# CLM MoE Conversion 001 — Frozen Protocol

Status: **FROZEN BEFORE FORMAL REAL-MODEL EXECUTION**

## 1. Question

Can a mature Hugging Face Granite MoE checkpoint be lifted into a CLM canonical substrate and materialized back into an ordinary Hugging Face checkpoint **without changing its MoE execution semantics**?

This protocol does **not** test continual learning, autonomous Cell discovery, or branch composition. It tests only whether CLM can sit above an existing MoE as an evolution layer without corrupting the underlying model.

## 2. Architectural boundary

The experiment freezes the following interpretation:

```text
CLM evolution plane
  canonical substrate
  tensor / expert-slice addresses
  future mutation lineage
  future fork / merge / rollback
            |
            v
MoE execution plane
  Transformer layers
  per-token MoE router
  packed experts
  attention / norms / embeddings
```

A MoE expert is **not** declared to be a CLM Cell.

Conversion 001 uses `kind=substrate_wrap`. The original checkpoint bytes remain canonical. CLM adds a manifest and logical addresses; it does not rewrite weights, retrain the router, split experts, or change forward execution.

## 3. Target

Formal real-model target:

- model: `ibm-granite/granite-3.1-1b-a400m-base`
- expected architecture: `GraniteMoeForCausalLM`
- expected `model_type`: `granitemoe`
- expected experts/layer: 32
- expected top-k: 8

The runner must resolve and record the concrete Hugging Face revision SHA. A run that records only the symbolic `main` revision is not a formal Stage B result.

## 4. Stage A — deterministic tiny Granite MoE

Stage A is an engineering prerequisite, not scientific evidence about model quality.

The runner creates a deterministic two-layer `GraniteMoeForCausalLM` with four experts and top-2 routing, saves it with safetensors, converts it into a CLM bundle, materializes it, and compares source versus materialized execution.

Command:

```bash
pip install -e '.[lm,dev]'
python scripts/research/moe_conversion_001/run.py \
  --stage tiny \
  --device cpu \
  --dtype float32 \
  --work-dir artifacts/moe-conversion-001-tiny
```

Stage A passes only if every gate in Section 7 passes.

## 5. Stage B — Granite 3.1 1B-A400M

Recommended Kaggle / CUDA command:

```bash
pip install -e '.[lm,dev]'
python scripts/research/moe_conversion_001/run.py \
  --stage real \
  --model-id ibm-granite/granite-3.1-1b-a400m-base \
  --device cuda \
  --dtype float16 \
  --tolerance 1e-5 \
  --work-dir artifacts/moe-conversion-001-real
```

`float16` is the recommended T4 execution dtype. The checkpoint itself remains byte-identical to the downloaded source; the dtype flag controls only the parity forward pass.

The formal artifact is `result.json` plus the generated `clm-bundle/clm_moe_manifest.json`. The resolved source revision in those artifacts is mandatory.

## 6. Conversion representation

The manifest schema is `clm.moe-substrate.v1`.

It must record:

- source model ID and resolved revision;
- every checkpoint file's byte length and SHA-256;
- safetensors tensor name, dtype, shape, byte length and SHA-256;
- tensor role (`shared_backbone`, `moe_router`, `moe_packed_experts`, or `moe_other`);
- logical expert-slice addresses for packed Granite MoE expert tensors;
- an identity SHA-256 over the manifest itself.

For Granite MoE packed expert tensors, a logical address has the form:

```text
model.layers.<L>.block_sparse_moe.input_linear.weight::expert[<E>]
model.layers.<L>.block_sparse_moe.output_linear.weight::expert[<E>]
```

These addresses are future mutation targets. They do not imply that one expert equals one Cell.

## 7. Frozen PASS gates

A run is `PASS` only if all of the following hold:

1. **Bundle integrity** — every canonical substrate file matches its recorded size and SHA-256.
2. **Checkpoint byte identity** — the materialized Hugging Face checkpoint has the same file identities as the source checkpoint.
3. **Forward logits parity** — maximum absolute source/materialized logit error is at most the configured tolerance.
4. **Logit decision identity** — source and materialized argmax token decisions are identical.
5. **Router value parity** — maximum absolute router-logit error is at most the configured tolerance.
6. **Router top-k identity** — selected expert IDs are identical at every returned MoE layer for every parity prompt.
7. **Greedy token identity** — for Stage B, the fixed prompts produce identical greedy continuation token IDs.
8. **Semantic boundary** — the manifest explicitly records `expert_is_cell=false`.
9. **Addressability** — at least one packed expert logical address is discovered.

No gate may be waived after seeing Stage B results. A changed threshold requires a new protocol version.

## 8. Interpretation

### PASS

A PASS supports only:

> An unchanged Granite MoE can be represented as a CLM canonical substrate with deterministic logical mutation addresses, then materialized back to a standard Hugging Face MoE without detectable execution drift under this protocol.

A PASS does **not** establish that mutations are safe, that Cells are naturally discoverable, that branches compose, or that continual learning is solved.

### FAIL

A FAIL blocks MoE-native CLM mutation experiments until the conversion/materialization mismatch is explained. The failure must not be interpreted as evidence against CLM composability itself unless the failure is specifically caused by an unavoidable representational conflict.

## 9. Next experiment after PASS

Only after Stage B passes should `MoE Mutation 001` be frozen. That experiment should create two independent sparse CLM mutations over the same immutable MoE substrate and test retention/composition against the current CLM composability gates.
