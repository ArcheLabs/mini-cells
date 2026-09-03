# Native CLM v0 — M3R Address Diagnostic

Status: **REGISTERED DIAGNOSTIC — NO NEW FORMAL SEEDS**

This stage follows the canonical M3R result:

```text
NATIVE_CLM_V0_M3R_READ_PRESERVING_GROWTH_NOT_SUPPORTED
```

M3R established that immutable root routing and function-preserving lineage birth remove the global read-ownership failure seen in M3, but the lineage-local parent/child split remains insufficiently selective and absolute retention remains far outside the registered target.

## Question

This diagnostic asks one narrow question:

> Inside a lineage that already has stable root ownership, is the old-A versus child-birth-domain functional split actually recoverable from the current frozen query representation, from local write/effect geometry, or from neither?

It does **not** retrain Native CLM, spawn new Cells, update certificates, or consume new formal seeds.

## Canonical inputs

The diagnostic is bound to the already-published M3R evidence:

- M3R formal-result commit: `986b043a5d2f5ee9140cf35b14f68aacc3b7a942`
- M3R formal seeds: `73611 / 73612 / 73613` — already consumed, used here only as checkpoint identities
- M3R protocol SHA-256: `c3e73545899ccf20f54411df701f22dd64b10cb46ff728e862c2d002a94f8627`
- M3R data-manifest SHA-256: `213ddb9d093ea44fd0524e6ba6318f86a61c54270bd5cad6ddeb3233470565b0`
- Hugging Face repo: `archelabsxyz/native-clm-v0`
- Hugging Face revision: `a23b521e137a7e44616809895d44d87cc7d6f87f`

Only the three canonical `lineage_growth` final checkpoints are required for the primary diagnostic. Their SHA-256 identities are taken from the published M3R `model-artifacts.json`.

## Lineage-conditional sampling

A naive A-vs-B/C comparison would mostly measure domain separation at the root level. That is not the M3R failure.

For every spawned edge:

```text
parent -> child
```

we first recover the edge's original root ancestor. A token is eligible for this edge only if:

1. that root is selected by the immutable root Top-K; and
2. for a deeper child, every earlier ancestor decision on the chain would already have routed the token down to the parent.

The probe therefore tests the actual local decision surface that M3R uses.

The negative class is always domain A (TinyStories). The positive class is the child's birth domain inferred from its registered global birth step:

```text
1..400      -> B / WikiText
401..800    -> C / code
801..1200   -> D / Dolly
```

## Registered feature families

### Current cosine rule

The current M3R decision statistic:

```text
q · k_child - q · k_parent
```

is evaluated directly with its fixed direction. This measures the rule we actually used, not an oracle re-fit.

### Query geometry

A held-out linear logistic probe is fit on the frozen normalized query vector `q`.

If this is strongly separable while the current cosine rule is weak, the representation is likely adequate and the lineage-local routing algorithm is the main problem.

### Write/effect geometry

Three diagnostic views are registered:

- `write_input`: Cell operator input `x`;
- `write_left`: downstream gradient `dL/dh_cell_out` at the Cellular Layer output boundary;
- `write_pair`: concatenated normalized `write_input` and `write_left`.

For a local linear Cell update, the per-token write signal is naturally related to the outer-product factors

```text
dW ~ write_left outer write_input
```

so these probes connect the Native CLM diagnostic back to the factorized write/effect geometry found in Core 009.

A fourth view, `certificate_residual`, evaluates the component of the Cell input left outside the parent's protected certificate subspace.

## Registered diagnostic outcomes

This stage produces a classification, not a continual-learning PASS/FAIL claim.

### `QUERY_GEOMETRY_SEPARABLE`

Use a learned lineage-local read gate next. The current cosine key is insufficient, but frozen query geometry already contains a strong boundary.

### `WRITE_EFFECT_GEOMETRY_SEPARABLE`

Separate read and write addressing next. Semantic/query geometry is insufficient, but the current write/effect signal contains a recoverable functional boundary.

### `NO_CLEAR_LOCAL_BOUNDARY`

Neither registered query nor write/effect views expose a sufficiently stable linear boundary. The next experiment should investigate a richer learned functional coordinate rather than another routing heuristic.

### `INCONCLUSIVE_COVERAGE`

Too many parent/child edges lack enough lineage-conditional A/birth-domain samples to support the registered probe.

## Anti-cheating boundary

Domain labels are allowed only inside these offline probes. They must never be supplied to:

- Native CLM routing;
- the write controller;
- growth decisions;
- Cell updates;
- certificate updates.

No output from this diagnostic may be used to retroactively change M3R gates or reclassify M3R as supported.

## Execution

The canonical Kaggle workflow will:

1. fetch the exact published M3R lineage checkpoints from Hugging Face and verify SHA-256;
2. reconstruct the exact M3R A/B/C/D snapshot from its pinned HF revisions and verify every corpus SHA-256;
3. run the three checkpoint diagnostics, using two GPUs concurrently when available;
4. aggregate all valid child edges under the frozen diagnostic thresholds;
5. Git-publish only lightweight JSON/CSV/Markdown evidence.

No new `.pt` file is created or uploaded by this stage.
