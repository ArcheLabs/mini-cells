---
library_name: mini-cells
tags:
  - minicells
  - hybrid-clm
  - moe
---

# MiniCells HybridCLM Cell — Granite 3.1 1B A400M / L7 / K64

**Artifact type:** Cell mutation only. Foundation weights are not included.

- Base model: `ibm-granite/granite-3.1-1b-a400m-base`
- Immutable base revision: `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`
- MiniCells version: `0.2.0a1`
- Release tag: `v0.2.0a1`
- Cell placement: layer `7`, budget `64`
- Engineering seed: `26090501`

## Evaluation summary

- Ranking OFF: `0.062500`
- Ranking ON: `0.820312`
- Ranking gain: `0.757812`
- Same-graph zero-state: pass
- Restoration: pass
- Locality at alpha=1: unresolved under the frozen gate

Scientific status: **Engineering Evidence · Formal Validation Pending**.
Formal seeds remain reserved and untouched. This release does not claim
standalone CLM conversion, superiority to LoRA, or solved locality.

## Installation and attach

```python
from minicells import CellMutation, HybridCLM

hybrid = HybridCLM.from_pretrained(
    "ibm-granite/granite-3.1-1b-a400m-base", revision="408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"
)
mutation = CellMutation.from_pretrained("<HF_MODEL_REPO>")
hybrid.cellularize(mutation.placements)
hybrid.attach(mutation).set_alpha(mutation, 1.0)
```

License relationship: MiniCells is Apache-2.0; the foundation model
remains under its own upstream license.
