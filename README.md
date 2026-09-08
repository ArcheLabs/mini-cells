[English] | [中文](README.zh-CN.md)

# MiniCells

MiniCells explores whether pretrained language models can be transformed into
Cellular Language Models through independently manageable neural Cells.

**HybridCLM is the current bridge:** a pretrained MoE remains the mature
computational substrate while trainable, reversible Cell mutations are attached
inside the model. The long-term objective is progressive dependency withdrawal
from the original model toward a standalone CLM. HybridCLM may also become a
modular post-training mechanism if controlled comparisons show a genuine
advantage over established PEFT methods.

This repository ships the public HybridCLM toolkit and preserves the historical
MiniCells research record. It does not claim that MoE → standalone CLM,
continual learning, catastrophic-forgetting control, or HybridCLM superiority
to LoRA has been solved.

## Engineering Evidence · Formal Validation Pending

PCU Hybrid Reattachment 001 v3 currently records:

- same-cellular zero-state equivalence passed;
- Cell OFF ranking 6.25%, Cell ON ranking 82.03%;
- causal hybrid consumption and exact restoration supported;
- the alpha=1 locality threshold failed;
- a coarse amplitude sweep found no pre-registered jointly passing point;
- formal validation has not run.

These are engineering boundaries, not a formal HybridCLM result.

## Public API

```python
from minicells import CellMutation, CellPlacement, HybridCLM

hybrid = HybridCLM.from_pretrained(
    "ibm-granite/granite-3.1-1b-a400m-base",
    revision="<immutable-hub-commit>",
)
hybrid.cellularize(CellPlacement(layer=7, experts="all"))
mutation = CellMutation.from_pretrained("<mutation-artifact>")
hybrid.attach(mutation)
hybrid.set_alpha(mutation, 0.75)
```

Install the optional Granite/Hugging Face integration with
`pip install "mini-cells[hybrid]"`. v0.1 supports explicit placement,
inspection, safe safetensors mutation artifacts, attach/detach, alpha scaling,
zero-state diagnostics, and rollback reporting. Only the tested Granite MoE
backend is supported; unknown architectures fail closed.

## Repository map

- [`src/minicells/hybrid/`](src/minicells/hybrid/): public HybridCLM API.
- [`docs/hybrid-clm/`](docs/hybrid-clm/): API, artifact, placement, safety, and status docs.
- [`research/stages/08-hybrid-clm/`](research/stages/08-hybrid-clm/): current research boundary and roadmap.
- [`research/`](research/README.md): historical protocols, reports, and evidence catalog.
- [`artifacts/`](artifacts/): durable evidence and release assets.
- [`tests/`](tests/): unit, integration, research, and public fixtures.
- [`docs/integrations/minijam.md`](docs/integrations/minijam.md): MiniCells/MiniJAM ownership boundary.

## Research status and non-goals

Automatic optimal placement, generic MoE support, standalone conversion, new
router training, formal HybridCLM execution, LoRA superiority, production JAM
execution, and a general continual-learning solution are explicitly out of
scope for this prerelease.

## License

See [LICENSE](LICENSE).
