[English] | [中文](README.zh-CN.md)

# MiniCells Research

Research is organized by long-lived project. Each project owns its questions,
experiment records, code, notebooks, reports, and project-specific results.


## Projects

| Project | Focus |
|---|---|
| [Cellular Computing](projects/cellular-computing/README.md) | Cellular computation, self-organization, routing, growth, and sparse execution. |
| [Continual Learning](projects/continual-learning/README.md) | Continual-learning mechanisms, validation, and known limits. |
| [Cellular Language Models](projects/cellular-language-models/README.md) | Native and hybrid cellular language-model architectures and language validation. |
| [Safe Model Evolution](projects/safe-model-evolution/README.md) | Safe, measurable changes to pretrained models. |
| [CLM Native](projects/clm-native/README.md) | Bounded-candidate dynamic sparse computation and causal/cost validation. |

## Shared research governance

[`governance/`](governance/README.md) contains the experiment registry, frozen
protocol records, audits, and historical indexes. These records govern status;
project folders own the research itself.

## Research discipline

- Keep every experiment under exactly one project.
- Preserve frozen protocols, source commits, run identities, and negative results.
- Keep experiment outputs with their owning experiment; repository-wide release
  artifacts remain under the top-level `artifacts/` tree.
- Shared code belongs under [`../shared/`](../shared/); project-specific logic
  stays with its project or experiment.
- A successful run, useful mechanism, or visualized path does not by itself
  establish a scientific claim.
