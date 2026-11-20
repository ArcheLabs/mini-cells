# Historical Research Ownership Map

The original numbered folders grouped material by stage. They have been
consolidated into four project homes. The old folder names remain only inside
each project's `context/` when they help preserve the history and meaning of the
research line; there are no compatibility symlinks.

| Former stage | Current project home |
|---|---|
| `01-foundations`, `02-self-organization`, `03-routing-and-growth` | [`cellular-computing`](../projects/cellular-computing/) |
| `04-continual-learning` | [`continual-learning`](../projects/continual-learning/) |
| `05-language-validation`, `06-native-clm`, `08-hybrid-clm` | [`cellular-language-models`](../projects/cellular-language-models/) |
| `07-safe-model-evolution` | [`safe-model-evolution`](../projects/safe-model-evolution/) |

Experiment-specific assets were moved with their owning study. Repository-level
release assets remain at the repository root, and genuinely shared software
resides under `shared/`. The move changes file location, not scientific status,
result content, run identity, or source commit.

## Research root cleanup

Historical launchers and evidence retain their original contents. Run frozen launchers against their recorded source revision; the table below maps their files to their current home.

| Former path | Current path |
|---|---|
| `research/ham-001` | [`projects/continual-learning/experiments/ham-001`](../projects/continual-learning/experiments/ham-001/) |
| `research/ham-001a` | [`projects/continual-learning/experiments/ham-001a`](../projects/continual-learning/experiments/ham-001a/) |
| `research/ham-002` | [`projects/continual-learning/experiments/ham-002`](../projects/continual-learning/experiments/ham-002/) |
| `research/ham-003` | [`projects/continual-learning/experiments/ham-003`](../projects/continual-learning/experiments/ham-003/) |
| `research/persistent-cognitive` | [`projects/continual-learning/experiments/persistent-cognitive`](../projects/continual-learning/experiments/persistent-cognitive/) |
| `research/portable-cell-v01` | [`projects/continual-learning/experiments/portable-cell-v01`](../projects/continual-learning/experiments/portable-cell-v01/) |
| `research/titans-mlp-qwen3-controlled` | [`projects/continual-learning/experiments/titans-mlp-qwen3-controlled`](../projects/continual-learning/experiments/titans-mlp-qwen3-controlled/) |
| `research/titans-qwen3-06b/cell-content-causality-v1` | [`projects/cellular-language-models/experiments/titans-qwen3-06b-cell-content-causality-v1`](../projects/cellular-language-models/experiments/titans-qwen3-06b-cell-content-causality-v1/) |
| `research/titans-qwen3-06b/layer-location-v1` | [`projects/cellular-language-models/experiments/titans-qwen3-06b-layer-location-v1`](../projects/cellular-language-models/experiments/titans-qwen3-06b-layer-location-v1/) |
| `research/titans-qwen3-06b/p1-1` | [`projects/cellular-language-models/experiments/titans-qwen3-06b-p1-1`](../projects/cellular-language-models/experiments/titans-qwen3-06b-p1-1/) |
| `research/titans-qwen3-06b/p1-2` | [`projects/cellular-language-models/experiments/titans-qwen3-06b-p1-2`](../projects/cellular-language-models/experiments/titans-qwen3-06b-p1-2/) |
| `research/titans-qwen3-06b/p1-2-training` | [`projects/cellular-language-models/experiments/titans-qwen3-06b-p1-2-training`](../projects/cellular-language-models/experiments/titans-qwen3-06b-p1-2-training/) |
| `research/titans-qwen3-06b/p1-3-diagnostic` | [`projects/cellular-language-models/experiments/titans-qwen3-06b-p1-3-diagnostic`](../projects/cellular-language-models/experiments/titans-qwen3-06b-p1-3-diagnostic/) |
| `research/titans-qwen3-06b/p1-4` | [`projects/cellular-language-models/experiments/titans-qwen3-06b-p1-4`](../projects/cellular-language-models/experiments/titans-qwen3-06b-p1-4/) |
| `research/titans-qwen3-06b/p1-5` | [`projects/cellular-language-models/experiments/titans-qwen3-06b-p1-5`](../projects/cellular-language-models/experiments/titans-qwen3-06b-p1-5/) |
| `research/titans-qwen3-06b/p1-meta` | [`projects/cellular-language-models/experiments/titans-qwen3-06b-p1-meta`](../projects/cellular-language-models/experiments/titans-qwen3-06b-p1-meta/) |
| `research/titans/audit` | [`projects/continual-learning/experiments/titans-audit`](../projects/continual-learning/experiments/titans-audit/) |
| `research/titans/phase1` | [`projects/continual-learning/experiments/titans-phase1`](../projects/continual-learning/experiments/titans-phase1/) |
| `research/titans/qwen-mvp` | [`projects/cellular-language-models/experiments/titans-qwen-mvp`](../projects/cellular-language-models/experiments/titans-qwen-mvp/) |
| `research/titans/stage6d` | [`projects/continual-learning/experiments/titans-stage6d`](../projects/continual-learning/experiments/titans-stage6d/) |
| `research/titans/stage7` | [`projects/continual-learning/experiments/titans-stage7`](../projects/continual-learning/experiments/titans-stage7/) |
| `research/08-hybrid-clm/experiments/005-compositional-cell-graph/artifacts` | [`projects/cellular-language-models/experiments/005-compositional-cell-graph/artifacts`](../projects/cellular-language-models/experiments/005-compositional-cell-graph/artifacts/) |

## Residual artifact migration

The ten tracked evidence files left under
`research/08-hybrid-clm/experiments/005-compositional-cell-graph/artifacts/`
move byte for byte to
[the owning experiment's artifacts directory](../projects/cellular-language-models/experiments/005-compositional-cell-graph/artifacts/).
Use the same relative path beneath that directory. Historical runs that pin a
source commit retain their original paths at that commit. The CLM-002 Gate 0
Notebook contains a historical sparse checkout pattern for the old directory;
its frozen launcher is preserved and should be read with its original source
identity. This move makes no change to any run or scientific conclusion.
