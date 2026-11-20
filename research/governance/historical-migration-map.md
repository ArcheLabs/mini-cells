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
