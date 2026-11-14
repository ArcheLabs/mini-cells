# Stage 7 development runner

The development notebook runs clean-slate adapter training, paired retrieval,
write-granularity and contradictory-write diagnostics, and a CPU artifact
round trip. It streams worker output and reserves time for summary publication.
The CLI accepts an identity-bound checkpoint through `--resume`.

The current artifact check consumes previously encoded queries while the source
process remains alive. Fresh Qwen re-encoding after source termination remains
required for the persistence gate. The distractor check is a single episode.
The full distractor ladder, open-set rejection, native Qwen baseline, TNT
ablation and gated fusion are pending. This notebook is a development entry
point. The MVP decision remains incomplete.
