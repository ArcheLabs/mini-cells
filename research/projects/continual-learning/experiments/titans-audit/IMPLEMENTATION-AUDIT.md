# Titans roadmap Phase 0

The frozen EPCLM-004 M3 memory is the implementation reference for both P4
and MiniCells 0.8. The existing finite-difference M3 gate checks the outer
gradient. Bridge v2 reran the historical P4 runners and reproduced both
small aligned recall and old-group collapse. Those runs serve as historical
reproduction evidence. Phase 1 still needs a unified-geometry capacity report
and the full common metric set before the mechanism ablations.

The implementation retains the core gradient, momentum and forgetting
recurrence. Its scalar controls and mean-gradient chunk transition are
simplifications whose impact remains an empirical question. Preserve frozen
code and compare new variants in isolated research runners.

See UPDATE-MECHANISM.md, REPRESENTATION-PIPELINE.md and TITANS-PAPER-COMPARISON.md
for the code-to-method comparison.
