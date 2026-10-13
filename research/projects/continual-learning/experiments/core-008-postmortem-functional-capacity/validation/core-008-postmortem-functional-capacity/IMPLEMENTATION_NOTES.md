# Implementation notes

The postmortem intentionally distinguishes geometric upper bounds from budget-matched controls.

- Per-write SVD and global PCA are not practical Cell parameterizations; they measure capacity ceilings.
- The factorized offline dictionary is constrained to 32 total rank units, matching Core 008's conceptual 4096 factor-scalar budget at `d=64`.
- The factorized dictionary is allowed to see all training write demands and has no certificate or deployable router. Therefore it is an optimistic capacity control. If it fails, the online certified mechanism has little reason to succeed under the same factor budget.
- All reported residuals use normalized `G` matrices and local action `Z G^T`, consistent with the Core 008 post-preflight metric choice.
- The hidden-state cache is operational only and is excluded from published artifacts.
