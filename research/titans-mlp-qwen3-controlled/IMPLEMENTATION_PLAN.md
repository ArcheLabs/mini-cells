# Implementation plan

1. CPU recurrence and synthetic capacity matrix, followed by pinned Qwen audit.
2. Extract and cache layer 23 train, validation and test representations.
3. Train independent addressing projections and adaptive controllers on
   episodes, starting with Smoke. Require validation recall improvement.
4. Evaluate semantic controls with disjoint held-out facts and templates.
5. Verify symmetric fresh-process migration on Correct, Donor and Empty.
6. Evaluate language integration only after migration acceptance.
