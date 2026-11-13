# Phase 1 closure status

OPEN. Historical reproduction is available under the Titans Bridge v2
experiment directory. The new native M3 screen uses the historical
normalized Gaussian generator at all four loads. It pairs data and orders
across methods and records failures as rows. See screen.py for the
configuration and PHASE1-CAPACITY-RESULTS.md for current evidence.

Implementation differences include the runtime/pilot token selection and
query context. Algorithmic differences include mean-gradient updates
versus tokenwise recurrence from chunk-start gradients. The effects on
held-out retrieval, the loss ablations, GPU cost, and confirmatory
confidence intervals remain unresolved. Phase 2 is gated on closure.
