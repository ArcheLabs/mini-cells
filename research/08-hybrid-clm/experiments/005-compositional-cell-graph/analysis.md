# Analysis plan

Report the three questions separately:

1. **B1:** paired held-out token-weighted NLL/PPL for S0 versus P0 under matched cell capacity, four visits, data and training-token budget.
2. **B2:** G0 versus R0 on every registered seed, then G0 versus S0 (and S0-shortlist if registered). Include edge-refresh history, fixed-degree perturbations, node/edge utilization, repeated visits, and matched compute. A missing strong-flat rule, edge-change rule, or anti-collapse gate remains `INCOMPLETE`.
3. **System gate:** G0 versus M0 on held-out PPL, per-token unique touched weights, full-model observed GEMM FLOPs, and actual batch-1 KV-cache decode p50/p95. Keep training forward compute and backup visits separate from inference cost; peak allocated memory is not DRAM traffic.

Use paired seed-level differences and paired bootstrap over a frozen held-out sampling unit. Do not compare raw cell IDs across seeds. Do not label unique touched weights as RAM use or DRAM bytes. A successful synthetic composition test is an implementation gate only. Positive results on this from-scratch 61M model do not establish 7B–30B pretrained MoE conversion or phone deployment.

The content-addressed analyzer reads `seed-<n>/<arm>/metrics.json` under a source-SHA archive and emits a separate immutable JSON report. It reports the experiment as incomplete until all registered confirmation seeds and the machine-readable protocol are present. An archive session being complete is not a scientific decision.
