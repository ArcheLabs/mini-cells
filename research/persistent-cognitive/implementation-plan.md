# Implementation sequence

1. Freeze tensor shapes, dtype, layer/head mapping and artifact allowlists.
2. Implement the new memory core in its own module. Verify its update
   and addressing equations against independent small-tensor references.
3. Integrate with pinned Qwen while preserving local attention, native
   gating and the Gated DeltaNet path. Verify causal output parity.
4. Implement Cell create, commit, export, restore and fork. Test that
   Session tensors and legacy fast weights are rejected. Verify restore
   after source process exit with an empty Session.
5. Implement a new training pipeline and independently version its slow
   weights. Keep TNT's hierarchical training and target inference
   adaptation explicit in any TNT comparison.
6. Run local unit and integration tests, then review the architecture
   before allocating training GPU time.

Execution status: audit and ownership boundaries recorded. Code contracts,
model implementation, tests and final architecture review remain open.
