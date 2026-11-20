# MA-001.4 — Multi-Cell Runtime Feasibility

Owner: Continual Learning. See [current status](RESULTS.md).

Run from this experiment directory with CPU PyTorch and the dependencies for
the HAM-001 adapter:

```sh
python run_experiment.py PINNED_TITANS_CHECKOUT NEW_OUTPUT --mode all \
    --index MA001_1_SAVED_INDEX --e5 PINNED_E5_SNAPSHOT
```

The fixture reuses the three target IDs of the frozen MA-001.1 query
(MiniJAM, MiniCells and AI Memory), rather than relabelling an index to force
a match. This is a fixture variation from the proposed JamScript example.
Use the original MA-001.1 writer to generate the matching index. No weights
are trained. An address case without its inputs is reported as blocked.
Checkpoint and migration workers use fresh processes. They compare
per-Cell read tensors and state hashes before and after the next write.
