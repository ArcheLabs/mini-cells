# HAM-001 — CPU mechanism benchmark

## Repository audit and implementation plan

The existing Qwen3 experiment uses the pinned external Titans implementation via
`p1-meta/p1.py`. Its P1.5 driver runs checkpoint diagnostics using the native
attention wrapper and episodic Titans state; its results are historical evidence,
not standalone HAM results. The research checkout has no validated standalone
Mamba-2 adapter for these projected K/V inputs. A Mamba-2 backbone dependency alone
would not establish how to map this protocol to a genuine Mamba-2 input and state.

HAM adds an isolated NumPy benchmark rather than changing existing configs,
tokenizers, wrappers, or training entrances. The synthetic protocol and baseline
parameters live in benchmark.py. Model-free local/full attention use fixed
scaled cosine softmax; the Full KV reference uses hard retrieval with latest
write tie breaking. Neither is a complete Transformer. Delta and Gated Delta
use fixed gates and optional key normalization.

Run on CPU:

    python -m unittest -v
    python benchmark.py /tmp/ham-001-new-run

H0 covers deterministic generation, causal step, batched read, multi-head state,
serialization round trip and fork isolation for implemented backends. B1/B4 are
explicitly unavailable. Multi-hop is deferred until the initial baseline matrix
is complete. Timing and persistent state bytes are measured; matched resource
claims, FLOPs and confirmation runs are pending.

Next validate a Titans adapter against native forward/store and a real Mamba-2
reference or document why it cannot support this interface. Freeze the remaining
task and resource protocols before H1 acceptance. H2 requires H1 acceptance;
H3/H4 require H2 evidence. H5 remains gated. P1.6/P1.7 remain separate.

### Optional B4: pinned Titans

Use the Titans checkout and CPU dependencies from the P1.5 launcher, with
`HAM_TITANS_UPSTREAM=/path/to/titans python -m unittest -v` before running
`python benchmark.py /tmp/ham-001-with-titans --titans-upstream /path/to/titans`.

The adapter calls the original `store_only` and `read_only`. Inputs are synthetic
projected K/V, with P1-style address scaling; chunk and update batch size are one
to satisfy HAM's token interface. This is a fresh seeded standalone memory with
synthetic geometry, not a replay of Qwen checkpoints. B4 uses FP32; the NumPy
arms use FP64. State bytes count all exported tensors, including duplicated
fields, and are not allocator or matched-budget measurements.
