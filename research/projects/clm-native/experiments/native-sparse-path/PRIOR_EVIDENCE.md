# M0 historical evidence review

The raw audit is [artifacts/m0/prior-evidence.json](artifacts/m0/prior-evidence.json). Archived hashes and original source identities are checked; historical GPU training was not rerun. An archived summary is weaker evidence than independent reproduction. Missing raw traces and any manifest mismatch are recorded explicitly, not repaired by rewriting frozen evidence.

| Study | Archived evidence inspected | Implication for this study |
|---|---|---|
| 003 hard transition | source 819df6113ca3; seed-42 metrics, worker logs, model code/dispatch | soft/hard completed, fixed/router controls incomplete; no full-matrix support. Count dispatch and avoid per-Cell Python loops. |
| 004 graph | source a59402238b68; seed-42 reference metrics and protocol | soft-cell PPL 12.1822 vs fixed 12.2870, but batch latency 88.48 vs 23.15 ms; development session only. Quality alone is not efficiency. |
| 007 30M | source 6860d3525a6f; decision, CSV, metadata, task spec; src/minicells/language_30m.py | PPL 5.35324 vs 5.32900; throughput 12605 vs 28392 tokens/s (~2.25× slower). Archived numeric claim verified, no historical retraining. |
| 015 topology | source ae7eec4f4baf; three worker replicates, decision, task spec; src/minicells/language_sparse_topology.py | dynamic/dense NLL ratio 1.474; partial topology signal, quality failed. Similarity einsum computes full row pairs; do not inherit it as sparse-cost code. |
| 015c | source bc059607c04d; workers, decision, protocol; src/minicells/language_emergent_topology.py | topology emergence passed; causal topology/plasticity failed; shuffled-topology causal replicates 0. A picture is not H1/H2 evidence. |
| 016 growth | source 77ab5765dfa3; workers, decision, task spec; src/minicells/language_growing_organism.py | G/T wall cost 5.888×, language retention failed. Structural change is insufficient; growth and lifecycle omitted here. |

The old TextNCA recurrent/graph operators are useful evidence of scheduling and state-IO costs, but would import fixed local/dense computation assumptions. This study therefore reuses proven infrastructure (seed/environment, atomic JSON, 005 Checkpoint/RNG/identity utilities, historical Git/Askpass publisher), while implementing the new bounded-candidate mechanism separately. No historical implementation or result is overwritten. No CLM-Bridge conversion training is run.

Source-level follow-up: 003 routes through `src/minicells/models/routing_research.py` into `routing/hard_transition.py`; 004 adapts that backbone in `models/graph_cell.py` and `routing/graph_router.py`. The latter explicitly iterates `for destination in range(experts + 1)` in backup dispatch. It is retained as historical evidence, not reused as this project's sparse performance executor.
