# Prior work on memory composition

This note refines the proposed R2 research scope in light of Titans, Nested Learning and TNT.

| Work | Mechanism | Implication for MiniCells |
|---|---|---|
| Titans | MAC, MAG and MAL integrate an online neural memory with attention. The paper also studies the internal MLP depth of the memory and uses input-independent persistent parameters. | Integration and internal depth provide baselines for modular memory. Track the number of independently evolving memory states separately from MLP depth. |
| Nested Learning / HOPE | The Continuum Memory System (CMS) organizes memory blocks by update frequency and context. The paper discusses nested initialization, sequential processing and independent head-wise aggregation. HOPE combines self-modifying Titans with CMS. | Treat initialization dependency, dataflow and aggregation as separate composition choices. |
| TNT | A sequential global memory with large chunks supplies long-range context. Local memory uses smaller chunks and periodic resets for context parallelism. Multiple local resolutions and Q–K projection are studied. | Global-plus-local deep memory is a direct precedent for the proposed parent/child arrangement. Local state lifetimes and reset boundaries matter when comparing it with persistent Cells. |

TNT focuses on chunkwise training efficiency and train–inference chunk resolution. Its reported 17.37× speedup is a time-to-loss comparison against Titans at loss 3.20 in the paper's 150M experiment. The measured comparison is 19.48 hours versus 1.12 hours. Use that result only for the specified comparison. [TNT, §4, Table 1 and Appendix E](https://arxiv.org/abs/2511.07343)

## R2 research scope

1. Establish reference behavior for Titans integration, CMS topologies and TNT global/local memory.
2. Test composition across component count, type, update cadence, lifetime and exchange topology. Use the reference topologies as comparators. Measure recall, retention, isolation, dynamic state bytes and update/recall cost.
3. Investigate query-conditioned routing, portable state, cross-architecture transfer, shared child membership, runtime composition and independent fork/restore. Each is an open MiniCells research question. The scope comparison concerns these papers' demonstrated mechanisms and does not establish novelty across the broader literature.

A useful R2 question is whether homogeneous and heterogeneous memory components can be composed recursively, evolve independently, and be dynamically routed through a common inference interface. This research note leaves the existing P4 gates in effect.

## Primary references

- [Titans: Learning to Memorize at Test Time, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html)
- [Nested Learning: The Illusion of Deep Learning Architectures](https://arxiv.org/abs/2512.24695)
- [TNT: Improving Chunkwise Training for Test-Time Memorization](https://arxiv.org/abs/2511.07343)
