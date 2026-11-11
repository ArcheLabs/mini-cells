# P4 implementation deviations and open dependencies

- M1.0 stays at the experiment root as frozen evidence. New experiments can live in sibling subdirectories without relocating existing files.
- M1.1 geometry comparisons preserve the common M1.0 target scale sqrt(32). Unit directions define the geometry, and the same amplitude reaches the write path in each group. QR at 64 is unavailable in dimension 32.
- The M1.0 full-set loop is Batch Replay. Sequential and Chunk use the frozen learn call with a different time dimension. No algorithm change is required.
- M2 can reuse frozen persistence and fork. Strict dtype, architecture and peer inventory checks require a thin validation layer. Tests must cover correctly rehashed semantic mismatches.
- Equal dynamic capacity cannot be represented exactly by changing only integer hidden width at dimension 32. Report the 248-byte conservative near-match explicitly.
- The latent interface validates shared Memory parameters and projections. Backbone/tokenizer integration, GPU screening and Kaggle M1.0 publication are separate verification items. No technical blocker was found for preparing the CPU workstreams.
- The present scope is audit and design. CPU smoke, M1.1 screening, M2 execution and P4-M3 gates remain pending.
