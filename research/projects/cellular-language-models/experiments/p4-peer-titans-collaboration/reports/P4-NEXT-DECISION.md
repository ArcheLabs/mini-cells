# P4 next-stage decision

M1.1 CPU smoke and M2 latent CPU isolation pass. The evidence supports broader screening before a peer composition experiment.

- At 128 exposures per association, Gaussian Batch Replay reaches 100% mean Recall@1 through 16 bindings, 46.88% at 32 and 5.21% at 64. QR Batch at 32 reaches 29.17%. QR at 64 exceeds dimension 32 and is omitted.
- Sequential, Chunk 4 and Chunk 8 each reach 100% on all three 64-binding Gaussian mappings. They use 8,192, 2,048 and 1,024 inner calls respectively versus 128 for Batch Replay. This smoke isolates exposure, while update count varies with chunk width. The measured operating range depends on budget, geometry and acquisition schedule.
- After learning two additional four-binding groups, the first group has zero retention in all nine phase-order runs. Final recall of the newly acquired group and all earlier group trajectories are recorded in the manifest.
- At 64 bindings, hard-negative FPR is 71.74% for Sequential, 52.34% for Chunk 4 and 36.85% for Chunk 8 at the fixed 0.8 threshold. Batch's zero hard-negative FPR coincides with weak recall. Review known acceptance alongside false positives before any threshold choice.
- A/B and forks have disjoint dynamic storage, preserve peer outputs under bidirectional writes and restore bitwise in a new process. Forks acquire distinct associations.
- The two-Cell byte count and conservative union width are known. Target-device runtime and screening uncertainty remain unverified. P4-M3 remains pending.

The base screening budget is 460,800 inner calls. CPU smoke acquisition rates project 13.07 minutes of acquisition. Full wall time also includes rejection curves, persistence, extended budgets and interference analysis. GPU costs require synchronized measurement on the target device.
