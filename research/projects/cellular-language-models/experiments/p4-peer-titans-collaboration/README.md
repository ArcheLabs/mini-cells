# P4 Peer Titans Collaboration

M0 audits the frozen M3 implementation and verifies the original archive. M1 measures one Cell in a latent association task. See the [implementation plan](docs/P4-IMPLEMENTATION-PLAN.md) and [CPU results](docs/P4-RESULTS.md).

Run `python run_m1.py --output /tmp/p4-m1-new` from this directory with the existing PyTorch environment. The output directory holds the manifest and independent Cell checkpoints. Render the report with `python report.py /tmp/p4-m1-new/run-manifest.json`.

The CPU run qualifies the single Cell development gate. Target GPU screening and subsequent Peer Cell phases are separate milestones.

- [Titans learning, forgetting, and Memory Policy proposal](docs/TITANS-MEMORY-POLICY.md)
