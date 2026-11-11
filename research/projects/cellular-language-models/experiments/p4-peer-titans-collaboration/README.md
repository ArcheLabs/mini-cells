# P4 Peer Titans Collaboration

M0 audits the frozen M3 implementation and verifies the original archive. M1 measures one Cell in a latent association task. See the [implementation plan](docs/P4-IMPLEMENTATION-PLAN.md) and [CPU results](docs/P4-RESULTS.md).

Run `python run_m1.py --output /tmp/p4-m1-new` from this directory with the existing PyTorch environment. The output directory holds the manifest and independent Cell checkpoints. Render the report with `python report.py /tmp/p4-m1-new/run-manifest.json`.

The CPU run qualifies the single Cell development gate. Target GPU screening and subsequent Peer Cell phases are separate milestones.

- [Titans learning, forgetting, and Memory Policy proposal](docs/TITANS-MEMORY-POLICY.md)

## P4-M3 Stage 3

The dedicated Peer Titans Screening runner and config are under
[m3-composition/peer-titans](m3-composition/peer-titans). See the
[Screening report](reports/M3C-PEER-TITANS-SCREENING-REPORT.md) and
[run evidence](m3-composition/peer-titans/runs/screening). Use the
locked PyTorch CPU environment and a new output directory.

```sh
python m3-composition/peer-titans/screen.py /tmp/p4-stage3-calibration-new --calibration
python m3-composition/peer-titans/screen.py /tmp/p4-stage3-official-new
python m3-composition/peer-titans/analyze.py /tmp/p4-stage3-official-new /tmp/p4-stage3-evidence-new /tmp/p4-stage3-report-new.md
```

The runner terminates the acquisition process before restoring Cell checkpoints
and comparing outputs in a fresh interpreter. The four calibration episodes
have a separate seed namespace and output directory. The official dataset
contains 80 paired episodes. Checkpoint tensor files stay in the local run
output. The committed evidence includes hashes and complete episode records.
