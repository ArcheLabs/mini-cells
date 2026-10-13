# Experiment 006 Kaggle real-data development pilot

This is real TinyStories GPU training, but an UNREGISTERED DEVELOPMENT PILOT.
It does not modify formal train/profile/publish gates, access confirmation
data, grant a seed approval, or claim registered scientific evidence.

Use the sibling notebook-dev-pilot.ipynb on Kaggle with Internet On and GPU T4 x2.
It pins source SHA, downloads only the first train Parquet shard at the exact
dataset revision, computes real SHA256 checksums, builds a train-only BPE
tokenizer, and verifies training/development story identities before training.

Defaults are a 16M candidate, six operations per token, a 1M-token pilot,
paired R0/R1 workers and seed 6060611. Switch the pair to R2/M0 for
the second comparison. Change TRAIN_TOKENS to 10000000 for a separate full
development pilot: use a new output workdir because the schedule differs.
Never promote the pilot seed to untouched formal confirmation.

Kaggle outputs include pilot-provenance.json, pilot-config.json, verified
development assets, per-arm model/optimizer/RNG checkpoint, learning_curve.json,
metrics.json and pilot-decode-profile.json. Pilot weights/results remain in
Kaggle outputs; no Git result publication is attempted without formal registration.
Resume with the same source, data, token budget and seed. Save Kaggle outputs
before the session expires to preserve checkpoints.

Formal confirmation is still sealed. Once the original registered evidence
and frozen manifests really exist in GitHub, use the original notebook.ipynb.
