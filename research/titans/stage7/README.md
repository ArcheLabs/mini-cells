# Stage 7 development runner

Use development_t4.ipynb with Tesla T4, Internet and GITHUB_TOKEN.
The notebook streams progress, runs clean-slate episodic training and
evaluation, and publishes a development report with verified Git SHA.
The CLI accepts an identity-bound checkpoint through --resume.

The development run includes paired direct recall, rejection calibrated
on validation, token-bucket write diagnostics, contradictory writes,
distractors, a bounded native baseline and a read-time span ablation.
The fresh worker re-encodes queries after the source exits. The optional
answer-token fusion diagnostic runs only after the retrieval gates pass.
Runtime artifacts are retained in Kaggle Output.

Full acceptance remains open. Current limitations include incomplete
multi-Cell composition, distractor coverage on one episode, no separately
trained TNT arm, and fusion without base-language regression measurement.
The independent worker compares shard readouts. It does not yet certify
the full post-restore accuracy gate. T4 runtime remains unverified.
