# R3i — Stable Co-Adaptation

Import `notebooks/kaggle-development.ipynb` into the original Kaggle Notebook. Select T4 ×2, enable Internet and existing HF_TOKEN/GITHUB_TOKEN, Run All once.

The fixed R3h private revision supplies historical H/T controls at steps 0,100,500. New CH/CT start from exactly the saved step0 model and optimizer, verified against the common R3c D500 identity. First-proposal step100 must reproduce the historical model, Cell optimizer, data cursor/RNG, routing settings and full saved CE/exposure curve exactly; no tolerance relaxation.

Twelve new models, 500 main Cell steps each. Proposals at100/200/300/400 reuse the R3f-v2 optimizer and 22-scale Admission scan. Successful admission enters paired K10 updates; selected updates count inside500, discarded updates count separately. Calibration acceptance is0.001. Four-cell/Connection crossover is descriptive, not a selection rule.

Candidate and transaction state is persisted before scanning. One coordinator writes HF; independent GPU workers wait for its acknowledgment. Every transaction remains immutable and resumes at its saved pair. Development is mapped only after all12 trained states pass the terminal barrier. Confirmation is sealed.

Re-running this same pinned Notebook restores the verified HF run. For publication recovery use `--report-existing --output <existing directory>`; this performs no training. Keep Kaggle Save Version Output as a second copy. Public reports contain aggregates, not weights or training tokens.

CPU tests exercise actual candidate/22-scale execution on tiny synthetic models, repeated accepted updates, committed-transaction replay, the job/checkpoint/report path, and historical mismatch rejection. No GPU test or real training is performed during implementation. Real T4 reproducibility and runtime remain to be observed; no hard time termination is installed.
