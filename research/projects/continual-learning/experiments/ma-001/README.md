# MA-001 — Memory Addressing Feasibility

Owner: Continual Learning. [Experiment protocol](PROTOCOL.md).

This directory contains the first implementation of the mock registry,
deterministic write routing, atomic index transactions, and ranking. No actual
Cell runtime or model is trained. See the completed synthetic CPU evaluation in [RESULTS.md](RESULTS.md).

Run from this directory with Python 3.11+, Torch and Transformers:

    python scripts/run_experiment.py --mode all --output /tmp/ma001-run

The pinned multilingual E5 encoder must be cached locally or explicitly
acquired with `--download`. No lexical substitute is used for Dense.
Each output directory is a separate run. The runner records input hashes and
fits thresholds on the 24 development queries before reading the 56 holdout
queries. The 10-cell, 120-event corpus is synthetic and reuses per-Cell context.
The evaluation does not establish production routing reliability.
