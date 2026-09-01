# Core Validation 007 Experiment Adapter

Canonical protocol: `research/validations/core-007-functional-boundary-discovery/protocol.json`.

Implementation: `src/minicells/real_representation_007_*.py`.

Discovery and confirmation are intentionally separate phases. Confirmation is rejected until `winner-lock.json` has been generated from the two frozen discovery seeds and committed.
