# Experiment 019 stable rerun and checkpoint policy

The original Experiment 019 run is preserved as an invalid numerical-oracle run. The corrected official rerun uses `scripts/research/run_proposal_utility_discovery_stable.py` and writes to `results/proposal-utility-discovery-stable-v1`.

## Scientific invariants

The stable rerun keeps the original seeds, Phase-1 recipe, six skill families, donor adaptation, epsilon, feature definitions, strong leave-one-family-out split, estimator definitions and preregistered thresholds. The only oracle change is a forward-equivalent numerical rewrite of the gated replicator so autodiff remains finite at recruitment `e=0`.

## Checkpoint protocol

The stable worker stores Kaggle-local checkpoints under:

`results/proposal-utility-discovery-stable-v1/checkpoints/`

Per replicate it stores:

- one Phase-1 checkpoint;
- six trained one-cell donor checkpoints;
- one RANDOM control checkpoint.

With three replicates the complete cache contains 24 checkpoint files. Each donor checkpoint contains the exact model state, the pre-adaptation `LocalizedLearningState`, donor summary and structural events. The Phase-1 checkpoint also stores validation starts and Phase-1 diagnostics.

Checkpoints are written atomically. They are for recovery/re-measurement and are intentionally not published into the GitHub result artifact.

## Commands

First official stable run:

```bash
python scripts/research/run_proposal_utility_discovery_stable.py
```

If the run is interrupted after some Phase-1 or donor trainings, run the same command again. Completed checkpoints are restored and only missing training plus oracle/feature measurement is executed.

To deliberately ignore the cache:

```bash
python scripts/research/run_proposal_utility_discovery_stable.py --force-retrain
```

If all stable worker CSV/JSON outputs already exist and only estimator/plot/decision postprocessing changed:

```bash
python scripts/research/run_proposal_utility_discovery_stable.py --postprocess-only
```

If oracle epsilon or feature extraction changes but donor learning does not, rerun the normal stable command. The 24 model checkpoints are reused and the observation matrix is remeasured without Phase-1/donor retraining.

## Provenance

The original failed run remains in `results/proposal-utility-discovery-v1` during the Kaggle session and must not be silently rewritten into a valid preregistered result. Stable results use the separate `proposal-utility-discovery-stable-v1` path.
