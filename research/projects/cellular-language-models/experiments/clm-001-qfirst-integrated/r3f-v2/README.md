# R3f-v2 — Local Trust-Region Connection Update

Independent experiment: `CLM-001-R3f-v2-Local-Connection-001`.

## Run

Import `notebooks/kaggle-development.ipynb` into the existing `bingjiangli/minicells` Notebook. Enable Internet and the existing `HF_TOKEN` and `GITHUB_TOKEN` secrets, select the T4 environment, and Run All once. The launcher does not run the CPU test suite, a Pilot, a Smoke or the historical regression suites in Kaggle.

The runner restores the pinned private historical manifest, imports only verified post-rejection **Step 100** official A-C and B-C states, and gives the new experiment a separate identity. It never substitutes Step 110, regenerates historical training, reads Development/Confirmation, or releases a 500-step matrix. If Step 100 evidence is unavailable, it stops that arm with an explicit failure and still checks the other arm.

Each arm generates one new candidate: 12 Train counterfactual states, original all-action rerouted terminal CE, original AdamW/objective and 8 Connection updates. This is **not** claimed to reconstruct the lost historical proposal. The full unscaled candidate, official state, optimizer, RNG, samples and parameter delta are saved and read back through the existing immutable Store and private HF transport **before the first scale evaluation**.

The 22 scales are `0` and `2**(-k)` for `k=0..20`. Admission uses the first 8 existing frozen **Admission** windows. Calibration is used only for the K=10 old/new Cell adaptation quality decision, with the historical gain threshold 0.001. Three original trust upper bounds remain unchanged. Nonempty admission requires a centered score change beyond numeric tolerance, changed hard actions and at least one changed full path. The 1% path-coverage statistic is descriptive only.

The paired adaptation transaction, Cell optimizer and Gumbel/epsilon streams reuse R3f. The 10 actual update pairs record input hashes, training action/path differences, Cell calls, gradient norms and the epsilon-mixture policy KL separately from deterministic inference differences. Proposal/scanning/observation/retained and discarded adaptation costs are reported separately; instrumentation is restricted to this bounded mechanism experiment.

A-C and B-C execute sequentially on their assigned devices to serialize private storage writes. Neither arm failure skips the other. No runtime termination timer is introduced. The work budget controls resource use.

## Recovery and publication

Run All with the same pinned launcher restores the new experiment's private HF pointer and reuses the archived candidate and completed transaction. No archived candidate is reoptimized and no committed adaptation pair is repeated. Every local execution is separated by an execution identity; the HF reference records the resumable experiment identity.

Public GitHub publication stages only the named aggregate reports under this research unit, on a main-descended results branch. Weights, sampled hidden states, training tokens and optimizer states stay private. A push is successful only after the remote SHA matches the local result commit. Publication can be retried without invoking training:

```bash
python scripts/run_r3f_v2.py --publish-existing /path/to/retained-output
```

Save Version retains the output if either private or public publication fails. A new cloud session cannot recover local-only unuploaded state automatically; attach the saved Output or restore its retained directory before retrying. Do not discard it on storage failure.

## Interpretation

- H1: a real candidate has a stable, nonempty Admission update.
- H2: an admitted candidate completes both K=10 branches and Calibration evaluation.
- H3: Calibration gain is at least 0.001. H3 is NOT_TESTED if adaptation did not run.

No engineering pass, raw parameter difference, surrogate objective improvement or path picture establishes quality benefit. No Development or statistical confirmation claim is made.

## Engineering validation

`python .../r3f-v2/checks.py` runs only synthetic CPU fixtures. It covers explicit parent import, rejection of Step 110 substitution, archive-before-scan and candidate reuse, all 22 scales, strict nonempty admission, fixed Admission windows, paired K=10 updates, interrupts at pairs 1/5/10 and exact restart, committed replay, and independent hypothesis statuses. Real parent receipts, candidate hashes, scan outcomes, timing and NLL are produced by Kaggle; engineering reports do not fabricate them.
