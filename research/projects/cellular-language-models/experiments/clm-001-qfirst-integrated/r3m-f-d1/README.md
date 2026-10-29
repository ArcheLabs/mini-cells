# R3m-F-D1: automatic training and GitHub results

Import [`notebooks/kaggle-r3mf-d1.ipynb`](notebooks/kaggle-r3mf-d1.ipynb) into the existing Kaggle Notebook. Select **T4 x2**, enable **Internet** and both existing secrets **HF_TOKEN** and **GITHUB_TOKEN**, then use **Save Version → Save & Run All** once. `RUN_BCD=True` is already set under the human authorization; no manual mode or hash is required.

The flow is CPU identity/data preflight → bounded GitHub write check → B/C/D training → result validation → Git commit/push → remote commit verification. Missing write access or a failed return-path check stops before GPU work. B/C/D each retain at most 200 logical updates, with evaluation at 0/50/100/200. A reuses the old result. Failed-attempt cumulative work is distinct from logical trajectory progress.

GitHub receives the full `RESULTS.json`, a runtime manifest snapshot, all three arm reports, budget proof, sample plan, reused A predictions, CPU preflight identity and an artifact hash index. Results are committed under this experiment's `reports/published/<run_id>/` on a `results/r3mf-d1-*` branch derived from current main. Logs print the report URL and verified remote commit SHA. Checkpoint weights remain in `/kaggle/working/r3mf-d1-output`; no weights or workflow edits are pushed to GitHub. The full R3m-F Pilot, Main and Confirmation are outside the authorization.

A successful run ends with `publication.status=PUBLISHED` and `remote_verified=true`. Local file creation, a local commit or a successful push alone are not reported as verified return. Kaggle Output `PUBLICATION.json` and the updated runtime `RUN_MANIFEST.json` contain the final remote SHA. The remote manifest is an immutable payload snapshot; the publication commit itself is identified through Git metadata and the runtime receipt.

If training is complete but publication fails, retained reports and checkpoints stay intact. Re-running the same Notebook in the same session retries publication without restoring a model or adding optimizer updates. `run.py --mode publish-existing --output <retained-run-or-output-root>` also publishes prior D1 runs without training. A fresh session needs the retained Output mounted/restored before it can publish those files; it must not silently substitute a fresh training run for missing output.

`KAGGLE_API_TOKEN` is not required. See [PUBLICATION.md](PUBLICATION.md) for the reused components, payload validation, failure states and CPU Git round-trip evidence.
